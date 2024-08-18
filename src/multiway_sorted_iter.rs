use crate::box_iterator;
use compare::Compare;
use std::cmp::{self, Ordering};
use std::iter::Peekable;
use std::marker::PhantomData;
use std::mem::MaybeUninit;

/// Peekable iterator whose `peek()` does not mutate self. Each
/// `IndexedPeekedIterator` admits an index of type `usize`, which will be
/// emitted with the inner iterator items.
struct IndexedPeekedIterator<I: Iterator> {
    inner: I,
    buf: Option<(I::Item, usize)>,
}

impl<I: Iterator> IndexedPeekedIterator<I> {
    fn new(mut inner: I, index: usize) -> Self {
        let buf = inner.next().map(|item| (item, index));
        Self { inner, buf }
    }

    fn peek(&self) -> Option<&(I::Item, usize)> {
        self.buf.as_ref()
    }
}

impl<I: Iterator> Iterator for IndexedPeekedIterator<I> {
    type Item = (I::Item, usize);

    fn next(&mut self) -> Option<Self::Item> {
        match self.buf.take() {
            None => None,
            Some((inner_item, index)) => {
                if let Some(next_item) =
                    self.inner.next().map(|inner_item| (inner_item, index))
                {
                    self.buf.get_or_insert(next_item);
                }
                Some((inner_item, index))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (min, max) = self.inner.size_hint();
        match self.buf {
            None => (min, max),
            Some(_) => {
                (
                    // We always need to add 1 to the lower bound in order to
                    // take into account the buffered item.
                    min + 1,
                    // When the upper bound of `self.inner` is tight, we will
                    // need to add 1 to the upper bound, taking account the
                    // buffered item.
                    max.map(|max| max + 1),
                )
            }
        }
    }
}

#[cfg(test)]
mod indexed_peeked_iterator_tests {
    use super::IndexedPeekedIterator;

    #[test]
    fn test_peek() {
        let v = vec![3, 2, 5];
        let ipi = IndexedPeekedIterator::new(v.into_iter(), 9);
        assert_eq!(ipi.peek(), Some(&(3, 9)));
        assert_eq!(ipi.peek(), Some(&(3, 9)));
        assert_eq!(ipi.peek(), Some(&(3, 9)));

        let v: Vec<i32> = vec![];
        let ipi = IndexedPeekedIterator::new(v.into_iter(), 9);
        assert_eq!(ipi.peek(), None);
        assert_eq!(ipi.peek(), None);
    }

    #[test]
    fn test_next() {
        let v = vec![3, 2, 5];
        let mut ipi = IndexedPeekedIterator::new(v.into_iter(), 9);
        assert_eq!(ipi.next(), Some((3, 9)));
        assert_eq!(ipi.next(), Some((2, 9)));
        assert_eq!(ipi.next(), Some((5, 9)));
        assert_eq!(ipi.next(), None);
        assert_eq!(ipi.next(), None);

        let v: Vec<i32> = vec![];
        let mut ipi = IndexedPeekedIterator::new(v.into_iter(), 9);
        assert_eq!(ipi.next(), None);
        assert_eq!(ipi.next(), None);
    }

    #[test]
    fn test_size_hint() {
        let v = vec![3, 2, 5];
        let mut ipi = IndexedPeekedIterator::new(v.into_iter(), 9);

        let (hmin, hmax) = ipi.size_hint();
        assert!(hmin <= 3);
        if let Some(hmax) = hmax {
            assert!(hmax >= 3);
        }

        ipi.next();
        let (hmin, hmax) = ipi.size_hint();
        assert!(hmin <= 2);
        if let Some(hmax) = hmax {
            assert!(hmax >= 2);
        }

        ipi.next();
        let (hmin, hmax) = ipi.size_hint();
        assert!(hmin <= 1);
        if let Some(hmax) = hmax {
            assert!(hmax >= 1);
        }

        ipi.next();
        let (hmin, hmax) = ipi.size_hint();
        assert_eq!(hmin, 0);
        if let Some(hmax) = hmax {
            assert_eq!(hmax, 0);
        }

        ipi.next();
        let (hmin, hmax) = ipi.size_hint();
        assert_eq!(hmin, 0);
        if let Some(hmax) = hmax {
            assert_eq!(hmax, 0);
        }
    }
}

struct IndexedPeekedIteratorComparator<T, C: Compare<T, T>> {
    inner: C,
    phantom: PhantomData<T>,
}

impl<T, C: Compare<T, T>> From<C> for IndexedPeekedIteratorComparator<T, C> {
    fn from(value: C) -> Self {
        Self {
            inner: value,
            phantom: PhantomData::default(),
        }
    }
}

impl<I: Iterator, C: Compare<I::Item, I::Item>>
    Compare<IndexedPeekedIterator<I>, IndexedPeekedIterator<I>>
    for IndexedPeekedIteratorComparator<I::Item, C>
{
    fn compare(
        &self,
        u: &IndexedPeekedIterator<I>,
        v: &IndexedPeekedIterator<I>,
    ) -> Ordering {
        match (&u.buf, &v.buf) {
            (Some((u_buf, u_index)), Some((v_buf, v_index))) => {
                let mut order = self.inner.compare(u_buf, v_buf);
                if let Ordering::Equal = order {
                    order = u_index.cmp(&v_index)
                }
                order
            }
            // None means infinity; so return Less.
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => Ordering::Equal,
        }
    }
}

/// Used as a sentinel data source.
struct NoneIter<T> {
    phantom: PhantomData<T>,
}

impl<T> NoneIter<T> {
    fn new() -> Self {
        Self {
            phantom: PhantomData::default(),
        }
    }
}

impl<T> Iterator for NoneIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(0))
    }
}

/// Visits the values representing the union of `K>0` *strictly* sorted
/// iterators, comparing with `compare`. Yields `Vec<Option<T>>` of length `K`.
///
/// Usage example:
///
/// ```
/// use sorted_iter::MultiWayUnion;
///
/// fn using_multi_way_union() {
///     let v1 = vec![3, 5];
///     let v2 = vec![2, 3];
///     let v3 = vec![2, 3, 5];
///     let mut um = MultiWayUnion::new(
///         [v1.into_iter(), v2.into_iter(), v3.into_iter()],
///         compare::natural(),
///     );
///     assert_eq!(um.next(), Some(vec![None, Some(2), Some(2)]));
///     assert_eq!(um.next(), Some(vec![Some(3), Some(3), Some(3)]));
///     assert_eq!(um.next(), Some(vec![Some(5), None, Some(5)]));
///     assert_eq!(um.next(), None);
/// }
/// ```
pub struct MultiWayUnion<'a, T, C: Compare<T, T>> {
    /// The tournament tree (loser tree).
    tt: Vec<IndexedPeekedIterator<Box<dyn Iterator<Item = T> + 'a>>>,
    len: usize,
    compare: IndexedPeekedIteratorComparator<T, C>,
}

/// Initialize the tournament tree. The len of `iters` should be a power of 2.
fn init_tt<'a, T, C: Compare<T, T>>(
    iters: Vec<IndexedPeekedIterator<Box<dyn Iterator<Item = T> + 'a>>>,
    compare: &IndexedPeekedIteratorComparator<T, C>,
) -> Vec<IndexedPeekedIterator<Box<dyn Iterator<Item = T> + 'a>>> {
    let n_iters = iters.len();
    assert!(n_iters.is_power_of_two());
    let mut tt: Vec<
        MaybeUninit<IndexedPeekedIterator<Box<dyn Iterator<Item = T>>>>,
    > = Vec::with_capacity(2 * n_iters);
    // Fill the first half with uninitialized values, and the second half with
    // elements from `iters`.
    unsafe {
        let (head, _) = tt.spare_capacity_mut().split_at_mut(n_iters);
        head.fill_with(|| MaybeUninit::uninit().assume_init());
        tt.set_len(n_iters);
    };
    for item in iters {
        tt.push(MaybeUninit::new(item));
    }
    assert_eq!(tt.len(), 2 * n_iters);

    // The uninitialized first half is initialized here.
    // For example, denote the uninitialized positions as `_`, and denote
    // sentinel elements as `.`, the tournament tree `tt` can be written as:
    // _ _ _ _ _ _ _ _ 7 2 5 8 4 1 . .
    //                 a b
    //
    // _ _ _ _ 7 _ _ _ 2 _ 5 8 4 1 . .
    //                     a b
    //
    // _ _ _ _ 7 8 _ _ 2 5 _ _ 4 1 . .
    //                         a b
    //
    // _ _ _ _ 7 8 4 _ 2 5 1 _ _ _ . .
    //                             a b
    //
    // _ _ _ _ 7 8 4 . 2 5 1 . _ _ _ _
    //                 a b
    //
    // _ _ 5 _ 7 8 4 . 2 _ 1 . _ _ _ _
    //                     a b
    //
    // _ _ 5 . 7 8 4 . 2 1 _ _ _ _ _ _
    //                 a b
    //
    // _ 2 5 . 7 8 4 . 1 _ _ _ _ _ _ _
    //
    // 1 2 5 . 7 8 4 . _ _ _ _ _ _ _ _
    //
    // The moving range of (a,b) can be bounded by [n_iters, n_iters + p)
    // for each traversal, and p is exclusively lower bounded by 1.

    /// Swap winner to its position.
    #[inline]
    fn swap_winner<T>(winner_idx: usize, n_iters: usize, tt: &mut [T]) {
        let target_idx = (winner_idx - n_iters) / 2 + n_iters;
        tt.swap(winner_idx, target_idx);
    }

    /// Swap loser to its position.
    #[inline]
    fn swap_loser<T>(loser_idx: usize, n_iters: usize, p: usize, tt: &mut [T]) {
        let target_idx = (loser_idx - n_iters + p) / 2;
        tt.swap(loser_idx, target_idx);
    }

    let mut p = n_iters;
    while p > 1 {
        for i in (n_iters..n_iters + p).filter(|x| x % 2 == 0) {
            // SAFETY: i < n_iters + p <= tt.len() == 2 * n_iters.
            let a = unsafe { tt.get_unchecked(i).assume_init_ref() };
            // SAFETY: i, p and n_iters are even, so i + 1 < n_iters + p.
            let b = unsafe { tt.get_unchecked(i + 1).assume_init_ref() };
            let (winner, loser) = match compare.compare(a, b) {
                // If a <= b, then winner is a, and loser is b.
                Ordering::Less | Ordering::Equal => (i, i + 1),
                // If a > b, then winner is b, and loser is a.
                Ordering::Greater => (i + 1, i),
            };
            // The order (loser, and then winner) is important.
            swap_loser(loser, n_iters, p, &mut tt);
            swap_winner(winner, n_iters, &mut tt);
        }
        p /= 2;
    }
    // Swap the output
    tt.swap(n_iters, 0);

    tt.truncate(n_iters);
    unsafe { tt.into_iter().map(|e| e.assume_init()).collect() }
}

/// Pop from the tournament tree, and find the next winner.
fn pop_and_find_next_tt<T, C: Compare<T, T>>(
    tt: &mut [IndexedPeekedIterator<Box<dyn Iterator<Item = T> + '_>>],
    compare: &IndexedPeekedIteratorComparator<T, C>,
) -> Option<(T, usize)> {
    assert!(tt.len().is_power_of_two());
    // SAFETY: tt.len() is a power of 2.
    match unsafe { tt.get_unchecked_mut(0).next() } {
        // tt.first is the smallest. If the smallest is None (infinity), then
        // all the others must also be None.
        None => None,
        Some((value, index)) => {
            let mut i = (index + tt.len()) / 2;
            while i > 0 {
                // SAFETY: tt.len() is a power of 2.
                let a = unsafe { tt.get_unchecked(0) };
                // SAFETY: index < tt.len(), so i < tt.len().
                let b = unsafe { tt.get_unchecked(i) };
                if let Ordering::Greater = compare.compare(a, b) {
                    // a is loser.
                    tt.swap(0, i);
                }
                i /= 2;
            }
            Some((value, index))
        }
    }
}

#[cfg(test)]
mod tournament_tree_tests {
    use super::{
        init_tt, pop_and_find_next_tt, IndexedPeekedIterator,
        IndexedPeekedIteratorComparator,
    };

    macro_rules! new_indexed_peeked_itr {
        ($idx:literal) => {
            {
                let it: IndexedPeekedIterator<Box<dyn Iterator<Item = i32>>> =
                    IndexedPeekedIterator::new(Box::new(vec![].into_iter()), $idx);
                it
            }
        };
        ($idx:literal; $( $elem:literal ),+) => {
            {
                let it: IndexedPeekedIterator<Box<dyn Iterator<Item = i32>>> =
                    IndexedPeekedIterator::new(Box::new(vec![$( $elem ),+].into_iter()), $idx);
                it
            }
        };
    }

    fn new_iters1() -> Vec<IndexedPeekedIterator<Box<dyn Iterator<Item = i32>>>>
    {
        vec![
            new_indexed_peeked_itr!(0; 7),
            new_indexed_peeked_itr!(1; 2),
            new_indexed_peeked_itr!(2; 2),
            new_indexed_peeked_itr!(3; 8),
            new_indexed_peeked_itr!(4; 4),
            new_indexed_peeked_itr!(5),
            new_indexed_peeked_itr!(6),
            new_indexed_peeked_itr!(7),
        ]
    }

    fn new_iters2() -> Vec<IndexedPeekedIterator<Box<dyn Iterator<Item = i32>>>>
    {
        vec![
            new_indexed_peeked_itr!(0; 2, 7),
            new_indexed_peeked_itr!(1; 5, 10),
            new_indexed_peeked_itr!(2; 3, 6),
            new_indexed_peeked_itr!(3; 4, 8),
        ]
    }

    fn new_iters3() -> Vec<IndexedPeekedIterator<Box<dyn Iterator<Item = i32>>>>
    {
        vec![new_indexed_peeked_itr!(0; 2, 4, 5)]
    }

    #[test]
    fn test_init_tt_1() {
        let iters = new_iters1();
        let compare = IndexedPeekedIteratorComparator::from(compare::natural());
        let tt = init_tt(iters, &compare);
        let peeks: Vec<_> = tt.iter().map(|it| it.peek()).collect();
        assert_eq!(
            peeks,
            vec![
                Some(&(2, 1)),
                Some(&(4, 4)),
                Some(&(2, 2)),
                None,
                Some(&(7, 0)),
                Some(&(8, 3)),
                None,
                None,
            ]
        );
    }

    #[test]
    fn test_init_tt_2() {
        let iters = new_iters2();
        let compare = IndexedPeekedIteratorComparator::from(compare::natural());
        let tt = init_tt(iters, &compare);
        let peeks: Vec<_> = tt.iter().map(|it| it.peek()).collect();
        assert_eq!(
            peeks,
            vec![Some(&(2, 0)), Some(&(3, 2)), Some(&(5, 1)), Some(&(4, 3)),]
        );
    }

    #[test]
    fn test_init_tt_3() {
        let iters = new_iters3();
        let compare = IndexedPeekedIteratorComparator::from(compare::natural());
        let tt = init_tt(iters, &compare);
        let peeks: Vec<_> = tt.iter().map(|it| it.peek()).collect();
        assert_eq!(peeks, vec![Some(&(2, 0))]);
    }

    #[test]
    fn test_pop_and_find_next_tt_1() {
        let iters = new_iters1();
        let compare = IndexedPeekedIteratorComparator::from(compare::natural());
        let mut tt = init_tt(iters, &compare);

        let popped = pop_and_find_next_tt(&mut tt, &compare);
        assert_eq!(popped, Some((2, 1)));
        let peeks: Vec<_> = tt.iter().map(|it| it.peek()).collect();
        assert_eq!(
            peeks,
            vec![
                Some(&(2, 2)),
                Some(&(4, 4)),
                Some(&(7, 0)),
                None,
                None,
                Some(&(8, 3)),
                None,
                None,
            ]
        );

        let popped = pop_and_find_next_tt(&mut tt, &compare);
        assert_eq!(popped, Some((2, 2)));
        let peeks: Vec<_> = tt.iter().map(|it| it.peek()).collect();
        assert_eq!(
            peeks,
            vec![
                Some(&(4, 4)),
                Some(&(7, 0)),
                Some(&(8, 3)),
                None,
                None,
                None,
                None,
                None,
            ]
        );

        let popped = pop_and_find_next_tt(&mut tt, &compare);
        assert_eq!(popped, Some((4, 4)));
        let peeks: Vec<_> = tt.iter().map(|it| it.peek()).collect();
        assert_eq!(
            peeks,
            vec![
                Some(&(7, 0)),
                None,
                Some(&(8, 3)),
                None,
                None,
                None,
                None,
                None,
            ]
        );

        let popped = pop_and_find_next_tt(&mut tt, &compare);
        assert_eq!(popped, Some((7, 0)));
        let peeks: Vec<_> = tt.iter().map(|it| it.peek()).collect();
        assert_eq!(
            peeks,
            vec![Some(&(8, 3)), None, None, None, None, None, None, None]
        );

        let popped = pop_and_find_next_tt(&mut tt, &compare);
        assert_eq!(popped, Some((8, 3)));
        let peeks: Vec<_> = tt.iter().map(|it| it.peek()).collect();
        assert_eq!(peeks, vec![None, None, None, None, None, None, None, None]);

        let popped = pop_and_find_next_tt(&mut tt, &compare);
        assert_eq!(popped, None);
        let peeks: Vec<_> = tt.iter().map(|it| it.peek()).collect();
        assert_eq!(peeks, vec![None, None, None, None, None, None, None, None]);
    }

    #[test]
    fn test_pop_and_find_next_tt_2() {
        let iters = new_iters2();
        let compare = IndexedPeekedIteratorComparator::from(compare::natural());
        let mut tt = init_tt(iters, &compare);

        let popped = pop_and_find_next_tt(&mut tt, &compare);
        assert_eq!(popped, Some((2, 0)));
        let peeks: Vec<_> = tt.iter().map(|it| it.peek()).collect();
        assert_eq!(
            peeks,
            vec![Some(&(3, 2)), Some(&(5, 1)), Some(&(7, 0)), Some(&(4, 3))]
        );

        let popped = pop_and_find_next_tt(&mut tt, &compare);
        assert_eq!(popped, Some((3, 2)));
        let peeks: Vec<_> = tt.iter().map(|it| it.peek()).collect();
        assert_eq!(
            peeks,
            vec![Some(&(4, 3)), Some(&(5, 1)), Some(&(7, 0)), Some(&(6, 2))]
        );

        let popped = pop_and_find_next_tt(&mut tt, &compare);
        assert_eq!(popped, Some((4, 3)));
        let peeks: Vec<_> = tt.iter().map(|it| it.peek()).collect();
        assert_eq!(
            peeks,
            vec![Some(&(5, 1)), Some(&(6, 2)), Some(&(7, 0)), Some(&(8, 3))]
        );

        let popped = pop_and_find_next_tt(&mut tt, &compare);
        assert_eq!(popped, Some((5, 1)));
        let peeks: Vec<_> = tt.iter().map(|it| it.peek()).collect();
        assert_eq!(
            peeks,
            vec![Some(&(6, 2)), Some(&(7, 0)), Some(&(10, 1)), Some(&(8, 3))]
        );

        let popped = pop_and_find_next_tt(&mut tt, &compare);
        assert_eq!(popped, Some((6, 2)));
        let peeks: Vec<_> = tt.iter().map(|it| it.peek()).collect();
        assert_eq!(
            peeks,
            vec![Some(&(7, 0)), Some(&(8, 3)), Some(&(10, 1)), None]
        );

        let popped = pop_and_find_next_tt(&mut tt, &compare);
        assert_eq!(popped, Some((7, 0)));
        let peeks: Vec<_> = tt.iter().map(|it| it.peek()).collect();
        assert_eq!(peeks, vec![Some(&(8, 3)), Some(&(10, 1)), None, None]);

        let popped = pop_and_find_next_tt(&mut tt, &compare);
        assert_eq!(popped, Some((8, 3)));
        let peeks: Vec<_> = tt.iter().map(|it| it.peek()).collect();
        assert_eq!(peeks, vec![Some(&(10, 1)), None, None, None]);

        let popped = pop_and_find_next_tt(&mut tt, &compare);
        assert_eq!(popped, Some((10, 1)));
        let peeks: Vec<_> = tt.iter().map(|it| it.peek()).collect();
        assert_eq!(peeks, vec![None, None, None, None]);

        let popped = pop_and_find_next_tt(&mut tt, &compare);
        assert_eq!(popped, None);
        let peeks: Vec<_> = tt.iter().map(|it| it.peek()).collect();
        assert_eq!(peeks, vec![None, None, None, None]);
    }

    #[test]
    fn test_pop_and_find_next_tt_3() {
        let iters = new_iters3();
        let compare = IndexedPeekedIteratorComparator::from(compare::natural());
        let mut tt = init_tt(iters, &compare);

        let popped = pop_and_find_next_tt(&mut tt, &compare);
        assert_eq!(popped, Some((2, 0)));
        let peeks: Vec<_> = tt.iter().map(|it| it.peek()).collect();
        assert_eq!(peeks, vec![Some(&(4, 0))]);

        let popped = pop_and_find_next_tt(&mut tt, &compare);
        assert_eq!(popped, Some((4, 0)));
        let peeks: Vec<_> = tt.iter().map(|it| it.peek()).collect();
        assert_eq!(peeks, vec![Some(&(5, 0))]);

        let popped = pop_and_find_next_tt(&mut tt, &compare);
        assert_eq!(popped, Some((5, 0)));
        let peeks: Vec<_> = tt.iter().map(|it| it.peek()).collect();
        assert_eq!(peeks, vec![None]);

        let popped = pop_and_find_next_tt(&mut tt, &compare);
        assert_eq!(popped, None);
        let peeks: Vec<_> = tt.iter().map(|it| it.peek()).collect();
        assert_eq!(peeks, vec![None]);
    }
}

impl<'a, T: 'a, C: Compare<T, T> + 'a> MultiWayUnion<'a, T, C> {
    /// Construct new instance from homogeneous collection of iterators. There
    /// should be at least one iterator.
    pub fn new<I: Iterator<Item = T> + 'a>(
        iters: impl IntoIterator<Item = I>,
        compare: C,
    ) -> Self {
        Self::from_boxed(iters.into_iter().map(box_iterator), compare)
    }

    /// Construct new instance from collection of boxed iterators. There should
    /// be at least one iterator.
    pub fn from_boxed(
        iters: impl IntoIterator<Item = Box<dyn Iterator<Item = T> + 'a>>,
        compare: C,
    ) -> Self {
        let iters: Vec<_> = iters.into_iter().collect();
        let n_iters = iters.len();
        assert!(n_iters > 0);
        let n_iters_total = n_iters.next_power_of_two();
        let sentinels: Vec<_> = (iters.len()..n_iters_total)
            .map(|_| Box::new(NoneIter::new()) as Box<dyn Iterator<Item = T>>)
            .collect();
        let iters: Vec<_> = iters
            .into_iter()
            .chain(sentinels.into_iter())
            .zip(0..n_iters_total)
            .map(|(it, index)| IndexedPeekedIterator::new(it, index))
            .collect();
        let compare = IndexedPeekedIteratorComparator::from(compare);
        let tt = init_tt(iters, &compare);
        Self {
            tt,
            len: n_iters,
            compare,
        }
    }

    pub fn into_boxed(self) -> Box<dyn Iterator<Item = Vec<Option<T>>> + 'a> {
        Box::new(self)
    }
}

/// Pop from the tournament tree (`tt`) all elements whose value is equal to
/// `value`, comparing using `compare`, and collect them if the output vec
/// (`out`) is provided. When `out` is provided, it should already be
/// initialized. If `out` is provided, it's guaranteed that after the call, it
/// will contain at least one element `value` at index `index`.
fn find_equal_value_and_collect<T, C: Compare<T, T>>(
    tt: &mut [IndexedPeekedIterator<Box<dyn Iterator<Item = T> + '_>>],
    value: T,
    index: usize,
    compare: &IndexedPeekedIteratorComparator<T, C>,
    mut out: Option<&mut Vec<Option<T>>>,
) {
    // Loop and pop from the tournament tree until we don't need to pop.
    loop {
        // Peek the tournament tree and decide if the tree needs to be popped.
        // The answer is yes, if the next value is equal to `value`, different
        // only in `index`.
        // SAFETY: tt.len() is a power of 2.
        let need_pop_next = match unsafe { tt.get_unchecked(0).peek() } {
            None => false,
            Some((value_to_pop, _)) => {
                match compare.inner.compare(&value, value_to_pop) {
                    Ordering::Less | Ordering::Greater => false,
                    Ordering::Equal => true,
                }
            }
        };
        // Break out from the loop.
        if !need_pop_next {
            break;
        }
        // Pop from the tournament tree.
        // SAFETY: already checked `need_pop_next`.
        let (next_value, next_index) =
            unsafe { pop_and_find_next_tt(tt, compare).unwrap_unchecked() };
        // If the output vec, `out`, is provided, and if the popped element is
        // not from a sentinel (if it's from a sentinel, `next_index` will be
        // `>= v.len()`), insert `next_value` to the output vector.
        if let Some(ref mut v) = out {
            if let Some(pos) = v.get_mut(next_index) {
                pos.get_or_insert(next_value);
            }
        }
    }
    // Finally, if the output vec, `out`, is provided, insert `value` to it.
    if let Some(ref mut v) = out {
        v.get_mut(index).unwrap().get_or_insert(value);
    }
}

impl<'a, T, C: Compare<T, T>> Iterator for MultiWayUnion<'a, T, C> {
    type Item = Vec<Option<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        match pop_and_find_next_tt(&mut self.tt, &self.compare) {
            None => None,
            Some((value, index)) => {
                // Why not use `vec![None; self.tt.len()]`: T is not Clone.
                let mut ret = Vec::with_capacity(self.len);
                for _ in 0..self.len {
                    ret.push(None);
                }
                find_equal_value_and_collect(
                    &mut self.tt,
                    value,
                    index,
                    &self.compare,
                    Some(&mut ret),
                );
                Some(ret)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.tt.iter().fold((0, Some(0)), |(cmin, cmax), it| {
            let (imin, imax) = it.size_hint();
            // Full overlap.
            let cmin = cmp::max(cmin, imin);
            // No overlap.
            let cmax = cmax
                .and_then(|cmax| imax.and_then(|imax| cmax.checked_add(imax)));
            (cmin, cmax)
        })
    }

    fn nth(&mut self, mut n: usize) -> Option<Self::Item> {
        // Similar to `next()`, but don't allocate memory for return in the
        // first n-1 rounds.
        loop {
            match pop_and_find_next_tt(&mut self.tt, &self.compare) {
                None => break None,
                Some((value, index)) => {
                    let mut ret = if n == 0 {
                        let mut v: Vec<Option<T>> =
                            Vec::with_capacity(self.len);
                        for _ in 0..self.len {
                            v.push(None);
                        }
                        Some(v)
                    } else {
                        None
                    };
                    find_equal_value_and_collect(
                        &mut self.tt,
                        value,
                        index,
                        &self.compare,
                        ret.as_mut(),
                    );
                    if n == 0 {
                        break ret;
                    }
                    n -= 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod multi_way_union_tests {
    use super::MultiWayUnion;
    use compare::Compare;
    use std::cmp::Ordering;

    macro_rules! assert_size_hint {
        ($itr:ident, $lb:expr, $ub:expr) => {{
            let (min, max) = $itr.size_hint();
            assert!(min <= $lb);
            match (max, $ub) {
                (Some(max), Some(ub)) => assert!(max >= ub),
                (Some(max), None) => panic!("ub `{}` is not inf", max),
                (None, Some(_)) => panic!("ub `inf` is too loose"),
                (None, None) => (),
            }
        }};
    }

    struct FirstComparator;

    impl Compare<(i32, char), (i32, char)> for FirstComparator {
        fn compare(&self, u: &(i32, char), v: &(i32, char)) -> Ordering {
            u.0.cmp(&v.0)
        }
    }

    #[test]
    fn test_multi_way_union_iterator() {
        let a = vec![(1, 'a'), (3, 'b'), (5, 'c'), (6, 'd')].into_iter();
        let b =
            vec![(0, 'e'), (1, 'f'), (4, 'g'), (5, 'h'), (7, 'i')].into_iter();
        let c =
            vec![(0, 'j'), (2, 'k'), (3, 'l'), (5, 'm'), (7, 'n'), (8, 'o')]
                .into_iter();
        let mut u = MultiWayUnion::new([a, b, c], FirstComparator).into_boxed();
        assert_size_hint!(u, 9, Some(9));
        assert_eq!(u.next(), Some(vec![None, Some((0, 'e')), Some((0, 'j'))]));
        assert_size_hint!(u, 8, Some(8));
        assert_eq!(u.next(), Some(vec![Some((1, 'a')), Some((1, 'f')), None]));
        assert_size_hint!(u, 7, Some(7));
        assert_eq!(u.next(), Some(vec![None, None, Some((2, 'k'))]));
        assert_size_hint!(u, 6, Some(6));
        assert_eq!(u.next(), Some(vec![Some((3, 'b')), None, Some((3, 'l'))]));
        assert_size_hint!(u, 5, Some(5));
        assert_eq!(u.next(), Some(vec![None, Some((4, 'g')), None]));
        assert_size_hint!(u, 4, Some(4));
        assert_eq!(
            u.next(),
            Some(vec![Some((5, 'c')), Some((5, 'h')), Some((5, 'm'))])
        );
        assert_size_hint!(u, 3, Some(3));
        assert_eq!(u.next(), Some(vec![Some((6, 'd')), None, None]));
        assert_size_hint!(u, 2, Some(2));
        assert_eq!(u.next(), Some(vec![None, Some((7, 'i')), Some((7, 'n'))]));
        assert_size_hint!(u, 1, Some(1));
        assert_eq!(u.next(), Some(vec![None, None, Some((8, 'o'))]));
        assert_size_hint!(u, 0, Some(0));
        assert_eq!(u.next(), None);
        assert_size_hint!(u, 0, Some(0));
        assert_eq!(u.next(), None);
    }

    #[test]
    fn test_multi_way_union_iterator_single() {
        let a = vec![(1, 'a'), (3, 'b'), (5, 'c'), (6, 'd')].into_iter();
        let mut u = MultiWayUnion::new([a], FirstComparator).into_boxed();
        assert_size_hint!(u, 4, Some(4));
        assert_eq!(u.next(), Some(vec![Some((1, 'a'))]));
        assert_size_hint!(u, 3, Some(3));
        assert_eq!(u.next(), Some(vec![Some((3, 'b'))]));
        assert_size_hint!(u, 2, Some(2));
        assert_eq!(u.next(), Some(vec![Some((5, 'c'))]));
        assert_size_hint!(u, 1, Some(1));
        assert_eq!(u.next(), Some(vec![Some((6, 'd'))]));
        assert_size_hint!(u, 0, Some(0));
        assert_eq!(u.next(), None);
        assert_size_hint!(u, 0, Some(0));
        assert_eq!(u.next(), None);
    }

    #[test]
    fn test_multi_way_union_nth() {
        let a = vec![(1, 'a'), (3, 'b'), (5, 'c'), (6, 'd')].into_iter();
        let b =
            vec![(0, 'e'), (1, 'f'), (4, 'g'), (5, 'h'), (7, 'i')].into_iter();
        let c =
            vec![(0, 'j'), (2, 'k'), (3, 'l'), (5, 'm'), (7, 'n'), (8, 'o')]
                .into_iter();
        let mut u = MultiWayUnion::new([a, b, c], FirstComparator).into_boxed();
        assert_eq!(u.nth(4), Some(vec![None, Some((4, 'g')), None]));
        assert_eq!(
            u.nth(0),
            Some(vec![Some((5, 'c')), Some((5, 'h')), Some((5, 'm'))])
        );
        assert_eq!(u.nth(5), None);
    }
}

/// Visits the values representing the intersection of `K>0` *strictly* sorted
/// iterators, comparing with `compare`. Yields `Vec<T>` of length `K`.
///
/// Usage example:
///
/// ```
/// use sorted_iter::MultiWayIntersection;
///
/// fn using_multi_way_union() {
///     let v1 = vec![3, 5];
///     let v2 = vec![2, 3];
///     let v3 = vec![2, 3, 5];
///     let mut um = MultiWayIntersection::new(
///         [v1.into_iter(), v2.into_iter(), v3.into_iter()],
///         compare::natural(),
///     );
///     assert_eq!(um.next(), Some(vec![3, 3, 3]));
///     assert_eq!(um.next(), None);
/// }
/// ```
pub struct MultiWayIntersection<'a, T, C: Compare<T, T>> {
    iters: Vec<Peekable<Box<dyn Iterator<Item = T> + 'a>>>,
    compare: C,
    exhausted: bool,
}

impl<'a, T: 'a, C: Compare<T, T> + 'a> MultiWayIntersection<'a, T, C> {
    /// Construct new instance from homogeneous collection of iterators. There
    /// should be at least one iterator.
    pub fn new<I: Iterator<Item = T> + 'a>(
        iters: impl IntoIterator<Item = I>,
        compare: C,
    ) -> Self {
        Self::from_boxed(iters.into_iter().map(box_iterator), compare)
    }

    /// Construct new instance from collection of boxed iterators. There should
    /// be at least one iterator.
    pub fn from_boxed(
        iters: impl IntoIterator<Item = Box<dyn Iterator<Item = T> + 'a>>,
        compare: C,
    ) -> Self {
        let iters: Vec<_> = iters.into_iter().collect();
        assert!(!iters.is_empty());
        Self {
            iters: iters.into_iter().map(|it| it.peekable()).collect(),
            compare,
            exhausted: false,
        }
    }

    pub fn into_boxed(self) -> Box<dyn Iterator<Item = Vec<T>> + 'a> {
        Box::new(self)
    }
}

enum MultiWayIntersectionState {
    /// If any sub-iterator is exhausted.
    Exhausted,
    /// If current value does not constitute an intersection.
    NotIntersection,
    /// An intersection is found.
    Ok,
}

/// Step each iterator `i` until `i.peek()` is either `None` or a value as
/// large as `value`. If any `i.peek()` is `None`, `Exhausted` is returned
/// immediately. Otherwise, return `NotIntersection` or `Ok` accordingly. If
/// the output vec `out` is provided initialized, it will be populated when
/// `Ok` is to be returned.
fn step_iters_until_as_large_as_value_and_collect<T, C: Compare<T, T>>(
    iters: &mut [Peekable<Box<dyn Iterator<Item = T> + '_>>],
    compare: &C,
    value: T,
    out: Option<&mut Vec<T>>,
) -> MultiWayIntersectionState {
    let mut form_intersection = true;
    for itr in iters.iter_mut() {
        while let Some(itr_value) = itr.peek() {
            match compare.compare(itr_value, &value) {
                Ordering::Less => itr.next(),
                Ordering::Equal => break,
                Ordering::Greater => {
                    form_intersection = false;
                    break;
                }
            };
        }
        if itr.peek().is_none() {
            return MultiWayIntersectionState::Exhausted;
        }
    }
    if form_intersection {
        if let Some(v) = out {
            v.push(value);
            for itr in iters.iter_mut() {
                v.push(itr.next().unwrap());
            }
        }
        MultiWayIntersectionState::Ok
    } else {
        MultiWayIntersectionState::NotIntersection
    }
}

impl<'a, T, C: Compare<T, T>> Iterator for MultiWayIntersection<'a, T, C> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        let mut ret = Vec::with_capacity(self.iters.len());
        let (head, tail) = self.iters.split_first_mut().unwrap();
        loop {
            match head.next() {
                None => break None,
                Some(value) => {
                    match step_iters_until_as_large_as_value_and_collect(
                        tail,
                        &self.compare,
                        value,
                        Some(&mut ret),
                    ) {
                        MultiWayIntersectionState::Exhausted => {
                            self.exhausted = true;
                            break None;
                        }
                        MultiWayIntersectionState::NotIntersection => (),
                        MultiWayIntersectionState::Ok => break Some(ret),
                    }
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.exhausted {
            (0, Some(0))
        } else {
            // No overlap.
            let rmin = 0usize;

            // Full overlap.
            let rmax = self.iters.iter().fold(None, |max_opt, it| {
                let (_, imax_opt) = it.size_hint();
                // When current accumulator `max_opt` is `None` (infinity),
                // return `imax_opt`. Otherwise, take the min of `imax_opt` and
                // `max_opt`.
                max_opt.map_or(imax_opt, |max| {
                    Some(imax_opt.map_or(max, |imax| cmp::min(max, imax)))
                })
            });
            (rmin, rmax)
        }
    }

    fn nth(&mut self, mut n: usize) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        let mut ret = Vec::with_capacity(self.iters.len());
        let (head, tail) = self.iters.split_first_mut().unwrap();
        loop {
            match head.next() {
                None => break None,
                Some(value) => {
                    match step_iters_until_as_large_as_value_and_collect(
                        tail,
                        &self.compare,
                        value,
                        if n == 0 { Some(&mut ret) } else { None },
                    ) {
                        MultiWayIntersectionState::Exhausted => {
                            self.exhausted = true;
                            break None;
                        }
                        MultiWayIntersectionState::NotIntersection => (),
                        MultiWayIntersectionState::Ok => {
                            if n == 0 {
                                break Some(ret);
                            } else {
                                n -= 1;
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod multi_way_intersection_tests {
    use super::MultiWayIntersection;
    use compare::Compare;
    use std::cmp::Ordering;

    macro_rules! assert_size_hint {
        ($itr:ident, $lb:expr, $ub:expr) => {{
            let (min, max) = $itr.size_hint();
            assert!(min <= $lb);
            match (max, $ub) {
                (Some(max), Some(ub)) => assert!(max >= ub),
                (Some(max), None) => panic!("ub `{}` is not inf", max),
                (None, Some(_)) => panic!("ub `inf` is too loose"),
                (None, None) => (),
            }
        }};
    }

    struct FirstComparator;

    impl Compare<(i32, char), (i32, char)> for FirstComparator {
        fn compare(&self, u: &(i32, char), v: &(i32, char)) -> Ordering {
            u.0.cmp(&v.0)
        }
    }

    #[test]
    fn test_multi_way_intersection_iterator_single() {
        let a = vec![(1, 'a'), (3, 'b'), (5, 'c'), (6, 'd')].into_iter();
        let mut u = MultiWayIntersection::new([a], FirstComparator);
        assert_size_hint!(u, 4, Some(4));
        assert_eq!(u.next(), Some(vec![(1, 'a')]));
        assert_size_hint!(u, 3, Some(3));
        assert_eq!(u.next(), Some(vec![(3, 'b')]));
        assert_size_hint!(u, 2, Some(2));
        assert_eq!(u.next(), Some(vec![(5, 'c')]));
        assert_size_hint!(u, 1, Some(1));
        assert_eq!(u.next(), Some(vec![(6, 'd')]));
        assert_size_hint!(u, 0, Some(0));
        assert_eq!(u.next(), None);
        assert_size_hint!(u, 0, Some(0));
        assert_eq!(u.next(), None);
    }

    #[test]
    fn test_multi_way_intersection_iterator() {
        let a = vec![
            (1, 'a'),
            (3, 'b'),
            (5, 'c'),
            (6, 'd'),
            (8, 'q'),
            (10, 'r'),
            (11, 's'),
            (15, 't'),
        ]
        .into_iter();
        let b =
            vec![(0, 'e'), (1, 'f'), (3, 'g'), (4, 'h'), (5, 'i'), (7, 'j')]
                .into_iter();
        let c =
            vec![(0, 'k'), (2, 'l'), (3, 'm'), (5, 'n'), (7, 'o'), (8, 'p')]
                .into_iter();
        let mut u = MultiWayIntersection::new([a, b, c], FirstComparator);
        assert_size_hint!(u, 2, Some(2));
        assert_eq!(u.next(), Some(vec![(3, 'b'), (3, 'g'), (3, 'm')]));
        assert_size_hint!(u, 1, Some(1));
        assert_eq!(u.next(), Some(vec![(5, 'c'), (5, 'i'), (5, 'n')]));
        assert_size_hint!(u, 0, Some(0));
        assert_eq!(u.next(), None);
        assert_size_hint!(u, 0, Some(0));
        assert_eq!(u.next(), None);
    }

    #[test]
    fn test_multi_way_intersection_nth() {
        let a = vec![
            (1, 'a'),
            (3, 'b'),
            (5, 'c'),
            (6, 'd'),
            (8, 'q'),
            (10, 'r'),
            (11, 's'),
            (15, 't'),
        ]
        .into_iter();
        let b =
            vec![(0, 'e'), (1, 'f'), (3, 'g'), (4, 'h'), (5, 'i'), (7, 'j')]
                .into_iter();
        let c =
            vec![(0, 'k'), (2, 'l'), (3, 'm'), (5, 'n'), (7, 'o'), (8, 'p')]
                .into_iter();
        let mut u = MultiWayIntersection::new([a, b, c], FirstComparator);
        assert_eq!(u.nth(0), Some(vec![(3, 'b'), (3, 'g'), (3, 'm')]));
        assert_size_hint!(u, 1, Some(1));
        assert_eq!(u.nth(4), None);
        assert_size_hint!(u, 0, Some(0));

        let a = vec![
            (1, 'a'),
            (3, 'b'),
            (5, 'c'),
            (6, 'd'),
            (8, 'q'),
            (10, 'r'),
            (11, 's'),
            (15, 't'),
        ]
        .into_iter();
        let b =
            vec![(0, 'e'), (1, 'f'), (3, 'g'), (4, 'h'), (5, 'i'), (7, 'j')]
                .into_iter();
        let c =
            vec![(0, 'k'), (2, 'l'), (3, 'm'), (5, 'n'), (7, 'o'), (8, 'p')]
                .into_iter();
        let mut u = MultiWayIntersection::new([a, b, c], FirstComparator);
        assert_eq!(u.nth(1), Some(vec![(5, 'c'), (5, 'i'), (5, 'n')]));
        assert_size_hint!(u, 0, Some(0));
        assert_eq!(u.nth(0), None);
        assert_size_hint!(u, 0, Some(0));
    }
}

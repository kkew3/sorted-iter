use crate::box_iterator;
use binary_heap_plus::BinaryHeap;
use compare::{Compare, Rev};
use std::cmp::{self, Ordering};
use std::iter::Peekable;
use std::marker::PhantomData;

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

struct Count {
    n: usize,
}

impl Count {
    fn new() -> Self {
        Self { n: 0 }
    }
}

impl Iterator for Count {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let n = self.n;
        self.n += 1;
        Some(n)
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
    bh: BinaryHeap<
        IndexedPeekedIterator<Box<dyn Iterator<Item = T> + 'a>>,
        Rev<IndexedPeekedIteratorComparator<T, C>>,
    >,
    /// A copy of the inner comparator used in the heap. This would be
    /// unnecessary if we were able to access the comparator of the heap.
    inner_comparator: C,
}

impl<'a, T: 'a, C: Compare<T, T> + Clone + 'a> MultiWayUnion<'a, T, C> {
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
        let iters: Vec<_> = iters
            .into_iter()
            .zip(Count::new())
            .map(|(it, n)| IndexedPeekedIterator::new(it, n))
            .collect();
        assert!(!iters.is_empty());
        // Was written as `let comparator = IndexedPeekedIteratorComparator::from(compare).rev()`.
        // Changed due to compiler's suggestion.
        let comparator = <IndexedPeekedIteratorComparator<T, C> as Compare<
            IndexedPeekedIterator<Box<dyn Iterator<Item = T>>>,
        >>::rev(IndexedPeekedIteratorComparator::from(
            compare.clone(),
        ));
        let bh = BinaryHeap::from_vec_cmp(iters, comparator);
        Self {
            bh,
            inner_comparator: compare,
        }
    }

    pub fn into_boxed(self) -> Box<dyn Iterator<Item = Vec<Option<T>>> + 'a> {
        Box::new(self)
    }
}

impl<'a, T, C: Compare<T, T>> MultiWayUnion<'a, T, C> {
    /// Initialize the output vec of `Iterator::next()`.
    #[inline]
    fn init_next_output(&self) -> Vec<Option<T>> {
        // Why not use `vec![None; self.tt.len()]`: T is not Clone.
        (0..self.bh.len()).map(|_| None).collect()
    }

    /// Isolate `self.bh.peek_mut` inside a function to prevent the mutable
    /// reference from being leaked.
    #[inline]
    fn peek_mut_next(&mut self) -> Option<(T, usize)> {
        self.bh.peek_mut().unwrap().next()
    }

    /// Pop from the heap all elements whose value is equal to `value`, and
    /// collect them if the output vec (`out`) is provided. When `out` is not
    /// `None`, it should already be initialized to contain `self.bh.len()`
    /// elements. `out` is guaranteed to contain at least one element `value`
    /// at index `index` if it's not `None`.
    fn pop_equal_value_and_collect(
        &mut self,
        value: T,
        index: usize,
        mut out: Option<&mut Vec<Option<T>>>,
    ) {
        // Loop and pop until the top element in the heap is not equal to
        // `value`. The equality is decided by `self.inner_comparator`.
        while self
            .bh
            .peek()
            .unwrap()
            .peek()
            .filter(|(value_to_pop, _)| {
                self.inner_comparator.compares_eq(value_to_pop, &value)
            })
            .is_some()
        {
            let (value_to_pop, index_to_pop) = self.peek_mut_next().unwrap();
            if let Some(ref mut v) = out {
                v.get_mut(index_to_pop).unwrap().get_or_insert(value_to_pop);
            }
        }
        if let Some(ref mut v) = out {
            v.get_mut(index).unwrap().get_or_insert(value);
        }
    }
}

impl<'a, T, C: Compare<T, T>> Iterator for MultiWayUnion<'a, T, C> {
    type Item = Vec<Option<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.peek_mut_next() {
            None => None,
            Some((value, index)) => {
                let mut v = self.init_next_output();
                self.pop_equal_value_and_collect(value, index, Some(&mut v));
                Some(v)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.bh.iter().fold((0, Some(0)), |(cmin, cmax), it| {
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
            match self.peek_mut_next() {
                None => break None,
                Some((value, index)) => {
                    if n == 0 {
                        let mut v = self.init_next_output();
                        self.pop_equal_value_and_collect(
                            value,
                            index,
                            Some(&mut v),
                        );
                        break Some(v);
                    } else {
                        self.pop_equal_value_and_collect(value, index, None);
                        n -= 1;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod multiway_union_tests {
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

    #[derive(Copy, Clone)]
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
        let mut u =
            MultiWayUnion::new([a, b, c], FirstComparator).into_boxed();
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
        let mut u =
            MultiWayUnion::new([a, b, c], FirstComparator).into_boxed();
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

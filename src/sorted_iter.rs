//! Heavily inspired by https://github.com/rklaehn/sorted-iter.

use super::Comparator;
use std::cmp::{max, Ordering};
use std::iter::Peekable;

/// Visits the values representing the union of two *strictly* sorted iterators,
/// comparing with `compare`, and deduplicate using `map`. The arguments to
/// `map` are guaranteed to contain at least one `Some(_)`.
///
/// Usage example:
///
/// ```
/// use sorted_iter::{NaturalComparator, UnionMap};
///
/// fn using_union_map() {
///     let v1 = vec![3, 5];
///     let v2 = vec![2, 3];
///     let mut um = UnionMap::new(
///         v1.into_iter(),
///         v2.into_iter(),
///         NaturalComparator::new(),
///         |x, y| (x, y),
///     );
///     assert_eq!(um.next(), Some((None, Some(2))));
///     assert_eq!(um.next(), Some((Some(3), Some(3))));
///     assert_eq!(um.next(), Some((Some(5), None)));
///     assert_eq!(um.next(), None);
/// }
/// ```
pub struct UnionMap<I, J, U, V, C, M, W>
where
    I: Iterator<Item = U>,
    J: Iterator<Item = V>,
    C: Comparator<U, V>,
    M: FnMut(Option<U>, Option<V>) -> W,
{
    iter1: Peekable<I>,
    iter2: Peekable<J>,
    compare: C,
    map: M,
}

impl<I, J, U, V, C, M, W> UnionMap<I, J, U, V, C, M, W>
where
    I: Iterator<Item = U>,
    J: Iterator<Item = V>,
    C: Comparator<U, V>,
    M: FnMut(Option<U>, Option<V>) -> W,
{
    pub fn new(iter1: I, iter2: J, compare: C, map: M) -> Self {
        Self {
            iter1: iter1.peekable(),
            iter2: iter2.peekable(),
            compare,
            map,
        }
    }
}

impl<I, J, U, V, C, M, W> Iterator for UnionMap<I, J, U, V, C, M, W>
where
    I: Iterator<Item = U>,
    J: Iterator<Item = V>,
    C: Comparator<U, V>,
    M: FnMut(Option<U>, Option<V>) -> W,
{
    type Item = W;

    fn next(&mut self) -> Option<Self::Item> {
        let (o1, o2) = match (self.iter1.peek(), self.iter2.peek()) {
            (Some(v1), Some(v2)) => match self.compare.compare(v1, v2) {
                Ordering::Less => (self.iter1.next(), None),
                Ordering::Greater => (None, self.iter2.next()),
                Ordering::Equal => (self.iter1.next(), self.iter2.next()),
            },
            (Some(_), None) => (self.iter1.next(), None),
            (None, Some(_)) => (None, self.iter2.next()),
            (None, None) => return None,
        };
        Some((self.map)(o1, o2))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (min1, max1) = self.iter1.size_hint();
        let (min2, max2) = self.iter2.size_hint();
        // Full overlap.
        let rmin = max(min1, min2);
        // No overlap.
        let rmax = match (max1, max2) {
            (Some(max1), Some(max2)) => max1.checked_add(max2),
            _ => None,
        };
        (rmin, rmax)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Comparator, UnionMap};
    use std::cmp::Ordering;

    #[derive(Debug, Eq, PartialEq)]
    struct KeyValue {
        key: i32,
        value: i32,
    }

    macro_rules! kv {
        ($k:expr, $v:expr) => {
            KeyValue { key: $k, value: $v }
        };
    }

    fn vec_of_keyvalues(
        keys: Vec<i32>,
        mut starting_value: i32,
    ) -> Vec<KeyValue> {
        keys.into_iter()
            .map(|key| {
                let item = KeyValue {
                    key,
                    value: starting_value,
                };
                starting_value += 1;
                item
            })
            .collect()
    }

    struct ComparatorOnKey;

    impl Comparator<KeyValue, KeyValue> for ComparatorOnKey {
        fn compare(&self, u: &KeyValue, v: &KeyValue) -> Ordering {
            u.key.cmp(&v.key)
        }
    }

    fn multiply_value(u: Option<KeyValue>, v: Option<KeyValue>) -> KeyValue {
        match (u, v) {
            (Some(u), Some(v)) => KeyValue {
                key: u.key,
                value: u.value * v.value,
            },
            (Some(u), None) => u,
            (None, Some(v)) => v,
            (None, None) => panic!(),
        }
    }

    #[test]
    fn test_union_map() {
        // === CASE 1: empty inputs ===
        let v1 = vec_of_keyvalues(vec![], 1);
        let v2 = vec_of_keyvalues(vec![], 11);
        let mut um = UnionMap::new(
            v1.into_iter(),
            v2.into_iter(),
            ComparatorOnKey,
            multiply_value,
        );
        assert_eq!(um.next(), None);

        // === CASE 2: left input is empty ===
        let v1 = vec_of_keyvalues(vec![], 1);
        let v2 = vec_of_keyvalues(vec![1, 3, 5], 11);
        let mut um = UnionMap::new(
            v1.into_iter(),
            v2.into_iter(),
            ComparatorOnKey,
            multiply_value,
        );
        assert_eq!(um.next(), Some(kv!(1, 11)));
        assert_eq!(um.next(), Some(kv!(3, 12)));
        assert_eq!(um.next(), Some(kv!(5, 13)));
        assert_eq!(um.next(), None);

        // === CASE 3: right input is empty ===
        let v1 = vec_of_keyvalues(vec![1, 3, 5], 1);
        let v2 = vec_of_keyvalues(vec![], 11);
        let mut um = UnionMap::new(
            v1.into_iter(),
            v2.into_iter(),
            ComparatorOnKey,
            multiply_value,
        );
        assert_eq!(um.next(), Some(kv!(1, 1)));
        assert_eq!(um.next(), Some(kv!(3, 2)));
        assert_eq!(um.next(), Some(kv!(5, 3)));
        assert_eq!(um.next(), None);

        // === CASE 4: disjoint inputs ===
        let v1 = vec_of_keyvalues(vec![1, 3, 5], 1);
        let v2 = vec_of_keyvalues(vec![2, 4, 6], 11);
        let mut um = UnionMap::new(
            v1.into_iter(),
            v2.into_iter(),
            ComparatorOnKey,
            multiply_value,
        );
        assert_eq!(um.next(), Some(kv!(1, 1)));
        assert_eq!(um.next(), Some(kv!(2, 11)));
        assert_eq!(um.next(), Some(kv!(3, 2)));
        assert_eq!(um.next(), Some(kv!(4, 12)));
        assert_eq!(um.next(), Some(kv!(5, 3)));
        assert_eq!(um.next(), Some(kv!(6, 13)));
        assert_eq!(um.next(), None);

        // === CASE 5: overlapping inputs ===
        let v1 = vec_of_keyvalues(vec![2, 3, 5], 1);
        let v2 = vec_of_keyvalues(vec![2, 3, 5], 11);
        let mut um = UnionMap::new(
            v1.into_iter(),
            v2.into_iter(),
            ComparatorOnKey,
            multiply_value,
        );
        assert_eq!(um.next(), Some(kv!(2, 1 * 11)));
        assert_eq!(um.next(), Some(kv!(3, 2 * 12)));
        assert_eq!(um.next(), Some(kv!(5, 3 * 13)));
        assert_eq!(um.next(), None);

        // === CASE 6: left subset right ===
        let v1 = vec_of_keyvalues(vec![3, 5, 8], 1);
        let v2 = vec_of_keyvalues(vec![2, 3, 4, 5, 8, 9], 11);
        let mut um = UnionMap::new(
            v1.into_iter(),
            v2.into_iter(),
            ComparatorOnKey,
            multiply_value,
        );
        assert_eq!(um.next(), Some(kv!(2, 11)));
        assert_eq!(um.next(), Some(kv!(3, 1 * 12)));
        assert_eq!(um.next(), Some(kv!(4, 13)));
        assert_eq!(um.next(), Some(kv!(5, 2 * 14)));
        assert_eq!(um.next(), Some(kv!(8, 3 * 15)));
        assert_eq!(um.next(), Some(kv!(9, 16)));
        assert_eq!(um.next(), None);

        // === CASE 7: right subset left ===
        let v1 = vec_of_keyvalues(vec![2, 3, 4, 5, 8, 9], 1);
        let v2 = vec_of_keyvalues(vec![3, 5, 8], 11);
        let mut um = UnionMap::new(
            v1.into_iter(),
            v2.into_iter(),
            ComparatorOnKey,
            multiply_value,
        );
        assert_eq!(um.next(), Some(kv!(2, 1)));
        assert_eq!(um.next(), Some(kv!(3, 2 * 11)));
        assert_eq!(um.next(), Some(kv!(4, 3)));
        assert_eq!(um.next(), Some(kv!(5, 4 * 12)));
        assert_eq!(um.next(), Some(kv!(8, 5 * 13)));
        assert_eq!(um.next(), Some(kv!(9, 6)));
        assert_eq!(um.next(), None);

        // CASE 8: random 1 ===
        let v1 = vec_of_keyvalues(vec![2, 5, 6, 8], 1);
        let v2 = vec_of_keyvalues(vec![2, 3, 7, 8, 9], 11);
        let mut um = UnionMap::new(
            v1.into_iter(),
            v2.into_iter(),
            ComparatorOnKey,
            multiply_value,
        );
        assert_eq!(um.next(), Some(kv!(2, 1 * 11)));
        assert_eq!(um.next(), Some(kv!(3, 12)));
        assert_eq!(um.next(), Some(kv!(5, 2)));
        assert_eq!(um.next(), Some(kv!(6, 3)));
        assert_eq!(um.next(), Some(kv!(7, 13)));
        assert_eq!(um.next(), Some(kv!(8, 4 * 14)));
        assert_eq!(um.next(), Some(kv!(9, 15)));
    }
}

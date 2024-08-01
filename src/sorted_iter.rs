//! Heavily inspired by https://github.com/rklaehn/sorted-iter.

use super::Comparator;
use std::cmp::{max, Ordering};
use std::iter::Peekable;

/// Visits the values representing the union of two *strictly* sorted iterators,
/// comparing with `compare`. Yields 2-tuples of type `(Option<U>, Option<V>)`,
///
/// Usage example:
///
/// ```
/// use sorted_iter::{NaturalComparator, Union};
///
/// fn using_union() {
///     let v1 = vec![3, 5];
///     let v2 = vec![2, 3];
///     let mut um = Union::new(
///         v1.into_iter(),
///         v2.into_iter(),
///         NaturalComparator::new(),
///     );
///     assert_eq!(um.next(), Some((None, Some(2))));
///     assert_eq!(um.next(), Some((Some(3), Some(3))));
///     assert_eq!(um.next(), Some((Some(5), None)));
///     assert_eq!(um.next(), None);
/// }
/// ```
pub struct Union<I, J, C>
where
    I: Iterator,
    J: Iterator,
    C: Comparator<I::Item, J::Item>,
{
    iter1: Peekable<I>,
    iter2: Peekable<J>,
    compare: C,
}

impl<I, J, C> Union<I, J, C>
where
    I: Iterator,
    J: Iterator,
    C: Comparator<I::Item, J::Item>,
{
    pub fn new(iter1: I, iter2: J, compare: C) -> Self {
        Self {
            iter1: iter1.peekable(),
            iter2: iter2.peekable(),
            compare,
        }
    }
}

impl<I, J, C> Clone for Union<I, J, C>
where
    I: Iterator + Clone,
    J: Iterator + Clone,
    C: Comparator<I::Item, J::Item> + Clone,
    I::Item: Clone,
    J::Item: Clone,
{
    fn clone(&self) -> Self {
        Self {
            iter1: self.iter1.clone(),
            iter2: self.iter2.clone(),
            compare: self.compare.clone(),
        }
    }
}

impl<I, J, C> Iterator for Union<I, J, C>
where
    I: Iterator,
    J: Iterator,
    C: Comparator<I::Item, J::Item>,
{
    type Item = (Option<I::Item>, Option<J::Item>);

    fn next(&mut self) -> Option<Self::Item> {
        match (self.iter1.peek(), self.iter2.peek()) {
            (Some(v1), Some(v2)) => match self.compare.compare(v1, v2) {
                Ordering::Less => Some((self.iter1.next(), None)),
                Ordering::Greater => Some((None, self.iter2.next())),
                Ordering::Equal => Some((self.iter1.next(), self.iter2.next())),
            },
            (Some(_), None) => Some((self.iter1.next(), None)),
            (None, Some(_)) => Some((None, self.iter2.next())),
            (None, None) => None,
        }
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
    use crate::{Comparator, Union};
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

    macro_rules! kv_pair {
        ($k1:expr, $v1:expr; $k2:expr, $v2:expr) => {
            (Some(kv!($k1, $v1)), Some(kv!($k2, $v2)))
        };
        ($k1:expr, $v1:expr; _) => {
            (Some(kv!($k1, $v1)), None)
        };
        (_; $k2:expr, $v2:expr) => {
            (None, Some(kv!($k2, $v2)))
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

    #[test]
    fn test_union() {
        // === CASE 1: empty inputs ===
        let v1 = vec_of_keyvalues(vec![], 1);
        let v2 = vec_of_keyvalues(vec![], 11);
        let mut um =
            Union::new(v1.into_iter(), v2.into_iter(), ComparatorOnKey);
        assert_eq!(um.next(), None);

        // === CASE 2: left input is empty ===
        let v1 = vec_of_keyvalues(vec![], 1);
        let v2 = vec_of_keyvalues(vec![1, 3, 5], 11);
        let mut um =
            Union::new(v1.into_iter(), v2.into_iter(), ComparatorOnKey);
        assert_eq!(um.next(), Some(kv_pair!(_; 1, 11)));
        assert_eq!(um.next(), Some(kv_pair!(_; 3, 12)));
        assert_eq!(um.next(), Some(kv_pair!(_; 5, 13)));
        assert_eq!(um.next(), None);

        // === CASE 3: right input is empty ===
        let v1 = vec_of_keyvalues(vec![1, 3, 5], 1);
        let v2 = vec_of_keyvalues(vec![], 11);
        let mut um =
            Union::new(v1.into_iter(), v2.into_iter(), ComparatorOnKey);
        assert_eq!(um.next(), Some(kv_pair!(1, 1; _)));
        assert_eq!(um.next(), Some(kv_pair!(3, 2; _)));
        assert_eq!(um.next(), Some(kv_pair!(5, 3; _)));
        assert_eq!(um.next(), None);

        // === CASE 4: disjoint inputs ===
        let v1 = vec_of_keyvalues(vec![1, 3, 5], 1);
        let v2 = vec_of_keyvalues(vec![2, 4, 6], 11);
        let mut um =
            Union::new(v1.into_iter(), v2.into_iter(), ComparatorOnKey);
        assert_eq!(um.next(), Some(kv_pair!(1, 1; _)));
        assert_eq!(um.next(), Some(kv_pair!(_; 2, 11)));
        assert_eq!(um.next(), Some(kv_pair!(3, 2; _)));
        assert_eq!(um.next(), Some(kv_pair!(_; 4, 12)));
        assert_eq!(um.next(), Some(kv_pair!(5, 3; _)));
        assert_eq!(um.next(), Some(kv_pair!(_; 6, 13)));
        assert_eq!(um.next(), None);

        // === CASE 5: overlapping inputs ===
        let v1 = vec_of_keyvalues(vec![2, 3, 5], 1);
        let v2 = vec_of_keyvalues(vec![2, 3, 5], 11);
        let mut um =
            Union::new(v1.into_iter(), v2.into_iter(), ComparatorOnKey);
        assert_eq!(um.next(), Some(kv_pair!(2, 1; 2, 11)));
        assert_eq!(um.next(), Some(kv_pair!(3, 2; 3, 12)));
        assert_eq!(um.next(), Some(kv_pair!(5, 3; 5, 13)));
        assert_eq!(um.next(), None);

        // === CASE 6: left subset right ===
        let v1 = vec_of_keyvalues(vec![3, 5, 8], 1);
        let v2 = vec_of_keyvalues(vec![2, 3, 4, 5, 8, 9], 11);
        let mut um =
            Union::new(v1.into_iter(), v2.into_iter(), ComparatorOnKey);
        assert_eq!(um.next(), Some(kv_pair!(_; 2, 11)));
        assert_eq!(um.next(), Some(kv_pair!(3, 1; 3, 12)));
        assert_eq!(um.next(), Some(kv_pair!(_; 4, 13)));
        assert_eq!(um.next(), Some(kv_pair!(5, 2; 5, 14)));
        assert_eq!(um.next(), Some(kv_pair!(8, 3; 8, 15)));
        assert_eq!(um.next(), Some(kv_pair!(_; 9, 16)));
        assert_eq!(um.next(), None);

        // === CASE 7: right subset left ===
        let v1 = vec_of_keyvalues(vec![2, 3, 4, 5, 8, 9], 1);
        let v2 = vec_of_keyvalues(vec![3, 5, 8], 11);
        let mut um =
            Union::new(v1.into_iter(), v2.into_iter(), ComparatorOnKey);
        assert_eq!(um.next(), Some(kv_pair!(2, 1; _)));
        assert_eq!(um.next(), Some(kv_pair!(3, 2; 3, 11)));
        assert_eq!(um.next(), Some(kv_pair!(4, 3; _)));
        assert_eq!(um.next(), Some(kv_pair!(5, 4; 5, 12)));
        assert_eq!(um.next(), Some(kv_pair!(8, 5; 8, 13)));
        assert_eq!(um.next(), Some(kv_pair!(9, 6; _)));
        assert_eq!(um.next(), None);

        // CASE 8: random 1 ===
        let v1 = vec_of_keyvalues(vec![2, 5, 6, 8], 1);
        let v2 = vec_of_keyvalues(vec![2, 3, 7, 8, 9], 11);
        let mut um =
            Union::new(v1.into_iter(), v2.into_iter(), ComparatorOnKey);
        assert_eq!(um.next(), Some(kv_pair!(2, 1; 2, 11)));
        assert_eq!(um.next(), Some(kv_pair!(_; 3, 12)));
        assert_eq!(um.next(), Some(kv_pair!(5, 2; _)));
        assert_eq!(um.next(), Some(kv_pair!(6, 3; _)));
        assert_eq!(um.next(), Some(kv_pair!(_; 7, 13)));
        assert_eq!(um.next(), Some(kv_pair!(8, 4; 8, 14)));
        assert_eq!(um.next(), Some(kv_pair!(_; 9, 15)));
    }
}

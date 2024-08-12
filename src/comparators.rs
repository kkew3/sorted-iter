use crate::Comparator;
use std::cmp::Ordering;
use std::marker::PhantomData;

/// A comparator that follow the natural ordering of objects.
pub struct NaturalComparator<U: Ord> {
    phantom: PhantomData<U>,
}

impl<U: Ord> NaturalComparator<U> {
    pub fn new() -> Self {
        Self {
            phantom: PhantomData::default(),
        }
    }
}

impl<U: Ord> Comparator<U, U> for NaturalComparator<U> {
    fn compare(&self, u: &U, v: &U) -> Ordering {
        u.cmp(v)
    }
}

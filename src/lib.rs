mod sorted_iter;

pub use sorted_iter::{Union, Intersection};

use std::cmp::Ordering;
use std::marker::PhantomData;

pub trait Comparator<U, V> {
    fn compare(&self, u: &U, v: &V) -> Ordering;
}

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

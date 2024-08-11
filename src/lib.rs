pub mod comparators;
mod sorted_iter;

pub use sorted_iter::{Difference, Intersection, Union};

use std::cmp::Ordering;

pub trait Comparator<U, V> {
    fn compare(&self, u: &U, v: &V) -> Ordering;
}

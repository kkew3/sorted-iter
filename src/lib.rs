pub mod comparators;
mod multiway_sorted_iter;
mod sorted_iter;

pub use sorted_iter::{Difference, Intersection, Union};
pub use multiway_sorted_iter::MultiWayUnion;

use std::cmp::Ordering;

pub trait Comparator<U, V> {
    fn compare(&self, u: &U, v: &V) -> Ordering;
}

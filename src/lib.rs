pub mod comparators;
mod multiway_sorted_iter;
mod sorted_iter;

pub use multiway_sorted_iter::{MultiWayIntersection, MultiWayUnion};
pub use sorted_iter::{Difference, Intersection, Union};

use std::cmp::Ordering;

pub trait Comparator<U, V> {
    fn compare(&self, u: &U, v: &V) -> Ordering;
}

pub fn box_iterator<'a, I: Iterator + 'a>(
    iter: I,
) -> Box<dyn Iterator<Item = I::Item> + 'a> {
    Box::new(iter)
}

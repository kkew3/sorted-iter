mod multiway_sorted_iter;
mod sorted_iter;

pub use multiway_sorted_iter::{
    MultiWayIntersection, MultiWayUnion, MultiWayUnionH,
};
pub use sorted_iter::{Difference, Intersection, Union};

pub fn box_iterator<'a, I: Iterator + 'a>(
    iter: I,
) -> Box<dyn Iterator<Item = I::Item> + 'a> {
    Box::new(iter)
}

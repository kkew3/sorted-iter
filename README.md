# Sorted Iter

## Introduction

`sorted-iter` contains a range of iterators that aggregate *strictly* sorted iterators into one iterator.
The provided iterators include two-way union (`sorted_iter::Union`), two-way intersection (`sorted_iter::Intersection`), difference (`sorted_iter::Difference`), K-way union (`sorted_iter::MultiWayUnion`) and K-way intersection (`sorted_iter::MultiWayIntersection`).
There are two main differences from other similar utilities.

1. The ordering of elements is decided by external comparators.
2. How to deduplicate elements is subject to the caller.

Point 1 indicates more flexibility, as now you may, when dealing with two-way operations, aggregate two different types of objects.
Point 2 implies that, e.g. when computing the intersection, the intersection iterator won't decide for you and drop all elements but the one from the first iterator.
This makes a lot of sense if you want to compute the intersection in terms of one ordering, but do subsequent works in terms of another ordering.

## Caveats

The ordering of the input iterators is assumed, and will be checked neither at compile-time nor during runtime.

## Installation

In `Cargo.toml`, under `[dependencies]` section,

```toml
sorted-iter = { git = "https://github.com/kkew3/sorted-iter.git" }
compare = "0.1"
```

## Tutorial

Steps to build an aggregate iterator:

1. Define your own comparator by implementing `compare::Compare` trait, or use `compare::natural()` if the input items already implement `Ord`.
2. Instantiate an aggregating iterator by calling its `new` associated function. For K-way operation, there's also a `from_boxed` associated function if your iterators are of different types. You may box your iterators using `sorted_iter::box_iterator` function.
3. Iterate over the instantiated iterator.

## Example

Two-way operation:

```rust
use sorted_iter::Union;

fn using_union() {
    let v1 = vec![3, 5];
    let v2 = vec![2, 3];
    let mut um = Union::new(
        v1.into_iter(),
        v2.into_iter(),
        compare::natural(),
    );
    assert_eq!(um.next(), Some((None, Some(2))));
    assert_eq!(um.next(), Some((Some(3), Some(3))));
    assert_eq!(um.next(), Some((Some(5), None)));
    assert_eq!(um.next(), None);
}
```

K-way operation:

```rust
use sorted_iter::MultiWayUnion;

fn using_multiway_union() {
    let v1 = vec![3, 5];
    let v2 = vec![2, 3];
    let v3 = vec![2, 3, 5];
    let mut um = MultiWayUnion::new(
        [v1.into_iter(), v2.into_iter(), v3.into_iter()],
        compare::natural(),
    );
    assert_eq!(um.next(), Some(vec![None, Some(2), Some(2)]));
    assert_eq!(um.next(), Some(vec![Some(3), Some(3), Some(3)]));
    assert_eq!(um.next(), Some(vec![Some(5), None, Some(5)]));
    assert_eq!(um.next(), None);
}
```

## Documentation

The iterators provided can be found in re-exports defined in `src/lib.rs`.

For detail, see doc comments in the source code.

## Acknowledgement

This project is partially inspired by [rklaehn/sorted-iter](https://github.com/rklaehn/sorted-iter).

## License

MIT.

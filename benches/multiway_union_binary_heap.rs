use crate::binary_heap_impl::BinaryHeapMultiWayUnion;
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};
use rand::prelude::*;
use rand_chacha::ChaChaRng;
use sorted_iter::MultiWayUnion;
use std::ops::Range;
use std::vec::IntoIter;

mod binary_heap_impl {
    use std::cmp::{Ordering, Reverse};
    use std::collections::BinaryHeap;

    struct IndexedPeekedIterator<I: Iterator> {
        inner: I,
        buf: Option<(I::Item, usize)>,
    }

    impl<I: Iterator> IndexedPeekedIterator<I> {
        fn new(mut inner: I, index: usize) -> Option<Self> {
            let buf = inner.next().map(|item| (item, index));
            buf.map(|buf| Self {
                inner,
                buf: Some(buf),
            })
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
                Some(_) => (min + 1, max.map(|max| max + 1)),
            }
        }
    }

    impl<I: Iterator> PartialEq for IndexedPeekedIterator<I>
    where
        I::Item: PartialEq,
    {
        fn eq(&self, other: &Self) -> bool {
            self.buf.eq(&other.buf)
        }
    }

    impl<I: Iterator> Eq for IndexedPeekedIterator<I> where I::Item: Eq {}

    impl<I: Iterator> PartialOrd for IndexedPeekedIterator<I>
    where
        I::Item: PartialOrd,
    {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            match (&self.buf, &other.buf) {
                (Some(a), Some(b)) => a.partial_cmp(b),
                (Some(_), None) => Some(Ordering::Less),
                (None, Some(_)) => Some(Ordering::Greater),
                (None, None) => Some(Ordering::Equal),
            }
        }
    }

    impl<I: Iterator> Ord for IndexedPeekedIterator<I>
    where
        I::Item: Ord,
    {
        fn cmp(&self, other: &Self) -> Ordering {
            match (&self.buf, &other.buf) {
                (Some(a), Some(b)) => a.cmp(b),
                (Some(_), None) => Ordering::Less,
                (None, Some(_)) => Ordering::Greater,
                (None, None) => Ordering::Equal,
            }
        }
    }

    pub struct BinaryHeapMultiWayUnion<'a, T: Ord> {
        bh: BinaryHeap<
            Reverse<IndexedPeekedIterator<Box<dyn Iterator<Item = T> + 'a>>>,
        >,
        len: usize,
    }

    impl<'a, T: Ord> BinaryHeapMultiWayUnion<'a, T> {
        pub fn new<I: Iterator<Item = T> + 'a>(
            iters: impl IntoIterator<Item = I>,
        ) -> Self {
            let mut i = 0usize;
            let bh =
                BinaryHeap::from_iter(iters.into_iter().filter_map(|it| {
                    let it = IndexedPeekedIterator::new(
                        Box::new(it) as Box<dyn Iterator<Item = T>>,
                        i,
                    );
                    i += 1;
                    it.map(Reverse)
                }));
            Self { bh, len: i }
        }
    }

    impl<'a, T: Ord> Iterator for BinaryHeapMultiWayUnion<'a, T> {
        type Item = Vec<Option<T>>;

        fn next(&mut self) -> Option<Self::Item> {
            match self.bh.pop() {
                None => None,
                Some(Reverse(mut p)) => {
                    let (value, index) = p.next().unwrap();
                    let mut ret: Vec<Option<T>> = Vec::with_capacity(self.len);
                    for _ in 0..self.len {
                        ret.push(None);
                    }

                    if p.peek().is_some() {
                        self.bh.push(Reverse(p));
                    }
                    while self
                        .bh
                        .peek()
                        .filter(|rit| rit.0.peek().unwrap().0.eq(&value))
                        .is_some()
                    {
                        let mut p = self.bh.pop().unwrap().0;
                        let (value, index) = p.next().unwrap();
                        ret.get_mut(index).unwrap().get_or_insert(value);
                        if p.peek().is_some() {
                            self.bh.push(Reverse(p));
                        }
                    }
                    ret.get_mut(index).unwrap().get_or_insert(value);
                    Some(ret)
                }
            }
        }
    }

    #[cfg(test)]
    mod multiway_union_tests {
        use std::cmp::Ordering;

        #[derive(Debug)]
        struct OrdByFirstWrapper((i32, char));

        impl From<(i32, char)> for OrdByFirstWrapper {
            fn from(value: (i32, char)) -> Self {
                Self(value)
            }
        }

        impl From<OrdByFirstWrapper> for (i32, char) {
            fn from(value: OrdByFirstWrapper) -> Self {
                value.0
            }
        }

        impl PartialEq for OrdByFirstWrapper {
            fn eq(&self, other: &Self) -> bool {
                self.0 .0 == other.0 .0
            }
        }

        impl Eq for OrdByFirstWrapper {}

        impl PartialOrd for OrdByFirstWrapper {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.0 .0.partial_cmp(&other.0 .0)
            }
        }

        impl Ord for OrdByFirstWrapper {
            fn cmp(&self, other: &Self) -> Ordering {
                self.0 .0.cmp(&other.0 .0)
            }
        }

        #[allow(unused_macros)]
        macro_rules! ordbyfirst_vec {
            () => {
                {
                    let v: Vec<OrdByFirstWrapper> = vec![];
                    v
                }
            };
            ( $( ($value:literal, $label:literal) ),+ ) => {
                {
                    let v = vec![ $( OrdByFirstWrapper::from(($value, $label)) ),+ ];
                    v
                }
            };
        }

        #[allow(unused)]
        trait IntoLiteralHelper {
            fn into_lit(self) -> Option<Vec<Option<(i32, char)>>>;
        }

        impl IntoLiteralHelper for Option<Vec<Option<OrdByFirstWrapper>>> {
            fn into_lit(self) -> Option<Vec<Option<(i32, char)>>> {
                self.map(|v| v.into_iter().map(|e| e.map(Into::into)).collect())
            }
        }

        #[test]
        fn test_iterator() {
            let a = ordbyfirst_vec![(1, 'a'), (3, 'b'), (5, 'c'), (6, 'd')]
                .into_iter();
            let b = ordbyfirst_vec![
                (0, 'e'),
                (1, 'f'),
                (4, 'g'),
                (5, 'h'),
                (7, 'i')
            ]
            .into_iter();
            let c = ordbyfirst_vec![
                (0, 'j'),
                (2, 'k'),
                (3, 'l'),
                (5, 'm'),
                (7, 'n'),
                (8, 'o')
            ]
            .into_iter();
            let mut u = super::BinaryHeapMultiWayUnion::new([a, b, c]);
            assert_eq!(u.bh.len(), 3);
            assert_eq!(
                u.next().into_lit(),
                Some(vec![None, Some((0, 'e')), Some((0, 'j'))])
            );
            assert_eq!(
                u.next().into_lit(),
                Some(vec![Some((1, 'a')), Some((1, 'f')), None])
            );
            assert_eq!(
                u.next().into_lit(),
                Some(vec![None, None, Some((2, 'k'))])
            );
            assert_eq!(
                u.next().into_lit(),
                Some(vec![Some((3, 'b')), None, Some((3, 'l'))])
            );
            assert_eq!(
                u.next().into_lit(),
                Some(vec![None, Some((4, 'g')), None])
            );
            assert_eq!(
                u.next().into_lit(),
                Some(vec![Some((5, 'c')), Some((5, 'h')), Some((5, 'm'))])
            );
            assert_eq!(
                u.next().into_lit(),
                Some(vec![Some((6, 'd')), None, None])
            );
            assert_eq!(
                u.next().into_lit(),
                Some(vec![None, Some((7, 'i')), Some((7, 'n'))])
            );
            assert_eq!(
                u.next().into_lit(),
                Some(vec![None, None, Some((8, 'o'))])
            );
            assert_eq!(u.next().into_lit(), None);
            assert_eq!(u.next().into_lit(), None);
        }

        #[test]
        fn test_iterator_single() {
            let a = ordbyfirst_vec![(1, 'a'), (3, 'b'), (5, 'c'), (6, 'd')]
                .into_iter();
            let mut u = super::BinaryHeapMultiWayUnion::new([a]);
            assert_eq!(u.next().into_lit(), Some(vec![Some((1, 'a'))]));
            assert_eq!(u.next().into_lit(), Some(vec![Some((3, 'b'))]));
            assert_eq!(u.next().into_lit(), Some(vec![Some((5, 'c'))]));
            assert_eq!(u.next().into_lit(), Some(vec![Some((6, 'd'))]));
            assert_eq!(u.next().into_lit(), None);
            assert_eq!(u.next().into_lit(), None);
        }
    }
}

/// Random initialize `n_seq` strictly ascending arrays of varying lengths.
fn build_bench_case(
    n_seq: usize,
    delta_range: Range<u32>,
    len_range: Range<usize>,
) -> Vec<IntoIter<u32>> {
    assert!(delta_range.start > 0);
    let mut rng = ChaChaRng::seed_from_u64(12345);
    (0..n_seq)
        .map(|_| {
            let len = rng.gen_range(len_range.clone());
            let v = (0..len).map(|_| rng.gen_range(delta_range.clone()));
            let mut acc: u32 = 0;
            let v: Vec<_> = v
                .map(|x| {
                    acc += x;
                    acc
                })
                .collect();
            v.into_iter()
        })
        .collect()
}

fn tt_count(bench_case: Vec<IntoIter<u32>>) {
    MultiWayUnion::new(bench_case, compare::natural()).count();
}

fn bh_count(bench_case: Vec<IntoIter<u32>>) {
    BinaryHeapMultiWayUnion::new(bench_case).count();
}

fn bench_multiway_union(c: &mut Criterion) {
    let mut group = c.benchmark_group("MultiWayUnion");
    let param_desc = "n_seq=256, delta_range=1..16, len_range=10000..20000";
    let bench_case = build_bench_case(256, 1..16, 10000..20000);
    group.bench_function(BenchmarkId::new("TT", param_desc), |b| {
        b.iter(|| tt_count(black_box(bench_case.clone())))
    });
    group.bench_function(BenchmarkId::new("BH", param_desc), |b| {
        b.iter(|| bh_count(black_box(bench_case.clone())))
    });
    group.finish();
}

criterion_group!(benches, bench_multiway_union);
criterion_main!(benches);

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Streaming selection for flat search. Sorting is deferred until the scan ends.

use std::{cmp::Ordering, collections::BinaryHeap, num::NonZeroUsize};

use diskann_wide::{SIMDMask, SIMDPartialOrd, SIMDVector};

use super::{DistanceHandler, DistancesUnordered};
use crate::{ANNResult, error::ErrorExt, neighbor::Neighbor};

#[inline]
pub(super) async fn collect<V: DistancesUnordered>(
    visitor: &mut V,
    k: NonZeroUsize,
    reservoir_min_k: usize,
) -> ANNResult<(Vec<Neighbor<V::Id>>, u32)> {
    // Choose outside the callback so the scan has a single, monomorphized collector.
    if k.get() == 1 {
        scan(visitor, Top1 { best: None }).await
    } else if k.get() < reservoir_min_k {
        scan(visitor, Heap::new(k.get())).await
    } else {
        scan(visitor, Reservoir::new(k.get())).await
    }
}

#[inline]
async fn scan<V, C>(visitor: &mut V, collector: C) -> ANNResult<(Vec<Neighbor<V::Id>>, u32)>
where
    V: DistancesUnordered,
    C: Collector<V::Id> + Send,
{
    let mut handler = CountingHandler { collector, cmps: 0 };
    visitor
        .distances_unordered_with(&mut handler)
        .await
        .escalate("flat scan must complete to produce correct k-NN results")?;
    Ok((handler.collector.finish(), handler.cmps))
}

struct CountingHandler<C> {
    collector: C,
    cmps: u32,
}

impl<I: Copy, C: Collector<I> + Send> DistanceHandler<I> for CountingHandler<C> {
    #[inline]
    fn prefers_buffered_batches(&self) -> bool {
        C::PREFER_BUFFERED_BATCHES
    }

    #[inline]
    fn add(&mut self, id: I, distance: f32) {
        self.cmps += 1;
        self.collector.insert(Neighbor::new(id, distance));
    }

    #[inline]
    #[track_caller]
    fn add_batch(&mut self, ids: &[I], distances: &[f32]) {
        assert_eq!(ids.len(), distances.len(), "ID and distance lengths differ");
        self.cmps += distances.len() as u32;
        self.collector.insert_batch(ids, distances);
    }
}

trait Collector<I: Copy> {
    const PREFER_BUFFERED_BATCHES: bool = true;

    fn threshold(&self) -> f32;
    fn insert(&mut self, neighbor: Neighbor<I>);
    fn finish(self) -> Vec<Neighbor<I>>;

    #[inline]
    fn insert_batch(&mut self, ids: &[I], distances: &[f32]) {
        let (groups, tail) = distances.as_chunks::<8>();
        for (group, distances) in groups.iter().enumerate() {
            let mut eligible = eligible_lanes(distances, self.threshold());
            if eligible == u8::MAX {
                // Dense groups (for example, improving distances) do not benefit
                // from walking a bitmask. Keep the scalar loop easy to unroll.
                for (&id, &distance) in ids[group * 8..group * 8 + 8].iter().zip(distances) {
                    self.insert(Neighbor::new(id, distance));
                }
                continue;
            }
            while eligible != 0 {
                let lane = eligible.trailing_zeros() as usize;
                eligible &= eligible - 1;
                // Earlier insertions can lower the threshold. insert rechecks it
                // and preserves later-arrival preference by consuming lanes in order.
                self.insert(Neighbor::new(ids[group * 8 + lane], distances[lane]));
            }
        }
        for (&id, &distance) in ids[groups.len() * 8..].iter().zip(tail) {
            self.insert(Neighbor::new(id, distance));
        }
    }
}

#[inline]
fn eligible_lanes(distances: &[f32; 8], threshold: f32) -> u8 {
    diskann_wide::alias!(f32s = f32x8);
    // SAFETY: f32x8 reads exactly the eight initialized values in distances.
    // ARCH only enables instructions supported by the compilation target.
    let values = unsafe { f32s::load_simd(diskann_wide::ARCH, distances.as_ptr()) };
    // Ordered <= excludes NaNs, admits infinities, and treats signed zeros as ties.
    values
        .le_simd(f32s::splat(diskann_wide::ARCH, threshold))
        .bitmask()
        .to_underlying()
}

/// A persistent collector for producers that interleave blocks from multiple queries.
#[cfg(feature = "flat-gemm")]
pub(super) struct BatchCollector<I>(BatchStrategy<I>);

#[cfg(feature = "flat-gemm")]
enum BatchStrategy<I> {
    Top1(Top1<I>),
    Heap(Heap<I>),
    Reservoir(Reservoir<I>),
}

#[cfg(feature = "flat-gemm")]
impl<I: Copy> BatchCollector<I> {
    pub(super) fn new(k: NonZeroUsize, reservoir_min_k: usize) -> Self {
        Self(if k.get() == 1 {
            BatchStrategy::Top1(Top1 { best: None })
        } else if k.get() < reservoir_min_k {
            BatchStrategy::Heap(Heap::new(k.get()))
        } else {
            BatchStrategy::Reservoir(Reservoir::new(k.get()))
        })
    }

    pub(super) fn add_batch(&mut self, ids: &[I], distances: &[f32]) {
        assert_eq!(ids.len(), distances.len(), "ID and distance lengths differ");
        match &mut self.0 {
            BatchStrategy::Top1(c) => c.insert_batch(ids, distances),
            BatchStrategy::Heap(c) => c.insert_batch(ids, distances),
            BatchStrategy::Reservoir(c) => c.insert_batch(ids, distances),
        }
    }

    pub(super) fn finish(self) -> Vec<Neighbor<I>> {
        match self.0 {
            BatchStrategy::Top1(c) => c.finish(),
            BatchStrategy::Heap(c) => c.finish(),
            BatchStrategy::Reservoir(c) => c.finish(),
        }
    }
}

struct Top1<I> {
    best: Option<Neighbor<I>>,
}

impl<I: Copy> Collector<I> for Top1<I> {
    // Updating one best candidate is cheap. Avoid creating distance buffers for
    // it, while retaining SIMD screening when a producer already has a block.
    const PREFER_BUFFERED_BATCHES: bool = false;

    #[inline]
    fn threshold(&self) -> f32 {
        self.best.as_ref().map_or(f32::INFINITY, |n| *n.distance())
    }

    #[inline]
    fn insert(&mut self, neighbor: Neighbor<I>) {
        let distance = *neighbor.distance();
        // Accept infinities even before any finite candidate, and replace ties
        // (including signed zero) to preserve the queue's later-arrival preference.
        if !distance.is_nan()
            && self
                .best
                .as_ref()
                .is_none_or(|best| distance <= *best.distance())
        {
            self.best = Some(neighbor);
        }
    }

    fn finish(self) -> Vec<Neighbor<I>> {
        self.best.into_iter().collect()
    }
}

struct Entry<I> {
    neighbor: Neighbor<I>,
    // Preserve the sorted queue's preference for later arrivals on distance ties.
    arrival: usize,
}

impl<I> PartialEq for Entry<I> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}
impl<I> Eq for Entry<I> {}
impl<I> PartialOrd for Entry<I> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<I> Ord for Entry<I> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Insertion excludes NaNs. partial_cmp treats signed zeros as equal, just
        // like NeighborPriorityQueue; total_cmp would change boundary membership.
        self.neighbor
            .distance()
            .partial_cmp(other.neighbor.distance())
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.arrival.cmp(&self.arrival))
    }
}

struct Heap<I> {
    entries: BinaryHeap<Entry<I>>,
    k: usize,
    arrival: usize,
}

impl<I> Heap<I> {
    fn new(k: usize) -> Self {
        Self {
            entries: BinaryHeap::with_capacity(k),
            k,
            arrival: 0,
        }
    }
}

impl<I: Copy> Collector<I> for Heap<I> {
    #[inline]
    fn threshold(&self) -> f32 {
        if self.entries.len() < self.k {
            f32::INFINITY
        } else {
            self.entries
                .peek()
                .map_or(f32::INFINITY, |e| *e.neighbor.distance())
        }
    }

    #[inline]
    fn insert(&mut self, neighbor: Neighbor<I>) {
        let distance = *neighbor.distance();
        if distance.is_nan() {
            return;
        }
        if self.entries.len() == self.k
            && self
                .entries
                .peek()
                .is_some_and(|worst| distance > *worst.neighbor.distance())
        {
            return;
        }
        let entry = Entry {
            neighbor,
            arrival: self.arrival,
        };
        self.arrival += 1;
        if self.entries.len() < self.k {
            self.entries.push(entry);
        } else if let Some(mut worst) = self.entries.peek_mut() {
            *worst = entry;
        }
    }

    fn finish(self) -> Vec<Neighbor<I>> {
        self.entries
            .into_sorted_vec()
            .into_iter()
            .map(|entry| entry.neighbor)
            .collect()
    }
}

struct Reservoir<I> {
    entries: Vec<Entry<I>>,
    k: usize,
    limit: usize,
    arrival: usize,
    threshold: f32,
}

impl<I> Reservoir<I> {
    fn new(k: usize) -> Self {
        let limit = k.saturating_mul(2);
        Self {
            entries: Vec::with_capacity(limit),
            k,
            limit,
            arrival: 0,
            threshold: f32::INFINITY,
        }
    }

    fn shrink(&mut self) {
        let (_, pivot, _) = self.entries.select_nth_unstable(self.k - 1);
        self.threshold = *pivot.neighbor.distance();
        self.entries.truncate(self.k);
    }
}

impl<I: Copy> Collector<I> for Reservoir<I> {
    #[inline]
    fn threshold(&self) -> f32 {
        self.threshold
    }

    #[inline]
    fn insert(&mut self, neighbor: Neighbor<I>) {
        let distance = *neighbor.distance();
        if distance.is_nan() || distance > self.threshold {
            return;
        }
        self.entries.push(Entry {
            neighbor,
            arrival: self.arrival,
        });
        self.arrival += 1;
        if self.entries.len() == self.limit {
            self.shrink();
        }
    }

    fn finish(mut self) -> Vec<Neighbor<I>> {
        if self.entries.len() > self.k {
            self.shrink();
        }
        self.entries.sort_unstable();
        self.entries
            .into_iter()
            .map(|entry| entry.neighbor)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neighbor::NeighborPriorityQueue;

    fn assert_matches_queue(distances: &[f32], k: usize) {
        let mut reference = NeighborPriorityQueue::new(k);
        let mut heap = Heap::new(k);
        let mut reservoir = Reservoir::new(k);
        let mut top1 = Top1 { best: None };
        for (id, &distance) in distances.iter().enumerate() {
            let neighbor = Neighbor::new(id as u128, distance);
            reference.insert(neighbor);
            heap.insert(neighbor);
            reservoir.insert(neighbor);
            top1.insert(neighbor);
        }
        let bits = |n: Neighbor<u128>| (*n.id(), n.distance().to_bits());
        let expected: Vec<_> = reference.iter().map(bits).collect();
        assert_eq!(
            top1.finish().into_iter().map(bits).collect::<Vec<_>>(),
            expected.iter().take(1).copied().collect::<Vec<_>>()
        );
        assert_eq!(
            heap.finish().into_iter().map(bits).collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            reservoir.finish().into_iter().map(bits).collect::<Vec<_>>(),
            expected
        );
        let ids: Vec<_> = (0..distances.len() as u128).collect();
        for batch_size in [1, 7, 8, 9, 31, 32, 33, 255] {
            let mut heap = CountingHandler {
                collector: Heap::new(k),
                cmps: 0,
            };
            let mut reservoir = CountingHandler {
                collector: Reservoir::new(k),
                cmps: 0,
            };
            let mut top1 = CountingHandler {
                collector: Top1 { best: None },
                cmps: 0,
            };
            for (ids, distances) in ids.chunks(batch_size).zip(distances.chunks(batch_size)) {
                heap.add_batch(ids, distances);
                reservoir.add_batch(ids, distances);
                top1.add_batch(ids, distances);
            }
            assert_eq!(heap.cmps as usize, distances.len());
            assert_eq!(reservoir.cmps, heap.cmps);
            assert_eq!(top1.cmps, heap.cmps);
            assert_eq!(
                heap.collector
                    .finish()
                    .into_iter()
                    .map(bits)
                    .collect::<Vec<_>>(),
                expected
            );
            assert_eq!(
                reservoir
                    .collector
                    .finish()
                    .into_iter()
                    .map(bits)
                    .collect::<Vec<_>>(),
                expected
            );
            assert_eq!(
                top1.collector
                    .finish()
                    .into_iter()
                    .map(bits)
                    .collect::<Vec<_>>(),
                expected.iter().take(1).copied().collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn matches_sorted_queue_across_orders_and_boundaries() {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(0x1824_99af);
        let random: Vec<f32> = (0..4097).map(|_| rng.random_range(-2.0..2.0)).collect();
        let ties: Vec<f32> = random.iter().map(|d| (d * 8.0).floor()).collect();
        let mut ascending = random.clone();
        ascending.sort_by(f32::total_cmp);
        let descending: Vec<_> = ascending.iter().rev().copied().collect();
        for distances in [&random, &ties, &ascending, &descending] {
            for k in [1, 2, 10, 99, 100, 101, 1000, 4096, 5000] {
                assert_matches_queue(distances, k);
            }
        }
    }

    #[test]
    fn empty_nan_infinity_and_signed_zero() {
        for k in [1, 3, 10, 100] {
            assert_matches_queue(&[], k);
            assert_matches_queue(&[f32::NAN; 257], k);
            let distances: Vec<_> = [
                f32::NAN,
                f32::NEG_INFINITY,
                0.0,
                -0.0,
                1.0,
                f32::INFINITY,
                0.0,
                -0.0,
            ]
            .into_iter()
            .cycle()
            .take(1025)
            .collect();
            assert_matches_queue(&distances, k);
            assert_matches_queue(&[f32::INFINITY; 257], k);
            assert_matches_queue(
                &[0.0, -0.0]
                    .into_iter()
                    .cycle()
                    .take(257)
                    .collect::<Vec<_>>(),
                k,
            );
        }
    }
}

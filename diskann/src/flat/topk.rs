/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Streaming selection for flat search. Sorting is deferred until the scan ends.

use std::{cmp::Ordering, collections::BinaryHeap, num::NonZeroUsize};

use super::DistancesUnordered;
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
async fn scan<V, C>(visitor: &mut V, mut collector: C) -> ANNResult<(Vec<Neighbor<V::Id>>, u32)>
where
    V: DistancesUnordered,
    C: Collector<V::Id> + Send,
{
    let mut cmps = 0;
    visitor
        .distances_unordered(|id, distance| {
            cmps += 1;
            collector.insert(Neighbor::new(id, distance));
        })
        .await
        .escalate("flat scan must complete to produce correct k-NN results")?;
    Ok((collector.finish(), cmps))
}

trait Collector<I> {
    fn insert(&mut self, neighbor: Neighbor<I>);
    fn finish(self) -> Vec<Neighbor<I>>;
}

struct Top1<I> {
    best: Option<Neighbor<I>>,
}

impl<I> Collector<I> for Top1<I> {
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

impl<I> Collector<I> for Heap<I> {
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

impl<I> Collector<I> for Reservoir<I> {
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

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The traits a provider implements: the accessors through which the index reads and writes
//! lists.

use std::{fmt::Debug, hash::Hash, num::NonZeroUsize};

use diskann_utils::future::SendFuture;

use super::region::Region;
use crate::{
    ANNResult,
    neighbor::{Neighbor, NeighborPriorityQueue, NeighborPriorityQueueIdType},
};

/// The list and vector ID types used by a subtrait.
pub trait HasIds {
    /// Internal ID of a point, as recorded in list memberships.
    type Id: Copy + Send;
    /// ID of a list.
    type ListId: Copy + Ord + Hash + Debug + Send + Sync + 'static;
}

///////////
// Build //
///////////

/// The main extension point for the build operation on an IVF index.
///
/// The index uses an accessor for one operation, split into stages, in this order:
///
/// 1. [`route`](Self::route) stages the batch of points `B`, and inserts them
///    to existing lists.
///
/// 2. Any number of stages follow. A stage calls [`fill`](Self::fill), records changes
///    through the returned [`Region`], and `finish`es the region. Between stages, the index
///    may look up lists through [`find`](Self::find).
///
/// 3. [`finish`](Self::finish) publishes the operation.
///
/// # Important
///
/// - Dropping the accessor without calling `finish` abandons the operation,
///   and nothing it staged or recorded should be committed.
///
/// - [`fill`](Self::fill) and [`Self::find`] reflect the routed
///   appends and **every** region `finish`ed earlier in the operation.
pub trait BuildAccessor<B>: Send + Sized + HasIds {
    /// Borrowed vectors of one list.
    ///
    /// This is the reference type of [`Region::VectorsRef`] in [`Self::Region`], and
    /// is lifted here for convenience.
    type VectorsRef<'a>;

    /// The [`Region`] object returned by [`Self::fill`].
    type Region<'a>: for<'b> Region<Self::Id, Self::ListId, VectorsRef<'b> = Self::VectorsRef<'b>>
    where
        Self: 'a;

    /// Lookup the nearest neighbor of a list, with filtering.
    ///
    /// Emit up to `nprobe` distinct [live](Region) lists nearest the centroid of list `id`,
    /// among those accepted by `pred`, and return how many were emitted. Implementations
    /// need not return exactly `nprobe` items.
    ///
    /// - `pred` filters candidates based on id. This can be used to exclude some ids when looking
    ///   for neighbors of a list.
    ///
    /// - `emit` is a callback applied to the ids and distances of the top results (that pass the filter).
    ///
    /// # Errors
    ///
    /// Fails if `id` is not live or the provider cannot complete the lookup.
    fn find<P, F>(
        &self,
        id: Self::ListId,
        nprobe: NonZeroUsize,
        pred: P,
        emit: F,
    ) -> impl SendFuture<ANNResult<usize>>
    where
        P: Fn(Self::ListId) -> bool + Send,
        F: FnMut(Self::ListId, f32) + Send;

    /// Stage a batch of `points`, assign each point to a list(s), and return the
    /// number of points staged.
    ///
    /// `emit` is callback that is applied to the (list id, list size) tuple. The list
    /// size should **include** the number of the points inserted in this call, in addition
    /// to existing points.
    ///
    /// # Errors
    ///
    /// Fails if `points` is empty, or if a point cannot be staged or routed.
    fn route<F>(&mut self, points: B, emit: F) -> impl SendFuture<ANNResult<NonZeroUsize>>
    where
        F: FnMut(Self::ListId, NonZeroUsize) + Send;

    /// Fill the centroids, members, and vectors of `lists` into a [`Region`].
    ///
    /// The region can be scoped to the list ids in `lists`, and may reject or throw errors when
    /// accessing or applying updates to ids outside of `lists`.
    fn fill<Itr>(&mut self, lists: Itr) -> impl SendFuture<ANNResult<Self::Region<'_>>>
    where
        Itr: Iterator<Item = Self::ListId> + Clone + Send + Sync;

    /// Publish the operation. The staged points, their appends, and the changes of every
    /// finished region become visible together.
    fn finish(self) -> impl SendFuture<ANNResult<()>>;
}

////////////
// Search //
////////////

/// Candidate collection with independent partial states.
pub trait Accumulator<I>: Send + Sized {
    /// Create an empty accumulator with the same selection policy and configuration.
    fn empty_like(&self) -> Self;

    /// Consider one candidate for inclusion in the results.
    fn push(&mut self, id: I, distance: f32);

    /// Combine a compatible partial state, such as one created by [`Self::empty_like`].
    fn merge(&mut self, other: Self);
}

/// Selects and scans lists for one query.
pub trait SearchAccessor: Send + Sized + HasIds {
    /// Scan the `nprobe` lists nearest the accessor's query, pushing **every** member's
    /// `(id, distance)` pair into `results`.
    ///
    /// A concrete implementation may scan fewer lists than `nprobe` if there aren't enough lists.
    ///
    /// Implementations can specialize their implementation
    /// and use [`Accumulator::empty_like`] to use indepedently accumulators that they merge into
    /// `results`.
    fn scan<A>(&self, nprobe: NonZeroUsize, results: &mut A) -> impl SendFuture<ANNResult<()>>
    where
        A: Accumulator<Self::Id>;
}

impl<I: NeighborPriorityQueueIdType> Accumulator<I> for NeighborPriorityQueue<I> {
    fn empty_like(&self) -> Self {
        Self::new(self.search_l())
    }

    fn push(&mut self, id: I, distance: f32) {
        self.insert(Neighbor::new(id, distance));
    }

    fn merge(&mut self, other: Self) {
        for neighbor in other.iter() {
            self.insert(neighbor);
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};

    use super::*;
    use crate::test::cmp::assert_eq_verbose;

    #[test]
    fn empty_like_keeps_capacity_and_drops_candidates() {
        let mut queue = NeighborPriorityQueue::new(3);
        queue.push(1u32, 1.0);
        queue.push(2, 2.0);

        let empty = queue.empty_like();
        assert_eq!(empty.size(), 0);
        assert_eq!(empty.search_l(), 3);
    }

    /// Compare merging randomly partitioned candidates with inserting them into one
    /// fixed-capacity queue.
    fn merge_fuzz_impl(capacity: usize, parts: usize, pushes: u32, rng: &mut StdRng) {
        // Distinct distances, so the best candidates are unique.
        let mut distances: Vec<f32> = (0..pushes).map(|d| d as f32).collect();
        distances.shuffle(rng);

        let mut whole = NeighborPriorityQueue::new(capacity);
        let mut partials: Vec<_> = (0..parts).map(|_| whole.empty_like()).collect();
        for (id, &distance) in (0..).zip(&distances) {
            whole.push(id, distance);
            partials[rng.random_range(0..parts)].push(id, distance);
        }

        let mut merged = whole.empty_like();
        for partial in partials {
            merged.merge(partial);
        }

        let best = |queue: &NeighborPriorityQueue<u32>| queue.iter().collect::<Vec<_>>();
        assert_eq_verbose!(best(&merged), best(&whole));
    }

    #[test]
    fn merge_fuzz() {
        let mut rng = StdRng::seed_from_u64(0x3c6ef372fe94f82b);
        for capacity in [1, 3, 10] {
            for parts in [1, 2, 4] {
                for pushes in [0, 5, 40] {
                    merge_fuzz_impl(capacity, parts, pushes, &mut rng);
                }
            }
        }
    }
}

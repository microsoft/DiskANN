/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The traits a provider implements: strategies that create accessors, and the accessors
//! through which the index reads and writes lists.

use std::{fmt::Debug, hash::Hash, num::NonZeroUsize};

use diskann_utils::future::SendFuture;

use super::region::Region;
use crate::{
    ANNResult,
    neighbor::{Neighbor, NeighborPriorityQueue, NeighborPriorityQueueIdType},
};

/// The list and vector ID types used by a subtrait.
///
/// The IDs are unconstrained here; each operation of the index states what it needs of them.
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
///    may look up lists through [`nearest`](Self::nearest).
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
    /// for neighbors of a list.  
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

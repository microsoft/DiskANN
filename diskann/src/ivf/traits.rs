/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Storage boundaries for IVF construction and search.

use std::num::NonZeroUsize;

use diskann_utils::future::SendFuture;

use super::region::{Update, View};
use crate::{
    ANNResult,
    error::StandardError,
    neighbor::{Neighbor, NeighborPriorityQueue, NeighborPriorityQueueIdType},
    provider::ExecutionContext,
    utils::VectorId,
};

pub trait HasIds {
    type Id: VectorId;
    type ListId: VectorId;
}

pub trait NearestList: Send {
    type Id;
    /// Select `nprobe` distinct live lists nearest to `id`'s centroid, that pass the filter.
    ///
    /// Results are nearest first and may include `id` itself.
    ///
    /// # Errors
    ///
    /// Fails if `id` is not live or the provider cannot complete the lookup.
    fn find<P, F>(
        &self,
        id: Self::Id,
        nprobe: NonZeroUsize,
        pred: P,
        emit: F,
    ) -> impl SendFuture<ANNResult<usize>>
    where
        P: Fn(&Self::Id) -> bool + Send,
        F: FnMut(&Self::Id, f32) + Send;
}

pub trait BuildAccessor<B>: Send + Sized + HasIds {
    type Nearest<'a>: NearestList<Id = Self::ListId>
    where
        Self: 'a;

    type VectorsRef<'a>;

    type Region<'a>: for<'b> View<Self::Id, Self::ListId, VectorsRef<'b> = Self::VectorsRef<'b>>
    where
        Self: 'a;

    type Update<'a>: Update<Self::Id, Self::ListId>
    where
        Self: 'a;

    /// Positive dimension of all canonical points and centroids.
    fn dim(&self) -> NonZeroUsize;

    fn nearest(&mut self) -> Self::Nearest<'_>;

    /// Stage points and append their internal IDs to exactly one routed list each.
    ///
    /// Emit operation-local list sizes including pending appends. Lists may be emitted
    /// more than once, but each touched list must report its final size.
    /// Staged points and memberships remain unpublished until [`Self::finish`].
    fn route<F>(&mut self, points: B, emit: F) -> impl SendFuture<ANNResult<NonZeroUsize>>
    where
        F: FnMut(&Self::ListId, NonZeroUsize) + Send;

    /// Make requested list's members, data and centroids available
    /// for split and assignment, and register updates to the region.
    ///
    /// Writes recorded through the update handle remain pending until [`Self::finish`].
    fn fill<Itr>(
        &mut self,
        lists: Itr,
    ) -> impl SendFuture<ANNResult<(Self::Region<'_>, Self::Update<'_>)>>
    where
        Itr: Iterator<Item = Self::ListId> + Clone + Send + Sync;

    /// Finalize the operation, publishing accessor-side appends and regional writes together.
    fn finish(self) -> impl SendFuture<ANNResult<()>>;
}

//////////////
// Strategy //
//////////////

pub trait BuildStrategy<'a, B, C: ExecutionContext>: Send + Sync + HasIds {
    /// [`BuildAccessor`] used to insert new points, and performs splits if necessary.
    type Accessor: BuildAccessor<B, Id = Self::Id, ListId = Self::ListId>;

    type Error: StandardError;

    fn build_accessor(&'a self, context: &'a C) -> Result<Self::Accessor, Self::Error>;
}

////////////
// Search //
////////////

pub trait SearchStrategy<'a, T: Sync, C: ExecutionContext>: Send + Sync + HasIds {
    /// Query-bound accessor responsible for list selection and scanning.
    type Accessor: SearchAccessor<T, ListId = Self::ListId, Id = Self::Id>;

    type Error: StandardError;

    fn search_accessor(&'a self, context: &'a C, query: T) -> Result<Self::Accessor, Self::Error>;
}

/// Candidate collection with independent partial states for partitioned scans.
pub trait Accumulator<I>: Send + Sized {
    /// Create an empty accumulator with the same selection policy and configuration.
    fn empty_like(&self) -> Self;

    /// Consider one candidate for inclusion in the results.
    fn push(&mut self, id: I, distance: f32);

    /// Combine a compatible partial state, such as one created by [`Self::empty_like`].
    fn merge(&mut self, other: Self);
}

/// Queue-backed accumulation requires unique candidate IDs across partial states.
/// Equal-distance selections may depend on merge order.
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

pub trait SearchAccessor<T: Sync>: Send + Sized + HasIds {
    /// Scan the `min(nprobe, live_list_count)` lists nearest the accessor's query.
    ///
    /// Implementations own selection and scheduling, and may reduce independent
    /// accumulators from parallel scans into `results`.
    ///
    /// # Errors
    ///
    /// A failure may leave partial results. The caller must discard them rather
    /// than publish them as a completed search.
    fn scan<A>(&self, nprobe: NonZeroUsize, results: &mut A) -> impl SendFuture<ANNResult<()>>
    where
        A: Accumulator<Self::Id>;
}

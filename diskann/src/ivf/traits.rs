/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Storage boundaries for IVF construction.

use std::num::NonZeroUsize;

use diskann_utils::future::SendFuture;

use super::region::{Update, View};
use crate::{
    ANNResult, error::StandardError, flat::DistancesUnordered, provider::ExecutionContext,
    utils::VectorId,
};

pub trait HasIds {
    type Id: VectorId;
    type ListId: VectorId;
}

/// Centroid selection within one build operation.
pub trait Router<T>: Send {
    type Id;
    type Ret;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Select `min(nprobe, len())` distinct live lists, nearest first.
    fn route<F>(&self, query: T, nprobe: NonZeroUsize, emit: F) -> impl SendFuture<ANNResult<()>>
    where
        F: FnMut(&Self::Id, Self::Ret) + Send;
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
    /// [`BuildAccessor`] used to insert new points, and performs splits if necessary.
    type Accessor: SearchAccessor<T, ListId = Self::ListId, Id = Self::Id>;

    type Error: StandardError;

    fn search_accessor(&'a self, context: &'a C, query: T) -> Result<Self::Accessor, Self::Error>;
}

pub trait SearchAccessor<T: Sync>: Send + Sized + HasIds {
    type Router<'a>: Router<T, Id = Self::ListId, Ret = Self::Dist<'a>>
    where
        Self: 'a;

    type Dist<'a>: DistancesUnordered<Id = Self::Id>
    where
        Self: 'a;

    fn router(&self) -> Self::Router<'_>;
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Storage boundaries for IVF construction.

use std::num::NonZeroUsize;

use diskann_utils::future::SendFuture;

use super::region::{RegionUpdate, RegionView};
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
    fn route<F>(
        &self,
        query: &T,
        nprobe: NonZeroUsize,
        emit: F,
    ) -> impl SendFuture<ANNResult<usize>>
    where
        F: FnMut(&Self::Id, Self::Ret) + Send;
}

pub trait RouterAndNearest<T>: Router<T> {
    /// Select `nprobe` distinct live lists nearest to `id`'s centroid, that pass the filter.
    ///
    /// Results are nearest first and may include `id` itself.
    ///
    /// # Errors
    ///
    /// Fails if `id` is not live or the provider cannot complete the lookup.
    fn nearest<P, F>(
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

pub trait BuildAccessor<T>: Send + Sized + HasIds {
    type Router<'a>: RouterAndNearest<T, Id = Self::ListId>
    where
        Self: 'a;

    type VectorsRef<'a>;

    type Region<'a>: for<'b> RegionView<Self::Id, Self::ListId, VectorsRef<'b> = Self::VectorsRef<'b>>
    where
        Self: 'a;

    type Update<'a>: RegionUpdate<Self::Id, Self::ListId>
    where
        Self: 'a;

    /// Positive dimension of all canonical points and centroids.
    fn dim(&self) -> NonZeroUsize;

    fn router(&mut self) -> Self::Router<'_>;

    /// Append members and return the resulting operation-local list size.
    fn append_to<Itr>(
        &mut self,
        list: Self::ListId,
        ids: Itr,
    ) -> impl SendFuture<ANNResult<NonZeroUsize>>
    where
        Itr: ExactSizeIterator<Item = Self::Id> + Clone + Send + Sync;

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

pub trait BuildStrategy<'a, T: Sync, C: ExecutionContext>: Send + Sync + HasIds {
    /// [`BuildAccessor`] used to insert new points, and performs splits if necessary.
    type Accessor: SetElements<T, C, Id = Self::Id>
        + BuildAccessor<T, Id = Self::Id, ListId = Self::ListId>;

    type Error: StandardError;

    fn build_accessor(&'a self, context: &'a C) -> Result<Self::Accessor, Self::Error>;
}

pub trait SetElements<T, C: ExecutionContext> {
    type Id: VectorId;

    fn set_elements(
        &mut self,
        context: &C,
        elements: &[T],
    ) -> impl std::future::Future<Output = ANNResult<Vec<Self::Id>>> + Send;
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

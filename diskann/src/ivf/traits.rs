/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Storage boundaries for IVF construction.

use std::num::NonZeroUsize;

use diskann_utils::future::SendFuture;

use super::region::{RegionUpdate, RegionView};
use crate::{ANNResult, error::StandardError, provider::ExecutionContext, utils::VectorId};

pub trait HasIds {
    type Id: VectorId;
    type ListId: VectorId;
}

/// Centroid selection within one build operation.
pub trait Router<T>: Send {
    type Id: VectorId;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Select `min(nprobe, len())` distinct live lists, nearest first.
    fn route(&self, query: &T, nprobe: usize) -> impl SendFuture<ANNResult<Vec<(Self::Id, f32)>>>;
}

pub trait RouterAndNearest<T>: Router<T> {
    /// Select `min(nprobe, len())` distinct live lists nearest to `id`'s centroid.
    ///
    /// Results are nearest first and may include `id` itself. Callers exclude
    /// lists being replaced when selecting retained neighbors.
    ///
    /// # Errors
    ///
    /// Fails if `id` is not live or the provider cannot complete the lookup.
    fn nearest(
        &self,
        id: Self::Id,
        nprobe: usize,
    ) -> impl SendFuture<ANNResult<Vec<(Self::Id, f32)>>>;
}

pub trait BuildAccessor<T>: Send + Sized + HasIds {
    type Router<'a>: RouterAndNearest<T, Id = Self::ListId>
    where
        Self: 'a;

    type Region<'a>: RegionView<Self::Id, Self::ListId>
    where
        Self: 'a;

    type Update: RegionUpdate<Self::Id, Self::ListId>;

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

    /// Make every requested list's members, data and centroids available
    /// for split and assignment. Register updates to the region.
    fn fill<Itr>(
        &mut self,
        lists: Itr,
    ) -> impl SendFuture<ANNResult<(Self::Region<'_>, Self::Update)>>
    where
        Itr: Iterator<Item = Self::ListId> + Clone + Send + Sync;

    /// Finalize the operation, publishing accessor-side appends and regional writes together.
    ///
    /// `None` means there are no regional re-writes.
    fn finish(self, update: Option<Self::Update>) -> impl SendFuture<ANNResult<()>>;
}

pub trait BuildStrategy<'a, T: Sync, C: ExecutionContext>: Send + Sync + HasIds {
    type Accessor: BuildAccessor<T, Id = Self::Id, ListId = Self::ListId>
        + SetElements<T, C, Id = Self::Id>;
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

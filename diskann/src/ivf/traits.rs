/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Storage boundaries for IVF construction.

use std::{fmt::Debug, num::NonZeroUsize, sync::Arc};

use diskann_utils::future::SendFuture;

use super::region::{Batch, RegionUpdate, RegionView};
use crate::{
    ANNResult,
    error::{StandardError, ToRanked},
    provider::ExecutionContext,
    utils::VectorId,
};

/// Point identity and execution context shared by an IVF provider's accessors.
///
/// Prepared points become translatable only after the build operation finishes.
pub trait Provider: Sized + Send + Sync + 'static {
    type Context: ExecutionContext;
    type InternalId: VectorId;
    type ExternalId: PartialEq + Send + Sync + 'static;
    /// Stable identity shared by a centroid and its inverted list. Never reused.
    type ListId: VectorId;
    type Error: ToRanked + Debug + Send + Sync + 'static;

    fn to_internal_id(
        &self,
        context: &Self::Context,
        id: &Self::ExternalId,
    ) -> Result<Self::InternalId, Self::Error>;

    fn to_external_id(
        &self,
        context: &Self::Context,
        id: Self::InternalId,
    ) -> Result<Self::ExternalId, Self::Error>;
}

pub trait HasIds {
    type Id: VectorId;
    type ListId: VectorId;
}

/// Centroid selection within one build operation.
pub trait Router: Send {
    type Id: VectorId;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Select `min(nprobe, len())` distinct live lists, nearest first.
    fn route(
        &self,
        query: &[f32],
        nprobe: usize,
    ) -> impl SendFuture<ANNResult<Vec<(Self::Id, f32)>>>;
}

pub trait RouterAndNearest: Router {
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

pub trait BuildAccessor: Send + Sized + HasIds {
    type Router<'a>: RouterAndNearest<Id = Self::ListId>
    where
        Self: 'a;

    type Region<'a>: RegionView<Self::Id, Self::ListId>
    where
        Self: 'a;

    type Update: RegionUpdate<Self::Id, Self::ListId>;

    /// Positive dimension of all canonical points and centroids.
    fn dim(&self) -> usize;

    fn router(&mut self) -> Self::Router<'_>;

    /// Append members and return the resulting operation-local list size.
    fn append_to<Itr>(
        &mut self,
        list: Self::ListId,
        members: Itr,
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
    fn finish(self, update: Option<&Self::Update>) -> impl SendFuture<ANNResult<()>>;
}

pub trait BuildStrategy<'a, P: Provider>: Send + Sync {
    type Accessor: BuildAccessor<Id = P::InternalId, ListId = P::ListId>;
    type Error: StandardError;

    fn build_accessor(
        &'a self,
        provider: &'a mut P,
        context: &'a P::Context,
    ) -> Result<Self::Accessor, Self::Error>;
}

/// Prepare typed inputs and seed a build accessor with their canonical data.
///
/// Only this boundary depends on `T`; list operations and clustering do not.
pub trait InsertStrategy<'a, P: Provider, T: Sync>: BuildStrategy<'a, P> {
    fn prepare(
        &self,
        accessor: &mut Self::Accessor,
        points: &[(P::ExternalId, T)],
    ) -> impl SendFuture<ANNResult<Arc<Batch<P::InternalId>>>>;
}

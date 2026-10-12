/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Storage boundaries for IVF construction.

use std::{fmt::Debug, sync::Arc};

use diskann_utils::future::SendFuture;

use super::workingset::Batch;
use crate::{
    ANNResult,
    error::{StandardError, ToRanked},
    ivf::workingset::RegionSet,
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
    type InternalId: VectorId;
    type ListId: VectorId;
}

/// Centroid selection within one build operation.
pub trait Router: Send {
    type ListId: VectorId;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Select `min(nprobe, len())` live lists, nearest first.
    fn select(
        &self,
        query: &[f32],
        nprobe: usize,
    ) -> impl SendFuture<ANNResult<Vec<(Self::ListId, f32)>>>;
}

pub trait BuildAccessor: Send + Sized + HasIds {
    /// Owned regional edits. The default records no regional changes.
    type Update: Send;

    type Router<'a>: Router<ListId = Self::ListId>
    where
        Self: 'a;

    type Region<'a>: RegionSet<Self::InternalId, Self::ListId, Update = Self::Update>
    where
        Self: 'a;

    /// Positive dimension of all canonical points and centroids.
    fn dim(&self) -> usize;

    fn router(&mut self) -> Self::Router<'_>;

    /// Append members and return the resulting operation-local list size.
    fn append_to(
        &mut self,
        list: Self::ListId,
        members: &[Self::InternalId],
    ) -> impl SendFuture<ANNResult<usize>>;

    /// Make every requested list's complete operation-local membership, vectors
    /// and centroids available for computation.
    fn fill(&mut self, lists: &[Self::ListId]) -> impl SendFuture<ANNResult<Self::Region<'_>>>;

    /// Publish accessor-side appends and owned regional edits together.
    fn finish(&mut self, update: Option<Self::Update>) -> impl SendFuture<ANNResult<()>>;
}

pub trait BuildStrategy<'a, P: Provider>: Send + Sync {
    type Accessor: BuildAccessor<InternalId = P::InternalId, ListId = P::ListId>;
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

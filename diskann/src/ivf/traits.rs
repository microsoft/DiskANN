/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Storage boundaries for IVF construction.

use std::{fmt::Debug, sync::Arc};

use diskann_utils::future::SendFuture;

use super::workingset::{Batch, WorkingSet};
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

/// A live list selected by centroid distance.
#[derive(Debug, Clone, Copy)]
pub struct SelectedList<L> {
    pub id: L,
    pub distance: f32,
}

/// Centroid selection within one build operation.
pub trait CoarseAccessor: Send {
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

/// Logical list mutations, private to the enclosing build operation.
///
/// Reads and working-set fills include pending writes. Publication occurs only
/// when the build accessor finishes.
pub trait ListAccessor: Send {
    type InternalId: VectorId;
    type ListId: VectorId;

    /// Append members and return the resulting operation-local list size.
    fn append_members(
        &mut self,
        list: Self::ListId,
        members: Vec<Self::InternalId>,
    ) -> impl SendFuture<ANNResult<usize>>;

    /// Reserve a fresh list ID and stage its centroid and complete membership.
    ///
    /// The centroid must be finite and have the canonical dimension. Empty
    /// memberships are valid, including during initialization.
    fn create_list(
        &mut self,
        centroid: &[f32],
        members: Vec<Self::InternalId>,
    ) -> impl SendFuture<ANNResult<Self::ListId>>;

    /// Replace a live list's complete membership, leaving its centroid unchanged.
    ///
    /// This supersedes earlier pending membership changes to the list.
    fn set_members(
        &mut self,
        list: Self::ListId,
        members: Vec<Self::InternalId>,
    ) -> impl SendFuture<ANNResult<()>>;

    /// Stage retirement of a live list and its centroid.
    ///
    /// All its members must be placed elsewhere before the build operation finishes.
    fn retire_list(&mut self, list: Self::ListId) -> impl SendFuture<ANNResult<()>>;
}

pub trait BuildAccessor: Send + Sized {
    type InternalId: VectorId;
    type ListId: VectorId;

    type Coarse<'a>: CoarseAccessor<ListId = Self::ListId>
    where
        Self: 'a;
    type Lists<'a>: ListAccessor<InternalId = Self::InternalId, ListId = Self::ListId>
    where
        Self: 'a;

    /// Positive dimension of all canonical points and centroids.
    fn dim(&self) -> usize;
    fn coarse(&mut self) -> Self::Coarse<'_>;
    fn lists(&mut self) -> Self::Lists<'_>;

    /// Make every requested list's complete current membership available.
    ///
    /// Lists appear in request order, with their canonical centroids and every
    /// member, including pending appends. Coordinates have the canonical dimension
    /// and are finite. Missing required data is an error, not a partial working set.
    /// TO DO: Make [`WorkingSet`] an associated type here.
    fn fill(
        &mut self,
        lists: &[Self::ListId],
    ) -> impl SendFuture<ANNResult<WorkingSet<'_, Self::InternalId, Self::ListId>>>;

    /// Publish the operation's point mappings, centroids, and memberships together.
    fn finish(self) -> impl SendFuture<ANNResult<()>>;
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

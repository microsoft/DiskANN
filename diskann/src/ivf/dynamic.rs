/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Contracts for an incrementally maintained IVF index.
//!
//! This module defines the boundary between the IVF algorithm and
//! provider-specific centroid navigation, inverted-list access, and partition
//! mutation. Concrete accessors are responsible for presenting a coherent view
//! across those components.

use std::fmt::Debug;

use diskann_utils::{future::SendFuture, views::MutMatrixView};

use crate::{
    error::{StandardError, ToRanked},
    ivf::update::InsertionUpdate,
    provider::{ExecutionContext, HasId},
    utils::VectorId,
};

/// One centroid/list selected for a query.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelectedList<L> {
    /// Stable logical list identifier.
    pub id: L,
    /// Distance from the query to this list's centroid.
    pub distance: f32,
}

/// Work performed while scanning the selected lists.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScanStats {
    /// Number of point representations scored against the query.
    pub comparisons: u64,
    /// Number of distinct lists read.
    pub lists_scanned: u32,
}

/// The storage behind a dynamic IVF index: point identity and the lists points are
/// partitioned into.
///
/// Reads and writes go through operation-scoped accessors; the provider itself only
/// names the identity types and translates the ids of visible points. Points staged
/// through [`StageElements`] are not translatable until an update that places them is
/// applied.
pub trait Provider: Sized + Send + Sync + 'static {
    /// Per-operation context handed to strategies.
    type Context: ExecutionContext;

    /// Internal point id used by lists and updates.
    type InternalId: VectorId;

    /// Caller-facing point id.
    type ExternalId: PartialEq + Send + Sync + 'static;

    /// Stable id shared by a centroid and its inverted list. Retired ids are never
    /// reused.
    type ListId: VectorId;

    /// Errors from id translation.
    type Error: ToRanked + Debug + Send + Sync + 'static;

    /// Translate an external id to its corresponding internal id.
    fn to_internal_id(
        &self,
        context: &Self::Context,
        gid: &Self::ExternalId,
    ) -> Result<Self::InternalId, Self::Error>;

    /// Translate an internal id to its corresponding external id.
    fn to_external_id(
        &self,
        context: &Self::Context,
        id: Self::InternalId,
    ) -> Result<Self::ExternalId, Self::Error>;
}

/// An in-memory index over the live centroids.
///
/// The centroid catalog is authoritative. Approximate implementations, such as
/// a graph navigator, must retain enough catalog information to reject retired
/// ids and recover with exact selection when navigation cannot produce a usable
/// result. Exact and graph implementations expose the same interface.
pub trait CentroidIndex: Send + Sync {
    /// Stable logical centroid id, also used as the inverted-list id.
    type ListId: VectorId;

    /// Errors encountered while reading or navigating the centroid index.
    type Error: ToRanked + Debug + Send + Sync + 'static;

    /// Number of live centroids.
    fn len(&self) -> usize;

    /// Whether there are no live centroids.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow one authoritative full-precision centroid vector.
    fn centroid(&self, id: Self::ListId) -> Option<&[f32]>;

    /// Select exactly `min(nprobe, self.len())` live centroids for `query`, nearest
    /// first.
    ///
    /// A graph implementation must use exact selection as a recovery path when
    /// graph navigation produces too few live ids.
    fn select(
        &self,
        query: &[f32],
        nprobe: usize,
    ) -> impl SendFuture<Result<Vec<SelectedList<Self::ListId>>, Self::Error>>;
}

/// Query-bound access to centroid selection and inverted-list scanning.
///
/// This is the dynamic IVF search algorithm's primary extension point, in the
/// same spirit as [`crate::graph::glue::SearchAccessor`]. Implementations are free
/// to batch reads, coalesce blob requests, prefetch, decode quantized payloads, or
/// fan work out across tasks.
pub trait SearchAccessor: HasId + Send + Sync {
    /// Stable logical centroid/list id, fixed to [`Provider::ListId`] by the
    /// strategy.
    type ListId: VectorId;

    /// Errors from list selection or scanning.
    type Error: ToRanked + Debug + Send + Sync + 'static;

    /// Select up to `nprobe` lists for the bound query, nearest first.
    fn select_lists(
        &mut self,
        nprobe: usize,
    ) -> impl SendFuture<Result<Vec<SelectedList<Self::ListId>>, Self::Error>>;

    /// Scan `lists`, scoring their members against the bound query.
    ///
    /// `lists` must come from [`Self::select_lists`] on this accessor, which keeps
    /// selection and scanning coherent using provider-specific coordination.
    fn scan_lists<F>(
        &mut self,
        lists: &[SelectedList<Self::ListId>],
        emit: F,
    ) -> impl SendFuture<Result<ScanStats, Self::Error>>
    where
        F: FnMut(Self::Id, f32) + Send;
}

/// Factory for one dynamic IVF search accessor.
pub trait SearchStrategy<'a, P: Provider, T>: Send + Sync {
    /// Query-bound accessor used for both coarse and fine search.
    type SearchAccessor: SearchAccessor<Id = P::InternalId, ListId = P::ListId>;

    /// Error constructing the accessor.
    type Error: StandardError;

    /// Construct an accessor around `query`.
    fn search_accessor(
        &'a self,
        provider: &'a P,
        context: &'a P::Context,
        query: T,
    ) -> Result<Self::SearchAccessor, Self::Error>;
}

/// Operation-scoped reads and writes needed by split/dissolve maintenance.
///
/// The accessor is the consistency boundary for one mutation. It must present a
/// unified view across point data, centroids, and inverted lists for all planning
/// reads. It also owns any provider-specific lock, epoch, transaction, staging area,
/// or poison state needed to apply the final update through [`Self::apply`].
///
/// List reads take one list and return its result. [`Self::read_vectors`], which
/// reads far more data, takes a batch so disk and blob providers can reorder,
/// coalesce, or parallelize I/O.
pub trait MaintenanceAccessor<P: Provider>: Send + Sized {
    /// In-memory centroid catalog and navigator in this accessor's unified view.
    type Centroids: CentroidIndex<ListId = P::ListId, Error = Self::Error>;

    /// Errors from planning reads, staging, or applying the update.
    type Error: ToRanked + Debug + Send + Sync + 'static;

    /// Dimension of every canonical vector and centroid in this accessor's view.
    ///
    /// Fixed for the provider's lifetime, including before the index is initialized.
    fn dim(&self) -> usize;

    /// Borrow the centroid index used for routing and maintenance neighborhoods.
    fn centroids(&self) -> &Self::Centroids;

    /// Read the size of a live list.
    fn list_size(&mut self, list: P::ListId) -> impl SendFuture<Result<usize, Self::Error>>;

    /// Read the member ids of a live list.
    fn read_members(
        &mut self,
        list: P::ListId,
    ) -> impl SendFuture<Result<&[P::InternalId], Self::Error>>;

    /// Write the canonical vector of `ids[i]` into row `i` of `out`, in any order.
    ///
    /// `out` has one row per id and [`Self::dim`] columns, and every row must be
    /// written. Covers visible points only; the canonical vectors of staged points
    /// come from [`StageElements::stage`].
    fn read_vectors(
        &mut self,
        ids: &[P::InternalId],
        out: MutMatrixView<'_, f32>,
    ) -> impl SendFuture<Result<(), Self::Error>>;

    /// Reserve fresh logical list ids that will not alias retired ids.
    /// TO DO: Re-evaluate we even need this.
    fn reserve_list_ids(
        &mut self,
        count: usize,
    ) -> impl SendFuture<Result<Vec<P::ListId>, Self::Error>>;

    /// Apply `update` and finish the operation, consuming the accessor.
    ///
    /// Only the index constructs updates, and it guarantees the invariants documented
    /// on [`InsertionUpdate`], so implementations may rely on them without re-checking.
    ///
    /// Returning `Ok(())` means all components expose one coherent resulting index.
    /// Rollback, durability, concurrent-reader visibility, and recovery after `Err` are
    /// intentionally provider-defined.
    fn apply(
        self,
        update: InsertionUpdate<P::InternalId, P::ListId>,
    ) -> impl SendFuture<Result<(), Self::Error>>;
}

/// Stages new points of element type `T` for one maintenance operation.
///
/// The provider owns id allocation, duplicate detection, and the conversion of `T` to
/// its stored and canonical representations. Staged points become visible only when
/// the accessor applies an update that places them, and are discarded if the accessor
/// is dropped first. A provider that accepts several input types implements this once
/// per type.
pub trait StageElements<P: Provider, T>: MaintenanceAccessor<P> {
    /// Stage `points` and return their internal ids in order, writing the canonical
    /// vector of `points[i]` into row `i` of `out`.
    ///
    /// `out` has one row per point and [`MaintenanceAccessor::dim`] columns, and every
    /// row must be written.
    fn stage(
        &mut self,
        points: &[(P::ExternalId, T)],
        out: MutMatrixView<'_, f32>,
    ) -> impl SendFuture<Result<Vec<P::InternalId>, Self::Error>>;
}

/// Factory for an operation-scoped dynamic IVF maintenance accessor.
///
/// The provider is borrowed exclusively for the lifetime of the accessor, so no
/// search or other mutation can observe it until the accessor is applied or
/// dropped. Providers may therefore mutate plain in-memory state in
/// [`MaintenanceAccessor::apply`] without interior synchronization. Providers that
/// share state outside this borrow (for example through an `Arc`) remain responsible
/// for coordinating those aliases.
pub trait MaintenanceStrategy<'a, P: Provider>: Send + Sync {
    /// Accessor used to plan and stage split/dissolve operations.
    type MaintenanceAccessor: MaintenanceAccessor<P>;

    /// Error constructing the accessor.
    type Error: StandardError;

    /// Construct one maintenance accessor.
    fn maintenance_accessor(
        &'a self,
        provider: &'a mut P,
        context: &'a P::Context,
    ) -> Result<Self::MaintenanceAccessor, Self::Error>;
}

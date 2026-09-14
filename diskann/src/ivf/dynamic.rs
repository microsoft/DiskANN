/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Contracts for an incrementally maintained IVF index.
//!
//! This module is an API skeleton. It defines the boundary between an IVF
//! algorithm and provider-specific centroid navigation, inverted-list access,
//! and partition mutation. Concrete accessors are responsible for presenting a
//! coherent view across those components. The online split/dissolve algorithm
//! is not implemented here yet.

use std::fmt::Debug;

use diskann_utils::future::SendFuture;

use crate::{
    error::{StandardError, ToRanked},
    provider::{DataProvider, HasId},
    utils::VectorId,
};

/// One centroid/list selected during coarse search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelectedList<ListId> {
    /// Stable logical list identifier.
    pub id: ListId,
    /// Distance from the bound query to this list's centroid.
    pub distance: f32,
}

/// The handoff from centroid selection to inverted-list scanning.
///
/// A plan must be consumed by the same [`SearchAccessor`] that produced it.
/// The accessor is responsible for keeping centroid selection and list reads
/// coherent using provider-specific coordination.
#[derive(Debug, Clone)]
pub struct SelectionPlan<ListId> {
    selected: Vec<SelectedList<ListId>>,
}

impl<ListId> SelectionPlan<ListId> {
    /// Construct a plan from selected logical lists.
    pub fn new(selected: Vec<SelectedList<ListId>>) -> Self {
        Self { selected }
    }

    /// Lists selected in increasing coarse-distance order.
    pub fn selected(&self) -> &[SelectedList<ListId>] {
        &self.selected
    }

    /// Consume the plan and return the selected lists.
    pub fn into_selected(self) -> Vec<SelectedList<ListId>> {
        self.selected
    }
}

/// Work performed while scanning the selected lists.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScanStats {
    /// Number of point representations scored against the query.
    pub comparisons: u64,
    /// Number of distinct lists read.
    pub lists_scanned: u32,
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

    /// Select exactly `min(nprobe, self.len())` live centroids for `query`.
    ///
    /// A graph implementation must use exact selection as a recovery path when
    /// graph navigation produces too few live ids.
    fn select(
        &self,
        query: &[f32],
        nprobe: usize,
    ) -> impl SendFuture<Result<SelectionPlan<Self::ListId>, Self::Error>>;
}

/// Query-bound access to centroid selection and inverted-list scanning.
///
/// This is the dynamic IVF search algorithm's primary extension point, in the
/// same spirit as [`crate::graph::SearchAccessor`]. Implementations are free to
/// batch reads, coalesce blob requests, prefetch, decode quantized payloads, or
/// fan work out across tasks.
pub trait SearchAccessor: HasId + Send + Sync {
    /// Stable logical centroid/list id.
    type ListId: VectorId;

    /// Errors from list selection or scanning.
    type Error: ToRanked + Debug + Send + Sync + 'static;

    /// Select exactly the available requested number of lists for the bound query.
    fn select_lists(
        &mut self,
        nprobe: usize,
    ) -> impl SendFuture<Result<SelectionPlan<Self::ListId>, Self::Error>>;

    /// Scan every list in `plan`, scoring members against the bound query.
    fn scan_lists<F>(
        &mut self,
        plan: SelectionPlan<Self::ListId>,
        emit: F,
    ) -> impl SendFuture<Result<ScanStats, Self::Error>>
    where
        F: FnMut(Self::Id, f32) + Send;
}

/// Factory for one dynamic IVF search accessor.
pub trait SearchStrategy<'a, Provider, T>: Send + Sync
where
    Provider: DataProvider,
{
    /// Query-bound accessor used for both coarse and fine search.
    type SearchAccessor: SearchAccessor<Id = Provider::InternalId>;

    /// Error constructing the accessor.
    type Error: StandardError;

    /// Construct an accessor around `query`.
    fn search_accessor(
        &'a self,
        provider: &'a Provider,
        context: &'a Provider::Context,
        query: T,
    ) -> Result<Self::SearchAccessor, Self::Error>;
}

/// Size metadata for one logical inverted list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ListMetadata<ListId> {
    /// Stable logical list identifier.
    pub id: ListId,
    /// Number of live point ids in the list.
    pub len: usize,
}

/// A new centroid to install as part of a partition update.
#[derive(Debug, Clone, PartialEq)]
pub struct CentroidRecord<ListId> {
    /// Fresh stable id. Installed ids must never be reused after retirement.
    pub id: ListId,
    /// Full-precision centroid used by maintenance and exact fallback.
    pub vector: Box<[f32]>,
}

/// Changes to the authoritative live-centroid catalog.
#[derive(Debug, Clone, PartialEq)]
pub struct CentroidDelta<ListId> {
    /// Centroids made live by this update.
    pub insert: Vec<CentroidRecord<ListId>>,
    /// Live centroids retired by this update.
    pub retire: Vec<ListId>,
}

/// One point membership change in a partition mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PointMove<Id, ListId> {
    /// Internal point id being inserted, deleted, or reassigned.
    pub id: Id,
    /// Previous list, or `None` for a newly inserted point.
    pub from: Option<ListId>,
    /// New list, or `None` for a deleted point.
    pub to: Option<ListId>,
}

/// A complete logical partition change.
///
/// The maintenance accessor applies this update across point identity,
/// canonical vectors, list membership, reverse assignments, scan payloads, and
/// centroid liveness according to its provider-specific consistency contract.
#[derive(Debug, Clone, PartialEq)]
pub struct PartitionUpdate<Id, ListId> {
    /// Centroids inserted and retired by this mutation.
    pub centroids: CentroidDelta<ListId>,
    /// Point membership changes, including inserts and deletes.
    pub point_moves: Vec<PointMove<Id, ListId>>,
}

/// Operation-scoped reads and writes needed by split/dissolve maintenance.
///
/// The accessor is the consistency boundary for one mutation. It must present a
/// unified view across point data, centroids, reverse assignments, and inverted
/// lists for all planning reads. It also owns any provider-specific lock, epoch,
/// transaction, staging area, or poison state needed to apply the final update.
///
/// Read methods are deliberately coarse grained so disk and blob providers can
/// batch I/O. Callbacks may be invoked in any order. `read_members` may invoke
/// its callback more than once for a list when the backend streams chunks.
pub trait MaintenanceAccessor<T>: HasId + Send + Sized
where
    T: Send,
{
    /// External point id accepted by the data provider.
    type ExternalId: PartialEq + Send + Sync + 'static;

    /// Stable centroid/list id.
    type ListId: VectorId;

    /// In-memory centroid catalog and navigator in this accessor's unified view.
    type Centroids: CentroidIndex<ListId = Self::ListId, Error = Self::Error>;

    /// Errors from planning reads, staging, or applying the update.
    type Error: ToRanked + Debug + Send + Sync + 'static;

    /// Borrow the centroid index used for routing and maintenance neighborhoods.
    fn centroids(&self) -> &Self::Centroids;

    /// Read list sizes for `lists`.
    fn list_metadata<I, F>(
        &mut self,
        lists: I,
        emit: F,
    ) -> impl SendFuture<Result<(), Self::Error>>
    where
        I: Iterator<Item = Self::ListId> + Send,
        F: FnMut(ListMetadata<Self::ListId>) + Send;

    /// Resolve current list assignments for point ids.
    fn assignments<I, F>(&mut self, ids: I, emit: F) -> impl SendFuture<Result<(), Self::Error>>
    where
        I: Iterator<Item = Self::Id> + Send,
        F: FnMut(Self::Id, Option<Self::ListId>) + Send;

    /// Stream the live member ids of selected lists.
    fn read_members<I, F>(&mut self, lists: I, emit: F) -> impl SendFuture<Result<(), Self::Error>>
    where
        I: Iterator<Item = Self::ListId> + Send,
        F: FnMut(Self::ListId, &[Self::Id]) + Send;

    /// Materialize canonical full-precision vectors for maintenance.
    fn read_vectors<I, F>(&mut self, ids: I, emit: F) -> impl SendFuture<Result<(), Self::Error>>
    where
        I: Iterator<Item = Self::Id> + Send,
        F: FnMut(Self::Id, &[f32]) + Send;

    /// Stage a canonical point and reserve its internal id.
    fn stage_insert(
        &mut self,
        id: &Self::ExternalId,
        element: T,
    ) -> impl SendFuture<Result<Self::Id, Self::Error>>;

    /// Materialize canonical vectors for points staged by this accessor.
    fn read_staged_vectors<I, F>(
        &mut self,
        ids: I,
        emit: F,
    ) -> impl SendFuture<Result<(), Self::Error>>
    where
        I: Iterator<Item = Self::Id> + Send,
        F: FnMut(Self::Id, &[f32]) + Send;

    /// Stage deletion of a currently visible internal point id.
    fn stage_delete(&mut self, id: Self::Id) -> impl SendFuture<Result<(), Self::Error>>;

    /// Reserve fresh logical list ids that will not alias retired ids.
    fn reserve_list_ids(
        &mut self,
        count: usize,
    ) -> impl SendFuture<Result<Vec<Self::ListId>, Self::Error>>;

    /// Apply the complete logical update and finish this mutation operation.
    ///
    /// Returning `Ok(())` means all components expose one coherent resulting
    /// index. Rollback, durability, concurrent-reader visibility, and recovery
    /// after `Err` are intentionally provider-defined.
    fn apply(
        self,
        update: PartitionUpdate<Self::Id, Self::ListId>,
    ) -> impl SendFuture<Result<(), Self::Error>>;
}

/// Factory for an operation-scoped dynamic IVF maintenance accessor.
pub trait MaintenanceStrategy<'a, Provider, T>: Send + Sync
where
    Provider: DataProvider,
    T: Send,
{
    /// Accessor used to plan and stage split/dissolve operations.
    type MaintenanceAccessor: MaintenanceAccessor<T, Id = Provider::InternalId, ExternalId = Provider::ExternalId>;

    /// Error constructing the accessor.
    type Error: StandardError;

    /// Construct one maintenance accessor.
    fn maintenance_accessor(
        &'a self,
        provider: &'a Provider,
        context: &'a Provider::Context,
    ) -> Result<Self::MaintenanceAccessor, Self::Error>;
}

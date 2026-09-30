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

use diskann_utils::{future::SendFuture, views::MutMatrixView};

use crate::{
    error::{StandardError, ToRanked},
    provider::{DataProvider, HasId},
    utils::VectorId,
};

/// One centroid/list selected during coarse search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelectedList<L> {
    /// Stable logical list identifier.
    pub id: L,
    /// Distance from the bound query to this list's centroid.
    pub distance: f32,
}

/// The handoff from centroid selection to inverted-list scanning.
///
/// A plan must be consumed by the same [`SearchAccessor`] that produced it.
/// The accessor is responsible for keeping centroid selection and list reads
/// coherent using provider-specific coordination.
#[derive(Debug, Clone)]
pub struct SelectionPlan<L> {
    selected: Vec<SelectedList<L>>,
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

/// A data provider whose points are partitioned into stable logical lists.
pub trait ListProvider: DataProvider {
    /// Stable id shared by a centroid and its inverted list.
    type ListId: VectorId;
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
    /// Stable logical centroid/list id, fixed to [`ListProvider::ListId`] by the
    /// strategy.
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
    Provider: ListProvider,
{
    /// Query-bound accessor used for both coarse and fine search.
    type SearchAccessor: SearchAccessor<Id = Provider::InternalId, ListId = Provider::ListId>;

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
pub struct ListMetadata {
    /// Number of live point ids in the list.
    pub len: usize,
}

/// Operation-scoped reads and writes needed by split/dissolve maintenance.
///
/// The accessor is the consistency boundary for one mutation. It must present a
/// unified view across point data, centroids, reverse assignments, and inverted
/// lists for all planning reads. It also owns any provider-specific lock, epoch,
/// transaction, staging area, or poison state needed to apply the final update
/// through [`Apply`].
///
/// List reads take one list and return its result. [`Self::read_vectors`], which
/// reads far more data, takes a batch so disk and blob providers can reorder,
/// coalesce, or parallelize I/O.
pub trait MaintenanceAccessor<T>: HasId + Send + Sized
where
    T: Send,
{
    /// External point id accepted by the data provider.
    type ExternalId: PartialEq + Send + Sync + 'static;

    /// Stable centroid/list id, fixed to [`ListProvider::ListId`] by the strategy.
    type ListId: VectorId;

    /// In-memory centroid catalog and navigator in this accessor's unified view.
    type Centroids: CentroidIndex<ListId = Self::ListId, Error = Self::Error>;

    /// Errors from planning reads, staging, or applying the update.
    type Error: ToRanked + Debug + Send + Sync + 'static;

    /// Dimension of every canonical vector and centroid in this accessor's view.
    ///
    /// Fixed for the provider's lifetime, including before the index is initialized.
    fn dim(&self) -> usize;

    /// Borrow the centroid index used for routing and maintenance neighborhoods.
    fn centroids(&self) -> &Self::Centroids;

    /// Read the size of a live list.
    fn list_metadata(
        &mut self,
        list: Self::ListId,
    ) -> impl SendFuture<Result<ListMetadata, Self::Error>>;

    /// Read the member ids of a live list.
    fn read_members(
        &mut self,
        list: Self::ListId,
    ) -> impl SendFuture<Result<&[Self::Id], Self::Error>>;

    /// Write the canonical vector of `ids[i]` into row `i` of `out`, in any order.
    ///
    /// `out` has one row per id and [`Self::dim`] columns, and every row must be
    /// written. Covers points staged by this accessor as well as visible points,
    /// without making staged points visible through ordinary provider reads.
    fn read_vectors(
        &mut self,
        ids: &[Self::Id],
        out: MutMatrixView<'_, f32>,
    ) -> impl SendFuture<Result<(), Self::Error>>;

    /// Stage a canonical point and reserve its internal id.
    fn stage_insert(
        &mut self,
        id: &Self::ExternalId,
        element: T,
    ) -> impl SendFuture<Result<Self::Id, Self::Error>>;

    /// Reserve fresh logical list ids that will not alias retired ids.
    fn reserve_list_ids(
        &mut self,
        count: usize,
    ) -> impl SendFuture<Result<Vec<Self::ListId>, Self::Error>>;
}

/// Applies one kind of partition update and finishes the maintenance operation.
///
/// Providers implement this once per update they support, from
/// [`crate::ivf::update`]. For example, a provider that never splits implements only
/// `Apply<Appends<_, _>>`. Only the index constructs updates, and it guarantees their
/// documented invariants, so implementations may rely on them without re-checking.
///
/// Returning `Ok(())` means all components expose one coherent resulting index.
/// Rollback, durability, concurrent-reader visibility, and recovery after `Err` are
/// intentionally provider-defined.
pub trait Apply<U>: Send + Sized {
    /// Errors from applying the update.
    type Error: ToRanked + Debug + Send + Sync + 'static;

    /// Apply `update`, consuming the accessor.
    fn apply(self, update: U) -> impl SendFuture<Result<(), Self::Error>>;
}

/// Factory for an operation-scoped dynamic IVF maintenance accessor.
///
/// The provider is borrowed exclusively for the lifetime of the accessor, so no
/// search or other mutation can observe it until the accessor is applied or
/// dropped. Providers may therefore mutate plain in-memory state in
/// [`Apply::apply`] without interior synchronization. Providers that share state
/// outside this borrow (for example through an `Arc`) remain responsible for
/// coordinating those aliases.
pub trait MaintenanceStrategy<'a, Provider, T>: Send + Sync
where
    Provider: ListProvider,
    T: Send,
{
    /// Accessor used to plan and stage split/dissolve operations.
    type MaintenanceAccessor: MaintenanceAccessor<
            T,
            Id = Provider::InternalId,
            ExternalId = Provider::ExternalId,
            ListId = Provider::ListId,
        >;

    /// Error constructing the accessor.
    type Error: StandardError;

    /// Construct one maintenance accessor.
    fn maintenance_accessor(
        &'a self,
        provider: &'a mut Provider,
        context: &'a Provider::Context,
    ) -> Result<Self::MaintenanceAccessor, Self::Error>;
}

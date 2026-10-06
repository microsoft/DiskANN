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

use std::{fmt::Debug, ops::Deref};

use diskann_utils::{future::SendFuture, views::rowmajor};

use crate::{
    ANNResult,
    error::{StandardError, ToRanked},
    ivf::update::Deltas,
    provider::{ExecutionContext, Guard, NoopGuard},
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

pub trait StageElements<P: Provider, T: Sync> {
    /// The kind of error yielded by `set_element`.
    type StageError: ToRanked + std::fmt::Debug + Send + Sync + 'static;

    /// Stage a new point; its id is provisional until `update` commits it.
    fn stage_point(
        &mut self,
        id: &P::ExternalId,
        element: T,
        out: &mut [f32],
    ) -> impl SendFuture<ANNResult<P::InternalId>>;

    /// Stage a new centroid and mint its list id, provisional until `update` commits it.
    fn stage_centroid(&mut self, centroid: &[f32]) -> impl SendFuture<ANNResult<P::ListId>>;
}

///////////////
// Centroids //
///////////////

/// An in-memory index over the live centroids.
///
/// The centroid catalog is authoritative. Approximate implementations, such as
/// a graph navigator, must retain enough catalog information to reject retired
/// ids and recover with exact selection when navigation cannot produce a usable
/// result. Exact and graph implementations expose the same interface.
pub trait Centroids: Send + Sync {
    /// Stable logical centroid id, also used as the inverted-list id.
    type ListId: VectorId;

    /// Errors encountered while reading or navigating the centroid index.
    type Error: ToRanked + Debug + Send + Sync + 'static;

    /// Number of live centroids.
    fn len(&self) -> usize;

    /// dimension of full precision centroid vectors.
    fn dim(&self) -> usize;

    /// Whether there are no live centroids.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow one authoritative full-precision centroid vector.
    fn centroid(&self, id: Self::ListId) -> Option<impl Deref<Target = [f32]>>;

    /// Select exactly `min(nprobe, self.len())` live centroids for `query`, nearest
    /// first.
    ///
    /// A graph implementation must use exact selection as a recovery path when
    /// graph navigation produces too few live ids.
    fn select(
        &self,
        query: &[f32],
        nprobe: usize,
    ) -> Result<Vec<SelectedList<Self::ListId>>, Self::Error>;
}

//////////////
// Accessor //
//////////////

pub trait InsertAccessor<P: Provider, T: Sync>: Send + Sized + StageElements<P, T> {
    /// In-memory centroid catalog and navigator in this accessor's unified view.
    type Centroids<'a>: Centroids<ListId = P::ListId, Error = Self::Error>
    where
        Self: 'a;

    // Scoped reader to read vectors mapped to list ids.
    type Reader<'a>: Reader<f32, Id = P::ListId>
    where
        Self: 'a;

    /// Errors from planning reads, staging, or applying the update.
    type Error: ToRanked + Debug + Send + Sync + 'static;

    /// Borrow the list reader to read vectors from posting lists.
    fn reader(&self) -> Self::Reader<'_>;

    /// Borrow the centroid index used for routing and maintenance neighborhoods.
    fn centroids(&self) -> Self::Centroids<'_>;

    fn dim(&self) -> usize;

    /// Read the size of a live list.
    fn list_size(&self, list: P::ListId) -> Result<usize, Self::Error>;

    /// Read the member ids of a live list.
    fn get_members(&self, list: P::ListId) -> Result<&[P::InternalId], Self::Error>;

    /// Consume the accessor and write the update
    fn update(self, value: Deltas<P::InternalId, P::ListId>) -> impl SendFuture<ANNResult<()>>;
}

pub trait Reader<T = f32>: Send + Sync {
    /// Error type in case a read fails.
    type Error: ToRanked + Debug + Send + Sync + 'static;

    /// Id type
    type Id;

    /// dimension of the output of each Id.
    fn dim(&self) -> usize;

    /// Write the canonical vector of `ids[i]` into row `i` of `out`, in any order.
    ///
    /// `out` has one row per id and [`Self::dim`] columns, and every row must be
    /// written.
    fn read_into(
        &self,
        id: Self::Id,
        out: rowmajor::Mut<'_, T>,
    ) -> impl SendFuture<Result<(), Self::Error>>;
}

//////////////
// Strategy //
//////////////

/// Factory for an operation-scoped dynamic IVF maintenance accessor.
///
/// The provider is borrowed exclusively for the lifetime of the accessor, so no
/// search or other mutation can observe it until the accessor is applied or
/// dropped. Providers may therefore mutate plain in-memory state in
/// [`InsertAccessor::apply`] without interior synchronization. Providers that
/// share state outside this borrow (for example through an `Arc`) remain responsible
/// for coordinating those aliases.
pub trait MaintenanceStrategy<'a, P: Provider, T: Sync>: Send + Sync {
    /// Accessor used to plan and stage split/dissolve operations.
    type MaintenanceAccessor: InsertAccessor<P, T>;

    /// Error constructing the accessor.
    type Error: StandardError;

    /// Construct one maintenance accessor.
    fn maintenance_accessor(
        &'a self,
        provider: &'a mut P,
        context: &'a P::Context,
    ) -> Result<Self::MaintenanceAccessor, Self::Error>;
}

/// Factory for one dynamic IVF search accessor.
pub trait SearchStrategy<'a, P: Provider, T>: Send + Sync {
    /// Query-bound accessor used for both coarse and fine search.
    type SearchAccessor: SearchAccessor<ListId = P::ListId, InternalId = P::InternalId>;

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

////////////
// Search //
////////////

/// Query-bound access to centroid selection and inverted-list scanning.
///
/// This is the dynamic IVF search algorithm's primary extension point, in the
/// same spirit as [`crate::graph::glue::SearchAccessor`]. Implementations are free
/// to batch reads, coalesce blob requests, prefetch, decode quantized payloads, or
/// fan work out across tasks.
pub trait SearchAccessor: Send + Sync {
    /// Stable logical centroid/list id, fixed by the strategy.
    type ListId;

    type InternalId;

    /// In-memory centroid view.
    type Centroids<'a>: Centroids<ListId = Self::ListId, Error = Self::Error>
    where
        Self: 'a;

    /// Errors from list selection or scanning.
    type Error: ToRanked + Debug + Send + Sync + 'static;

    /// Borrow the centroid index used for routing.
    fn centroids(&self) -> Self::Centroids<'_>;

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
        F: FnMut(Self::InternalId, f32) + Send;
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::fmt::Debug;

use diskann_utils::future::SendFuture;

use crate::{
    error::{StandardError, ToRanked},
    graph::SearchOutputBuffer,
    provider::{DataProvider, HasId},
};

/////////////////
// Search path //
/////////////////

/// Selects candidate inverted lists for a bound query or insert vector.
pub trait ListAccessor: Send + Sync {
    /// Opaque handle identifying an inverted list within the index.
    type Id: Copy + Send + Sync;

    /// The error type for [`Self::select_lists`].
    type Error: ToRanked + Debug + Send + Sync + 'static;

    /// Push the selected lists into `output` in distance order.
    fn select_lists<B>(
        &mut self,
        nprobe: usize,
        output: &mut B,
    ) -> impl SendFuture<Result<(), Self::Error>>
    where
        B: SearchOutputBuffer<Self::Id> + Send + ?Sized;
}

/// Scans a set of lists for search.
pub trait SearchAccessor: HasId + Send + Sync {
    /// Opaque handle identifying an inverted list within the index.
    type ListId: Copy + Send + Sync;

    type Error: ToRanked + Debug + Send + Sync + 'static;

    /// Score members of `lists` and invoke `f` for each `(id, distance)` pair.
    fn scan_lists<Itr, F>(&mut self, lists: Itr, f: F) -> impl SendFuture<Result<(), Self::Error>>
    where
        Itr: Iterator<Item = Self::ListId> + Send,
        F: Send + FnMut(Self::Id, f32);
}

/// Per-call factory for IVF search.
pub trait SearchStrategy<'a, Provider, T>: Send + Sync
where
    Provider: DataProvider,
{
    /// The inverted-list handle type, shared by both accessors.
    type ListId: Copy + Send + Sync;

    /// The fine accessor, keyed to the provider's internal id and the shared list handle.
    type SearchAccessor: SearchAccessor<Id = Provider::InternalId, ListId = Self::ListId>;

    /// The coarse accessor, keyed to the shared list handle.
    type ListAccessor: ListAccessor<Id = Self::ListId>;

    /// An error that can occur when constructing either accessor.
    type Error: StandardError;

    /// Construct the fine scan accessor.
    fn search_accessor(
        &'a self,
        provider: &'a Provider,
        context: &'a Provider::Context,
        query: T,
    ) -> Result<Self::SearchAccessor, Self::Error>;

    /// Construct the coarse list-selection accessor.
    fn list_accessor(
        &'a self,
        provider: &'a Provider,
        context: &'a Provider::Context,
        query: T,
    ) -> Result<Self::ListAccessor, Self::Error>;
}

/////////////////
// Insert path //
/////////////////

/// Appends a vector to a chosen list during insert.
pub trait InsertAccessor<T>: HasId + Send + Sync {
    /// Opaque handle identifying an inverted list within the index.
    type ListId: Copy + Send + Sync;

    /// The error type for [`Self::append`].
    type Error: ToRanked + Debug + Send + Sync + 'static;

    /// Append `vector` to `list` under `id`.
    fn append(
        &mut self,
        list: Self::ListId,
        id: Self::Id,
        vector: T,
    ) -> impl SendFuture<Result<(), Self::Error>>;
}

/// Per-call factory for IVF insert.
pub trait InsertStrategy<'a, Provider, T>: Send + Sync
where
    Provider: DataProvider,
{
    /// The inverted-list handle type, shared by both accessors.
    type ListId: Copy + Send + Sync;

    /// The append accessor, keyed to the provider's internal id and the shared list handle.
    type InsertAccessor: InsertAccessor<T, Id = Provider::InternalId, ListId = Self::ListId>;

    /// The coarse accessor, keyed to the shared list handle.
    type ListAccessor: ListAccessor<Id = Self::ListId>;

    /// An error that can occur when constructing either accessor.
    type Error: StandardError;

    /// Construct the append accessor.
    fn insert_accessor(
        &'a self,
        provider: &'a Provider,
        context: &'a Provider::Context,
    ) -> Result<Self::InsertAccessor, Self::Error>;

    /// Construct the coarse list-selection accessor.
    fn list_accessor(
        &'a self,
        provider: &'a Provider,
        context: &'a Provider::Context,
        vector: T,
    ) -> Result<Self::ListAccessor, Self::Error>;
}

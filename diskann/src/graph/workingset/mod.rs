/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

//! The working-set is a core scratch data structure used during pruning.
//!
//! # Overview
//!
//! This interface consists of two traits for all insert types and one additional trait for
//! batch operations like [`multi_insert`](crate::graph::DiskANNIndex::multi_insert).
//! These are summarized below.
//!
//! * [`View`] (all inserts): Read-only random access to cached data. Data access should
//!   be reasonably efficient.
//!
//! * [`Fill`] (all inserts): Extension point for fetching data associated with internal IDs
//!   and producing a [`View`] to access the retrieved data. This trait uses an opaque
//!   `WorkingSet` that implementations are free to use in any way they see fit.
//!
//! * [`AsWorkingSet`] (bulk operations): Deferred creation of a `WorkingSet` to pass to
//!   [`Fill`]. This allows [`MultiInsertStrategy`](crate::graph::glue::MultiInsertStrategy)
//!   to seed [`View`]s with elements of the input.
//!
//! Traits that produce and otherwise interact with working sets are:
//!
//! * [`PruneStrategy`](crate::graph::glue::PruneStrategy): Creates a `WorkingSet` opaque type.
//! * [`MultiInsertStrategy`](crate::graph::glue::MultiInsertStrategy): Creates an implementation
//!   of [`AsWorkingSet`] to defer creation of the final `WorkingSet` until needed by worker
//!   tasks.
//!
//! # Implementation Strategies
//!
//! The `WorkingSet` type passed to [`Fill`] is opaque. Instead only the returned [`View`]
//! is accessed directly. This enables the following implementation strategies:
//!
//! * Allocate in `WorkingSet`, making `View` a shallow wrapper. This is the most
//!   straightforward approach. Elements are stored directly in `WorkingSet` and the accessor
//!   simply populates data.
//!
//! * Allocate in `Accessor`: Another approach is to allocate scratch space directly in `Self`
//!   for the elements in `itr`. This works since [`Fill::View`] borrows from `Self`.
//!
//! * Passthrough: If random access into the [`Accessor`] is synchronous and fast, then an
//!   implementation of [`View`] can simply reach through the [`Accessor`], use zero-sized
//!   types for the `WorkingSet`, and avoid allocation entirely.
//!
//! Within these, there are multiple strategies as well. For example, if data is known to
//! be standard slices of standard types, then an implementation of `WorkingSet` can pack
//! this data contiguously in memory for better memory locality.
//!
//! # Reuse in a `WorkingSet`
//!
//! Any particular `WorkingSet` can be reused across multiple prunes. It's up to the
//! [`Accessor`] implementation to decide whether this reuse can serve as a cache for vectors
//! or not. The default implementation [`Map`] offers both cached and uncached modes.
//!
//! Trade offs include:
//!
//! * Without any cross-fill reuse, more vector retrievals are made, but the data in the
//!   associated [`View`] is up-to-date.
//!
//! * With cross-fill reuse, the memory used by the working set increases.
//!
//! Working sets are used in the indexing algorithm within fairly tight temporal windows so
//! the risk of stale entries in the cache causing incorrect graphs is minimal.

use diskann_utils::{Reborrow, future::SendFuture};

use crate::{ANNError, provider::Accessor};

/////////////
// Exports //
/////////////

pub mod map;
pub use map::Map;

////////////
// Traits //
////////////

/// Populate a `WorkingSet` with data from an accessor and return a [`View`] over the set.
///
/// For each `i` in `itr` - the accessor should make the data behind `i` available, either
/// by storing it in `working_set` or through direct storage in the returned [`View`].
///
/// The `WorkingSet` type is constructed by either:
///
/// * [`PruneStrategy`](crate::graph::glue::PruneStrategy::create_working_set): Direct
///   construction of a working set for use in multiple prunes.
///
/// * [`MultiInsertStrategy::Seed`](crate::graph::glue::MultiInsertStrategy::Seed): Indirect
///   creation that uses [`AsWorkingSet`] for the final conversion. This allows the elements
///   in the input batch to [`multi_insert`](crate::graph::DiskANNIndex::multi_insert) to be
///   directly accessible by the `WorkingSet`/[`View`] types.
///
/// See Also: [`View`], [`AsWorkingSet`], [`Map`].
pub trait Fill<WorkingSet>: Accessor {
    /// Any critical error that occurs during [`fill`](Self::fill).
    ///
    /// Implementations of `fill` are expected to swallow any non-critical errors.
    type Error: Into<ANNError> + std::fmt::Debug + Send + Sync;

    /// The post-fill [`View`] used to access the retrieved data.
    type View<'a>: for<'b> View<Self::Id, ElementRef<'b> = Self::ElementRef<'b>> + Send + Sync
    where
        Self: 'a,
        WorkingSet: 'a;

    /// Make the data elements for items in `itr` available in the returned [`View`].
    ///
    /// Implementations may use `working_set` as scratch and to persist work across multiple calls.
    ///
    /// The input `itr` is `Clone` and is expected that the implementation of `Clone` is cheap
    /// and without interior mutability. This allows implementers to perform multiple passes
    /// of `itr` if needed.
    ///
    /// ## Missing Entries
    ///
    /// While it's a good idea to ensure all items in `itr` are fetched, callers are
    /// designed to tolerate a small number of missing entries without serious performance
    /// degradation.
    ///
    /// If a ID really needs to be fetched for algorithmic purposes, it will be the first
    /// item yielded from `itr`.
    fn fill<'a, Itr>(
        &'a mut self,
        working_set: &'a mut WorkingSet,
        itr: Itr,
    ) -> impl SendFuture<Result<Self::View<'a>, Self::Error>>
    where
        Itr: ExactSizeIterator<Item = Self::Id> + Clone + Send + Sync,
        Self: 'a;
}

/// Read-only view into a [working set](Fill).
///
/// Used by `occlude_list` and distance computations.
///
/// See Also: [`Fill`], [`Map`], [`map::View`].
pub trait View<I> {
    /// The reborrowed element with an arbitrarily short lifetime.
    ///
    /// Corresponds to [`crate::provider::Accessor::ElementRef`].
    type ElementRef<'a>;

    /// The element returned from [`Self::get`].
    ///
    /// The reborrow bound allows this type to be decoupled from the type provided to
    /// distance computers, and as a proxy for `Copy`.
    ///
    /// This does **not** need to be the same as [`crate::provider::Accessor::Element`]
    /// provided [`Self::ElementRef`] matches the accessor's `ElementRef`.
    type Element<'a>: for<'b> Reborrow<'b, Target = Self::ElementRef<'b>>
    where
        Self: 'a;

    /// Retrieve element associated with `id` if available.
    ///
    /// Users can expect that if `Self` was constructed from [`Fill::fill`], then `id` will
    /// belong to the iterator argument of that function.
    ///
    /// Retrieval should be reasonably efficient, but is not called in the hot-loop of prune.
    fn get(&self, id: I) -> Option<Self::Element<'_>>;
}

/// Use `Self` as a seed for `WorkingSet`.
///
/// This is used by [`crate::graph::glue::MultiInsertStrategy::Seed`] to package the data
/// provided to the input batch in a way that enables zero-copy access for multi-insert
/// worker threads.
///
/// See Also: [`Fill`], [`map::Builder`].
pub trait AsWorkingSet<WorkingSet> {
    /// Create a working set capable of holding `capacity` elements.
    fn as_working_set(&self, capacity: usize) -> WorkingSet;
}

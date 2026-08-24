/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # Layering
//!
//! A simplified interface for [`crate::Provider`] to use for building a graph index.

use std::num::NonZeroU16;

use diskann::ANNResult;
use thiserror::Error;

use crate::{
    counters::LocalCounters,
    num::{Capacity, IdLimit, MaxDegree},
};

pub mod full;
pub use full::{Full, FullPrecision};

/// Deferred creation of [`Layer`]s.
///
/// This is used in APIs like [`crate::Provider::new`] to defer allocation of large
/// in-memory data structures.
pub trait LayerConfig {
    /// The type of the resulting [`Layer`].
    type Layer: Layer;

    /// Build the target [`Layer`].
    fn build(self) -> ANNResult<Self::Layer>;
}

/// Configurable data layer for [`crate::Provider`].
///
/// Layers consist of the adjacency list and data for a concurrent in-memory graph index.
/// These are expected to be indexed using `u32` IDs from `0..self.id_limit()`, with
/// internal IDs in `0..self.capacity()` available for writing.
///
/// See also:
///
/// - [`Set`]: For assigning into the store.
/// - [`Search`]: Search compatibility with [`crate::Provider`].
/// - [`Insert`]: Insert compatibility with [`crate::Provider`].
pub trait Layer: Send + Sync + 'static {
    /// Return the [`MaxDegree`] of the internal graph.
    fn max_degree(&self) -> MaxDegree;

    /// Return the [`IdLimit`] for the data store.
    fn id_limit(&self) -> IdLimit;

    /// Return the functional [`Capacity`] for the data store.
    fn capacity(&self) -> Capacity;

    /// Retire the internal ID `i`. Such IDs will eventually be recycled for reuse.
    fn retire(&self, i: u32) -> ANNResult<()>;

    /// Return `true` if internal ID `i` is currently readable, returning `None` if `i`
    /// is outside `0..self.id_limit()`.
    fn is_readable(&self, i: u32) -> Option<bool>;
}

/// Attempt to write data into a [`Layer`].
///
/// This will attempt to find an available internal ID to which `element` can be assigned,
/// failing if no such ID can be found. The write is not eagerly committed. Instead, a
/// [`Guard`] is returned, allowing writes to be aborted if necessary.
pub trait Set<T>: Layer {
    /// The type of the [`Guard`] used to defer commitment of the write.
    type Guard<'a>: Guard;

    /// Attempt to write the data in `element` into the [`Layer`].
    ///
    /// Returns [`Self::Guard`] to retrieve the allocated internal ID for `element` and to
    /// defer commitment of the write until external code is ready.
    ///
    /// Dropping an unpublished [`Guard`] must abort the write and leave its data unpublished.
    fn set(&self, element: T) -> ANNResult<Self::Guard<'_>>;
}

/// An insert guard for [`Set`] providing deferred commitment of pending writes.
///
/// Guards provide several services:
///
/// * [`Guard::id`] returns the associated internal ID.
/// * [`Guard::publish`] commits the change, making the data at the slot publicly visible.
///
/// Dropping a guard without calling [`Guard::publish`] indicates a failed insert and
/// implementations should abort the write and leave the data unpublished.
pub trait Guard {
    fn id(&self) -> u32;
    fn publish(self);
}

/// Enable search over vectors defined by a [`Layer`].
pub trait Search: Send + Sync + 'static {
    /// The type of the query. This should be equivalent to the generic parameter in
    /// [`Set`], but needs to be replicated here due to limitations in the current trait
    /// design.
    type Query<'a>;

    /// Construct a [`crate::provider::SearchAccessor`] for the query.
    #[doc(hidden)]
    fn search_accessor<'a>(
        &'a self,
        query: Self::Query<'a>,
        provider: &'a (dyn std::any::Any + Send + Sync),
        counters: LocalCounters<'a>,
    ) -> ANNResult<crate::provider::SearchAccessor<'a>>;
}

/// An insert-specific specialization of [`Search`].
///
/// Note that the bounds for this trait are unnecessarily complicated, but require changes
/// to [`diskann`] to fully resolve.
pub trait Insert: Search + for<'a> Set<Self::Query<'a>> {
    #[doc(hidden)]
    fn insert_search_accessor<'a>(
        &'a self,
        query: Self::Query<'a>,
        provider: &'a (dyn std::any::Any + Send + Sync),
        counters: LocalCounters<'a>,
    ) -> ANNResult<crate::provider::SearchAccessor<'a>> {
        self.search_accessor(query, provider, counters)
    }

    #[doc(hidden)]
    fn prune_accessor<'a>(
        &'a self,
        counters: LocalCounters<'a>,
    ) -> ANNResult<crate::provider::PruneAccessor<'a>>;
}

//-----------------//
// Internal Traits //
//-----------------//

/// Trait-object-based implementation of
/// [`diskann::graph::glue::SearchAccessor::expand_beam`].
///
/// Dynamic dispatch is used to enable aggressive specialization of this primitive without
/// monomorphizing the entire search algorithm. Example specializations include:
///
/// * Optimizing for certain fixed dimensions.
/// * Inlining metric-specific distance functions.
/// * Tailoring prefetching to the dimension.
///
/// # Safety
///
/// Implementors must ensure that every ID accepted by [`Self::id_limit`] can be passed to
/// [`Self::expand_beam`] without an out-of-bounds memory access, including accesses performed
/// for prefetching.
pub(crate) unsafe trait ExpandBeam: Send + Sync + std::fmt::Debug {
    /// Evaluate a raw distance against index `i`.
    fn evaluate(&self, i: u32) -> ANNResult<Option<f32>>;

    /// Return an [`IdLimit`] for this primitive.
    ///
    /// Callers must be able to use this limit to satisfy the safety pre-conditions for
    /// [`Self::expand_beam`].
    ///
    /// See also: [`IdLimit::is_in_bounds`].
    fn id_limit(&self) -> IdLimit;

    /// Compute the distance between the query and each neighbor in `list`.
    ///
    /// Unreadable entries may be omitted. Return the number of entries in `buffer` that were
    /// written so that `buffer[..returned]` contains the expansion IDs and distances.
    ///
    /// # Safety
    ///
    /// * All items in `list` must be in bounds with respect to [`Self::id_limit`].
    /// * `buffer.len() >= list.len()`.
    unsafe fn expand_beam(&self, list: &[u32], buffer: &mut [(u32, f32)]) -> ANNResult<usize>;
}

/// Trait-object-based implementation for [`diskann::graph::glue::PruneAccessor`].
///
/// [`Self::prepare`] assigns a [`PruneKey`] to each retrieved entry, allowing implementations
/// to buffer data in a representation suitable for pruning. A prune session consists of one
/// call to [`Self::prepare`] followed by calls to [`Self::evaluate`] using the keys it
/// produced.
///
/// [`Self::prepare`] can be called multiple times for a single [`Prune`] object.
/// Implementations may assume that each call starts a new prune session and do not need to
/// buffer entries from all calls.
pub(crate) trait Prune: Send + Sync + std::fmt::Debug {
    /// Prepare items for pruning.
    ///
    /// The `items` iterator contains all the internal IDs that need to be pruned as keys.
    /// Each value in `items` is an output slot that should be populated with a [`PruneKey`]
    /// if retrieval succeeds. Implementations are responsible for providing unique
    /// [`PruneKey`]s and may assume these slots are initialized to `None`.
    ///
    /// Returns the total number of internal IDs that were successfully buffered.
    fn prepare(
        &mut self,
        items: hashbrown::hash_map::IterMut<'_, u32, Option<PruneKey>>,
    ) -> ANNResult<usize>;

    /// Compute the distance between the elements referenced by the two [`PruneKey`]s.
    ///
    /// Implementations may assume that `a` and `b` were produced by the most recent call to
    /// [`Self::prepare`] and are allowed to panic if this is violated. This property may
    /// **not** be relied on for `unsafe` code.
    fn evaluate(&self, a: PruneKey, b: PruneKey) -> f32;
}

/// A [`Prune`]-local miniature internal ID.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PruneKey(NonZeroU16);

impl PruneKey {
    const ONE: Self = Self(NonZeroU16::new(1).unwrap());

    /// Return the initial [`PruneKey`] in a sequence.
    pub(crate) fn counter() -> Self {
        Self::ONE
    }

    /// Increment `self` by 1.
    ///
    /// Overflows if called approximately 2^16 times.
    pub(crate) fn increment(self) -> Result<Self, Overflow> {
        match self.0.checked_add(1) {
            Some(v) => Ok(Self(v)),
            None => Err(Overflow),
        }
    }

    /// Return the zero-based index represented by this key.
    pub(crate) fn index(self) -> usize {
        usize::from(self.0.get()) - 1
    }
}

/// Incrementing a [`PruneKey`] overflowed.
#[derive(Debug, Error)]
#[error("prune list exceeded u16::MAX")]
pub(crate) struct Overflow;

diskann::convert_error!(Overflow);

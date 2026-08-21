/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A concurrent in-memory data store for driving [`plugin::Plugin`]s.
//!
//! This supports concurrent data access, deletes, and inserts through a safe interface.
//! Data is stored internally in slots indexed from `[0..N)` with `K` points reserved at the
//! end at positions `[N..N+K)`.
//!
//! ## Reading
//!
//! A [`Store`] provides no direct way of reading data. Instead, the [`plugin::Plugin`] is
//! responsible for exposing an appropriate reader (e.g., [`invasive::Invasive::reader`]) in
//! accordance with its lifecycle implementation. [`Store::guard`] can be used for
//! this purpose by acquiring an [`epoch::Guard`] for a [`Store`].
//!
//! ## Writing
//!
//! [`Store::acquire`] is used to find and claim an unused internal [`Slot`]. A [`Slot`]
//! provides write access to its corresponding [`Slot::data`]. Either [`Slot::publish`] or
//! [`Slot::freeze`] can be used to make data readable.
//!
//! If a [`Slot`] is dropped, its corresponding slot is returned to the [`Store`] without
//! publishing its contents.
//!
//! The index of the slot chosen may be obtained via [`Slot::slot`].
//!
//! ## Deleting
//!
//! Data is deleted via [`Store::retire`]. This immediately marks the corresponding slot as
//! unavailable for future readers. However, the retired slot will not be reused until the
//! [`Store`] can guarantee that no readers that could be using the data are active.
//!
//! Slots are automatically reclaimed as part of slot acquisition in the "writing" phase.
//!
//! ## Neighbor Access
//!
//! The [`Store`] also contains a [`Neighbors`] instance to store adjacency lists. Since
//! neighbors are generally accessed less frequently than data with a higher volume of write
//! traffic, fine-grained locks are used for this data structure.
//!
//! # Details
//!
//! This uses an implementation of the epoch-based reclamation (EBR) provided by [`Registry`].
//! Plugins follow the lifecycle process defined in the [plugin module docs](plugin).
//!
//! The EBR scheme allows readers to safely access data while only generating read traffic to
//! the CPU caches. The cost is that there is a delay between when slots are retired and when
//! they can be reused, with a long lived reader blocking this reclamation. As such, users of
//! this data structure should ensure that readers are reasonably short lived.

use std::{
    iter::repeat_n,
    mem::ManuallyDrop,
    num::{NonZeroU32, NonZeroUsize},
    sync::atomic::Ordering,
};

use diskann::{ANNError, utils::IntoUsize};
use thiserror::Error;

use crate::{
    buffer::BufferError,
    epoch::{self, Registry},
    freelist::{self, Freelist},
    neighbors::{Neighbors, NeighborsError},
    num::{Capacity, IdLimit, MaxDegree},
    tag::{AtomicTag, Tag},
};

pub(crate) mod invasive;
pub(crate) mod plugin;

#[cfg(any(test, feature = "integration-test"))]
pub(crate) mod checked;

/// To make extra sure that [`plugin::Plugin`] life-cycle arguments are not callable outside
/// of this module (i.e., elsewhere in this crate), this [`Lifecycle`] marker type is used
/// that is only constructible in this module.
#[derive(Debug)]
pub(crate) struct Lifecycle(());

impl Lifecycle {
    /// Construct a new [`Lifecycle`].
    ///
    /// DO NOT MAKE THIS `pub(anything)`. It helps prevent accidentally interacting with
    /// plugins when all uses should be managed in this file instead.
    const fn new() -> Self {
        Self(())
    }
}

/// Configuration for the concurrent store.
#[derive(Debug, Clone)]
pub struct Config {
    /// The number of epoch guard slots.
    ///
    /// Increasing this number will increase the number of threads that can work concurrently
    /// on the index at the cost of longer scan times for epoch advancement.
    epoch_guard_slots: NonZeroUsize,

    /// The capacity of the fast free list.
    freelist_recycle_capacity: NonZeroU32,
}

impl Config {
    /// Create a new [`Config`] with default concurrency parameters.
    pub fn new() -> Self {
        const DEFAULT_FREELIST_RECYCLE_CAPACITY: NonZeroU32 = NonZeroU32::new(1024).unwrap();
        Self {
            epoch_guard_slots: Registry::default_guard_slots(),
            freelist_recycle_capacity: DEFAULT_FREELIST_RECYCLE_CAPACITY,
        }
    }

    /// Overwrite the number of epoch guard slots.
    ///
    /// Increasing this number will increase the number of threads that can work concurrently
    /// on the index at the cost of longer scan times for epoch advancement.
    pub fn epoch_guard_slots(&mut self, epoch_guard_slots: NonZeroUsize) -> &mut Self {
        self.epoch_guard_slots = epoch_guard_slots;
        self
    }

    /// Overwrite the capacity of the freelist recycle queue.
    ///
    /// Increasing the capacity of the queue will allow more recycled IDs to be retrieved
    /// without triggering a scan, but will cost more memory.
    pub fn freelist_recycle_capacity(
        &mut self,
        freelist_recycle_capacity: NonZeroU32,
    ) -> &mut Self {
        self.freelist_recycle_capacity = freelist_recycle_capacity;
        self
    }

    /// An exhaustive constructor initializing every element.
    ///
    /// This is exposed under "integration-test" since it will change to reflect the state
    /// of the underlying data structure, potentially causing more churn for users if it were
    /// unconditionally exposed.
    #[cfg(any(test, feature = "integration-test"))]
    #[doc(hidden)]
    pub fn __exhaustive(
        epoch_guard_slots: NonZeroUsize,
        freelist_recycle_capacity: NonZeroU32,
    ) -> Self {
        Self {
            epoch_guard_slots,
            freelist_recycle_capacity,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

/// Layout parameters for [`Store`] and the corresponding [`plugin::Plugin`].
#[derive(Debug, Clone)]
pub(crate) struct Layout {
    /// The number of non-frozen slots to create space for.
    capacity: Capacity,

    /// The maximum number of neighbors in each adjacency list.
    max_degree: MaxDegree,

    /// The number of immutable points to reserve at the end of the [`Store`].
    frozen: u32,
}

impl Layout {
    /// Create a new [`Layout`] capable of holding `capacity` non-frozen points and `frozen`
    /// reserved points at the end.
    ///
    /// After construction, the only distinction between non-frozen and frozen points is that
    /// IDs for frozen points will not be selected through [`Store::acquire`]. Frozen slots
    /// IDs can be obtained via [`Store::frozen`], and direct acquisition can be done with
    /// [`Store::slot`].
    ///
    /// All adjacency lists will have a maximum capacity of `max_degree`.
    pub(crate) fn new(capacity: Capacity, max_degree: MaxDegree, frozen: u32) -> Self {
        Self {
            capacity,
            max_degree,
            frozen,
        }
    }
}

/// A concurrent data and graph store.
#[derive(Debug)]
pub(crate) struct Store<P> {
    // The [`plugin::Plugin`] managed by this [`Store`].
    plugin: P,

    // The number of unfrozen points.
    unfrozen: Capacity,

    // The authoritative source of truth for the state of each slot.
    tags: Vec<AtomicTag>,

    // Acceleration of finding free slot IDs.
    freelist: Freelist,

    // EBR registry.
    registry: Registry,

    // Graph.
    neighbors: Neighbors,
}

// TODO: This is a guess and probably needs tuning.
const RETRY_LIMIT: usize = 20;

impl<P> Store<P>
where
    P: plugin::Plugin,
{
    /// Create a new [`Store`].
    pub(crate) fn new<C>(layout: Layout, config: Config, plugin: C) -> Result<Self, StoreError>
    where
        C: plugin::PluginConfig<Plugin = P>,
    {
        let Layout {
            capacity,
            max_degree,
            frozen,
        } = layout;

        let Config {
            epoch_guard_slots,
            freelist_recycle_capacity,
        } = config;

        let too_many_entries = || StoreError::too_many_entries(capacity, frozen);

        // We have a hard upper-bound of `u32::MAX` total slots.
        //
        // This enforces that bound.
        let entries: u32 = capacity
            .value()
            .try_into()
            .map_err(|_| too_many_entries())?;

        let id_limit = IdLimit::new(entries.checked_add(frozen).ok_or_else(too_many_entries)?);

        let max_degree: u32 = max_degree
            .value()
            .try_into()
            .map_err(|_| StoreError::too_many_neighbors(max_degree))?;

        let plugin = plugin::PluginConfig::build(plugin, id_limit).map_err(StoreError::plugin)?;

        let plugin_id_limit = plugin.id_limit();
        if plugin_id_limit != id_limit {
            return Err(StoreError::invalid_construction(plugin_id_limit, id_limit));
        }

        let me = Self {
            plugin,
            unfrozen: capacity,
            tags: repeat_n(Tag::AVAILABLE, id_limit.as_usize())
                .map(AtomicTag::new)
                .collect(),

            // NOTE: The `Freelist` is initialized to `entries` and not `total` because
            // we do not want it to release frozen IDs.
            freelist: Freelist::new(entries, freelist_recycle_capacity),
            registry: Registry::with_capacity(epoch_guard_slots),
            neighbors: Neighbors::new(id_limit, max_degree)?,
        };

        Ok(me)
    }

    /// Return the [`plugin::Plugin`] for this store.
    pub(crate) fn plugin(&self) -> &P {
        &self.plugin
    }

    /// Return the range of slots containing frozen items in `self`.
    pub(crate) fn frozen(&self) -> std::ops::Range<u32> {
        (self.unfrozen.value() as u32)..self.neighbors.entries()
    }

    /// Return the [`IdLimit`] for this store.
    pub(crate) fn id_limit(&self) -> IdLimit {
        // The numeric cast is safe: We verify during construction that `self.tags.len()` fits
        // in a `u32`.
        IdLimit::new(self.tags.len() as u32)
    }

    /// Return the [`Capacity`] for this store.
    pub(crate) fn capacity(&self) -> Capacity {
        self.unfrozen
    }

    /// Return the [`Neighbors`] for this store.
    pub(crate) fn neighbors(&self) -> &Neighbors {
        &self.neighbors
    }

    /// Attempt to reclaim retired slots.
    ///
    /// If successful, returns the number of slots reclaimed.
    pub(crate) fn try_drain(&self) -> Option<usize> {
        let drain = self.registry.try_advance()?;
        let items = drain.len();
        for i in drain {
            #[expect(clippy::panic, reason = "this is an unrecoverable program bug")]
            let Some(tag) = self.tags.get(i.into_usize()) else {
                panic!(
                    "received an invalid ID ({}) while reclaiming slots - max allowed is {}",
                    i,
                    self.neighbors.entries(),
                );
            };

            // We release the plugin before the main tag. The other direction would
            // prematurely advertise availability.
            //
            // SAFETY: IDs only get added to the `epoch::Registry` upon retiring, and are
            // not released until the registry confirms that all guards active when the id
            // was retired have been dropped.
            //
            // Therefore, this slot has no accessors and is ready to be reclaimed.
            unsafe { plugin::Plugin::reclaim(self.plugin(), i, Lifecycle::new()) };

            // Use `Release` ordering to ensure that the store to the mirror cannot get moved
            // after the store to the authoritative list.
            //
            // The `load + check` is just runtime validation. The calling thread is expected
            // to have exclusive ownership of this tag.
            //
            // Using a load followed by a store avoids a CAS loop, which may be cheaper for
            // bulk reclamation at the cost of detecting concurrent modification only through
            // the assertion.
            assert_eq!(
                tag.load(Ordering::Relaxed),
                Tag::RETIRING,
                "CONCURRENCY VIOLATION",
            );

            tag.store(Tag::AVAILABLE, Ordering::Release);
            self.freelist.push(i);
        }
        Some(items)
    }

    /// Create an [`epoch::Guard`] for the [`epoch::Registry`] within `self` and invoke `f`
    /// with that guard, returning the result.
    ///
    /// This can be used by [`plugin::Plugin`] readers to establish a verifiable chain of
    /// custody for an [`epoch::Guard`] over the plugin.
    pub(crate) fn guard<'a, F, R>(&'a self, f: F) -> Result<R, epoch::Unavailable>
    where
        F: FnOnce(&'a P, epoch::Guard<'a>) -> R,
    {
        let guard = self.registry.guard()?;
        Ok(f(self.plugin(), guard))
    }

    /// Attempt to acquire a new [`Slot`] for writing.
    ///
    /// This method first consults the freelist and falls back to scanning the tags list
    /// if no ID is available from the fast path.
    pub(crate) fn acquire(&self) -> Option<Slot<'_, <P as plugin::Plugin>::Slot<'_>>> {
        for _ in 0..RETRY_LIMIT {
            match self.freelist.pop() {
                freelist::Id::Found(id) => {
                    if let Some(slot) = self.slot(id) {
                        return Some(slot);
                    }
                }
                freelist::Id::Scan => match self.scan_acquire() {
                    Some(slot) => return Some(slot),
                    None => {
                        self.try_drain();
                    }
                },
            }
        }
        None
    }

    /// Attempt to retire slot `i`. If successful, this slot will be placed in an internal
    /// retirement queue for reclamation once we can prove no readers are active that could
    /// have observed this transition.
    ///
    /// Returns `Ok(())` if the slot was successfully retired.
    ///
    /// # Errors
    ///
    /// Returns an error in any of the following conditions:
    ///
    /// * The slot index `i` is out-of-bounds.
    /// * The slot is not in a state that can be retired (e.g., it is already retired or
    ///   is owned by a different thread).
    /// * An [`epoch::Guard`] could not be obtained due to registration slot exhaustion.
    /// * An attempt to acquire the slot after these checks races with another thread and
    ///   the race was lost.
    pub(crate) fn retire(&self, i: usize) -> Result<(), RetireError> {
        let tag = self.tags.get(i).ok_or(RetireError::OutOfBounds)?;
        let current = tag.load(Ordering::Relaxed);

        // We can only perform a deletion if the generation is not in a reserved state.
        if current.is_reserved() {
            return Err(RetireError::SlotIsReserved { tag: current });
        }

        let guard = self
            .registry
            .guard()
            .map_err(RetireError::GuardUnavailable)?;

        let retiring = Tag::RETIRING;

        // Even if we make this change, we can't access any data until we wait for the
        // epoch to be bumped. As such, relaxed semantics are fine.
        match tag.compare_exchange(current, retiring, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => {
                // Set the metadata in the mirror as well.
                //
                // SAFETY: The above compare-exchange ensures that we transitioned the
                // authoritative state from "published" to "retired" and prevents other
                // threads from attempting the same transition.
                unsafe { plugin::Plugin::retire(self.plugin(), i as u32, Lifecycle::new()) };
                guard.retire(i as u32);
                Ok(())
            }
            Err(_) => Err(RetireError::CouldNotClaimSlot),
        }
    }

    /// A somewhat crude algorithm for cooperatively performing slot scanning.
    ///
    /// This uses [`Freelist::scan`] to acquire a disjoint chunk of the ID space for scanning,
    /// spreading out the search across multiple threads.
    ///
    /// If we successfully acquire a slot, we continue for the rest of the bucket returned
    /// by [`Freelist::scan`] and add any available slots to the freelist (allowing other
    /// threads to find them).
    ///
    /// Periodically, the freelist is checked to see if another thread has found an available
    /// slot for us.
    fn scan_acquire(&self) -> Option<Slot<'_, <P as plugin::Plugin>::Slot<'_>>> {
        // This is potentially quite slow, so scan approximately `1 / RETRY_LIMIT` of the
        // writable range. The outer retry loop provides broader coverage.
        let mut remaining = self.unfrozen.value().div_ceil(RETRY_LIMIT);
        let mut chunks_since_freelist_check = 0;
        let mut acquired: Option<Slot<'_, <P as plugin::Plugin>::Slot<'_>>> = None;

        while remaining != 0 {
            let chunk = self.freelist.scan();
            remaining = remaining.saturating_sub(chunk.len());

            for slot in chunk {
                #[expect(
                    clippy::expect_used,
                    reason = "this is a serious bug with the freelist"
                )]
                let tag = self
                    .tags
                    .get(slot.into_usize())
                    .expect("freelist scan should not give out invalid IDs");

                // If this slot is available and we haven't claimed a slot yet, try to
                // claim it. Otherwise, continue with the scan to partially repopulate the
                // freelist for other threads.
                if tag.load(Ordering::Relaxed) == Tag::AVAILABLE {
                    if acquired.is_none() {
                        // SAFETY: We're guaranteed that `tag` belongs to `slot`.
                        acquired = unsafe { self.try_acquire(tag, slot) };
                    } else {
                        self.freelist.push(slot);
                    }
                }
            }

            if acquired.is_some() {
                return acquired;
            }

            chunks_since_freelist_check += 1;
            if chunks_since_freelist_check == 4 {
                if let Some(id) = self.freelist.pop_recycled()
                    && let Some(slot) = self.slot(id)
                {
                    return Some(slot);
                }
                chunks_since_freelist_check = 0;
            }
        }
        None
    }

    /// Attempt to directly acquire a [`Slot`] to id `i`.
    ///
    /// Returns `None` if `i` is not within [`Self::id_limit`] or if the slot is not currently
    /// acquirable.
    pub(crate) fn slot(&self, i: u32) -> Option<Slot<'_, <P as plugin::Plugin>::Slot<'_>>> {
        let tag = &self.tags.get(i.into_usize())?;

        // SAFETY: We've guaranteed that `tag` belongs to `slot`.
        unsafe { self.try_acquire(tag, i) }
    }

    /// Try to acquire `slot` with the associated `tag`.
    ///
    /// # Safety
    ///
    /// Caller asserts that `tag` was obtained from `self.tags[slot]`. This is meant as
    /// a performance optimization where `tag` is first queried for potential availability.
    unsafe fn try_acquire<'a>(
        &'a self,
        tag: &'a AtomicTag,
        slot: u32,
    ) -> Option<Slot<'a, <P as plugin::Plugin>::Slot<'a>>> {
        if tag.load(Ordering::Relaxed) != Tag::AVAILABLE {
            return None;
        }

        match tag.compare_exchange(
            Tag::AVAILABLE,
            Tag::OWNED,
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                // SAFETY: The above compare-exchange ensures that this slot was previously
                // "available" and prevents other threads from trying to acquire this slot.
                // The acquire ordering synchronizes with the release transition to
                // "available", making plugin reclamation or abort work visible before
                // `Plugin::acquire`.
                //
                // The `Slot` data structure ensures that exactly one of the terminal methods
                // for `plugin::Slot` is called.
                let data =
                    unsafe { plugin::Plugin::acquire(self.plugin(), slot, Lifecycle::new()) };

                Some(Slot {
                    tag,
                    data: ManuallyDrop::new(data),
                    slot,
                })
            }
            Err(_) => None,
        }
    }

    /// Return whether or not it is probably okay to read from the slot `i`.
    ///
    /// This check is approximate and non-synchronizing. A full check requires the
    /// plugin-specific reader.
    ///
    /// Returns `None` if `i` is not within [`Self::id_limit`].
    pub(crate) fn can_read_approximate(&self, i: usize) -> Option<bool> {
        self.tags
            .get(i)
            .map(|tag| tag.load(Ordering::Relaxed).can_read())
    }

    #[cfg(test)]
    fn writable(&self) -> std::ops::Range<u32> {
        0..self.unfrozen.value() as u32
    }
}

/// Errors occurring during [`Store::new`].
#[derive(Debug, Error)]
#[error(transparent)]
pub(crate) struct StoreError(StoreErrorInner);

impl StoreError {
    fn too_many_entries(capacity: Capacity, frozen: u32) -> Self {
        Self(StoreErrorInner::TooManyEntries {
            entries: capacity.value(),
            frozen,
        })
    }

    fn too_many_neighbors(neighbors: MaxDegree) -> Self {
        Self(StoreErrorInner::TooManyNeighbors {
            neighbors: neighbors.value(),
        })
    }

    #[track_caller]
    fn plugin<E>(err: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self(StoreErrorInner::PluginError(ANNError::new(err)))
    }

    fn invalid_construction(got: IdLimit, expected: IdLimit) -> Self {
        Self(StoreErrorInner::InvalidConstruction { got, expected })
    }
}

impl From<BufferError> for StoreError {
    fn from(err: BufferError) -> Self {
        Self(err.into())
    }
}

impl From<NeighborsError> for StoreError {
    fn from(err: NeighborsError) -> Self {
        Self(err.into())
    }
}

diskann::convert_error!(StoreError);

#[derive(Debug, Error)]
enum StoreErrorInner {
    #[error(
        "total points ({} + {} frozen) must not exceed `u32::MAX`",
        entries,
        frozen
    )]
    TooManyEntries { entries: usize, frozen: u32 },
    #[error("number of neighbors ({}) may not exceed `u32::MAX`", neighbors)]
    TooManyNeighbors { neighbors: usize },
    #[error(transparent)]
    BufferError(#[from] BufferError),
    #[error(transparent)]
    NeighborsError(#[from] NeighborsError),
    #[error("error creating plugin")]
    PluginError(ANNError),
    #[error("requested {} but the plugin returned {}", expected, got)]
    InvalidConstruction { got: IdLimit, expected: IdLimit },
}

/// Error conditions for [`Store::retire`].
#[derive(Debug, Error)]
pub(crate) enum RetireError {
    /// Slot index was out-of-bounds.
    #[error("index out of bounds")]
    OutOfBounds,
    /// The slot cannot be retired because it is in a reserved state.
    #[error("slot is reserved: {}", tag)]
    SlotIsReserved { tag: Tag },
    /// An [`epoch::Guard`] could not be acquired.
    #[error(transparent)]
    GuardUnavailable(epoch::Unavailable),
    /// Another thread won the retirement race.
    #[error("could not claim slot")]
    CouldNotClaimSlot,
}

diskann::convert_error!(RetireError);

/// A writable buffer into the data managed by a [`Store`], obtained from [`Store::acquire`].
///
/// This is the only safe way to interact with a [`plugin::Slot`] since this ensures that one
/// of the terminal methods is called. Dropping a [`Slot`] without calling [`Slot::publish`]
/// or [`Slot::freeze`] automatically invokes [`plugin::Slot::abort`].
#[derive(Debug)]
pub(crate) struct Slot<'a, S>
where
    S: plugin::Slot,
{
    tag: &'a AtomicTag,
    data: ManuallyDrop<S>,
    slot: u32,
}

impl<'a, S> Slot<'a, S>
where
    S: plugin::Slot,
{
    /// View the raw inner slot.
    pub(crate) fn data(&mut self) -> &mut S {
        &mut self.data
    }

    /// Return the slot associated with this write.
    pub(crate) fn slot(&self) -> u32 {
        self.slot
    }

    pub(crate) fn freeze(self) {
        // Suppress normal `Drop`.
        let mut me = ManuallyDrop::new(self);

        // Freeze the inner slot.
        plugin::Slot::freeze(
            // SAFETY: The `ManuallyDrop` `data` is not used after this call.
            unsafe { ManuallyDrop::take(&mut me.data) },
            Lifecycle::new(),
        );

        // Update the authoritative store.
        me.tag.store(Tag::FROZEN, Ordering::Release);
    }

    /// Consume the slot and publish the written data for all readers.
    ///
    /// Return the internal slot ID.
    pub(crate) fn publish(self) -> u32 {
        let id = self.slot();

        // Suppress normal `Drop`.
        let mut me = ManuallyDrop::new(self);

        // Publish the inner slot.
        plugin::Slot::publish(
            // SAFETY: The `ManuallyDrop` `data` is not used after this call.
            unsafe { ManuallyDrop::take(&mut me.data) },
            Lifecycle::new(),
        );

        // Update the authoritative store.
        me.tag.store(Tag::PUBLISHED, Ordering::Release);
        id
    }
}

impl<S> Drop for Slot<'_, S>
where
    S: plugin::Slot,
{
    fn drop(&mut self) {
        plugin::Slot::abort(
            // SAFETY: The `ManuallyDrop` `data` is not used after this call.
            unsafe { ManuallyDrop::take(&mut self.data) },
            Lifecycle::new(),
        );
        self.tag.store(Tag::AVAILABLE, Ordering::Release);
    }
}

///////////
// Tests //
///////////

/// These tests are basic functionality tests for the store.
///
/// Longer running concurrency tests are in the integration test suite.
#[cfg(test)]
mod tests {
    use super::{checked::Checked, *};

    use std::assert_matches;

    /// A faulty config for [`Checked`] that doesn't respect the [`IdLimit`].
    #[derive(Debug)]
    struct FaultyConfig;

    impl plugin::PluginConfig for FaultyConfig {
        type Plugin = Checked;
        type Error = diskann::error::Infallible;

        fn build(self, id_limit: IdLimit) -> Result<Checked, diskann::error::Infallible> {
            let faulty = id_limit.value().checked_sub(1).unwrap_or(1);
            Ok(Checked::new(IdLimit::new(faulty)))
        }
    }

    // Build a store with `entries` writable slots of `entry_bytes` each, backed by `frozen`
    // zeroed frozen points. The frozen points occupy the highest slot indices.
    fn store(entries: usize, frozen: u32) -> Result<Store<Checked>, StoreError> {
        let config =
            Config::__exhaustive(NonZeroUsize::new(10).unwrap(), NonZeroU32::new(16).unwrap());

        let layout = Layout::new(Capacity::new(entries), MaxDegree::new(0), frozen);
        let store = Store::new(layout, config, Checked::config())?;
        assert_eq!(store.frozen().len(), frozen.into_usize());

        for (i, id) in store.frozen().enumerate() {
            let mut slot = store.slot(id).unwrap();
            slot.data().set(i as u64);
            slot.freeze();
        }

        Ok(store)
    }

    fn reader(store: &Store<Checked>) -> checked::Reader<'_> {
        Checked::reader(store).unwrap()
    }

    //------------------------//
    // Constructor validation //
    //------------------------//

    #[test]
    fn new_rejects_total_slot_overflow() {
        // `entries` alone fits in u32, but `entries + frozen` overflows it.
        let err = Store::new(
            Layout::new(Capacity::new(u32::MAX as usize), MaxDegree::new(0), 1),
            Config::default(),
            Checked::config(),
        )
        .unwrap_err();
        assert_matches!(err.0, StoreErrorInner::TooManyEntries { .. });
    }

    #[test]
    fn new_rejects_too_many_neighbors() {
        let err = Store::new(
            Layout::new(
                Capacity::new(4),
                MaxDegree::new(u32::MAX.into_usize() + 1),
                0,
            ),
            Config::default(),
            Checked::config(),
        )
        .unwrap_err();
        assert_matches!(err.0, StoreErrorInner::TooManyNeighbors { .. });
    }

    #[test]
    fn new_rejects_faulty_plugin() {
        let err = Store::new(
            Layout::new(
                Capacity::new(4),
                MaxDegree::new(10),
                0,
            ),
            Config::default(),
            FaultyConfig,
        ).unwrap_err();
        assert_matches!(err.0, StoreErrorInner::InvalidConstruction { .. });
    }

    //--------//
    // Layout //
    //--------//

    #[test]
    fn frozen_range_follows_writable_slots() {
        let s = store(4, 2).unwrap();

        // Writable slots are [0, 4); frozen points occupy [4, 6).
        assert_eq!(s.frozen(), 4..6);

        let reader = reader(&s);
        for i in 0u32..4 {
            assert!(!s.can_read_approximate(i.into_usize()).unwrap());
            assert!(reader.read(i).is_none());
        }

        assert!(s.can_read_approximate(4).unwrap());
        assert_eq!(reader.read(4).unwrap().get(), 0);

        assert!(s.can_read_approximate(5).unwrap());
        assert_eq!(reader.read(5).unwrap().get(), 1);

        assert!(s.can_read_approximate(6).is_none());
        assert!(reader.read(6).is_none());
    }

    ///////////////
    // Lifecycle //
    ///////////////

    #[test]
    fn acquire_write_publish_read_roundtrip() {
        let s = store(4, 1).unwrap();

        let reader = reader(&s);

        let idx = {
            let mut slot = s.acquire().expect("a fresh store has free slots");
            let idx = slot.slot();
            slot.data().set(10);

            // Before the slot is dropped - we should not be able to read it.
            assert!(reader.read(idx).is_none());
            assert!(!s.can_read_approximate(idx.into_usize()).unwrap());
            slot.publish();
            idx
        };

        assert_eq!(reader.read(idx).unwrap().get(), 10,);
        assert!(s.can_read_approximate(idx.into_usize()).unwrap());
    }

    #[test]
    fn unpublished_slots_are_immediately_available() {
        let s = store(4, 1).unwrap();

        let reader = reader(&s);

        let idx = {
            let mut slot = s.acquire().expect("a fresh store has free slots");
            let idx = slot.slot();
            slot.data().set(100);

            // Before the slot is dropped - we should not be able to read it.
            assert!(reader.read(idx).is_none());
            assert!(!s.can_read_approximate(idx.into_usize()).unwrap());

            // NOTE: We do not explicitly publish the slot.
            idx
        };

        assert!(reader.read(idx).is_none());
        assert!(!s.can_read_approximate(idx.into_usize()).unwrap());
    }

    #[test]
    fn acquire_exhausts_then_reports_none() {
        let s = store(2, 1).unwrap();
        // Hold the guards so the slots stay owned.
        let _a = s.acquire().expect("first writable slot");
        let _b = s.acquire().expect("second writable slot");
        assert!(
            s.acquire().is_none(),
            "all writable slots are owned, so acquire must fail"
        );
    }

    //--------//
    // Retire //
    //--------//

    #[test]
    fn retire_out_of_bounds() {
        let s = store(4, 1).unwrap();
        assert!(matches!(s.retire(999), Err(RetireError::OutOfBounds)));
    }

    #[test]
    fn retire_rejects_reserved_slots() {
        let s = store(4, 1).unwrap();
        // An untouched writable slot is AVAILABLE, which is a reserved state.
        assert!(matches!(
            s.retire(0),
            Err(RetireError::SlotIsReserved { .. })
        ));
        // A frozen slot is likewise reserved.
        let frozen = s.frozen().start as usize;
        assert!(matches!(
            s.retire(frozen),
            Err(RetireError::SlotIsReserved { .. })
        ));
        // An owned slot is not retirable.
        let slot = s.acquire().unwrap();
        assert!(matches!(
            s.retire(slot.slot() as usize),
            Err(RetireError::SlotIsReserved { .. })
        ));
    }

    #[test]
    fn retire_published_slot_then_unreadable() {
        let s = store(4, 1).unwrap();

        let idx = {
            let mut slot = s.acquire().unwrap();
            slot.data().set(101);
            slot.publish()
        };

        assert!(s.retire(idx.into_usize()).is_ok());

        // A reader opened after retirement must not observe the retired slot.
        let reader = reader(&s);
        assert_matches!(reader.read(idx), None);

        // The slot can also not be retired again.
        assert!(matches!(
            s.retire(idx.into_usize()),
            Err(RetireError::SlotIsReserved { .. })
        ));
    }

    //---------//
    // Recycle //
    //---------//

    #[test]
    fn test_recycling() {
        let entries = if cfg!(miri) { 16 } else { 2048 };

        let s = store(entries, 2).unwrap();

        assert_eq!(s.writable().len(), entries);

        // Claim all slots.
        let mut count = 0;
        while let Some(mut slot) = s.acquire() {
            slot.data().set(count as u64);
            slot.publish();
            count += 1;
        }

        assert_eq!(count, entries);

        {
            let reader = reader(&s);
            for i in 0..entries {
                assert_eq!(reader.read(i as u32).unwrap().get(), i as u64);
            }
        }

        // Now that all slots are claimed - retire all slots.
        for i in s.writable() {
            s.retire(i.into_usize()).unwrap();
        }

        // Verify that we can claim all slots again.
        let mut count = 0;
        while let Some(mut slot) = s.acquire() {
            slot.data().set(count as u64);
            slot.publish();
            count += 1;
        }

        assert_eq!(count, entries);
    }
}

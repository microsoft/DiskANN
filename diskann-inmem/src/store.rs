/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A concurrent in-memory data store for uniformly sized data.
//!
//! This supports concurrent data access, deletes, and inserts through a safe interface.
//! Data is stored internally in slots indexed from `[0..N)` with `K` points reserved at the
//! end at positions `[N..N+K)`.
//!
//! ## Reading
//!
//! Read access requires a [`Reader`] produced by [`Store::reader`]. [`Reader::read`]
//! provides read-only access to data at slot `i` if the data is valid for reads.
//!
//! ## Writing
//!
//! [`Store::acquire`] is used to find and claim an unused internal [`Slot`]. A [`Slot`]
//! provides write access to its coresponding data which is published when the [`Slot`] is
//! dropped.
//!
//! The index of the slot chosen may be obtained via [`Slot::slot`].k
//!
//! ## Deleting
//!
//! Data is deleted via [`Store::retire`]. This immediately marks the corresponding slot as
//! unavailable for future readers. However, the retired slot will not be reused until the
//! [`Store`] can guarantee that no [`Reader`]s that could be using the data are active.
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
//! Concurrency tags are mirrored inline with the stored data (just after the data payload)
//! to keep memory access localized. As such, high-performance implementations will want to
//! fetch the last cache line of data first to ensure the tag is resident in cache for faster
//! data checks.
//!
//! The EBR scheme allows readers to safely access data while only generating read traffic to
//! the CPU caches. The cost is that there is a delay between when slots are retired and when
//! they can be reused, with a long lived [`Reader`] blocking this reclamation. As such,
//! users of this data structure should ensure that [`Reader`]s are reasonably short lived.
//!
//! Internally, the data belongs to a single allocation.

use std::{
    iter::repeat_n,
    num::{NonZeroU32, NonZeroUsize},
    sync::atomic::Ordering,
};

use diskann::utils::IntoUsize;
use diskann_utils::views::MatrixView;
use thiserror::Error;

use crate::{
    buffer::{Buffer, BufferError, RawSlice},
    epoch::{self, Registry},
    freelist::{self, Freelist},
    neighbors::{Neighbors, NeighborsError},
    num::{Align, Bytes},
    tag::{AtomicTag, Tag},
};

/// A concurrent data and graph store.
#[derive(Debug)]
pub(crate) struct Store {
    // The invasive store where concurrency tags are stored inline with the data.
    //
    // These tags are mirrored from `tags` - with the latter being used for secondary scans
    // offering slightly better locality.
    //
    // The inline tags are stored after the data.
    buffer: Buffer,

    // The unpadded size of each row in `buffer`. This includes both the data **and** the
    // 1-byte tag. Tags are located at byte `unpadded - 1`.
    unpadded: Bytes,

    // The number of unfrozen points. This is guaranteed to be less than `buffer`.
    unfrozen: usize,

    // The authoritative source of truth for the state of each slot.
    tags: Vec<AtomicTag>,
    freelist: Freelist,

    // EBR registry.
    registry: Registry,

    // Graph.
    neighbors: Neighbors,
}

/// The number of bytes occupied by the in-line concurrency tag.
pub(crate) const TAG_SIZE: Bytes = Bytes::size_of::<AtomicTag>();

const TWO: NonZeroUsize = NonZeroUsize::new(2).unwrap();

// TODO: This is a guess and probably needs tuning.
const RETRY_LIMIT: usize = 20;

impl Store {
    /// Create a new [`Store`] capable of holding [`entries`] non-frozen slots each of
    /// length `bytes`.
    pub(crate) fn new(
        entries: usize,
        bytes: Bytes,
        max_neighbors: usize,
        init: MatrixView<'_, u8>,
    ) -> Result<Self, StoreError> {
        if init.ncols() != bytes.value() {
            return Err(StoreError::mismatched_frozen_point_dim(init.ncols(), bytes));
        }

        if init.nrows() == 0 {
            return Err(StoreError::need_frozen_point());
        }

        #[expect(
            clippy::expect_used,
            reason = "we expect `init` to have at least one row, so this should never happen"
        )]
        let unpadded = bytes
            .checked_add(TAG_SIZE)
            .expect("unreachable because `init` cannot exceed `isize::MAX` bytes");

        // Pad to half a cache line. When data occupies just part of a cache line, this
        // results in the same total number of cache lines being fetched while potentially
        // enabling more compact memory.
        #[expect(
            clippy::expect_used,
            reason = "we expect `init` to have at least one row, so this should never happen"
        )]
        let padded_bytes = unpadded
            .checked_next_multiple_of(Bytes::CACHELINE.div(TWO))
            .expect("unreachabel because `init` cannot exceed `isize::MAX` bytes");

        let too_many_entries = || StoreError::too_many_entries(entries, init.nrows());

        // We have a hard upper-bound of `u32::MAX` total slots.
        //
        // Thiis enforces that bound.
        let entries: u32 = entries.try_into().map_err(|_| too_many_entries())?;

        let frozen: u32 = init.nrows().try_into().map_err(|_| too_many_entries())?;

        let total: u32 = entries.checked_add(frozen).ok_or_else(too_many_entries)?;

        let max_neighbors: u32 = max_neighbors
            .try_into()
            .map_err(|_| StoreError::too_many_neighbors(max_neighbors))?;

        const FREELIST_SIZE: NonZeroU32 = NonZeroU32::new(1024).unwrap();

        let me = Self {
            buffer: Buffer::new(total.into_usize(), padded_bytes, Align::_128)?,
            unpadded,
            unfrozen: entries.into_usize(),
            tags: repeat_n(Tag::AVAILABLE, total.into_usize())
                .map(AtomicTag::new)
                .collect(),

            // NOTE: The `Freelist` is initialized to `entries` and not `total` because
            // we do not want it to release frozen IDs.
            freelist: Freelist::new(entries, FREELIST_SIZE),
            registry: Registry::new(),
            neighbors: Neighbors::new(total, max_neighbors)?,
        };

        // Populate frozen points.
        for (i, data) in init.row_iter().enumerate() {
            // We have checked that the total number of entries fits in `u32`, so this
            // arithmetic cannot overflow.
            #[expect(clippy::expect_used, reason = "this should always succeed")]
            let mut slot = me
                .slot(entries + (i as u32))
                .expect("store was just created - claiming the slot must succeed");

            slot.as_mut_slice().copy_from_slice(data);
            slot.freeze();
        }

        Ok(me)
    }

    /// Return the range of slots containing frozen items in `self`.
    pub(crate) fn frozen(&self) -> std::ops::Range<u32> {
        (self.unfrozen as u32)..(self.buffer.len() as u32)
    }

    /// Return the number of bytes occupied by each entry.
    pub(crate) fn bytes(&self) -> Bytes {
        self.unpadded
    }

    /// Return the maximum degree that can be stored in the graph.
    pub(crate) fn max_degree(&self) -> usize {
        self.neighbors.max_length()
    }

    /// Attempt to reclaim retired slots.
    ///
    /// If successful, returns the number of slots reclaimed.
    pub(crate) fn try_drain(&self) -> Option<usize> {
        #[expect(clippy::panic, reason = "we cannot proceed if we observe this")]
        fn release(tag: &AtomicTag, kind: &'static str) {
            // Relaxed ordering is sufficient as all readers/writers are synchronized on
            // the central generation.
            if let Err(got) = tag.compare_exchange(
                Tag::RETIRING,
                Tag::AVAILABLE,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                panic!(
                    "CONCURRENCY VIOLATION: {} - expected {} - got {}",
                    kind,
                    Tag::AVAILABLE,
                    got,
                );
            }
        }

        let drain = self.registry.try_advance()?;
        let items = drain.len();
        for i in drain {
            assert!(
                i.into_usize() < self.buffer.len(),
                "received an invalid ID ({}) while reclaiming slots - max allowed is {}",
                i,
                self.buffer.len(),
            );

            // We release the mirror before the main tag. The other direction would
            // prematurely advertise availability.
            //
            // SAFETY: We've verified that `i` is in-bounds.
            let (mirror, _) = unsafe { self.data_unchecked(i.into_usize()) };
            release(mirror, "mirror");
            release(&self.tags[i.into_usize()], "tag");
            self.freelist.push(i);
        }
        Some(items)
    }

    /// Return a [`Reader`] into the store.
    ///
    /// # Errors
    ///
    /// Returns [`epoch::Unavailable`] if there are too many active readers.
    pub(crate) fn reader(&self) -> Result<Reader<'_>, epoch::Unavailable> {
        Ok(Reader {
            buffer: &self.buffer,
            unpadded: self.unpadded,
            neighbors: &self.neighbors,
            _guard: self.registry.guard()?,
        })
    }

    /// Attempt to acquire a new [`Slot`] for writing.
    ///
    /// This method first consults the freelist and falls back to scanning the tags list
    /// if no ID is available from the fast path.
    pub(crate) fn acquire(&self) -> Option<Slot<'_>> {
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
                // SAFETY: We've checked that `i` is in-bounds.
                let (mirror, _) = unsafe { self.data_unchecked(i) };
                mirror.store(retiring, Ordering::Relaxed);
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
    fn scan_acquire(&self) -> Option<Slot<'_>> {
        // This is potentially quite slow - but stop if we've scanned the entire range
        // without finding anything.
        let mut remaining = self.unfrozen.div_ceil(RETRY_LIMIT);
        let mut chunks_since_freelist_check = 0;
        let mut acquired: Option<Slot<'_>> = None;

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

    fn slot(&self, i: u32) -> Option<Slot<'_>> {
        let tag = &self.tags.get(i.into_usize())?;

        // SAFETY: We've guaranteed that `tag` belongs to `slot`.
        unsafe { self.try_acquire(tag, i) }
    }

    /// Try to acquire `slot` with the associated `tag`.
    ///
    /// # Safety
    ///
    /// Caller asserts that `tag` was obtained from `self.tags[slot]`. This is meant as
    /// a perfomance optimization where `tag` is first queried for potential availability.
    unsafe fn try_acquire<'a>(&'a self, tag: &'a AtomicTag, slot: u32) -> Option<Slot<'a>> {
        match tag.compare_exchange(
            Tag::AVAILABLE,
            Tag::OWNED,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                // SAFETY: Inherited from caller - `slot` is in-bounds.
                let (mirror, data) = unsafe { self.data_unchecked(slot.into_usize()) };
                Some(Slot {
                    tag,
                    mirror,
                    data,
                    slot,
                })
            }
            Err(_) => None,
        }
    }

    /// Return the data at position `i` without bound-checking.
    ///
    /// # Safety
    ///
    /// The index `i` must be less then `self.buffer.len()`.
    unsafe fn data_unchecked(&self, i: usize) -> (&AtomicTag, RawSlice<'_>) {
        // SAFETY: inherited from caller.
        let (data, mirror) = unsafe { self.buffer.get_unchecked(i) }
            .truncate(self.unpadded)
            .split(self.unpadded.unchecked_sub(TAG_SIZE));
        (
            // SAFETY: We're careful in this module to ensure the inline tags are only
            // ever accessed atomically.
            unsafe { AtomicTag::from_ptr(mirror.as_mut_ptr().cast()) },
            data,
        )
    }

    /// Return whether or not it is probably okay to read from the slot `i`.
    ///
    /// This check is approximate and non-synchronizing. To fully check, [`Reader::can_read`]
    /// must be used.
    ///
    /// Returns `None` is index `i` is out-of-bounds.
    pub(crate) fn can_read_approximate(&self, i: usize) -> Option<bool> {
        self.tags
            .get(i)
            .map(|tag| tag.load(Ordering::Relaxed).can_read())
    }

    #[cfg(test)]
    fn writable(&self) -> std::ops::Range<u32> {
        0..self.unfrozen as u32
    }
}

/// Errors occurring during [`Store::new`].
#[derive(Debug, Error)]
#[error(transparent)]
pub(crate) struct StoreError(StoreErrorInner);

impl StoreError {
    fn mismatched_frozen_point_dim(dim: usize, bytes: Bytes) -> Self {
        Self(StoreErrorInner::MismatchedFrozenPointDim { dim, bytes })
    }

    fn need_frozen_point() -> Self {
        Self(StoreErrorInner::NeedFrozenPoint)
    }

    fn too_many_entries(entries: usize, frozen: usize) -> Self {
        Self(StoreErrorInner::TooManyEntries { entries, frozen })
    }

    fn too_many_neighbors(neighbors: usize) -> Self {
        Self(StoreErrorInner::TooManyNeighbors { neighbors })
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

#[derive(Debug, Error)]
enum StoreErrorInner {
    #[error(
        "frozen point dim ({}) must have the same dimensionality as requested bytes ({})",
        dim,
        bytes
    )]
    MismatchedFrozenPointDim { dim: usize, bytes: Bytes },
    #[error("at least one frozen point must be provided")]
    NeedFrozenPoint,
    #[error(
        "total points ({} + {} frozen) must not exceed `u32::MAX`",
        entries,
        frozen
    )]
    TooManyEntries { entries: usize, frozen: usize },
    #[error("number of neighbors ({}) may not exceed `u32::MAX`", neighbors)]
    TooManyNeighbors { neighbors: usize },
    #[error(transparent)]
    BufferError(#[from] BufferError),
    #[error(transparent)]
    NeighborsError(#[from] NeighborsError),
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

/// An epoch protected reader into a [`Store`].
///
/// Created via [`Store::reader`].
#[derive(Debug)]
pub(crate) struct Reader<'a> {
    buffer: &'a Buffer,
    unpadded: Bytes,
    neighbors: &'a Neighbors,
    // It's important that we hold onto this, even if we don't use it.
    _guard: epoch::Guard<'a>,
}

impl<'a> Reader<'a> {
    /// Attempt to read the value at index `i`. This can fail for any of the
    /// following reasons:
    ///
    /// 1. Index `i` is out-of-bounds.
    /// 2. The read cannot be guaranteed to be race-free.
    #[inline]
    pub(crate) fn read(&self, i: usize) -> Option<&[u8]> {
        if self.is_in_bounds(i) {
            // SAFETY: `i` is in-bounds.
            unsafe { self.read_in_bounds(i) }
        } else {
            None
        }
    }

    /// Return `true` if the index `i` is in-bounds.
    #[inline]
    #[must_use = "this function has no side-effects"]
    pub(crate) fn is_in_bounds(&self, i: usize) -> bool {
        i < self.buffer.len()
    }

    /// Return `true` if it is safe to read the data at position `i`.
    ///
    /// This guarantee only holds while `self` is alive. Construction of a new [`Reader`]
    /// requires a separate check.
    #[cfg(test)]
    pub(crate) fn can_read(&self, i: usize) -> Option<bool> {
        if !self.is_in_bounds(i) {
            return None;
        }

        // SAFETY: We've checked that `i` is in-bounds.
        //
        // Further, we guarantee that `self.unpadded >= TAG_SIZE`, so the pointer arithmetic
        // is in-bounds.
        let tag_ptr = unsafe {
            self.buffer
                .get_unchecked(i)
                .as_mut_ptr()
                .add(self.unpadded.unchecked_sub(TAG_SIZE).value())
        };

        // SAFETY: We only access tag pointers atomically.
        let can_read = unsafe { AtomicTag::from_ptr(tag_ptr.cast()) }
            .load(Ordering::Acquire)
            .can_read();

        Some(can_read)
    }

    /// Read the data as position `i` if it is guaranteed to be race-free without bounds
    /// checking.
    ///
    /// # Safety
    ///
    /// The index `i` must satisfy [`Self::is_in_bounds`].
    #[inline]
    pub(crate) unsafe fn read_in_bounds(&self, i: usize) -> Option<&[u8]> {
        debug_assert!(self.is_in_bounds(i));

        // SAFETY:
        //
        // * The caller asserts `i` is in-bounds.
        // * We maintain an internal invariant that `self.buffer.stride() <= self.unpadded`.
        // * Further, we maintain that `self.unpadded >= TAG_SIZE`.
        let (data, tag_ptr) = unsafe {
            self.buffer
                .get_unchecked(i)
                .truncate_unchecked(self.unpadded)
                .split_unchecked(self.unpadded.unchecked_sub(TAG_SIZE))
        };

        // NOTE: Must be `Acquire` to correctly synchronize with writes.
        //
        // SAFETY: We are careful in this module to ensure that inline tags are only accessed
        // atomically.
        let can_read = unsafe { AtomicTag::from_ptr(tag_ptr.as_mut_ptr().cast()) }
            .load(Ordering::Acquire)
            .can_read();

        if can_read {
            // SAFETY: We've passed the `can_read` check - `_guard` will ensure the read
            // slice is valid and race-free.
            Some(unsafe { data.as_slice() })
        } else {
            None
        }
    }

    /// Return the raw data slice for index `i` without any race guarantees.
    ///
    /// # Safety
    ///
    /// The index `i` must be satisfy [`Self::is_in_bounds`].
    #[inline]
    pub(crate) unsafe fn read_raw_unchecked(&self, i: usize) -> RawSlice<'_> {
        // SAFETY: Inherited from caller: `i` is inbounds.
        unsafe { self.buffer.get_unchecked(i) }.truncate(self.unpadded)
    }

    /// Return the number of bytes for each entry.
    pub(crate) fn bytes(&self) -> Bytes {
        self.unpadded
    }

    /// Return [`Neighbors`].
    pub(crate) fn neighbors(&self) -> &Neighbors {
        self.neighbors
    }
}

/// A writable buffer into the data managed by a [`Store`], obtained from [`Store::acquire`].
#[derive(Debug)]
pub(crate) struct Slot<'a> {
    tag: &'a AtomicTag,
    mirror: &'a AtomicTag,
    data: RawSlice<'a>,
    slot: u32,
}

impl<'a> Slot<'a> {
    /// View the managed data as a mutable slice.
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: The slot guarantees exclusive access to its corresponding data.
        unsafe { self.data.as_mut_slice() }
    }

    /// Return the slot associated with this write.
    pub(crate) fn slot(&self) -> u32 {
        self.slot
    }

    fn freeze(self) {
        let me = std::mem::ManuallyDrop::new(self);
        me.mirror.store(Tag::FROZEN, Ordering::Release);
        me.tag.store(Tag::FROZEN, Ordering::Release);
    }

    /// Consume the slot and publish the written data for all readers.
    ///
    /// Return the internal slot ID.
    pub(crate) fn publish(self) -> u32 {
        let id = self.slot();
        let me = std::mem::ManuallyDrop::new(self);
        me.mirror.store(Tag::PUBLISHED, Ordering::Release);
        me.tag.store(Tag::PUBLISHED, Ordering::Release);
        id
    }
}

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        self.mirror.store(Tag::AVAILABLE, Ordering::Release);
        self.tag.store(Tag::AVAILABLE, Ordering::Release);
    }
}

///////////
// Tests //
///////////

/// These tests are basic functionality tests for the store.
///
/// Longer running conurrency tests are in the integration test suite.
#[cfg(test)]
mod tests {
    use super::*;

    use diskann_utils::views::Matrix;

    // Build a store with `entries` writable slots of `entry_bytes` each, backed by `frozen`
    // zeroed frozen points. The frozen points occupy the highest slot indices.
    fn store(entries: usize, entry_bytes: usize, frozen: usize) -> Result<Store, StoreError> {
        let mut data = Matrix::new(0u8, frozen, entry_bytes);
        let mut base = 0u8;
        for row in data.row_iter_mut() {
            row.fill(base);
            base = base.wrapping_add(1);
        }

        Store::new(entries, Bytes::new(entry_bytes), 0, data.as_view())
    }

    //------------------------//
    // Constructor validation //
    //------------------------//

    #[test]
    fn new_rejects_mismatched_frozen_dim() {
        // Frozen point has 8 columns but the store is asked for 16-byte entries.
        let data = Matrix::new(0u8, 1, 8);
        let err = Store::new(4, Bytes::new(16), 0, data.as_view()).unwrap_err();
        assert!(matches!(
            err.0,
            StoreErrorInner::MismatchedFrozenPointDim { dim: 8, .. }
        ));
    }

    #[test]
    fn new_requires_a_frozen_point() {
        let err = store(4, 8, 0).unwrap_err();
        assert!(matches!(err.0, StoreErrorInner::NeedFrozenPoint));
    }

    #[test]
    fn new_rejects_total_slot_overflow() {
        // `entries` alone fits in u32, but `entries + frozen` overflows it.
        let data = Matrix::new(0u8, 1, 8);
        let err = Store::new(u32::MAX as usize, Bytes::new(8), 0, data.as_view()).unwrap_err();
        assert!(matches!(err.0, StoreErrorInner::TooManyEntries { .. }));
    }

    #[test]
    fn new_rejects_too_many_neighbors() {
        let data = Matrix::new(0u8, 1, 8);
        let err =
            Store::new(4, Bytes::new(8), u32::MAX.into_usize() + 1, data.as_view()).unwrap_err();
        assert!(matches!(err.0, StoreErrorInner::TooManyNeighbors { .. }));
    }

    //--------//
    // Layout //
    //--------//

    #[test]
    fn frozen_range_follows_writable_slots() {
        let s = store(4, 8, 2).unwrap();

        // Writable slots are [0, 4); frozen points occupy [4, 6).
        assert_eq!(s.frozen(), 4..6);

        let reader = s.reader().unwrap();
        for i in 0..4 {
            assert!(!s.can_read_approximate(i).unwrap());
            assert!(!reader.can_read(i).unwrap());
            assert!(reader.read(i).is_none());
        }

        assert!(s.can_read_approximate(4).unwrap());
        assert!(reader.can_read(4).unwrap());
        assert_eq!(reader.read(4).unwrap(), &[0, 0, 0, 0, 0, 0, 0, 0]);

        assert!(s.can_read_approximate(5).unwrap());
        assert!(reader.can_read(5).unwrap());
        assert_eq!(reader.read(5).unwrap(), &[1, 1, 1, 1, 1, 1, 1, 1]);

        assert!(s.can_read_approximate(6).is_none());
        assert!(reader.can_read(6).is_none());
        assert!(reader.read(6).is_none());
    }

    ///////////////
    // Lifecycle //
    ///////////////

    #[test]
    fn acquire_write_publish_read_roundtrip() {
        let s = store(4, 8, 1).unwrap();

        let reader = s.reader().expect("reader guard available");

        let idx = {
            let mut slot = s.acquire().expect("a fresh store has free slots");
            let idx = slot.slot() as usize;
            slot.as_mut_slice()
                .copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);

            // Before the slot is dropped - we should not be able to read it.
            assert!(reader.read(idx).is_none());
            assert!(!s.can_read_approximate(idx).unwrap());
            slot.publish();
            idx
        };

        assert_eq!(reader.read(idx), Some([1, 2, 3, 4, 5, 6, 7, 8].as_slice()));
        assert!(s.can_read_approximate(idx).unwrap());
    }

    #[test]
    fn unpublished_slots_are_immediately_available() {
        let s = store(4, 8, 1).unwrap();

        let reader = s.reader().expect("reader guard available");

        let idx = {
            let mut slot = s.acquire().expect("a fresh store has free slots");
            let idx = slot.slot() as usize;
            slot.as_mut_slice()
                .copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);

            // Before the slot is dropped - we should not be able to read it.
            assert!(reader.read(idx).is_none());
            assert!(!s.can_read_approximate(idx).unwrap());

            // NOTE: We do not explicitly publish the slot.
            idx
        };

        assert!(reader.read(idx).is_none());
        assert!(!s.can_read_approximate(idx).unwrap());
    }

    #[test]
    fn acquire_exhausts_then_reports_none() {
        let s = store(2, 8, 1).unwrap();
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
        let s = store(4, 8, 1).unwrap();
        assert!(matches!(s.retire(999), Err(RetireError::OutOfBounds)));
    }

    #[test]
    fn retire_rejects_reserved_slots() {
        let s = store(4, 8, 1).unwrap();
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
        let s = store(4, 8, 1).unwrap();

        let idx = {
            let slot = s.acquire().unwrap();
            slot.publish() as usize
        };

        assert!(s.retire(idx).is_ok());

        // A reader opened after retirement must not observe the retired slot.
        let reader = s.reader().unwrap();
        assert_eq!(reader.read(idx), None);
        assert_eq!(reader.can_read(idx), Some(false));

        // The slot can also not be retired again.
        assert!(matches!(
            s.retire(idx),
            Err(RetireError::SlotIsReserved { .. })
        ));
    }

    //---------//
    // Recycle //
    //---------//

    #[test]
    fn test_recycling() {
        let entries = if cfg!(miri) { 16 } else { 2048 };

        let s = store(entries, 4, 2).unwrap();

        // Claim all slots.
        let mut count = 0;
        while let Some(slot) = s.acquire() {
            slot.publish();
            count += 1;
        }

        assert_eq!(count, s.writable().len());

        // Now that all slots are claimed - retire all slots.
        for i in s.writable() {
            s.retire(i.into_usize()).unwrap();
        }

        // Verify that we can claim all slots again.
        let mut count = 0;
        while let Some(slot) = s.acquire() {
            slot.publish();
            count += 1;
        }

        assert_eq!(count, s.writable().len());
    }
}

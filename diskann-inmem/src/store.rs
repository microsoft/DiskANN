/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    iter::repeat_n,
    num::{NonZeroU32, NonZeroUsize},
    sync::atomic::Ordering,
};

use diskann::utils::IntoUsize;
use diskann_utils::views::MatrixView;
use thiserror::Error;

use crate::{
    buffer::{Buffer, RawSlice},
    epoch::{self, Registry},
    freelist::{self, Freelist},
    neighbors::Neighbors,
    num::{Align, Bytes},
    tag::{AtomicTag, Tag},
};

#[derive(Debug)]
pub(crate) struct Store {
    // The invasive store where concurrency tags are stored inline with the data.
    //
    // These tags are mirrored from `tags` - with the latter being used for secondary scans
    // offering slightly better locality.
    buffer: Buffer,
    unpadded: Bytes,

    // The number of unfrozen points. This is guaranteed to be less than `buffer`.
    unfrozen: usize,
    tags: Vec<AtomicTag>,
    freelist: Freelist,
    registry: Registry,
    neighbors: Neighbors,
}

const SPLIT: Bytes = Bytes::size_of::<AtomicTag>();
const TWO: NonZeroUsize = NonZeroUsize::new(2).unwrap();

// TODO: This is a guess and probably needs tuning.
const RETRY_LIMIT: usize = 20;

impl Store {
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

        let unpadded = bytes
            .checked_add(SPLIT)
            .expect("unreachable because `init` cannot exceed `isize::MAX` bytes");

        // Pad to half a cache line. When data occupies just part of a cache line, this
        // results in the same total number of cache lines being fetched while potentially
        // enabling more compact memory.
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

        let me = Self {
            buffer: Buffer::new(total.into_usize(), padded_bytes, Align::_128).unwrap(),
            unpadded,
            unfrozen: entries.into_usize(),
            tags: repeat_n(Tag::AVAILABLE, total.into_usize())
                .map(|v| AtomicTag::new(v))
                .collect(),

            // NOTE: The `Freelist` is initialized to `entries` and not `total` because
            // we do not want it to release frozen IDs.
            freelist: Freelist::new(entries, NonZeroU32::new(1024).unwrap()),
            registry: Registry::new(),
            neighbors: Neighbors::new(total, max_neighbors).unwrap(),
        };

        // Populate frozen points.
        for (i, data) in init.row_iter().enumerate() {
            // We have checked that the total number of entries fits in `u32`, so this
            // arithmetic cannot overflow.
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

    /// Return the number of unfrozen slots managed by `self`.
    pub(crate) fn capacity(&self) -> usize {
        self.buffer.len() - self.unfrozen
    }

    /// Return the number of bytes occupied by each entry.
    pub(crate) fn bytes(&self) -> Bytes {
        self.unpadded
    }

    /// Attempt to reclaim retired slots.
    ///
    /// If successful, returns the number of slots reclaimed.
    pub(crate) fn try_drain(&self) -> Option<usize> {
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
            // We release the mirror before the main tag. The other direction would
            // prematurely advertise availability.
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

    pub(crate) fn delete(&self, i: usize) -> bool {
        let guard = self.registry.guard().unwrap();
        let tag = self.tags.get(i).unwrap();
        let current = tag.load(Ordering::Relaxed);

        // We can only perform a deletion if the generation is not in a reserved state.
        if current.is_reserved() {
            return false;
        }

        let retiring = Tag::RETIRING;

        // Even if we make this change, we can't access any data until we wait for the
        // epoch to be bumped. As such, relaxed semantics are fine.
        match tag.compare_exchange(current, retiring, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => {
                // Set the metadata in the mirror as well.
                let (mirror, _) = unsafe { self.data_unchecked(i) };
                mirror.store(retiring, Ordering::Relaxed);
                guard.retire(i as u32);
                true
            }
            Err(_) => false,
        }
    }

    fn scan_acquire(&self) -> Option<Slot<'_>> {
        // This is potentially quite slow - but stop if we've scanned the entire range
        // without finding anything.
        let mut remaining = self.unfrozen;
        let mut chunks_since_freelist_check = 0;
        let mut acquired: Option<Slot<'_>> = None;

        while remaining != 0 {
            let chunk = self.freelist.scan();
            remaining = remaining.saturating_sub(chunk.len());

            for slot in chunk {
                let tag = self.tags.get(slot.into_usize()).unwrap();

                // If this slot is available and we haven't claimed a slot yet, try to
                // claim it. Otherwise, continue with the scan to partially repopulate the
                // freelist for other threads.
                if tag.load(Ordering::Relaxed) == Tag::AVAILABLE {
                    if acquired.is_none() {
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
        let tag = &self.tags.get(i.into_usize()).unwrap();
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
        let (data, mirror) = unsafe { self.buffer.get_unchecked(i) }
            .truncate(self.unpadded)
            .split(self.unpadded.unchecked_sub(SPLIT));
        (
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
}

#[derive(Debug, Error)]
#[error(transparent)]
pub struct StoreError(StoreErrorInner);

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

impl From<StoreErrorInner> for StoreError {
    fn from(inner: StoreErrorInner) -> Self {
        Self(inner)
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
}

/// An epoch protect reader into [`Store`].
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
    pub(crate) fn can_read(&self, i: usize) -> Option<bool> {
        if !self.is_in_bounds(i) {
            return None;
        }

        let tag_ptr = unsafe {
            self.buffer
                .get_unchecked(i)
                .as_mut_ptr()
                .add(self.unpadded.unchecked_sub(SPLIT).value())
        };

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

        let (data, tag_ptr) = unsafe {
            self.buffer
                .get_unchecked(i)
                .truncate_unchecked(self.unpadded)
                .split_unchecked(self.unpadded.unchecked_sub(SPLIT))
        };

        // NOTE: Must be `Acquire` to correctly synchronize with writes.
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
        unsafe { self.buffer.get_unchecked(i) }.truncate(self.unpadded)
    }

    /// Return the number of bytes for each entry.
    pub(crate) fn bytes(&self) -> Bytes {
        self.unpadded
    }

    /// Return [`Neighbors`].
    pub(crate) fn neighbors(&self) -> &Neighbors {
        &self.neighbors
    }
}

/// A writable buffer into the data managed by a [`Store`], obtained from [`Store::Acquire`].
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
}

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        self.mirror.store(Tag::PUBLISHED, Ordering::Release);
        self.tag.store(Tag::PUBLISHED, Ordering::Release);
    }
}

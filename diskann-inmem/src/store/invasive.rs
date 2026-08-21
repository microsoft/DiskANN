/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A store [`plugin::Plugin`] that maintains an invasive slot state where the data in each
//! slot is a contiguous slice of memory.
//!
//! Slot state is stored as an [`AtomicTag`] immediately after the slot data.
//!
//! ## Lifecycle Details
//!
//! The plugin lifecycle details are relatively straightforward. The invasive [`AtomicTag`]
//! mostly follows the transitions made by the [`Store`]. A [`Reader`] checks the tag for
//! readability before creating a shared reference to the data payload.
//!
//! The problematic transition from "published" to "retiring" is made safe because
//!
//! 1. Each [`Reader`] stores an [`epoch::Guard`].
//! 2. Slices yielded by [`Reader`] are scoped to the **borrow** of [`Reader`].
//!
//! This ensures that slices cannot outlive the [`epoch::Guard`] protecting their slot. EBR
//! prevents the slot from transitioning from "retiring" back to "available" and being
//! reused while the guard remains active.
//!
//! The transitions [`Slot::publish`] and [`Slot::freeze`] use release stores. Since these
//! are terminal slot operations, their release stores occur after all payload writes. The
//! acquire load in [`Reader::read`] makes those writes visible before creating a shared
//! slice.
//!
//! ## Safety
//!
//! The safety of this module depends on [`Invasive`] being embedded in a [`Store`] that
//! observes the plugin lifecycle. Every lifecycle operation requires a [`Lifecycle`] token,
//! which is constructible only by the parent store module. The unsafe [`plugin::Plugin`]
//! methods additionally rely on [`Store`] to satisfy their documented state and exclusivity
//! preconditions.

use std::sync::atomic::Ordering;

use diskann::utils::IntoUsize;
use thiserror::Error;

use crate::{
    buffer::{Buffer, BufferError, RawSlice},
    epoch,
    num::{Align, Bytes, IdLimit},
    store::{Lifecycle, Store, plugin},
    tag::{AtomicTag, Tag},
};

/// A [`plugin::PluginConfig`] for [`Invasive`].
#[derive(Debug, Clone)]
pub(crate) struct Config {
    /// The number of bytes held in each slot.
    bytes: Bytes,
}

impl Config {
    /// Create a new [`Config`] for [`Invasive`] reserving `bytes` bytes for each slot.
    pub(crate) fn new(bytes: Bytes) -> Self {
        Self { bytes }
    }

    /// Build an [`Invasive`] store holding `id_limit` slots.
    pub(crate) fn build(self, id_limit: IdLimit) -> Result<Invasive, InvasiveError> {
        let Self { bytes } = self;
        Invasive::new(id_limit, bytes)
    }
}

impl plugin::PluginConfig for Config {
    type Plugin = Invasive;
    type Error = InvasiveError;
    fn build(self, id_limit: IdLimit) -> Result<Invasive, InvasiveError> {
        <Config>::build(self, id_limit)
    }
}

/// The invasive store where concurrency tags are stored inline just after the data.
#[derive(Debug)]
pub(crate) struct Invasive {
    // The inline tags are `AtomicTag`s stored after the data.
    buffer: Buffer,

    // The unpadded size of each row in `buffer`. This includes both the data **and** the
    // 1-byte tag. Tags are located at byte `unpadded - 1`.
    unpadded: Bytes,
}

impl Invasive {
    /// Construct the [`Config`] for [`Self`].
    ///
    /// See also: [`Config::new`].
    pub(crate) fn config(bytes: Bytes) -> Config {
        Config::new(bytes)
    }

    /// Create a new [`Invasive`] with capacity for `id_limit` slots of `bytes`.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal buffer allocation exceeds `isize::MAX` or
    /// computation of the padded, invasive bytes exceeds `usize::MAX`.
    pub(crate) fn new(id_limit: IdLimit, bytes: Bytes) -> Result<Self, InvasiveError> {
        let Some(unpadded) = bytes.checked_add(AtomicTag::SIZE) else {
            return Err(InvasiveError::bytes_overflowed());
        };
        let Some(padded_bytes) = unpadded.checked_next_multiple_of(Bytes::CACHELINE) else {
            return Err(InvasiveError::bytes_overflowed());
        };

        let buffer = match Buffer::new(id_limit.as_usize(), padded_bytes, Align::_128) {
            Ok(buffer) => buffer,
            Err(err) => return Err(InvasiveError::buffer_error(err)),
        };

        Ok(Self { buffer, unpadded })
    }

    /// Return the [`IdLimit`] for this store.
    pub(crate) fn id_limit(&self) -> IdLimit {
        // The numeric cast is safe because `Invasive::new` takes an `IdLimit` in its
        // constructor, and thus `self.buffer.len()` cannot exceed `u32::MAX`.
        IdLimit::new(self.buffer.len() as u32)
    }

    /// Return a [`Reader`] over [`Self`] inside `store`.
    pub(crate) fn reader(store: &Store<Self>) -> Result<Reader<'_>, epoch::Unavailable> {
        store.guard(|this, guard: epoch::Guard<'_>| Reader {
            buffer: &this.buffer,
            unpadded: this.unpadded,
            _guard: guard,
        })
    }

    /// Return the data at position `i` without bound-checking.
    ///
    /// # Safety
    ///
    /// The index `i` must be less than `self.buffer.len()`.
    unsafe fn data_unchecked(&self, i: usize) -> (&AtomicTag, RawSlice<'_>) {
        // SAFETY: inherited from caller.
        let (data, mirror) = unsafe { self.buffer.get_unchecked(i) }
            .truncate(self.unpadded)
            .split(self.unpadded.unchecked_sub(AtomicTag::SIZE));
        (
            // SAFETY: The tag byte lies within the zero-initialized row, is sufficiently
            // aligned for `AtomicTag`, and is only accessed through atomic operations.
            unsafe { AtomicTag::from_ptr(mirror.as_mut_ptr().cast()) },
            data,
        )
    }

    fn data(&self, i: usize) -> Option<(&AtomicTag, RawSlice<'_>)> {
        if i >= self.buffer.len() {
            None
        } else {
            // SAFETY: We've checked that `i` is in-bounds.
            Some(unsafe { self.data_unchecked(i) })
        }
    }
}

#[derive(Debug, Error)]
#[error(transparent)]
pub(crate) struct InvasiveError(InvasiveErrorInner);

impl InvasiveError {
    fn bytes_overflowed() -> Self {
        Self(InvasiveErrorInner::BytesOverflowed)
    }

    fn buffer_error(err: BufferError) -> Self {
        Self(InvasiveErrorInner::BufferError(err))
    }
}

#[derive(Debug, Error)]
enum InvasiveErrorInner {
    #[error("computation of the bytes per slot overflowed")]
    BytesOverflowed,
    #[error(transparent)]
    BufferError(BufferError),
}

impl plugin::Plugin for Invasive {
    type Slot<'a> = Slot<'a>;

    fn id_limit(&self) -> IdLimit {
        <Invasive>::id_limit(self)
    }

    #[expect(clippy::panic, reason = "out-of-bounds is a hard program bug")]
    unsafe fn acquire(&self, i: u32, _: Lifecycle) -> Self::Slot<'_> {
        let Some((tag, data)) = self.data(i.into_usize()) else {
            panic!("index {i} is out-of-bounds");
        };

        // This is a pessimistic check to ensure that the caller is correctly using the
        // `plugin` API.
        debug_assert_eq!(
            tag.load(Ordering::Relaxed),
            Tag::AVAILABLE,
            "concurrency violation",
        );

        // While we can leave this tag as `Tag::AVAILABLE` since it's just a mirror, setting
        // it to `Tag::OWNED` lets us more precisely detect misuse from the caller.
        tag.store(Tag::OWNED, Ordering::Relaxed);
        Slot { tag, data }
    }

    #[expect(clippy::panic, reason = "out-of-bounds is a hard program bug")]
    unsafe fn reclaim(&self, i: u32, _: Lifecycle) {
        let Some((tag, _)) = self.data(i.into_usize()) else {
            panic!("index {i} is out-of-bounds");
        };

        tag.store(Tag::AVAILABLE, Ordering::Release);
    }

    #[expect(clippy::panic, reason = "out-of-bounds is a hard program bug")]
    unsafe fn retire(&self, i: u32, _: Lifecycle) {
        let Some((tag, _)) = self.data(i.into_usize()) else {
            panic!("index {i} is out-of-bounds");
        };

        tag.store(Tag::RETIRING, Ordering::Relaxed);
    }
}

/// A reader into an [`Invasive`] store.
#[derive(Debug)]
pub(crate) struct Reader<'a> {
    buffer: &'a Buffer,
    unpadded: Bytes,
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

    /// Return the [`IdLimit`] for this collection.
    #[inline]
    #[must_use = "this function has no side-effects"]
    pub(crate) fn id_limit(&self) -> IdLimit {
        // Like `Invasive::id_limit`, the numeric cast is safe because by construction,
        // the underlying buffer is limited to `u32::MAX`.
        IdLimit::new(self.buffer.len() as u32)
    }

    /// Return `true` if it is safe to read the data at position `i`.
    ///
    /// This guarantee only holds while `self` is alive. Construction of a new [`Reader`]
    /// requires a separate check.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "this is non-trivial method that is likely to be used in the future"
        )
    )]
    pub(crate) fn can_read(&self, i: usize) -> Option<bool> {
        if !self.is_in_bounds(i) {
            return None;
        }

        // SAFETY: We've checked that `i` is in-bounds.
        //
        // Further, we guarantee that `self.unpadded >= AtomicTag::SIZE`, so the pointer
        // arithmetic is in-bounds.
        let tag_ptr = unsafe {
            self.buffer
                .get_unchecked(i)
                .as_mut_ptr()
                .add(self.unpadded.unchecked_sub(AtomicTag::SIZE).value())
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
        // * We maintain the internal invariant that `self.unpadded <= self.buffer.stride()`.
        // * Further, we maintain that `self.unpadded >= AtomicTag::SIZE`.
        let (data, tag_ptr) = unsafe {
            self.buffer
                .get_unchecked(i)
                .truncate_unchecked(self.unpadded)
                .split_unchecked(self.unpadded.unchecked_sub(AtomicTag::SIZE))
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
    /// This includes both the data **and** the invasive tag.
    ///
    /// # Safety
    ///
    /// The index `i` must satisfy [`Self::is_in_bounds`].
    ///
    /// The returned [`RawSlice`] may only be used for prefetching. Callers must never
    /// materialize it as a proper slice or reference.
    #[inline]
    pub(crate) unsafe fn read_raw_unchecked(&self, i: usize) -> RawSlice<'_> {
        // SAFETY: Inherited from caller: `i` is in bounds.
        unsafe { self.buffer.get_unchecked(i) }.truncate(self.unpadded)
    }

    /// Return the number of bytes for each entry.
    pub(crate) fn bytes(&self) -> Bytes {
        self.unpadded.unchecked_sub(AtomicTag::SIZE)
    }
}

/// A [`plugin::Slot`] for [`Invasive`].
#[derive(Debug)]
pub(crate) struct Slot<'a> {
    // NOTE: `tag` and `data` must belong to the same slot.
    tag: &'a AtomicTag,
    data: RawSlice<'a>,
}

impl<'a> Slot<'a> {
    /// Return the data within this slot as a mutable slice.
    ///
    /// The length of this slice is guaranteed to be the number of bytes passed to
    /// [`Invasive::new`] or [`Config::new`].
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: Users of the `plugin::Slot` are obligated to ensure exclusivity.
        //
        // Since `Reader` obeys the plugin life-cycle requirements, a concurrent reader
        // of this data should not be possible.
        unsafe { self.data.as_mut_slice() }
    }
}

impl plugin::Slot for Slot<'_> {
    fn publish(self, _: Lifecycle) {
        self.tag.store(Tag::PUBLISHED, Ordering::Release);
    }
    fn freeze(self, _: Lifecycle) {
        self.tag.store(Tag::FROZEN, Ordering::Release);
    }
    fn abort(self, _: Lifecycle) {
        self.tag.store(Tag::AVAILABLE, Ordering::Release);
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::{
        assert_matches,
        num::{NonZeroU32, NonZeroUsize},
    };

    use crate::{
        num::{Capacity, MaxDegree},
        store,
    };

    // Build a store with `entries` writable slots of `entry_bytes` each, backed by `frozen`
    // zeroed frozen points. The frozen points occupy the highest slot indices.
    fn store(
        entries: usize,
        entry_bytes: usize,
        frozen: usize,
    ) -> Result<Store<Invasive>, store::StoreError> {
        let store = Store::new(
            store::Layout::new(
                Capacity::new(entries),
                MaxDegree::new(0),
                frozen.try_into().unwrap(),
            ),
            store::Config::__exhaustive(
                NonZeroUsize::new(10).unwrap(),
                NonZeroU32::new(16).unwrap(),
            ),
            Config::new(Bytes::new(entry_bytes)),
        )?;

        for (base, id) in store.frozen().enumerate() {
            let mut slot = store.slot(id).unwrap();
            slot.data().as_mut_slice().fill(base as u8);
            slot.freeze();
        }

        Ok(store)
    }

    //--------//
    // Layout //
    //--------//

    #[test]
    fn frozen_range_follows_writable_slots() {
        let s = store(4, 8, 2).unwrap();

        // Writable slots are [0, 4); frozen points occupy [4, 6).
        assert_eq!(s.frozen(), 4..6);

        let reader = Invasive::reader(&s).unwrap();
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

        let reader = Invasive::reader(&s).expect("reader guard available");

        let idx = {
            let mut slot = s.acquire().expect("a fresh store has free slots");
            let idx = slot.slot() as usize;
            slot.data()
                .as_mut_slice()
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

        let reader = Invasive::reader(&s).expect("reader guard available");

        let idx = {
            let mut slot = s.acquire().expect("a fresh store has free slots");
            let idx = slot.slot() as usize;
            slot.data()
                .as_mut_slice()
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
        assert_matches!(s.retire(999), Err(store::RetireError::OutOfBounds));
    }

    #[test]
    fn retire_rejects_reserved_slots() {
        let s = store(4, 8, 1).unwrap();
        // An untouched writable slot is AVAILABLE, which is a reserved state.
        assert_matches!(s.retire(0), Err(store::RetireError::SlotIsReserved { .. }));
        // A frozen slot is likewise reserved.
        let frozen = s.frozen().start as usize;
        assert_matches!(
            s.retire(frozen),
            Err(store::RetireError::SlotIsReserved { .. })
        );
        // An owned slot is not retirable.
        let slot = s.acquire().unwrap();
        assert_matches!(
            s.retire(slot.slot() as usize),
            Err(store::RetireError::SlotIsReserved { .. })
        );
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
        let reader = Invasive::reader(&s).unwrap();
        assert_eq!(reader.read(idx), None);
        assert_eq!(reader.can_read(idx), Some(false));

        // The slot can also not be retired again.
        assert_matches!(
            s.retire(idx),
            Err(store::RetireError::SlotIsReserved { .. })
        );
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

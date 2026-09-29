/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A [`slots::Slots`] that does not maintain any additional metadata, instead using a
//! [`tag::ReadOnly`] of its authoritative tags for concurrency.

use diskann::utils::IntoUsize;
use thiserror::Error;

use crate::{
    buffer::{Buffer, BufferError, RawSlice},
    epoch,
    num::{Align, Bytes, IdLimit},
    store::{Lifecycle, slots},
    tag,
};

/// A [`slots::SlotsConfig`] for [`Simple`].
#[derive(Debug, Clone)]
pub(crate) struct Config {
    /// The number of bytes held in each slot.
    bytes: Bytes,
}

impl Config {
    /// Create a new [`Config`] for [`Simple`] reserving `bytes` bytes for each slot.
    pub(crate) fn new(bytes: Bytes) -> Self {
        Self { bytes }
    }
}

impl slots::SlotsConfig for Config {
    type Slots = Simple;
    type Error = SimpleError;

    unsafe fn build(
        self,
        handle: epoch::RegistryHandle,
        tags: &tag::Authoritative,
    ) -> Result<Simple, SimpleError> {
        let Self { bytes } = self;

        // SAFETY: Inherited from caller.
        unsafe { Simple::new(bytes, handle, tags.read_only()) }
    }
}

/// A store where concurrency tags use [`tag::ReadOnly`] to determine data readability.
#[derive(Debug)]
pub(crate) struct Simple {
    /// The raw data, aligned to cache-line boundaries.
    buffer: Buffer,

    /// The unpadded length of the data in each entry in `buffer`.
    bytes: Bytes,

    /// The parent registry.
    ///
    /// This **must** be a handle to the [`epoch::Registry`] in the [`crate::store::Store`]
    /// driving `self`.
    handle: epoch::RegistryHandle,

    /// The parent tags.
    ///
    /// This **must** be a handle to the [`tag::Authoritative`] in the [`crate::store::Store`]
    /// driving `self`.
    tags: tag::ReadOnly,
}

impl Simple {
    /// Create a new [`Simple`] holding `bytes` bytes per entry.
    ///
    /// The number of entries is derived from the length of `tags`.
    ///
    /// # Safety
    ///
    /// This struct assumes it is being driven by a [`crate::store::Store`] and that:
    ///
    /// * `handle` is a handle to the [`epoch::Registry`] embedded in the parent store.
    /// * `tags` is a readonly handle to the [`tag::Authoritative`] in the parent store.
    unsafe fn new(
        bytes: Bytes,
        handle: epoch::RegistryHandle,
        tags: tag::ReadOnly,
    ) -> Result<Self, SimpleError> {
        let Some(padded) = bytes.checked_next_multiple_of(Bytes::CACHELINE) else {
            return Err(SimpleError::bytes_overflowed());
        };

        let buffer = match Buffer::new(tags.id_limit().as_usize(), padded, Align::_128) {
            Ok(buffer) => buffer,
            Err(err) => return Err(SimpleError::buffer_error(err)),
        };

        Ok(Self {
            buffer,
            bytes,
            handle,
            tags,
        })
    }

    /// Create the [`Config`] for `Self`.
    pub(crate) fn config(bytes: Bytes) -> Config {
        Config::new(bytes)
    }

    /// Return the [`IdLimit`] for this store.
    pub(crate) fn id_limit(&self) -> IdLimit {
        IdLimit::new(self.buffer.len() as u32)
    }

    /// Return a [`Reader`] over [`Self`].
    ///
    /// # Panics
    ///
    /// Panics if `guard` does not belong to `self`'s [`epoch::Registry`].
    pub(crate) fn reader<'a>(&'a self, guard: epoch::Guard<'a>) -> Reader<'a> {
        self.handle.assert_guard_belongs(&guard);
        Reader {
            buffer: &self.buffer,
            bytes: self.bytes,
            tags: &self.tags,
            _guard: guard,
        }
    }

    /// Return the data at position `i` without bound-checking.
    ///
    /// # Safety
    ///
    /// The index `i` must be less than `self.buffer.len()`.
    unsafe fn data_unchecked(&self, i: usize) -> RawSlice<'_> {
        // SAFETY: inherited from caller.
        unsafe { self.buffer.get_unchecked(i) }.truncate(self.bytes)
    }

    fn data(&self, i: usize) -> Option<RawSlice<'_>> {
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
pub(crate) struct SimpleError(SimpleErrorInner);

impl SimpleError {
    fn bytes_overflowed() -> Self {
        Self(SimpleErrorInner::BytesOverflowed)
    }

    fn buffer_error(err: BufferError) -> Self {
        Self(SimpleErrorInner::BufferError(err))
    }
}

#[derive(Debug, Error)]
enum SimpleErrorInner {
    #[error("computation of the bytes per slot overflowed")]
    BytesOverflowed,
    #[error(transparent)]
    BufferError(BufferError),
}

impl slots::Slots for Simple {
    type Exclusive<'a> = Exclusive<'a>;

    fn id_limit(&self) -> IdLimit {
        <Simple>::id_limit(self)
    }

    #[expect(clippy::panic, reason = "indices must be in-bounds")]
    unsafe fn acquire(&self, i: u32, _: Lifecycle) -> Exclusive<'_> {
        let Some(data) = self.data(i.into_usize()) else {
            panic!("index {i} is out-of-bounds");
        };

        Exclusive { data }
    }

    unsafe fn reclaim(&self, _i: u32, _: Lifecycle) {}
    unsafe fn retire(&self, _i: u32, _: Lifecycle) {}
}

/// A reader into an [`Simple`] store.
#[derive(Debug)]
pub(crate) struct Reader<'a> {
    buffer: &'a Buffer,
    bytes: Bytes,
    tags: &'a tag::ReadOnly,
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

    /// Read the data as position `i` if it is guaranteed to be race-free without bounds
    /// checking.
    ///
    /// # Safety
    ///
    /// The index `i` must satisfy [`Self::is_in_bounds`].
    #[inline]
    unsafe fn read_in_bounds(&self, i: usize) -> Option<&[u8]> {
        debug_assert!(self.is_in_bounds(i));

        if self.tags.can_read(i) {
            // SAFETY: The caller attests that `i` is in-bounds and we have checked that this
            // entry has a readable tag.
            //
            // Therefore, it is safe to:
            //
            // * Retrieve the data at position `i` since it is in-bounds.
            // * Truncate to `self.bytes` (by construction, this is less than `buffer.bytes()`.
            // * Turn the result into a slice - `self._guard` protects the immutability of the
            //   slice.
            Some(unsafe {
                self.buffer
                    .get_unchecked(i)
                    .truncate_unchecked(self.bytes)
                    .as_slice()
            })
        } else {
            None
        }
    }
}

/// A [`slots::Exclusive`] for [`Simple`].
#[derive(Debug)]
pub(crate) struct Exclusive<'a> {
    data: RawSlice<'a>,
}

impl<'a> Exclusive<'a> {
    /// Return the data within this slot as a mutable slice.
    ///
    /// The length of this slice is guaranteed to be the number of bytes passed to
    /// [`Simple::new`] or [`Config::new`].
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: Users of the `slots::Exclusive` are obligated to ensure exclusivity.
        //
        // Since `Reader` obeys the slots life-cycle requirements, a concurrent reader
        // of this data should not be possible.
        unsafe { self.data.as_mut_slice() }
    }
}

impl slots::Exclusive for Exclusive<'_> {
    fn publish(self, _: Lifecycle) {}
    fn freeze(self, _: Lifecycle) {}
    fn abort(self, _: Lifecycle) {}
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
        store::{self, Store},
    };

    // Build a store with `entries` writable slots of `entry_bytes` each, backed by `frozen`
    // zeroed frozen points. The frozen points occupy the highest slot indices.
    fn store(
        entries: usize,
        entry_bytes: usize,
        frozen: usize,
    ) -> Result<Store<Simple>, store::StoreError> {
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

        let reader = s.guard(|intrusive, guard| intrusive.reader(guard)).unwrap();
        for i in 0..4 {
            assert!(!s.can_read_approximate(i).unwrap());
            assert!(reader.read(i).is_none());
        }

        assert!(s.can_read_approximate(4).unwrap());
        assert_eq!(reader.read(4).unwrap(), &[0, 0, 0, 0, 0, 0, 0, 0]);

        assert!(s.can_read_approximate(5).unwrap());
        assert_eq!(reader.read(5).unwrap(), &[1, 1, 1, 1, 1, 1, 1, 1]);

        assert!(s.can_read_approximate(6).is_none());
        assert!(reader.read(6).is_none());
    }

    ///////////////
    // Lifecycle //
    ///////////////

    #[test]
    fn acquire_write_publish_read_roundtrip() {
        let s = store(4, 8, 1).unwrap();

        let reader = s
            .guard(|simple, guard| simple.reader(guard))
            .expect("reader guard available");

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

        let reader = s
            .guard(|simple, guard| simple.reader(guard))
            .expect("reader guard available");

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
        let reader = s
            .guard(|simple, guard| simple.reader(guard))
            .expect("reader guard available");

        assert_eq!(reader.read(idx), None);

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

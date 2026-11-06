/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A [`slots::Slots`] that maintains readability state using an atomic bitmap.
//!
//! With this, readability state is dense in memory (125 kB for 1 million vectors). This
//! improves the chances of readability checks being served from the CPU cache rather than
//! main memory, and can offset the extra checks `diskann-inmem` performs over a static index.

use std::sync::atomic::{AtomicU64, Ordering};

use diskann::utils::IntoUsize;
use thiserror::Error;

use crate::{
    buffer::{Buffer, BufferError, RawSlice},
    epoch,
    num::{Align, Bytes, IdLimit},
    prefetch,
    store::{Lifecycle, slots},
    tag,
};

/// A [`slots::SlotsConfig`] for [`Packed`].
#[derive(Debug, Clone)]
pub(crate) struct Config {
    bytes: Bytes,
}

impl Config {
    /// Create a new [`Config`] to build a [`Packed`] with the requested `bytes` for each entry.
    pub(crate) fn new(bytes: Bytes) -> Self {
        Self { bytes }
    }
}

impl slots::SlotsConfig for Config {
    type Slots = Packed;
    type Error = PackedError;

    unsafe fn build(
        self,
        handle: epoch::RegistryHandle,
        tags: &tag::Authoritative,
    ) -> Result<Packed, PackedError> {
        // SAFETY: Inherited from caller.
        unsafe { Packed::new(self.bytes, handle, tags.id_limit()) }
    }
}

/// An atomic bit-vector.
///
/// For an index `i`, the `i % 64`th bit of the `i / 64`th entry is set to 1 if slot `i`
/// is readable. Otherwise, it is zero.
#[derive(Debug)]
struct Readable {
    slots: Box<[AtomicU64]>,
}

impl Readable {
    /// Construct a new [`Readable`] for tracking up to `id_limit` slots.
    ///
    /// After construction, slot indices in `[0, id_limit)` are guaranteed to be in-bounds for
    /// the purposes of [`Self::check_bounds`] and the unchecked APIs.
    fn new(id_limit: IdLimit) -> Self {
        Self {
            slots: std::iter::repeat_with(AtomicU64::default)
                .take(
                    id_limit
                        .value()
                        .into_usize()
                        .div_ceil(Bytes::CACHELINE.value()),
                )
                .collect(),
        }
    }

    /// Return the mask needed to extract entry `i` from its bucket.
    fn mask(i: u32) -> u64 {
        1u64 << (i % 64)
    }

    /// Return the index of the word containing the state for slot `i`.
    fn slot(i: u32) -> usize {
        (i / 64).into_usize()
    }

    /// Return `true` if the slot `i` is tracked by `self`.
    fn check_bounds(&self, i: u32) -> bool {
        Self::slot(i) < self.slots.len()
    }

    /// Return the word containing slot `i`.
    ///
    /// # Safety
    ///
    /// The index `i` must be less than the [`IdLimit`] used to construct `self`.
    unsafe fn word_unchecked(&self, i: u32) -> &AtomicU64 {
        debug_assert!(self.check_bounds(i));

        // SAFETY: Since `i` is less than the ID limit, `i / 64` is less-than
        // `id_limit.div_ceil(64)`, which is the length of `slots`. So the access to
        // `self.slots` is in-bounds.
        unsafe { self.slots.get_unchecked(Self::slot(i)) }
    }

    /// Return the word containing the readability bit for slot `i`.
    ///
    /// Returns `None` if `i` is out-of-bounds.
    fn word(&self, i: u32) -> Option<&AtomicU64> {
        if !self.check_bounds(i) {
            None
        } else {
            // SAFETY: `i` is in-bounds.
            Some(unsafe { self.word_unchecked(i) })
        }
    }

    /// Return a [`BitRef`] for deferred setting of slot `i`.
    ///
    /// # Safety
    ///
    /// The index `i` must be less than the [`IdLimit`] used to construct `self`.
    unsafe fn acquire_unchecked(&self, i: u32) -> BitRef<'_> {
        BitRef {
            // SAFETY: Inherited from caller.
            word: unsafe { self.word_unchecked(i) },
            mask: Self::mask(i),
        }
    }

    /// Return `true` if slot `i` is readable without bounds checking.
    ///
    /// This has [`Ordering::Acquire`] ordering.
    ///
    /// # Safety
    ///
    /// The index `i` must be less than the [`IdLimit`] used to construct `self`.
    #[inline]
    unsafe fn can_read_unchecked(&self, i: u32) -> bool {
        // SAFETY: Inherited from caller.
        let word = unsafe { self.word_unchecked(i) };
        word.load(Ordering::Acquire) & Self::mask(i) != 0
    }

    /// Return `true` if slot `i` can be safely read under an [`epoch::Guard`].
    ///
    /// This has [`Ordering::Acquire`] ordering.
    #[inline]
    fn can_read(&self, i: u32) -> bool {
        if !self.check_bounds(i) {
            false
        } else {
            // SAFETY: `i` is in-bounds.
            unsafe { self.can_read_unchecked(i) }
        }
    }

    /// Mark slot `i` as retired.
    ///
    /// # Panics
    ///
    /// Panics if `i` is out-of-bounds or if the bit tracking slot `i` is already 0.
    #[expect(
        clippy::expect_used,
        reason = "this is called as part of the `Slots` API - a panic means an internal error occurred"
    )]
    fn retire(&self, i: u32) {
        let word = self.word(i).expect("an out-of-bounds index was supplied");
        let mask = Self::mask(i);

        // Retirement does not synchronize, so relaxed ordering is fine.
        let previous = word.fetch_and(!mask, Ordering::Relaxed);

        // Check that the bit we're clearing was previously set. Otherwise, we have a
        // concurrency violation - someone tried to retire this slot multiple times!
        assert_ne!(
            previous & mask,
            0,
            "concurrency violation - multiple retirements detected"
        );
    }

    /// Prefetch the cache line containing the readability bit for slot `i`.
    ///
    /// # Safety
    ///
    /// The index `i` must be less than the [`IdLimit`] used to construct `self`.
    #[inline]
    unsafe fn prefetch_unchecked(&self, i: u32) {
        debug_assert!(self.check_bounds(i));
        let i = i.into_usize();

        // SAFETY: Inherited from caller.
        unsafe { prefetch::prefetch_cacheline(self.slots.as_ptr().add(i / 64).cast::<u8>()) }
    }

    // Safe Test Helpers //

    #[cfg(test)]
    fn acquire(&self, i: u32) -> BitRef<'_> {
        assert!(self.check_bounds(i), "i = {i}");
        // SAFETY: `i` is in-bounds.
        unsafe { self.acquire_unchecked(i) }
    }
}

/// A deferred publishing mechanism for a bit within [`Readable`].
///
/// If dropped outside of [`BitRef::publish`] the referenced slot will not be marked as
/// readable.
#[derive(Debug)]
#[must_use = "BitRef doesn't do anything unless published"]
struct BitRef<'a> {
    word: &'a AtomicU64,
    mask: u64,
}

impl BitRef<'_> {
    /// Mark the referenced slot as readable with [`Ordering::Release`] ordering.
    ///
    /// # Panics
    ///
    /// Panics if the referenced slot is already readable.
    fn publish(self) {
        let previous = self.word.fetch_or(self.mask, Ordering::Release);
        assert_eq!(
            previous & self.mask,
            0,
            "concurrency violation - multiple publishes detected"
        );
    }
}

/// A packed store where readability bits are stored in a dense bit-vector.
#[derive(Debug)]
pub(crate) struct Packed {
    /// The actual data.
    ///
    /// Each entry is at least [`Self::bytes`] long.
    buffer: Buffer,

    /// The unpadded size of each entry in [`Self::buffer`].
    bytes: Bytes,

    /// The readability state of each entry in [`Self::buffer`].
    ///
    /// This has the same [`IdLimit`] as [`Self::buffer`].
    readable: Readable,

    /// A handle to the [`epoch::Registry`] managing this store.
    handle: epoch::RegistryHandle,
}

impl Packed {
    /// Construct the [`Config`] for [`Self`].
    ///
    /// See also: [`Config::new`].
    pub(crate) fn config(bytes: Bytes) -> Config {
        Config::new(bytes)
    }

    /// Create a new [`Packed`] with capacity for `id_limit` slots of `bytes`.
    ///
    /// # Safety
    ///
    /// This assumes that the resulting [`Packed`] is embedded in a [`crate::store::Store`]
    /// and that the following hold:
    ///
    /// * `handle` is an [`epoch::RegistryHandle`] into the [`epoch::Registry`] in the
    ///   containing `Store`.
    ///
    /// * `id_limit` matches the containing `Store`'s [`IdLimit`].
    ///
    /// # Errors
    ///
    /// Returns an error if rounding `bytes` up to a cache line overflows, or if the
    /// padded buffer allocation, including alignment requirements, exceeds `isize::MAX`.
    pub(crate) unsafe fn new(
        bytes: Bytes,
        handle: epoch::RegistryHandle,
        id_limit: IdLimit,
    ) -> Result<Self, PackedError> {
        let padded = bytes
            .checked_next_multiple_of(Bytes::CACHELINE)
            .ok_or(PackedError::BytesOverflowed)?;

        let buffer = Buffer::new(id_limit.value().into_usize(), padded, Align::_128)?;
        let readable = Readable::new(id_limit);

        Ok(Self {
            buffer,
            bytes,
            handle,
            readable,
        })
    }

    /// Return the [`IdLimit`] for this store.
    pub(crate) fn id_limit(&self) -> IdLimit {
        // The numeric cast is safe because `Packed::new` takes an `IdLimit` in its
        // constructor, and thus `self.buffer.len()` cannot exceed `u32::MAX`.
        IdLimit::new(self.buffer.len() as u32)
    }

    /// Return the number of bytes for each entry.
    pub(crate) fn bytes(&self) -> Bytes {
        self.bytes
    }

    /// Return a [`Reader`] over [`Self`].
    ///
    /// # Panics
    ///
    /// Panics if `guard` does not belong to `self`'s [`epoch::Registry`].
    pub(crate) fn reader<'a>(&'a self, guard: epoch::Guard<'a>) -> Reader<'a> {
        self.handle.assert_guard_belongs(&guard);
        Reader {
            store: self,
            _guard: guard,
        }
    }
}

#[derive(Debug, Error)]
pub(crate) enum PackedError {
    #[error("computation of the bytes per slot overflowed")]
    BytesOverflowed,
    #[error(transparent)]
    Buffer(#[from] BufferError),
}

impl slots::Slots for Packed {
    type Exclusive<'a> = Exclusive<'a>;

    fn id_limit(&self) -> IdLimit {
        self.id_limit()
    }

    unsafe fn acquire(&self, i: u32, _: Lifecycle) -> Exclusive<'_> {
        assert!(i.into_usize() < self.buffer.len(), "slot out-of-bounds");
        debug_assert!(!self.readable.can_read(i));

        // SAFETY: Bounds were checked. The caller guarantees exclusivity.
        unsafe {
            Exclusive {
                data: self
                    .buffer
                    .get_unchecked(i.into_usize())
                    .truncate(self.bytes),
                tag: self.readable.acquire_unchecked(i),
            }
        }
    }

    unsafe fn retire(&self, i: u32, _: Lifecycle) {
        assert!(i.into_usize() < self.buffer.len(), "slot out-of-bounds");
        self.readable.retire(i);
    }

    unsafe fn reclaim(&self, i: u32, _: Lifecycle) {
        assert!(i.into_usize() < self.buffer.len(), "slot out-of-bounds");
    }
}

/// A reader into a [`Packed`] store.
#[derive(Debug)]
pub(crate) struct Reader<'a> {
    store: &'a Packed,
    _guard: epoch::Guard<'a>,
}

impl Reader<'_> {
    /// Attempt to read the value at index `i`. This can fail for any of the
    /// following reasons:
    ///
    /// 1. Index `i` is out-of-bounds.
    /// 2. The read cannot be guaranteed to be race-free.
    #[inline]
    pub(crate) fn read(&self, i: u32) -> Option<&[u8]> {
        if self.is_in_bounds(i) {
            // SAFETY: Bounds checked above.
            unsafe { self.read_in_bounds(i) }
        } else {
            None
        }
    }

    /// Check if `i` is in-bounds, returning `true` if `i < self.id_limit()`.
    #[inline]
    pub(crate) fn is_in_bounds(&self, i: u32) -> bool {
        i.into_usize() < self.store.buffer.len()
    }

    /// Return the [`IdLimit`] for the reader.
    pub(crate) fn id_limit(&self) -> IdLimit {
        self.store.id_limit()
    }

    /// Return the number of bytes held in each slot.
    pub(crate) fn bytes(&self) -> Bytes {
        self.store.bytes
    }

    /// Read the data at position `i`, without bounds checking, if reading is guaranteed
    /// to be race-free.
    ///
    /// # Safety
    ///
    /// The index `i` must satisfy [`Self::is_in_bounds`].
    #[inline]
    pub(crate) unsafe fn read_in_bounds(&self, i: u32) -> Option<&[u8]> {
        debug_assert!(self.is_in_bounds(i));

        // SAFETY: Caller asserts that `i` is in-bounds.
        if unsafe { self.store.readable.can_read_unchecked(i) } {
            // SAFETY: Publication was observed. The `epoch::Guard` prevents recycling while
            // the returned reference is borrowed from this reader.
            Some(unsafe { self.read_raw_unchecked(i).as_slice() })
        } else {
            None
        }
    }

    /// Return the raw data slice for slot `i` without any race guarantees.
    ///
    /// # Safety
    ///
    /// The index `i` must satisfy [`Self::is_in_bounds`].
    ///
    /// The returned [`RawSlice`] may be used for prefetching. Materializing it as a slice
    /// or reference requires ensuring race freedom and satisfying the borrowing and
    /// aliasing requirements of [`RawSlice::as_slice`] or [`RawSlice::as_mut_slice`].
    #[inline]
    pub(crate) unsafe fn read_raw_unchecked(&self, i: u32) -> RawSlice<'_> {
        // SAFETY: Caller guarantees bounds; bytes does not exceed the row stride.
        unsafe {
            self.store
                .buffer
                .get_unchecked(i.into_usize())
                .truncate_unchecked(self.store.bytes)
        }
    }

    /// Prefetch the cache line containing the readability bit for slot `i`.
    ///
    /// # Safety
    ///
    /// The index `i` must satisfy [`Self::is_in_bounds`].
    #[inline]
    pub(crate) unsafe fn prefetch_tag(&self, i: u32) {
        // SAFETY: Inherited from caller.
        unsafe { self.store.readable.prefetch_unchecked(i) };
    }

    // Safe Test Helpers //

    #[cfg(test)]
    fn read_raw(&self, i: u32) -> RawSlice<'_> {
        assert!(self.is_in_bounds(i));

        // SAFETY: `i` is in-bounds.
        unsafe { self.read_raw_unchecked(i) }
    }
}

/// A [`slots::Exclusive`] for [`Packed`].
#[derive(Debug)]
pub(crate) struct Exclusive<'a> {
    data: RawSlice<'a>,
    tag: BitRef<'a>,
}

impl Exclusive<'_> {
    /// Return the data within this slot as a mutable slice.
    ///
    /// The length of this slice is guaranteed to be the number of bytes passed to
    /// [`Packed::new`] or [`Config::new`].
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: Users of the `slots::Exclusive` are obligated to ensure exclusivity.
        //
        // Since `Reader` obeys the slots life-cycle requirements, a concurrent reader
        // of this data should not be possible.
        unsafe { self.data.as_mut_slice() }
    }
}

impl slots::Exclusive for Exclusive<'_> {
    fn publish(self, _: Lifecycle) {
        self.tag.publish()
    }

    fn freeze(self, _: Lifecycle) {
        self.tag.publish()
    }

    fn abort(self, _: Lifecycle) {}
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    //! # Notes on Testing
    //!
    //! Since the readability bits are densely packed, we need to use larger store sizes
    //! than the other [`slots::Slots`] implementations. A 1,025-slot bitmap occupies
    //! 17 words (136 bytes), spanning multiple cache lines regardless of alignment.

    use super::*;

    use std::{
        assert_matches,
        num::{NonZeroU32, NonZeroUsize},
        sync::Barrier,
        thread,
    };

    use crate::{
        num::MaxDegree,
        store::{self, Store},
    };

    const ENTRIES: u32 = 1025;
    const BOUNDARIES: &[u32] = &[0, 1, 62, 63, 64, 65, 510, 511, 512, 513, 1023, 1024];
    const PAYLOAD_BYTES: Bytes = Bytes::new(8);

    fn store(id_limit: IdLimit, bytes: Bytes) -> Result<Store<Packed>, store::StoreError> {
        Store::new(
            store::Layout::new(id_limit, MaxDegree::new(0)),
            store::Config::__exhaustive(
                NonZeroUsize::new(10).unwrap(),
                NonZeroU32::new(16).unwrap(),
            ),
            Config::new(bytes),
        )
    }

    fn reader(s: &Store<Packed>) -> Reader<'_> {
        s.guard(|packed, guard| packed.reader(guard)).unwrap()
    }

    fn payload(i: u32, tag: u32) -> [u8; 8] {
        (u64::from(i) | (u64::from(tag) << 32)).to_le_bytes()
    }

    fn publish(s: &Store<Packed>, i: u32, tag: u32) {
        let mut slot = s.slot(i).unwrap();
        slot.data().as_mut_slice().copy_from_slice(&payload(i, tag));

        assert_eq!(slot.publish(), i);
    }

    fn is_set(x: &bool) -> bool {
        *x
    }

    fn is_not_set(x: &bool) -> bool {
        !x
    }

    //----------//
    // Readable //
    //----------//

    #[test]
    fn test_readable_bounds() {
        for len in [0, 1, 63, 64, 65, 511, 512, 513, 1024, ENTRIES] {
            let readable = Readable::new(IdLimit::new(len));
            for i in 0..len {
                assert!(readable.check_bounds(i), "len={len}, i={i}");
                assert!(!readable.can_read(i), "len={len}, i={i}");
            }

            for i in [len.next_multiple_of(64), u32::MAX] {
                assert!(!readable.check_bounds(i));
                assert!(readable.word(i).is_none());
                assert!(!readable.can_read(i));
            }

            if len != 0 {
                let last = len - 1;
                readable.acquire(last).publish();
                for i in 0..len {
                    if i == last {
                        assert!(readable.can_read(i), "len={len}, i={i}");
                    } else {
                        assert!(!readable.can_read(i), "len={len}, i={i}");
                    }
                }
                readable.retire(last);
                assert!(!readable.can_read(last));
            }
        }
    }

    #[test]
    fn test_readable_bit_isolation() {
        let readable = Readable::new(IdLimit::new(ENTRIES));
        let mut expected = vec![false; ENTRIES.into_usize()];
        for (i, bit) in expected.iter_mut().enumerate() {
            let i = u32::try_from(i).unwrap();

            if i % 2 == 0 || BOUNDARIES.contains(&i) {
                readable.acquire(i).publish();
                *bit = true;
            } else {
                // Dropped references should not publish.
                let _ = readable.acquire(i);
            }
        }
        for (i, &bit) in expected.iter().enumerate() {
            let i = u32::try_from(i).unwrap();
            assert_eq!(readable.can_read(i), bit, "published slot {i}");
        }

        for (i, bit) in expected.iter_mut().enumerate() {
            let i = u32::try_from(i).unwrap();

            if *bit && i % 3 == 0 {
                readable.retire(i);
                *bit = false;
            }
        }
        for (i, &bit) in expected.iter().enumerate() {
            let i = u32::try_from(i).unwrap();

            assert_eq!(readable.can_read(i), bit, "retired slot {i}");
        }
    }

    //--------//
    // Packed //
    //--------//

    #[test]
    fn test_acquire_publish_and_abort() {
        let s = store(IdLimit::new(ENTRIES), PAYLOAD_BYTES).unwrap();
        let reader = reader(&s);

        for &i in BOUNDARIES {
            {
                let mut slot = s.slot(i).unwrap();
                slot.data().as_mut_slice().copy_from_slice(&payload(i, 0));

                assert!(reader.read(i).is_none());
                assert_eq!(s.can_read_approximate(i.into_usize()), Some(false));
                // Dropping the unpublished writer aborts the acquisition.
            }
            assert!(reader.read(i).is_none());
            publish(&s, i, 1);
        }

        for i in 0..ENTRIES {
            if BOUNDARIES.contains(&i) {
                let slice = reader.read(i).unwrap();
                assert_eq!(slice, payload(i, 1));

                // The raw read should return the same span.
                let raw = reader.read_raw(i);
                assert_eq!(raw.as_ptr(), slice.as_ptr());
                assert_eq!(raw.len(), Bytes::new(slice.len()));
            } else {
                assert!(reader.read(i).is_none(), "untouched slot {i}");
            }

            assert_eq!(
                s.can_read_approximate(i.into_usize()),
                Some(BOUNDARIES.contains(&i))
            );
        }

        for i in [ENTRIES, ENTRIES + 1, u32::MAX] {
            assert!(!reader.is_in_bounds(i));
            assert!(reader.read(i).is_none());
            assert!(s.can_read_approximate(i.into_usize()).is_none());
        }
    }

    #[test]
    fn retirement_preserves_published_and_frozen_neighbors() {
        let id_limit = IdLimit::new(ENTRIES);
        let s = store(id_limit, PAYLOAD_BYTES).unwrap();
        assert_eq!(s.id_limit(), id_limit);

        let mut num_frozen = 0;
        let mut num_published = 0;

        fn is_frozen(i: u32) -> bool {
            i.is_multiple_of(2)
        }

        for &i in BOUNDARIES {
            let mut slot = s.slot(i).unwrap();
            slot.data().as_mut_slice().copy_from_slice(&payload(i, 0));
            if is_frozen(i) {
                slot.freeze();
                num_frozen += 1;
            } else {
                slot.publish();
                num_published += 1;
            }
        }

        assert_ne!(num_frozen, 0);
        assert_ne!(num_published, 0);

        let reader = reader(&s);
        for &i in BOUNDARIES {
            assert_eq!(reader.read(i), Some(payload(i, 0).as_slice()));
            assert!(
                s.slot(i).is_none(),
                "this slot should be previously acquired - it cannot be acquired again"
            );

            // If a point is not frozen, we should be able to retire it.
            if is_frozen(i) {
                assert_matches!(
                    s.retire(i.into_usize()),
                    Err(store::RetireError::SlotIsReserved { .. }),
                    "frozen points cannot be retired",
                );
            } else {
                s.retire(i.into_usize()).unwrap();
                assert!(
                    reader.read(i).is_none(),
                    "readers should observe the retirement of a slot",
                );

                assert_matches!(
                    s.retire(i.into_usize()),
                    Err(store::RetireError::SlotIsReserved { .. }),
                    "the slot has not been reclaimed - we cannot retire it again",
                );
            }
        }

        // Perform a scan over all entries, making sure that the view we get from `Packed`
        // matches our model of what should be happening.
        for i in 0..ENTRIES {
            let readable = BOUNDARIES.contains(&i) && is_frozen(i);

            assert_eq!(s.can_read_approximate(i.into_usize()), Some(readable));

            if readable {
                assert_eq!(reader.read(i), Some(payload(i, 0).as_slice()));
            } else {
                assert!(reader.read(i).is_none(), "slot {i}");
            }
        }

        // Ensure that trying to retire a free slot doesn't work.
        {
            let free_slot = 2;
            assert!(!BOUNDARIES.contains(&free_slot));
            assert_matches!(
                s.retire(free_slot.into_usize()),
                Err(store::RetireError::SlotIsReserved { .. })
            );
        }

        // Ensure that we cannot retire a slot that is in the `Exclusive` state.
        {
            let exclusive = 3;
            assert!(!BOUNDARIES.contains(&exclusive));

            let owned = s.slot(exclusive).unwrap();
            assert_matches!(
                s.retire(owned.slot() as usize),
                Err(store::RetireError::SlotIsReserved { .. }),
                "a slot in the exclusive state cannot be retired",
            );
        }

        // Out-of-bounds slots cannot be retired.
        for i in [ENTRIES, u32::MAX] {
            assert_matches!(
                s.retire(i.into_usize()),
                Err(store::RetireError::OutOfBounds),
                "out-of-bounds slots cannot be retired",
            );
        }
    }

    #[test]
    fn acquire_exhausts_then_abort_restores_availability() {
        // Use a smaller number of entries than the main tests - just spanning a couple
        // of words. This is mainly to help Miri run faster.
        let entries = 140;
        let s = store(IdLimit::new(entries), PAYLOAD_BYTES).unwrap();

        // Acquire all slots in the store.
        let mut slots: Vec<_> = (0..entries).map(|_| s.acquire().unwrap()).collect();

        // Ensure that each slot is uniquely seen.
        let mut seen = vec![false; entries.into_usize()];
        for slot in &slots {
            let i = slot.slot() as usize;
            assert!(!seen[i], "duplicate slot {i}");
            seen[i] = true;
        }
        assert!(seen.iter().all(is_set));
        assert!(s.acquire().is_none(), "there are no more slots left",);

        // Drop the last writer to make its slot available again.
        let last = slots.pop().unwrap();
        let i = last.slot();
        drop(last);

        let replacement = s.acquire().unwrap();
        assert_eq!(replacement.slot(), i);
        assert!(s.acquire().is_none());
    }

    #[test]
    fn reader_guard_delays_reuse_of_retired_slots() {
        let s = store(IdLimit::new(ENTRIES), PAYLOAD_BYTES).unwrap();
        for &i in BOUNDARIES {
            publish(&s, i, 0);
        }
        {
            let reader = reader(&s);
            let borrowed: Vec<_> = BOUNDARIES
                .iter()
                .map(|&i| reader.read(i).unwrap())
                .collect();

            for &i in BOUNDARIES {
                s.retire(i.into_usize()).unwrap();
                assert!(reader.read(i).is_none());
            }

            // One epoch advancement is possible while this reader is active; further
            // advancement must wait for its guard to be dropped.
            assert_matches!(
                s.try_drain(),
                Some(0),
                "one epoch can be advanced while the reader is active",
            );
            assert_matches!(s.try_drain(), None);
            assert_matches!(s.try_drain(), None);
            assert_matches!(s.try_drain(), None);

            for (&i, data) in BOUNDARIES.iter().zip(borrowed) {
                assert!(s.slot(i).is_none(), "slot has been retired",);
                assert_eq!(
                    data,
                    payload(i, 0).as_slice(),
                    "even though the slot has been retired, it's still readable",
                );
            }
        }

        let reclaimed: usize = (0..3).map(|_| s.try_drain().unwrap()).sum();
        assert_eq!(
            reclaimed,
            BOUNDARIES.len(),
            "every retired slot should now be available",
        );

        for &i in BOUNDARIES {
            publish(&s, i, 1);
        }

        let reader = reader(&s);
        for &i in BOUNDARIES {
            assert_eq!(reader.read(i), Some(payload(i, 1).as_slice()));
        }
    }

    #[test]
    fn test_recycling() {
        // Use a smaller number of entries than the main tests - just spanning a couple
        // of words. This is mainly to help Miri run faster.
        let entries = 129;

        let s = store(IdLimit::new(entries), PAYLOAD_BYTES).unwrap();

        // Freeze the last slot.
        let frozen = entries - 1;
        let mut slot = s.slot(frozen).unwrap();
        slot.data()
            .as_mut_slice()
            .copy_from_slice(&payload(frozen, 0));
        slot.freeze();

        // Here, we go through the whole lifecycle multiple times, acquiring all slots,
        // retiring, reclaiming, and again.
        for generation in 0..2 {
            let mut seen = vec![false; entries.into_usize()];
            while let Some(mut slot) = s.acquire() {
                let i = slot.slot();
                assert_ne!(i, frozen);
                assert!(!seen[i.into_usize()], "duplicate slot {i}");
                seen[i.into_usize()] = true;
                slot.data()
                    .as_mut_slice()
                    .copy_from_slice(&payload(i, generation));
                slot.publish();
            }

            assert!(seen[..frozen.into_usize()].iter().all(is_set));
            assert!(seen[frozen.into_usize()..].iter().all(is_not_set));

            {
                let reader = reader(&s);
                for i in 0..frozen {
                    assert_eq!(reader.read(i), Some(payload(i, generation).as_slice()));
                }
                for i in frozen..entries {
                    assert_eq!(reader.read(i), Some(payload(i, 0).as_slice()));
                }
            }

            if generation == 0 {
                for i in 0..frozen {
                    s.retire(i.into_usize()).unwrap();
                }
                let reader = reader(&s);
                for i in 0..frozen {
                    assert!(reader.read(i).is_none());
                }
                assert_eq!(reader.read(frozen), Some(payload(frozen, 0).as_slice()));
            }
        }
    }

    #[test]
    fn test_parallel_publish_and_retire() {
        let entries = 151;
        const WORKERS: u32 = 4;
        let s = &store(IdLimit::new(entries), PAYLOAD_BYTES).unwrap();
        let barrier = &Barrier::new(WORKERS.into_usize());

        let frozen = entries - 1;
        let mut slot = s.slot(frozen).unwrap();
        slot.data()
            .as_mut_slice()
            .copy_from_slice(&payload(frozen, 0));
        slot.freeze();

        // This test operates in two phases.
        //
        // The first one publishes all slots indicated by `published_gen0` (except for the
        // frozen point, which is already published).
        //
        // The second phase publishes the rest of the space and retires a subset of those
        // published in the first phase.
        //
        // As such:
        // * `retired_gen0` must return `true` for a subset of `published_gen0`.
        // * `published_gen1` must be disjoint from `published_gen0`.
        let published_gen0 = |i: u32| -> bool { i == frozen || i.is_multiple_of(2) };
        let retired_gen0 = |i: u32| -> bool { i != frozen && i.is_multiple_of(4) };
        let published_gen1 = |i: u32| -> bool { !published_gen0(i) };

        // Each worker gets its own interleaved portion of the ID space.
        thread::scope(|scope| {
            for worker in 0..WORKERS {
                scope.spawn(move || {
                    barrier.wait();
                    for i in (worker..frozen).step_by(WORKERS.into_usize()) {
                        if published_gen0(i) {
                            publish(s, i, 0);
                        }
                    }
                });
            }
        });

        // Ensure that we can observe writes from all workers.
        {
            let reader = reader(s);
            for i in 0..entries {
                if published_gen0(i) {
                    assert_eq!(reader.read(i), Some(payload(i, 0).as_slice()), "i = {i}");
                } else {
                    assert!(reader.read(i).is_none(), "i = {i}",);
                }
            }
        }

        thread::scope(|scope| {
            for worker in 0..WORKERS {
                scope.spawn(move || {
                    barrier.wait();
                    for i in (worker..frozen).step_by(WORKERS.into_usize()) {
                        if retired_gen0(i) {
                            s.retire(i.into_usize()).unwrap();
                        } else if published_gen1(i) {
                            publish(s, i, 1);
                        }
                    }
                });
            }
        });

        let reader = reader(s);
        for i in 0..entries {
            let readable = i == frozen || !retired_gen0(i);
            assert_eq!(s.can_read_approximate(i.into_usize()), Some(readable));

            if readable {
                let generation = if published_gen0(i) { 0 } else { 1 };
                assert_eq!(reader.read(i), Some(payload(i, generation).as_slice()));
            } else {
                assert!(reader.read(i).is_none(), "retired slot {i}");
            }
        }
    }
}

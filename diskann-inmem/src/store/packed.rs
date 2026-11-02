/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A [`slots::Slots`] that maintains readability state using an atomic bitmap.
//!
//! With this, readability state is dense in memory (125KB for 1 million vectors). This
//! improves the chances of the readable check being served from somewhere in the CPU cache
//! instead of always going to memory, and can offset the extra checks `diskann-inmem`
//! performs over a static index.

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

impl slots::SlotsConfig for Config {
    type Slots = Packed;
    type Error = PackedError;

    unsafe fn build(
        self,
        handle: epoch::RegistryHandle,
        tags: &tag::Authoritative,
    ) -> Result<Packed, PackedError> {
        let padded = self
            .bytes
            .checked_next_multiple_of(Bytes::CACHELINE)
            .ok_or(PackedError::BytesOverflowed)?;

        let entries = tags.id_limit().as_usize();
        let buffer = Buffer::new(entries, padded, Align::_128)?;
        let readable = Readable::new(entries);

        Ok(Packed {
            buffer,
            bytes: self.bytes,
            handle,
            readable,
        })
    }
}

#[derive(Debug)]
struct Readable {
    slots: Box<[AtomicU64]>,
}

impl Readable {
    fn new(len: usize) -> Self {
        Self {
            slots: std::iter::repeat_with(AtomicU64::default)
                .take(len.div_ceil(64))
                .collect(),
        }
    }

    /// Return the mask needed to extract entry `i` from its bucket.
    fn mask(i: usize) -> u64 {
        1u64 << (i % 64)
    }

    fn slot(i: usize) -> usize {
        i / 64
    }

    fn check_bounds(&self, i: usize) -> bool {
        Self::slot(i) < self.slots.len()
    }

    unsafe fn word_unchecked(&self, i: usize) -> &AtomicU64 {
        debug_assert!(self.check_bounds(i));
        unsafe { self.slots.get_unchecked(Self::slot(i)) }
    }

    fn word(&self, i: usize) -> Option<&AtomicU64> {
        if !self.check_bounds(i) {
            None
        } else {
            Some(unsafe { self.word_unchecked(i) })
        }
    }

    unsafe fn acquire_unchecked(&self, i: usize) -> BitRef<'_> {
        BitRef {
            word: unsafe { self.word_unchecked(i) },
            mask: Self::mask(i),
        }
    }

    /// Return `true` if slot `i` is readable without boundschecking.
    #[inline]
    unsafe fn can_read_unchecked(&self, i: usize) -> bool {
        self.word_unchecked(i).load(Ordering::Acquire) & Self::mask(i) != 0
    }

    #[inline]
    fn can_read(&self, i: usize) -> bool {
        if !self.check_bounds(i) {
            false
        } else {
            unsafe { self.can_read_unchecked(i) }
        }
    }

    fn retire(&self, i: usize) {
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

    #[inline]
    unsafe fn prefetch_unchecked(&self, i: usize) {
        debug_assert!(self.check_bounds(i));

        unsafe { prefetch::prefetch_cacheline(self.slots.as_ptr().add(i / 64).cast::<u8>()) }
    }
}

#[derive(Debug)]
struct BitRef<'a> {
    word: &'a AtomicU64,
    mask: u64,
}

impl BitRef<'_> {
    fn publish(self) {
        let previous = self.word.fetch_or(self.mask, Ordering::Release);
        assert_eq!(
            previous & self.mask,
            0,
            "cocurrency violation - multiple publishes detected"
        );
    }
}

#[derive(Debug)]
pub(crate) struct Packed {
    buffer: Buffer,
    bytes: Bytes,
    handle: epoch::RegistryHandle,
    readable: Readable,
}

impl Packed {
    pub(crate) fn config(bytes: Bytes) -> Config {
        Config { bytes }
    }

    pub(crate) fn id_limit(&self) -> IdLimit {
        IdLimit::new(self.buffer.len() as u32)
    }

    pub(crate) fn bytes(&self) -> Bytes {
        self.bytes
    }

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
        let i = i.into_usize();
        assert!(i < self.buffer.len(), "slot out-of-bounds");

        debug_assert!(!self.readable.can_read(i));

        Exclusive {
            // SAFETY: Bounds were checked; the caller guarantees exclusivity.
            data: unsafe { self.buffer.get_unchecked(i) }.truncate(self.bytes),
            tag: unsafe { self.readable.acquire_unchecked(i) },
        }
    }

    unsafe fn retire(&self, i: u32, _: Lifecycle) {
        assert!(i.into_usize() < self.buffer.len(), "slot out-of-bounds");
        self.readable.retire(i.into_usize());
    }

    unsafe fn reclaim(&self, i: u32, _: Lifecycle) {
        assert!(i.into_usize() < self.buffer.len(), "slot out-of-bounds");
    }
}

#[derive(Debug)]
pub(crate) struct Reader<'a> {
    store: &'a Packed,
    _guard: epoch::Guard<'a>,
}

impl Reader<'_> {
    #[inline]
    pub(crate) fn is_in_bounds(&self, i: usize) -> bool {
        i < self.store.buffer.len()
    }

    pub(crate) fn id_limit(&self) -> IdLimit {
        self.store.id_limit()
    }

    pub(crate) fn bytes(&self) -> Bytes {
        self.store.bytes
    }

    #[inline]
    pub(crate) fn read(&self, i: usize) -> Option<&[u8]> {
        if self.is_in_bounds(i) {
            // SAFETY: Bounds checked above.
            unsafe { self.read_in_bounds(i) }
        } else {
            None
        }
    }

    /// # Safety
    ///
    /// `i` must be in bounds.
    #[inline]
    pub(crate) unsafe fn read_in_bounds(&self, i: usize) -> Option<&[u8]> {
        debug_assert!(self.is_in_bounds(i));
        if unsafe { self.store.readable.can_read_unchecked(i) } {
            // SAFETY: Publication was observed with Acquire. The guard prevents
            // recycling while the returned reference is borrowed from this reader.
            Some(unsafe { self.read_raw_unchecked(i).as_slice() })
        } else {
            None
        }
    }

    /// # Safety
    ///
    /// `i` must be in bounds. The result is only for prefetching unless the
    /// caller independently establishes readability under this reader's guard.
    #[inline]
    pub(crate) unsafe fn read_raw_unchecked(&self, i: usize) -> RawSlice<'_> {
        // SAFETY: Caller guarantees bounds; bytes does not exceed the row stride.
        unsafe {
            self.store
                .buffer
                .get_unchecked(i)
                .truncate_unchecked(self.store.bytes)
        }
    }

    /// # Safety
    ///
    /// `i` must be in bounds.
    #[inline]
    pub(crate) unsafe fn prefetch_tag(&self, i: usize) {
        self.store.readable.prefetch_unchecked(i);
    }
}

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

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{num::NonZeroUsize, sync::atomic::Ordering};

use diskann::utils::IntoUsize;

use crate::{
    buffer::{Buffer, RawSlice},
    epoch,
    num::{Align, Bytes},
    tag::{AtomicTag, Tag},
};

/// The invasive store where concurrency tags are stored inline with the data.
#[derive(Debug)]
pub(crate) struct Invasive {
    // The inline tags are `AtomicTag`s stored after the data.
    buffer: Buffer,

    // The unpadded size of each row in `buffer`. This includes both the data **and** the
    // 1-byte tag. Tags are located at byte `unpadded - 1`.
    unpadded: Bytes,
}

const TWO: NonZeroUsize = NonZeroUsize::new(2).unwrap();

impl Invasive {
    pub(crate) fn new(entries: usize, bytes: Bytes) -> Self {
        let unpadded = bytes.checked_add(AtomicTag::SIZE).unwrap();
        let padded_bytes = unpadded
            .checked_next_multiple_of(Bytes::CACHELINE.div(TWO))
            .unwrap();

        Self {
            buffer: Buffer::new(entries, padded_bytes, Align::_128).unwrap(),
            unpadded,
        }
    }

    pub(crate) fn bytes(&self) -> Bytes {
        self.unpadded
    }

    pub(crate) unsafe fn reader<'a>(&'a self, guard: epoch::Guard<'a>) -> Reader<'a> {
        Reader {
            buffer: &self.buffer,
            unpadded: self.unpadded,
            _guard: guard,
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
            .split(self.unpadded.unchecked_sub(AtomicTag::SIZE));
        (
            // SAFETY: We're careful in this module to ensure the inline tags are only
            // ever accessed atomically.
            unsafe { AtomicTag::from_ptr(mirror.as_mut_ptr().cast()) },
            data,
        )
    }

    fn data(&self, i: usize) -> Option<(&AtomicTag, RawSlice<'_>)> {
        if i >= self.buffer.len() {
            None
        } else {
            Some(unsafe { self.data_unchecked(i) })
        }
    }
}

impl super::plugin::Plugin for Invasive {
    type Slot<'a> = Slot<'a>;

    unsafe fn acquire(&self, i: u32) -> Self::Slot<'_> {
        let Some((tag, data)) = self.data(i.into_usize()) else {
            panic!("index {i} is out-of-bounds");
        };

        // This is a pessimistic check to ensure that the caller is correctly using the
        // `plugin` API.
        assert_eq!(
            tag.load(Ordering::Relaxed),
            Tag::AVAILABLE,
            "concurrency violation",
        );

        // While we can leave this tag as `Tag::AVAILABLE` since it's just a mirror, setting
        // it to `Tag::OWNED` lets us more precisely detect misuse from the caller.
        tag.store(Tag::OWNED, Ordering::Relaxed);
        Slot { tag, data }
    }

    fn reclaim(&self, i: u32) {
        let Some((tag, _)) = self.data(i.into_usize()) else {
            panic!("index {i} is out-of-bounds");
        };

        tag.store(Tag::AVAILABLE, Ordering::Release);
    }

    fn retire(&self, i: u32) {
        let Some((tag, _)) = self.data(i.into_usize()) else {
            panic!("index {i} is out-of-bounds");
        };

        tag.store(Tag::RETIRING, Ordering::Relaxed);
    }
}

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

    /// Return `true` if it is safe to read the data at position `i`.
    ///
    /// This guarantee only holds while `self` is alive. Construction of a new [`Reader`]
    /// requires a separate check.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "this is non-trivial method that likely be used in the future"
        )
    )]
    pub(crate) fn can_read(&self, i: usize) -> Option<bool> {
        if !self.is_in_bounds(i) {
            return None;
        }

        // SAFETY: We've checked that `i` is in-bounds.
        //
        // Further, we guarantee that `self.unpadded >= AtomicTag:::SIZE`, so the pointer
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
        // * We maintain an internal invariant that `self.buffer.stride() <= self.unpadded`.
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
        self.unpadded.unchecked_sub(AtomicTag::SIZE)
    }
}

#[derive(Debug)]
pub(crate) struct Slot<'a> {
    tag: &'a AtomicTag,
    data: RawSlice<'a>,
}

impl<'a> Slot<'a> {
    pub(crate) unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { self.data.as_mut_slice() }
    }
}

impl super::plugin::Slot for Slot<'_> {
    fn publish(self) {
        self.tag.store(Tag::PUBLISHED, Ordering::Release);
    }
    fn freeze(self) {
        self.tag.store(Tag::FROZEN, Ordering::Release);
    }
    fn abort(self) {
        self.tag.store(Tag::AVAILABLE, Ordering::Release);
    }
}

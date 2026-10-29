/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Payload storage with a packed atomic readability mirror.

use std::sync::atomic::{AtomicU64, Ordering};

use diskann::utils::IntoUsize;
use thiserror::Error;

use crate::{
    buffer::{Buffer, BufferError, RawSlice},
    epoch,
    num::{Align, Bytes, IdLimit},
    store::{Lifecycle, slots},
    tag,
};

#[derive(Debug, Clone)]
pub(crate) struct Config {
    bytes: Bytes,
}

impl slots::SlotsConfig for Config {
    type Slots = Sidecar;
    type Error = SidecarError;

    unsafe fn build(
        self,
        handle: epoch::RegistryHandle,
        tags: &tag::Authoritative,
    ) -> Result<Sidecar, SidecarError> {
        let padded = self
            .bytes
            .checked_next_multiple_of(Bytes::CACHELINE)
            .ok_or(SidecarError::BytesOverflowed)?;
        let entries = tags.id_limit().as_usize();
        let buffer = Buffer::new(entries, padded, Align::_128)?;
        let mirror = Mirror {
            slots: std::iter::repeat_with(|| AtomicU64::new(0))
                .take(entries.div_ceil(64))
                .collect(),
        };
        Ok(Sidecar {
            buffer,
            bytes: self.bytes,
            handle,
            mirror,
        })
    }
}

#[derive(Debug)]
struct Mirror {
    slots: Box<[AtomicU64]>,
}

impl Mirror {
    fn word(&self, i: usize) -> &AtomicU64 {
        &self.slots[i / 64]
    }

    fn mask(i: usize) -> u64 {
        1u64 << (i % 64)
    }

    #[inline]
    fn can_read(&self, i: usize) -> bool {
        self.word(i).load(Ordering::Acquire) & Self::mask(i) != 0
    }

    fn publish(&self, i: usize) {
        // Other slots share this word. RMWs preserve their bits and the release
        // sequences synchronizing readers with each slot's payload writes.
        self.word(i).fetch_or(Self::mask(i), Ordering::Release);
    }

    fn retire(&self, i: usize) {
        self.word(i).fetch_and(!Self::mask(i), Ordering::Relaxed);
    }

    #[inline]
    fn prefetch(&self, i: usize) {
        let ptr = std::ptr::from_ref(self.word(i)).cast::<i8>();
        // SAFETY: The pointer addresses a live atomic word. No reference to its
        // non-atomic contents is materialized.
        unsafe { diskann_vector::prefetch_cache_line(ptr) };
    }
}

#[derive(Debug)]
pub(crate) struct Sidecar {
    buffer: Buffer,
    bytes: Bytes,
    handle: epoch::RegistryHandle,
    mirror: Mirror,
}

impl Sidecar {
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
pub(crate) enum SidecarError {
    #[error("computation of the bytes per slot overflowed")]
    BytesOverflowed,
    #[error(transparent)]
    Buffer(#[from] BufferError),
}

impl slots::Slots for Sidecar {
    type Exclusive<'a> = Exclusive<'a>;

    fn id_limit(&self) -> IdLimit {
        self.id_limit()
    }

    unsafe fn acquire(&self, i: u32, _: Lifecycle) -> Exclusive<'_> {
        let i = i.into_usize();
        assert!(i < self.buffer.len(), "slot out-of-bounds");
        debug_assert!(!self.mirror.can_read(i));
        Exclusive {
            // SAFETY: Bounds were checked; the caller guarantees exclusivity.
            data: unsafe { self.buffer.get_unchecked(i) }.truncate(self.bytes),
            mirror: &self.mirror,
            i,
        }
    }

    unsafe fn retire(&self, i: u32, _: Lifecycle) {
        assert!(i.into_usize() < self.buffer.len(), "slot out-of-bounds");
        self.mirror.retire(i.into_usize());
    }

    unsafe fn reclaim(&self, i: u32, _: Lifecycle) {
        assert!(i.into_usize() < self.buffer.len(), "slot out-of-bounds");
        debug_assert!(!self.mirror.can_read(i.into_usize()));
    }
}

#[derive(Debug)]
pub(crate) struct Reader<'a> {
    store: &'a Sidecar,
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
        if self.store.mirror.can_read(i) {
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
    pub(crate) unsafe fn prefetch_bitmap(&self, i: usize) {
        debug_assert!(self.is_in_bounds(i));
        self.store.mirror.prefetch(i);
    }
}

#[derive(Debug)]
pub(crate) struct Exclusive<'a> {
    data: RawSlice<'a>,
    mirror: &'a Mirror,
    i: usize,
}

impl Exclusive<'_> {
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: The Slots caller guarantees exclusivity until a terminal hook.
        unsafe { self.data.as_mut_slice() }
    }
}

impl slots::Exclusive for Exclusive<'_> {
    fn publish(self, _: Lifecycle) {
        self.mirror.publish(self.i);
    }

    fn freeze(self, _: Lifecycle) {
        self.mirror.publish(self.i);
    }

    fn abort(self, _: Lifecycle) {
        debug_assert!(!self.mirror.can_read(self.i));
    }
}

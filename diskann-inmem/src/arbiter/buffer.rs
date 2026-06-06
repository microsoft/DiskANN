/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{alloc::Layout, marker::PhantomData, ptr::NonNull, sync::atomic::AtomicU64};

use crate::num::{Align, Bytes};

#[derive(Debug)]
pub struct Buffer {
    ptr: NonNull<u8>,
    stride: Bytes,
    entries: usize,
    layout: Layout,
}

impl Buffer {
    pub fn new(entries: usize, bytes_per_entry: Bytes, align: Align) -> Self {
        let size = bytes_per_entry.0.checked_mul(entries).unwrap();
        let layout = std::alloc::Layout::from_size_align(size, align.0).unwrap();

        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = match NonNull::new(ptr) {
            Some(ptr) => ptr,
            None => std::alloc::handle_alloc_error(layout),
        };

        Self {
            ptr,
            stride: bytes_per_entry,
            entries,
            layout,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.entries
    }

    #[inline]
    pub fn stride(&self) -> Bytes {
        self.stride
    }

    #[inline]
    pub fn align(&self) -> Align {
        Align(self.layout.align())
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Issue prefetch hints for the entry at index `i`.
    ///
    /// `bytes` controls how many bytes to prefetch (clamped to `stride`).
    /// Uses `wrapping_add` to avoid UB on out-of-bounds indices — prefetching
    /// a bad address is architecturally harmless.
    #[inline(always)]
    pub fn prefetch(&self, i: usize, bytes: usize) {
        let offset = self.stride.0.wrapping_mul(i);
        let ptr = self.ptr.as_ptr().wrapping_add(offset);
        let bytes = bytes.min(self.stride.0);
        prefetch_cachelines(ptr, bytes);
    }

    #[inline]
    pub fn get(&self, i: usize) -> Option<Slice<'_>> {
        if i >= self.entries {
            None
        } else {
            // SAFETY: We have validated that `i < self.entries`. This does two things:
            //
            // 1. Ensure that the multiplication will not overflow.
            // 2. Ensures that the computed offset is within the original allocation.
            Some(unsafe { self.get_unchecked(i) })
        }
    }

    /// Get the slice for entry `i` without bounds checking.
    ///
    /// # Safety
    ///
    /// `i` must be less than [`len`](Self::len).
    #[inline]
    pub unsafe fn get_unchecked(&self, i: usize) -> Slice<'_> {
        debug_assert!(i < self.entries);
        let ptr = unsafe { self.ptr.add(self.stride.0 * i) };
        Slice {
            ptr,
            len: self.stride.0,
            _lifetime: PhantomData,
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: This is the same pointer and allocation that was previously returned
        // from a successful `alloc_zeroed`.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) }
    }
}

// SAFETY: We're safe to pass around the `Buffer`. It's just use of the returned `Slice`
// the needs to be arbitrated.
unsafe impl Send for Buffer {}

// SAFETY: We're safe to pass around the `Buffer`. It's just use of the returned `Slice`
// the needs to be arbitrated.
unsafe impl Sync for Buffer {}

#[derive(Debug, Clone, Copy)]
pub struct Slice<'a> {
    ptr: NonNull<u8>,
    len: usize,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a> Slice<'a> {
    unsafe fn new(ptr: NonNull<u8>, len: usize) -> Self {
        Self {
            ptr,
            len,
            _lifetime: PhantomData,
        }
    }

    #[inline]
    pub fn truncate(&self, n: usize) -> Slice<'a> {
        unsafe { Self::new(self.ptr, self.len.min(n)) }
    }

    #[inline]
    pub fn skip(&self, n: usize) -> Slice<'a> {
        let advance_by = self.len.min(n);
        unsafe { Self::new(self.ptr.add(advance_by), self.len - advance_by) }
    }

    #[inline]
    pub fn split(&self, n: usize) -> (Slice<'a>, Slice<'a>) {
        let n = self.len.min(n);
        unsafe {
            (
                Self::new(self.ptr, n),
                Self::new(self.ptr.add(n), self.len - n),
            )
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_ptr(&self) -> NonNull<u8> {
        self.ptr
    }

    #[inline]
    pub unsafe fn as_slice(&self) -> &'a [u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub unsafe fn as_mut_slice(&mut self) -> &'a mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

/// Issue prefetch hints for `bytes` starting at `ptr`.
///
/// This is purely a performance hint and cannot cause undefined behavior,
/// even if `ptr` is invalid or out of bounds.
#[inline(always)]
pub fn prefetch_cachelines(ptr: *const u8, bytes: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        let lines = bytes.div_ceil(64);
        for i in 0..lines {
            // SAFETY: _mm_prefetch is a hint; invalid addresses are silently ignored.
            unsafe { _mm_prefetch(ptr.wrapping_add(i * 64) as *const i8, _MM_HINT_T0) };
        }
    }
}

/// Issue a prefetch hint for a single generation tag.
#[inline(always)]
pub fn prefetch_tag(tag: &AtomicU64) {
    prefetch_cachelines(tag as *const AtomicU64 as *const u8, 8);
}

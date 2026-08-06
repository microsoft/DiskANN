// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! POC-local single-threaded resettable bump arena for the staged quantized
//! kernel.
//!
//! Unlike [`BumpAllocator`](crate::alloc::BumpAllocator) — a `Clone`/`Send`/`Sync`
//! grow-only arena reclaimed only when its last clone drops — this is a
//! *single-owner, single-threaded* arena with O(1)
//! [`reset`](ResettableArena::reset). It is owned by one
//! [`QuantStagedQuery`](super::i8::QuantStagedQuery) and reused across
//! `compute_max_sim` calls, so the staged driver's per-call `partial` / `scored`
//! scratch allocates from it instead of the global heap (zero heap traffic in
//! steady state).
//!
//! `reset` is sound *because* the arena is not shareable: it takes `&mut self`,
//! and the borrow checker therefore guarantees no
//! [`ScopedAllocator`](crate::alloc::ScopedAllocator) borrowing this arena — hence
//! no outstanding allocation — is alive at the rewind point. `deallocate` is a
//! no-op; storage is reclaimed wholesale by `reset` or when the arena drops.

use std::cell::{Cell, UnsafeCell};
use std::ptr::NonNull;

use crate::alloc::{AlignedAllocator, AllocatorCore, AllocatorError, Poly};

/// A single-owner, single-threaded resettable bump arena over an owned,
/// 64-byte-aligned byte buffer. Hands out aligned sub-slices by bumping a
/// non-atomic `head`; [`reset`](Self::reset) rewinds it.
pub(crate) struct ResettableArena {
    /// Backing storage. `UnsafeCell` legitimizes handing out `*mut` ranges while
    /// `allocate` holds only `&self` (mirrors `BumpAllocator`'s buffer).
    buffer: Poly<UnsafeCell<[u8]>, AlignedAllocator>,
    /// Bump cursor (non-atomic — this arena is single-threaded).
    head: Cell<usize>,
}

impl ResettableArena {
    /// Allocate a fresh arena with room for `capacity` bytes, base-aligned to 64.
    pub(crate) fn with_capacity(capacity: usize) -> Result<Self, AllocatorError> {
        let buffer = Poly::<[u8], _>::new_uninit_slice(capacity.max(1), AlignedAllocator::A64)?;
        let (ptr, alloc) = Poly::into_raw(buffer);

        // SAFETY: `UnsafeCell<[u8]>` shares the layout of `[u8]`, and `MaybeUninit<u8>`
        // is layout-compatible with `u8` (`u8` is valid for any bit pattern, and bytes
        // are only read after the allocator hands them out and the caller writes them).
        // `ptr` is non-null, having come from `Poly::into_raw`.
        let buffer = unsafe {
            Poly::from_raw(
                NonNull::new_unchecked(ptr.as_ptr() as *mut UnsafeCell<[u8]>),
                alloc,
            )
        };

        Ok(Self {
            buffer,
            head: Cell::new(0),
        })
    }

    /// Rewind the arena, freeing every prior allocation in O(1).
    ///
    /// The `&mut self` receiver is load-bearing: it makes the borrow checker
    /// forbid calling `reset` while any [`ScopedAllocator`](crate::alloc::ScopedAllocator)
    /// borrowing this arena (and hence any live allocation) exists — which is what
    /// makes rewinding the cursor sound.
    pub(crate) fn reset(&mut self) {
        self.head.set(0);
    }

    /// Total capacity in bytes.
    fn capacity(&self) -> usize {
        self.buffer.get().len()
    }

    /// Base pointer of the backing buffer.
    fn base(&self) -> *mut u8 {
        self.buffer.get().cast::<u8>()
    }
}

// SAFETY: on success `allocate` returns a slice of exactly `layout.size()` bytes
// whose base is aligned to at least `layout.align()` (the running offset is aligned
// up to `layout.align()` relative to the real base address); a request that cannot
// fit the fixed capacity returns an error. `deallocate` is a no-op — storage is
// reclaimed only by `reset` or on drop.
unsafe impl AllocatorCore for ResettableArena {
    fn allocate(&self, layout: std::alloc::Layout) -> Result<NonNull<[u8]>, AllocatorError> {
        let base = self.base() as usize;
        let head = self.head.get();
        // Align the current free address up to `layout.align()`, then reserve `size`.
        let cur = base.checked_add(head).ok_or(AllocatorError)?;
        let aligned = cur
            .checked_next_multiple_of(layout.align())
            .ok_or(AllocatorError)?;
        let pad = aligned - cur;
        let new_head = head
            .checked_add(pad)
            .and_then(|h| h.checked_add(layout.size()))
            .ok_or(AllocatorError)?;
        if new_head > self.capacity() {
            return Err(AllocatorError);
        }
        self.head.set(new_head);

        // SAFETY: `head + pad <= new_head <= capacity`, so the offset is in-bounds of
        // the backing buffer and the range `[off, off + size)` lies within it.
        let ptr = unsafe { self.base().add(head + pad) };
        NonNull::new(std::ptr::slice_from_raw_parts_mut(ptr, layout.size())).ok_or(AllocatorError)
    }

    unsafe fn deallocate(&self, _ptr: NonNull<[u8]>, _layout: std::alloc::Layout) {}
}

impl std::fmt::Debug for ResettableArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResettableArena")
            .field("capacity", &self.capacity())
            .field("head", &self.head.get())
            .finish()
    }
}

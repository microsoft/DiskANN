// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Single-owner resettable bump arena (copy of `staged::arena`, kept local so the
//! `tiler` experiment doesn't reach into its sibling).

use std::cell::{Cell, UnsafeCell};
use std::ptr::NonNull;

use crate::alloc::{AlignedAllocator, AllocatorCore, AllocatorError, Poly};

/// Single-owner, single-threaded resettable bump over a 64-byte-aligned buffer.
pub(crate) struct ResettableArena {
    buffer: Poly<UnsafeCell<[u8]>, AlignedAllocator>,
    head: Cell<usize>,
}

impl ResettableArena {
    pub(crate) fn with_capacity(capacity: usize) -> Result<Self, AllocatorError> {
        let buffer = Poly::<[u8], _>::new_uninit_slice(capacity.max(1), AlignedAllocator::A64)?;
        let (ptr, alloc) = Poly::into_raw(buffer);

        // SAFETY: `UnsafeCell<[u8]>` shares the layout of `[u8]`; `MaybeUninit<u8>` is
        // layout-compatible with `u8` (bytes are only read after being handed out and
        // written). `ptr` is non-null, from `Poly::into_raw`.
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

    /// Rewind in O(1). `&mut self` makes the borrow checker forbid resetting while any
    /// [`ScopedAllocator`](crate::alloc::ScopedAllocator) borrowing this arena is live.
    pub(crate) fn reset(&mut self) {
        self.head.set(0);
    }

    fn capacity(&self) -> usize {
        self.buffer.get().len()
    }

    fn base(&self) -> *mut u8 {
        self.buffer.get().cast::<u8>()
    }
}

// SAFETY: `allocate` returns exactly `layout.size()` bytes aligned to `layout.align()`
// within the fixed buffer, or errors. `deallocate` is a no-op — storage is reclaimed by
// `reset` or on drop.
unsafe impl AllocatorCore for ResettableArena {
    fn allocate(&self, layout: std::alloc::Layout) -> Result<NonNull<[u8]>, AllocatorError> {
        let base = self.base() as usize;
        let head = self.head.get();
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

        // SAFETY: `head + pad <= new_head <= capacity`, so the range is in-bounds.
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

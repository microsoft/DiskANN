/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{alloc::Layout, marker::PhantomData, ptr::NonNull};

#[derive(Debug)]
pub struct Buffer {
    ptr: NonNull<u8>,
    stride: usize,
    entries: usize,
    layout: Layout,
}

impl Buffer {
    pub fn new(bytes_per_entry: usize, entries: usize) -> Self {
        let size = bytes_per_entry.checked_mul(entries).unwrap();
        let layout = std::alloc::Layout::from_size_align(size, 128).unwrap();

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

    pub fn len(&self) -> usize {
        self.entries
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, i: usize) -> Option<Slice<'_>> {
        if i >= self.entries {
            None
        } else {
            // SAFETY: We have validated that `i < self.entries`. This does two things:
            //
            // 1. Ensure that the multiplication will not overflow.
            // 2. Ensures that the computed offset is within the original allocation.
            let ptr = unsafe { self.ptr.add(self.stride * i) };

            Some(Slice {
                ptr,
                len: self.stride,
                _lifetime: PhantomData,
            })
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
    pub fn truncate(&self, n: usize) -> Slice<'a> {
        Slice {
            ptr: self.ptr,
            len: self.len.min(n),
            _lifetime: PhantomData,
        }
    }

    pub fn skip(&self, n: usize) -> Slice<'a> {
        let advance_by = self.len.min(n);
        Slice {
            ptr: unsafe { self.ptr.add(advance_by) },
            len: self.len - advance_by,
            _lifetime: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub unsafe fn as_slice(&self) -> &'a [u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub unsafe fn as_mut_slice(&mut self) -> &'a mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

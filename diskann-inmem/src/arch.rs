/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::num::Bytes;

/// Prefetch `len` bytes beginning at `ptr`.
///
/// The last cache line prefetched first, followed by the rest in ascending order.
///
/// # Safety
///
/// The memory range `[ptr, ptr.add(len))` must be valid.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn prefetch(ptr: *const u8, len: usize) {
    use std::arch::x86_64::*;

    // Fetch the last cache line (the one with the tag) first.
    let stride = Bytes::CACHELINE.value();
    let ptr = ptr.cast::<i8>();
    let lines = len.div_ceil(stride);
    if lines == 0 {
        return;
    }

    // SAFETY: Inherited from caller.
    unsafe { _mm_prefetch(ptr.add(stride * (lines - 1)), _MM_HINT_T0) };
    for i in 0..(lines - 1) {
        // SAFETY: Inherited from caller.
        unsafe {
            _mm_prefetch(ptr.add(stride * i), _MM_HINT_T0);
        }
    }
}

/// Prefetch `len` bytes beginning at `ptr`.
///
/// The last cache line prefetched first, followed by the rest in ascending order.
///
/// # Safety
///
/// The memory range `[ptr, ptr.add(len))` must be valid.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
pub(crate) unsafe fn prefetch(_ptr: *const u8, _len: usize) {}

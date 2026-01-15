/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// x86 intrinsics
use std::arch::x86_64::*;

use super::{V3, v3::i32x4};
use crate::SIMDVector;

/// Perform a load of the first `first` bytes beginning at `ptr` into
/// an unsigned 128-bit integer.
///
/// For clarity, the first byte beginning at `ptr` will occupy the lowest 8-bits of the
/// returned integer. The next byte will occupy bits 8 to 15 etc.
///
/// Memory addresses equal to and above `ptr + first` will not be accessed.
///
/// Note: This is actually faster than memcpy (since memcpy cannot be inlined).
///
/// # Safety
///
/// The memory addresses in  `[ptr, ptr + first)` must all be readable and valid.
///
/// Guarantee: Memory addresses in `[ptr + first, ptr + 16)` will not be accessed.
#[inline(always)]
pub(crate) unsafe fn __load_first_of_16_bytes(_: V3, mut ptr: *const u8, first: usize) -> u128 {
    let mut remaining = first;
    if remaining >= 16 {
        // SAFETY:
        // * Pointer Cast: The instruction `_mm_loadu_si128` does not have any alignment
        //     restrictions, so if `[ptr, ptr + first)` is valid, the cast will be valid.
        // * `_mm_loadu_si128`: Use of the intrinsic is gated by the `cfg` macro.
        //     The load is valid since the caller passed a value greater than 16.
        // *`__m128i` and `u128` are both the same size, do not own any resources, and are
        //     valid for all bit patterns.
        return unsafe {
            std::mem::transmute::<__m128i, u128>(_mm_loadu_si128(ptr as *const __m128i))
        };
    }

    // Move the pointer to one-past the end of the memory we are going to load.
    //
    // SAFETY: The caller asserts that the memory in `[ptr, ptr + first)` is valid.
    ptr = unsafe { ptr.add(first) };

    let mut buffer: u128 = 0;

    // SAFETY: We emit in-bounds unaligned reads that are in the range specified by the
    // caller to be safe.
    unsafe {
        if remaining >= 8 {
            ptr = ptr.sub(8);
            let v: u64 = std::ptr::read_unaligned(ptr as *const u64);
            buffer |= v as u128;
            remaining -= 8;
        }
        if remaining >= 4 {
            ptr = ptr.sub(4);
            let v: u32 = std::ptr::read_unaligned(ptr as *const u32);
            buffer = (buffer << (8 * std::mem::size_of::<u32>())) | (v as u128);
            remaining -= 4;
        }
        if remaining >= 2 {
            ptr = ptr.sub(2);
            let v: u16 = std::ptr::read_unaligned(ptr as *const u16);
            buffer = (buffer << (8 * std::mem::size_of::<u16>())) | (v as u128);
            remaining -= 2;
        }
        if remaining >= 1 {
            ptr = ptr.sub(1);
            let v: u8 = std::ptr::read(ptr);
            buffer = (buffer << 8) | (v as u128);
        }
    }
    buffer
}

/// Load the first `first` 16-bit words from `ptr` and return the result as a `__m128i`.
///
/// # Safety
///
/// The memory addresses in `[ptr, ptr + first)` must all be readable and valid.
///
/// This function guarantees that the memory addresses in `[ptr + first, ptr + 16)` will not
/// be accessed.
#[inline(always)]
pub(crate) unsafe fn __load_first_u16_of_16_bytes(
    arch: V3,
    ptr: *const u16,
    first: usize,
) -> __m128i {
    if first >= 8 {
        // SAFETY: All lanes are readable. The intrinsic can be used because `arch` is present.
        return unsafe { _mm_loadu_si128(ptr as *const __m128i) };
    }

    // Strategy: Use a masked load to load elements at 4-byte granularities.
    // Then, we have at most one 2-byte load left.
    //
    // We use `_mm_insert_epi16` to insert the last element.
    //
    // SAFETY: The reads emitted are in the range `[ptr, ptr + first)` asserted by the caller
    // to be safe. The use of the intrinsic is safe by the presence of `arch`.
    unsafe {
        let mut reg = i32x4::load_simd_first(arch, ptr as *const i32, first / 2).to_underlying();
        if first == 1 {
            reg = _mm_insert_epi16::<0>(reg, std::ptr::read_unaligned(ptr.add(first - 1)).into());
        } else if first == 3 {
            reg = _mm_insert_epi16::<2>(reg, std::ptr::read_unaligned(ptr.add(first - 1)).into());
        } else if first == 5 {
            reg = _mm_insert_epi16::<4>(reg, std::ptr::read_unaligned(ptr.add(first - 1)).into());
        } else if first == 7 {
            reg = _mm_insert_epi16::<6>(reg, std::ptr::read_unaligned(ptr.add(first - 1)).into());
        }
        reg
    }
}

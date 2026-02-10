/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// x86 intrinsics
use std::arch::x86_64::*;

use super::V3;

/// Efficiently load the first `8 < bytes < 16` bytes from `ptr` without accessing memory
/// outside of `[ptr, ptr + bytes)`.
///
/// # Safety
///
/// * `bytes` must be in the range `(8, 16)`.
/// * The memory in `[ptr, ptr + bytes)` must be readable and valid.
#[inline(always)]
unsafe fn __load_8_to_16_bytes(_: V3, ptr: *const u8, bytes: usize) -> __m128i {
    debug_assert!(bytes > 8 && bytes < 16);

    // An identity shuffle adjusted by subtracting the shift amount. Lanes that underflow
    // become negative (high bit set), which `pshufb` zeroes. Lanes beyond the loaded 8
    // bytes read from the zero-extended upper half of `_mm_loadl_epi64`, producing zeros
    // that are harmless under OR.
    //
    // SAFETY: Both reads are within `[ptr, ptr + bytes)`. The intrinsics require SSSE3/SSE2,
    // available on V3.
    unsafe {
        let base = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let lo = _mm_loadl_epi64(ptr as *const __m128i);
        let hi = _mm_loadl_epi64(ptr.add(bytes - 8) as *const __m128i);
        let mask = _mm_sub_epi8(base, _mm_set1_epi8((bytes - 8) as i8));
        _mm_or_si128(lo, _mm_shuffle_epi8(hi, mask))
    }
}

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
pub(crate) unsafe fn __load_first_of_16_bytes(arch: V3, ptr: *const u8, first: usize) -> u128 {
    if first >= 16 {
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

    // For first > 8, use two overlapping 8-byte loads combined with `pshufb`.
    if first > 8 {
        // SAFETY: `first` is in `(8, 16)` and `[ptr, ptr + first)` is valid.
        return unsafe {
            std::mem::transmute::<__m128i, u128>(__load_8_to_16_bytes(arch, ptr, first))
        };
    }

    // For first <= 8, everything fits in general purpose registers.
    //
    // Use two overlapping reads whose results are combined with a single shift + OR.
    //
    // SAFETY: All reads are within `[ptr, ptr + first)`, which the caller asserts is valid.
    unsafe {
        if first == 8 {
            std::ptr::read_unaligned(ptr as *const u64) as u128
        } else if first >= 4 {
            let lo = std::ptr::read_unaligned(ptr as *const u32) as u64;
            let hi = std::ptr::read_unaligned(ptr.add(first - 4) as *const u32) as u64;
            (lo | (hi << ((first - 4) * 8))) as u128
        } else if first >= 2 {
            let lo = std::ptr::read_unaligned(ptr as *const u16) as u64;
            let hi = std::ptr::read_unaligned(ptr.add(first - 2) as *const u16) as u64;
            (lo | (hi << ((first - 2) * 8))) as u128
        } else if first == 1 {
            std::ptr::read(ptr) as u128
        } else {
            0
        }
    }
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

    let byte_ptr = ptr as *const u8;
    let bytes = first * 2;

    // For bytes > 8 (i.e., first >= 5), use two overlapping 8-byte loads
    // combined with `pshufb`.
    if bytes > 8 {
        // SAFETY: `bytes` is in `(8, 16)` and `[byte_ptr, byte_ptr + bytes)` is valid.
        return unsafe { __load_8_to_16_bytes(arch, byte_ptr, bytes) };
    }

    // For bytes <= 8, everything fits in general purpose registers.
    //
    // SAFETY: All reads are within `[ptr, ptr + first)`, which the caller
    // asserts is valid.
    unsafe {
        if bytes == 8 {
            let v = std::ptr::read_unaligned(byte_ptr as *const u64);
            _mm_cvtsi64_si128(v as i64)
        } else if bytes >= 4 {
            let lo = std::ptr::read_unaligned(byte_ptr as *const u32) as u64;
            let hi = std::ptr::read_unaligned(byte_ptr.add(bytes - 4) as *const u32) as u64;
            _mm_cvtsi64_si128((lo | (hi << ((bytes - 4) * 8))) as i64)
        } else if bytes >= 2 {
            _mm_cvtsi32_si128(std::ptr::read_unaligned(byte_ptr as *const u16) as i32)
        } else {
            _mm_setzero_si128()
        }
    }
}

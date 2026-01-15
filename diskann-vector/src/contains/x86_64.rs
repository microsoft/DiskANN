/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::arch::x86_64::*;

/// Contains operation on a vector of u32 using SIMD.
/// Performance critical function. If you change this - please compare with the benchmark.
pub(crate) fn contains_simd_u32(vector: &[u32], target: u32) -> bool {
    // SAFETY: Just using intrinsics.
    let target_lane = unsafe { _mm256_set1_epi32(target as i32) };

    // Check in sizes of 32.
    let mut remaining = vector;
    while remaining.len() >= 32 {
        if compare_32_items_u32(remaining.as_ptr(), target_lane) {
            return true;
        }

        // SAFETY: We are checking the length before doing this.
        remaining = unsafe { remaining.get_unchecked(32..) };
        continue;
    }

    // Check in sizes of 16. Note use of if instead of while.
    if remaining.len() >= 16 {
        if compare_16_items_u32(remaining.as_ptr(), target_lane) {
            return true;
        }

        // SAFETY: We are checking the length before doing this.
        remaining = unsafe { remaining.get_unchecked(16..) };
    }

    // Check in sizes of 8.
    if remaining.len() >= 8 {
        if compare_8_items_u32(remaining.as_ptr(), target_lane) {
            return true;
        }

        // SAFETY: We are checking the length before doing this.
        remaining = unsafe { remaining.get_unchecked(8..) };
    }

    // Check in sizes of 4.
    if remaining.len() >= 4 {
        if compare_4_items_u32(remaining.as_ptr(), target) {
            return true;
        }

        // SAFETY: We are checking the length before doing this.
        remaining = unsafe { remaining.get_unchecked(4..) };
    }

    // Check the rest.
    for val in remaining {
        if *val == target {
            return true;
        }
    }

    false
}

#[inline(always)]
fn compare_32_items_u32(chunk: *const u32, target_lane: __m256i) -> bool {
    compare_16_items_u32(chunk, target_lane)
        || compare_16_items_u32(
            // SAFETY: We are checking the length before doing this.
            unsafe { chunk.add(16) },
            target_lane,
        )
}

#[inline(always)]
fn compare_16_items_u32(chunk: *const u32, target_lane: __m256i) -> bool {
    compare_8_items_u32(chunk, target_lane)
        || compare_8_items_u32(
            // SAFETY: We are checking the length before doing this.
            unsafe { chunk.add(8) },
            target_lane,
        )
}

#[inline(always)]
fn compare_8_items_u32(chunk: *const u32, target_lane: __m256i) -> bool {
    // SAFETY: Using intrinsics.
    unsafe {
        let current_block = _mm256_loadu_si256(chunk as *const __m256i);
        let comparison = _mm256_cmpeq_epi32(current_block, target_lane);

        // Cast and compare - small latency win.
        // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=3693,3693,4640,4640,707,4640,4635&text=_mm256_movemask&techs=AVX_ALL
        _mm256_movemask_ps(_mm256_castsi256_ps(comparison)) != 0
    }
}

#[inline(always)]
fn compare_4_items_u32(chunk: *const u32, target: u32) -> bool {
    // SAFETY: Using intrinsics.
    unsafe {
        let current_block = _mm_loadu_si128(chunk as *const __m128i);
        let target_lane = _mm_set1_epi32(target as i32);
        let comparison = _mm_cmpeq_epi32(current_block, target_lane);
        _mm_movemask_ps(_mm_castsi128_ps(comparison)) != 0
    }
}

/// Contains operation on a vector of u64 using SIMD.
/// Performance critical function. If you change this - please compare with the benchmark.
pub(crate) fn contains_simd_u64(vector: &[u64], target: u64) -> bool {
    // SAFETY: Just using intrinsics.
    let target_lane = unsafe { _mm256_set1_epi64x(target as i64) };
    // Check in sizes of 32.
    let mut remaining = vector;
    while remaining.len() >= 32 {
        if compare_32_items_u64(remaining.as_ptr(), target_lane) {
            return true;
        }
        // SAFETY: We are checking the length before doing this.
        remaining = unsafe { remaining.get_unchecked(32..) };
        continue;
    }
    // Check in sizes of 16. Note use of if instead of while.
    if remaining.len() >= 16 {
        if compare_16_items_u64(remaining.as_ptr(), target_lane) {
            return true;
        }
        // SAFETY: We are checking the length before doing this.
        remaining = unsafe { remaining.get_unchecked(16..) };
    }
    // Check in sizes of 8.
    if remaining.len() >= 8 {
        if compare_8_items_u64(remaining.as_ptr(), target_lane) {
            return true;
        }
        // SAFETY: We are checking the length before doing this.
        remaining = unsafe { remaining.get_unchecked(8..) };
    }
    // Check in sizes of 4.
    if remaining.len() >= 4 {
        if compare_4_items_u64(remaining.as_ptr(), target_lane) {
            return true;
        }
        // SAFETY: We are checking the length before doing this.
        remaining = unsafe { remaining.get_unchecked(4..) };
    }
    // Check in sizes of 2.
    if remaining.len() >= 2 {
        if compare_2_items_u64(remaining.as_ptr(), target) {
            return true;
        }
        // SAFETY: We are checking the length before doing this.
        remaining = unsafe { remaining.get_unchecked(2..) };
    }
    // Check the rest.
    for val in remaining {
        if *val == target {
            return true;
        }
    }
    false
}
#[inline(always)]
fn compare_32_items_u64(chunk: *const u64, target_lane: __m256i) -> bool {
    compare_16_items_u64(chunk, target_lane)
        || compare_16_items_u64(
            // SAFETY: We are checking the length before doing this.
            unsafe { chunk.add(16) },
            target_lane,
        )
}
#[inline(always)]
fn compare_16_items_u64(chunk: *const u64, target_lane: __m256i) -> bool {
    compare_8_items_u64(chunk, target_lane)
        || compare_8_items_u64(
            // SAFETY: We are checking the length before doing this.
            unsafe { chunk.add(8) },
            target_lane,
        )
}
#[inline(always)]
fn compare_8_items_u64(chunk: *const u64, target_lane: __m256i) -> bool {
    compare_4_items_u64(chunk, target_lane)
        || compare_4_items_u64(
            // SAFETY: We are checking the length before doing this.
            unsafe { chunk.add(4) },
            target_lane,
        )
}
#[inline(always)]
fn compare_4_items_u64(chunk: *const u64, target_lane: __m256i) -> bool {
    // SAFETY: Using intrinsics.
    unsafe {
        let current_block = _mm256_loadu_si256(chunk as *const __m256i);
        let comparison = _mm256_cmpeq_epi64(current_block, target_lane);
        // Cast and compare - small latency win.
        // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=3693,3693,4640,4640,707,4640,4635&text=_mm256_movemask&techs=AVX_ALL
        _mm256_movemask_pd(_mm256_castsi256_pd(comparison)) != 0
    }
}
#[inline(always)]
fn compare_2_items_u64(chunk: *const u64, target: u64) -> bool {
    // SAFETY: Using intrinsics.
    unsafe {
        let current_block = _mm_loadu_si128(chunk as *const __m128i);
        let target_lane = _mm_set1_epi64x(target as i64);
        let comparison = _mm_cmpeq_epi64(current_block, target_lane);
        _mm_movemask_pd(_mm_castsi128_pd(comparison)) != 0
    }
}

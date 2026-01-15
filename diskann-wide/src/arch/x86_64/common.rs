/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// x86 intrinsics
use std::arch::x86_64::*;

/// A trait to get a SIMD intrinsic completely full of 1's
pub(crate) trait AllOnes {
    fn all_ones() -> Self;
}

impl AllOnes for __m128i {
    fn all_ones() -> Self {
        // SAFETY: Gated by CFG
        unsafe { _mm_set1_epi32(-1) }
    }
}

impl AllOnes for __m256i {
    fn all_ones() -> Self {
        // SAFETY: Gated by CFG
        unsafe { _mm256_set1_epi32(-1) }
    }
}

impl AllOnes for __m512i {
    fn all_ones() -> Self {
        // SAFETY: Gated by CFG
        unsafe { _mm512_set1_epi32(-1) }
    }
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// x86 intrinsics
use std::arch::x86_64::*;

use half::f16;

use crate::{
    arch::x86_64::{
        V4,
        macros::{self, X86Default, X86LoadStore, X86Splat},
        v3,
        v4::f16x8_::f16x8,
    },
    bitmask::BitMask,
    constant::Const,
    traits::SIMDVector,
};

/////////////////////
// 16-bit floating //
/////////////////////

macros::x86_define_register!(f16x16, __m256i, BitMask<16, V4>, f16, 16, V4);
macros::x86_define_default!(f16x16, _mm256_setzero_si256, "avx");
macros::x86_retarget!(f16x16 => v3::f16x16);
macros::x86_splitjoin!(
    f16x16,
    f16x8,
    _mm256_extracti128_si256,
    _mm256_set_m128i,
    "avx2"
);

impl X86Splat for f16x16 {
    #[inline(always)]
    fn x86_splat(_: V4, value: f16) -> Self {
        // Unpacking the conversion sequence:
        //
        // (1) .to_bits() -> Returns the underlying `u16` from the `f16`.
        // (2) as i16 -> Bit-cast to `i16` to give to the intrinsic.
        //
        // SAFETY: `_mm256_set1_epi16` requires AVX - implied by V4.
        Self(unsafe { _mm256_set1_epi16(value.to_bits() as i16) })
    }
}

impl X86LoadStore for f16x16 {
    #[inline(always)]
    unsafe fn load_simd(_: V4, ptr: *const f16) -> Self {
        // SAFETY: Pointer access guaranteed by caller.
        //
        // `_mm256_loadu_si256` requires AVX - implied by V4.
        Self(unsafe { _mm256_loadu_si256(ptr as *const Self::Underlying) })
    }

    #[inline(always)]
    unsafe fn load_simd_masked_logical(_: V4, ptr: *const f16, mask: Self::Mask) -> Self {
        // SAFETY: Pointer access guaranteed by caller.
        //
        // `_mm256_maskz_loadu_epi16` requires AVX512BW + AVX512VL - implied by V4.
        Self(unsafe { _mm256_maskz_loadu_epi16(mask.0, ptr.cast()) })
    }

    #[inline(always)]
    unsafe fn store_simd(self, ptr: *mut f16) {
        // SAFETY: Pointer access guaranteed by caller.
        //
        // `_mm256_storeu_si256` requires AVX - implied by V4.
        unsafe { _mm256_storeu_si256(ptr as *mut Self::Underlying, self.to_underlying()) }
    }

    #[inline(always)]
    unsafe fn store_simd_masked_logical(self, ptr: *mut f16, mask: Self::Mask) {
        // SAFETY: Pointer access guaranteed by caller.
        //
        // `_mm256_mask_storeu_epi16` requires AVX512BW + AVX512VL - implied by V4.
        unsafe { _mm256_mask_storeu_epi16(ptr.cast(), mask.0, self.0) }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_x86_f16 {
    use super::*;
    use crate::test_utils;

    // Miri currently does not understand the `cvtph2ps` function.
    // We need to supply a Miri-friendly version of this instruction to get coverage.
    #[test]
    fn test_load() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_load_simd::<f16, 16, f16x16>(arch);
        }
    }

    #[test]
    fn test_store() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_store_simd::<f16, 16, f16x16>(arch);
        }
    }

    // constructors
    #[cfg(not(miri))]
    #[test]
    fn test_constructors() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::ops::test_splat::<f16, 16, f16x16>(arch);
        }
    }

    test_utils::ops::test_splitjoin!(f16x16 => f16x8, 0xa4d00a4d04293967, V4::new_checked_uncached());
}

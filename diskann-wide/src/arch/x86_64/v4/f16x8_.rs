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
    },
    bitmask::BitMask,
    constant::Const,
    traits::SIMDVector,
};

/////////////////////
// 16-bit floating //
/////////////////////

macros::x86_define_register!(f16x8, __m128i, BitMask<8, V4>, f16, 8, V4);
macros::x86_define_default!(f16x8, _mm_setzero_si128, "sse2");
macros::x86_retarget!(f16x8 => v3::f16x8);

impl X86Splat for f16x8 {
    #[inline(always)]
    fn x86_splat(_: V4, value: f16) -> Self {
        // Unpacking the conversion sequence:
        //
        // (1) .to_bits() -> Returns the underlying `u16` from the `f16`.
        // (2) as i16 -> Bit-cast to `i16` to give to the intrinsic.
        //
        // SAFETY: `_mm_set1_epi16` requires SSE2 - implied by V4.
        Self(unsafe { _mm_set1_epi16(value.to_bits() as i16) })
    }
}

impl X86LoadStore for f16x8 {
    #[inline(always)]
    unsafe fn load_simd(_: V4, ptr: *const f16) -> Self {
        // SAFETY: Pointer access guaranteed by caller.
        //
        // `_mm_loadu_si128` requires SSE2 - implied by V4.
        Self(unsafe { _mm_loadu_si128(ptr as *const Self::Underlying) })
    }

    #[inline(always)]
    unsafe fn load_simd_masked_logical(_: V4, ptr: *const f16, mask: Self::Mask) -> Self {
        // SAFETY: Pointer access guaranteed by caller.
        //
        // `_mm_maskz_loadu_epi16` requires AVX512BW + AVX512VL - implied by V4.
        Self(unsafe { _mm_maskz_loadu_epi16(mask.0, ptr.cast()) })
    }

    #[inline(always)]
    unsafe fn store_simd(self, ptr: *mut f16) {
        // SAFETY: Pointer access guaranteed by caller.
        //
        // `_mm_storeu_si128` requires SSE2 - implied by V4.
        unsafe { _mm_storeu_si128(ptr as *mut Self::Underlying, self.to_underlying()) }
    }

    #[inline(always)]
    unsafe fn store_simd_masked_logical(self, ptr: *mut f16, mask: Self::Mask) {
        // SAFETY: Pointer access guaranteed by caller.
        //
        // `_mm_maskz_loadu_epi16` requires AVX512BW + AVX512VL - implied by V4.
        unsafe { _mm_mask_storeu_epi16(ptr.cast(), mask.0, self.0) }
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
            test_utils::test_load_simd::<f16, 8, f16x8>(arch);
        }
    }

    #[test]
    fn test_store() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_store_simd::<f16, 8, f16x8>(arch);
        }
    }

    // constructors
    #[cfg(not(miri))]
    #[test]
    fn test_constructors() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::ops::test_splat::<f16, 8, f16x8>(arch);
        }
    }
}

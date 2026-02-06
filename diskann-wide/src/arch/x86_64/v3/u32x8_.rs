/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// x86 intrinsics
use std::arch::x86_64::*;

use crate::{
    arch::x86_64::{
        V3,
        common::AllOnes,
        macros::{self, X86Default, X86LoadStore, X86Splat},
        v3::{masks::mask32x8, u32x4},
    },
    constant::Const,
    helpers,
    traits::{
        SIMDMask, SIMDMulAdd, SIMDPartialEq, SIMDPartialOrd, SIMDSelect, SIMDSumTree, SIMDVector,
    },
};

/////
///// 32-bit unsigned integer
/////

macros::x86_define_register!(u32x8, __m256i, mask32x8, u32, 8, V3);
macros::x86_define_splat!(u32x8 as i32, _mm256_set1_epi32, "avx");
macros::x86_define_default!(u32x8, _mm256_setzero_si256, "avx");
macros::x86_splitjoin!(
    u32x8,
    u32x4,
    _mm256_extracti128_si256,
    _mm256_set_m128i,
    "avx2"
);

helpers::unsafe_map_binary_op!(u32x8, std::ops::Add, add, _mm256_add_epi32, "avx2");
helpers::unsafe_map_binary_op!(u32x8, std::ops::Sub, sub, _mm256_sub_epi32, "avx2");
helpers::unsafe_map_binary_op!(u32x8, std::ops::Mul, mul, _mm256_mullo_epi32, "avx2");

helpers::unsafe_map_binary_op!(u32x8, std::ops::BitAnd, bitand, _mm256_and_si256, "avx2");
helpers::unsafe_map_binary_op!(u32x8, std::ops::BitOr, bitor, _mm256_or_si256, "avx2");
helpers::unsafe_map_binary_op!(u32x8, std::ops::BitXor, bitxor, _mm256_xor_si256, "avx2");
helpers::unsafe_map_binary_op!(u32x8, std::ops::Shr, shr, _mm256_srlv_epi32, "avx2");
helpers::unsafe_map_binary_op!(u32x8, std::ops::Shl, shl, _mm256_sllv_epi32, "avx2");
helpers::scalar_shift_by_splat!(u32x8, u32);

impl std::ops::Not for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        self ^ Self::from_underlying(self.arch(), <Self as SIMDVector>::Underlying::all_ones())
    }
}

impl SIMDMulAdd for u32x8 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        self * rhs + accumulator
    }
}

impl X86LoadStore for u32x8 {
    #[inline(always)]
    unsafe fn load_simd(_: V3, ptr: *const u32) -> Self {
        // SAFETY: The caller asserts this pointer access is safe and the presence of `V3`
        // means we can use this intrinsic.
        Self(unsafe { _mm256_loadu_si256(ptr as *const __m256i) })
    }

    #[inline(always)]
    unsafe fn load_simd_masked_logical(_: V3, ptr: *const u32, mask: Self::Mask) -> Self {
        // MIRI does not support `_mm256_maskload_epi32`.
        // So we go through a kind of convoluted dance to let this be tested by miri.
        //
        // SAFETY: The caller asserts this pointer access is safe and the presence of `V3`
        // means we can use these intrinsics.
        Self(unsafe {
            _mm256_castps_si256(_mm256_maskload_ps(ptr as *const f32, mask.to_underlying()))
        })
    }

    #[inline(always)]
    unsafe fn store_simd(self, ptr: *mut u32) {
        // SAFETY: The caller asserts this pointer access is safe and the presence of `Self`
        // (implicitly `V3`) means we can use this intrinsic.
        unsafe { _mm256_storeu_si256(ptr.cast::<__m256i>(), self.to_underlying()) }
    }

    #[inline(always)]
    unsafe fn store_simd_masked_logical(self, ptr: *mut u32, mask: Self::Mask) {
        // SAFETY: The caller asserts this pointer access is safe and the presence of `Self`
        // (implicitly `V3`) means we can use this intrinsic.
        unsafe {
            _mm256_maskstore_ps(
                ptr.cast::<f32>(),
                mask.to_underlying(),
                _mm256_castsi256_ps(self.to_underlying()),
            )
        };
    }
}

impl SIMDPartialEq for u32x8 {
    #[inline(always)]
    fn eq_simd(self, other: Self) -> Self::Mask {
        // SAFETY: Gated by CFG
        Self::Mask::from_underlying(self.arch(), unsafe { _mm256_cmpeq_epi32(self.0, other.0) })
    }

    #[inline(always)]
    fn ne_simd(self, other: Self) -> Self::Mask {
        // SAFETY: Gated by CFG
        let m =
            unsafe { _mm256_xor_si256(_mm256_cmpeq_epi32(self.0, other.0), __m256i::all_ones()) };
        Self::Mask::from_underlying(self.arch(), m)
    }
}

impl SIMDPartialOrd for u32x8 {
    #[inline(always)]
    fn lt_simd(self, other: Self) -> Self::Mask {
        // SAFETY: Gated by CFG
        let m = unsafe {
            let max = _mm256_max_epu32(self.0, other.0);
            _mm256_xor_si256(_mm256_cmpeq_epi32(self.0, max), __m256i::all_ones())
        };
        Self::Mask::from_underlying(self.arch(), m)
    }

    #[inline(always)]
    fn le_simd(self, other: Self) -> Self::Mask {
        // SAFETY: Gated by CFG
        let m = unsafe { _mm256_cmpeq_epi32(self.0, _mm256_min_epu32(self.0, other.0)) };
        Self::Mask::from_underlying(self.arch(), m)
    }
}

impl SIMDSumTree for u32x8 {
    #[inline(always)]
    fn sum_tree(self) -> u32 {
        let x = self.to_underlying();
        // SAFETY: Invoking intrinsics on value SIMD types without touching memory.
        unsafe {
            let hi_quad = _mm256_extracti128_si256(x, 0x1);
            let lo_quad = _mm256_castsi256_si128(x);
            let sum_quad = _mm_add_epi32(lo_quad, hi_quad);

            // Do a dance through the `ps` instructions.
            let lo_dual = sum_quad;
            let hi_dual = _mm_castps_si128(_mm_movehl_ps(
                _mm_castsi128_ps(sum_quad),
                _mm_castsi128_ps(sum_quad),
            ));
            let sum_dual = _mm_add_epi32(lo_dual, hi_dual);

            // Sum the last two elements.
            let lo = sum_dual;
            let hi = _mm_shuffle_epi32(sum_dual, 0x1);
            let sum = _mm_add_epi32(lo, hi);
            _mm_cvtsi128_si32(sum) as u32
        }
    }
}

impl SIMDSelect<u32x8> for mask32x8 {
    #[inline(always)]
    fn select(self, x: u32x8, y: u32x8) -> u32x8 {
        // SAFETY: Compilation of this trait implementation is predicated on the invoked
        // intrinsics being available at compile time.
        u32x8::from_underlying(self.arch(), unsafe {
            _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(y.to_underlying()),
                _mm256_castsi256_ps(x.to_underlying()),
                _mm256_castsi256_ps(self.to_underlying()),
            ))
        })
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_x86_u32 {
    use super::*;
    use crate::{reference::ReferenceScalarOps, test_utils};

    #[test]
    fn miri_test_load() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_load_simd::<u32, 8, u32x8>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_store_simd::<u32, 8, u32x8>(arch);
        }
    }

    // constructors
    #[cfg(not(miri))]
    #[test]
    fn test_constructors() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::ops::test_splat::<u32, 8, u32x8>(arch);
        }
    }

    // Binary Ops
    test_utils::ops::test_add!(u32x8, 0x63e8990efd20cef4, V3::new_checked_uncached());
    test_utils::ops::test_sub!(u32x8, 0xf3a68a9a6b747caa, V3::new_checked_uncached());
    test_utils::ops::test_mul!(u32x8, 0x7ab79dd0ec063c5f, V3::new_checked_uncached());
    test_utils::ops::test_fma!(u32x8, 0xea99d33ca3337639, V3::new_checked_uncached());

    test_utils::ops::test_cmp!(u32x8, 0xbc62480ada063710, V3::new_checked_uncached());
    test_utils::ops::test_splitjoin!(u32x8 => u32x4, 0xb151fcd6141b10c9, V3::new_checked_uncached());
    test_utils::ops::test_select!(u32x8, 0xfc34afd214cfb57e, V3::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(u32x8, 0xd8bf787696c2abfe, V3::new_checked_uncached());

    // Reductions
    test_utils::ops::test_sumtree!(u32x8, 0xd6780b08573e203b, V3::new_checked_uncached());
}

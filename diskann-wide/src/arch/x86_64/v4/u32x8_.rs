/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// x86 intrinsics
use std::arch::x86_64::*;

use crate::{
    BitMask,
    arch::x86_64::{
        V4,
        common::AllOnes,
        macros::{self, X86Default, X86LoadStore, X86Splat},
        v3,
        v4::u32x4_::u32x4, // direct import for Miri compat
    },
    constant::Const,
    helpers,
    traits::{SIMDMask, SIMDMulAdd, SIMDSelect, SIMDSumTree, SIMDVector},
};

/////
///// 32-bit floating point
/////

macros::x86_define_register!(u32x8, __m256i, BitMask<8, V4>, u32, 8, V4);
macros::x86_define_splat!(u32x8 as i32, _mm256_set1_epi32, "avx");
macros::x86_define_default!(u32x8, _mm256_setzero_si256, "avx");
macros::x86_retarget!(u32x8 => v3::u32x8);
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

macros::x86_avx512_load_store!(
    u32x8,
    _mm256_loadu_epi32,
    _mm256_maskz_loadu_epi32,
    _mm256_storeu_epi32,
    _mm256_mask_storeu_epi32,
    i32,
    "avx512f,avx512vl"
);

macros::x86_avx512_int_comparisons!(u32x8, _mm256_cmp_epu32_mask, "avx512f,avx512vl");

impl SIMDSumTree for u32x8 {
    #[inline(always)]
    fn sum_tree(self) -> u32 {
        self.retarget().sum_tree()
    }
}

impl SIMDSelect<u32x8> for BitMask<8, V4> {
    #[inline(always)]
    fn select(self, x: u32x8, y: u32x8) -> u32x8 {
        // SAFETY: `_mm256_mask_blend_epi32` requires AVX512F + AVX512VL - implied by V4
        u32x8::from_underlying(self.arch(), unsafe {
            _mm256_mask_blend_epi32(self.to_underlying(), y.to_underlying(), x.to_underlying())
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
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_load_simd::<u32, 8, u32x8>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_store_simd::<u32, 8, u32x8>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::ops::test_splat::<u32, 8, u32x8>(arch);
        }
    }

    // Binary Ops
    test_utils::ops::test_add!(u32x8, 0x63e8990efd20cef4, V4::new_checked_uncached());
    test_utils::ops::test_sub!(u32x8, 0xf3a68a9a6b747caa, V4::new_checked_uncached());
    test_utils::ops::test_mul!(u32x8, 0x7ab79dd0ec063c5f, V4::new_checked_uncached());
    test_utils::ops::test_fma!(u32x8, 0xea99d33ca3337639, V4::new_checked_uncached());

    test_utils::ops::test_cmp!(u32x8, 0xbc62480ada063710, V4::new_checked_uncached());
    test_utils::ops::test_splitjoin!(u32x8 => u32x4, 0xb151fcd6141b10c9, V4::new_checked_uncached());
    test_utils::ops::test_select!(u32x8, 0xf1da67c57b7324f7, V4::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(u32x8, 0x417f8adb857645a8, V4::new_checked_uncached());

    // Reductions
    test_utils::ops::test_sumtree!(u32x8, 0xd6780b08573e203b, V4::new_checked_uncached());
}

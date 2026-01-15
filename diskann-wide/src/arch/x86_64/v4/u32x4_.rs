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
    },
    constant::Const,
    helpers,
    traits::{SIMDMask, SIMDMulAdd, SIMDSelect, SIMDSumTree, SIMDVector},
};

/////
///// 32-bit floating point
/////

macros::x86_define_register!(u32x4, __m128i, BitMask<4, V4>, u32, 4, V4);
macros::x86_define_splat!(u32x4 as i32, _mm_set1_epi32, "sse2");
macros::x86_define_default!(u32x4, _mm_setzero_si128, "sse2");
macros::x86_retarget!(u32x4 => v3::u32x4);

helpers::unsafe_map_binary_op!(u32x4, std::ops::Add, add, _mm_add_epi32, "sse2");
helpers::unsafe_map_binary_op!(u32x4, std::ops::Sub, sub, _mm_sub_epi32, "sse2");
helpers::unsafe_map_binary_op!(u32x4, std::ops::Mul, mul, _mm_mullo_epi32, "sse4.1");

helpers::unsafe_map_binary_op!(u32x4, std::ops::BitAnd, bitand, _mm_and_si128, "sse2");
helpers::unsafe_map_binary_op!(u32x4, std::ops::BitOr, bitor, _mm_or_si128, "sse2");
helpers::unsafe_map_binary_op!(u32x4, std::ops::BitXor, bitxor, _mm_xor_si128, "sse2");
helpers::unsafe_map_binary_op!(u32x4, std::ops::Shr, shr, _mm_srlv_epi32, "avx2");
helpers::unsafe_map_binary_op!(u32x4, std::ops::Shl, shl, _mm_sllv_epi32, "avx2");
helpers::scalar_shift_by_splat!(u32x4, u32);

impl std::ops::Not for u32x4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        self ^ Self::from_underlying(self.arch(), <Self as SIMDVector>::Underlying::all_ones())
    }
}

impl SIMDMulAdd for u32x4 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        self * rhs + accumulator
    }
}

macros::x86_avx512_load_store!(
    u32x4,
    _mm_loadu_epi32,
    _mm_maskz_loadu_epi32,
    _mm_storeu_epi32,
    _mm_mask_storeu_epi32,
    i32,
    "avx512f,avx512vl"
);

macros::x86_avx512_int_comparisons!(u32x4, _mm_cmp_epu32_mask, "avx512f,avx512vl");

impl SIMDSumTree for u32x4 {
    #[inline(always)]
    fn sum_tree(self) -> u32 {
        self.retarget().sum_tree()
    }
}

impl SIMDSelect<u32x4> for BitMask<4, V4> {
    #[inline(always)]
    fn select(self, x: u32x4, y: u32x4) -> u32x4 {
        // SAFETY: `_mm_mask_blend_epi32` requires AVX512F + AVX512VL - implied by V4
        u32x4::from_underlying(self.arch(), unsafe {
            _mm_mask_blend_epi32(self.to_underlying(), y.to_underlying(), x.to_underlying())
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
            test_utils::test_load_simd::<u32, 4, u32x4>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_store_simd::<u32, 4, u32x4>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::ops::test_splat::<u32, 4, u32x4>(arch);
        }
    }

    // Binary Ops
    test_utils::ops::test_add!(u32x4, 0x8d7bf28b1c6e2545, V4::new_checked_uncached());
    test_utils::ops::test_sub!(u32x4, 0x4a1c644a1a910bed, V4::new_checked_uncached());
    test_utils::ops::test_mul!(u32x4, 0xf42ee707a808fd10, V4::new_checked_uncached());
    test_utils::ops::test_fma!(u32x4, 0x28540d9936a9e803, V4::new_checked_uncached());

    test_utils::ops::test_cmp!(u32x4, 0xfae27072c6b70885, V4::new_checked_uncached());
    test_utils::ops::test_select!(u32x4, 0xa9acc59495a3a1f7, V4::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(u32x4, 0xbe927713ea310164, V4::new_checked_uncached());

    // Reductions
    test_utils::ops::test_sumtree!(u32x4, 0xb9ac82ab23a855da, V4::new_checked_uncached());
}

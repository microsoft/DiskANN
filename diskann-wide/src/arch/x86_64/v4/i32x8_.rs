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
        v4::{i8x32_::i8x32, i16x16_::i16x16, i32x4_::i32x4, u8x32_::u8x32}, // direct import for Miri compat
    },
    constant::Const,
    helpers,
    traits::{SIMDAbs, SIMDDotProduct, SIMDMask, SIMDMulAdd, SIMDSelect, SIMDSumTree, SIMDVector},
};

/////
///// 32-bit floating point
/////

macros::x86_define_register!(i32x8, __m256i, BitMask<8, V4>, i32, 8, V4);
macros::x86_define_splat!(i32x8, _mm256_set1_epi32, "avx");
macros::x86_define_default!(i32x8, _mm256_setzero_si256, "avx");
macros::x86_retarget!(i32x8 => v3::i32x8);
macros::x86_splitjoin!(
    i32x8,
    i32x4,
    _mm256_extracti128_si256,
    _mm256_set_m128i,
    "avx2"
);

helpers::unsafe_map_binary_op!(i32x8, std::ops::Add, add, _mm256_add_epi32, "avx2");
helpers::unsafe_map_binary_op!(i32x8, std::ops::Sub, sub, _mm256_sub_epi32, "avx2");
helpers::unsafe_map_binary_op!(i32x8, std::ops::Mul, mul, _mm256_mullo_epi32, "avx2");
helpers::unsafe_map_unary_op!(i32x8, SIMDAbs, abs_simd, _mm256_abs_epi32, "sse3");

helpers::unsafe_map_binary_op!(i32x8, std::ops::BitAnd, bitand, _mm256_and_si256, "avx2");
helpers::unsafe_map_binary_op!(i32x8, std::ops::BitOr, bitor, _mm256_or_si256, "avx2");
helpers::unsafe_map_binary_op!(i32x8, std::ops::BitXor, bitxor, _mm256_xor_si256, "avx2");
helpers::unsafe_map_binary_op!(i32x8, std::ops::Shr, shr, _mm256_srav_epi32, "avx2");
helpers::unsafe_map_binary_op!(i32x8, std::ops::Shl, shl, _mm256_sllv_epi32, "avx2");
helpers::scalar_shift_by_splat!(i32x8, i32);

impl std::ops::Not for i32x8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        self ^ Self::from_underlying(self.arch(), <Self as SIMDVector>::Underlying::all_ones())
    }
}

impl SIMDMulAdd for i32x8 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        self * rhs + accumulator
    }
}

macros::x86_avx512_load_store!(
    i32x8,
    _mm256_loadu_epi32,
    _mm256_maskz_loadu_epi32,
    _mm256_storeu_epi32,
    _mm256_mask_storeu_epi32,
    i32,
    "avx512f,avx512vl"
);

macros::x86_avx512_int_comparisons!(i32x8, _mm256_cmp_epi32_mask, "avx512f,avx512vl");

impl SIMDSumTree for i32x8 {
    #[inline(always)]
    fn sum_tree(self) -> i32 {
        self.retarget().sum_tree()
    }
}

impl SIMDSelect<i32x8> for BitMask<8, V4> {
    #[inline(always)]
    fn select(self, x: i32x8, y: i32x8) -> i32x8 {
        // SAFETY: `_mm256_mask_blend_epi32` requires AVX512F + AVX512VL - implied by V4
        i32x8::from_underlying(self.arch(), unsafe {
            _mm256_mask_blend_epi32(self.to_underlying(), y.to_underlying(), x.to_underlying())
        })
    }
}

//--------------//
// Dot Products //
//--------------//

impl SIMDDotProduct<i16x16> for i32x8 {
    #[inline(always)]
    fn dot_simd(self, left: i16x16, right: i16x16) -> Self {
        // SAFETY: `_mm256_dpwssd_epi32` requires AVX512_VNNI + AVX512VL - implied by V4
        let r = unsafe { _mm256_dpwssd_epi32(self.0, left.0, right.0) };
        Self::from_underlying(self.arch(), r)
    }
}

impl SIMDDotProduct<u8x32, i8x32> for i32x8 {
    #[inline(always)]
    fn dot_simd(self, left: u8x32, right: i8x32) -> Self {
        // SAFETY: `_mm512_dpbusd_epi32` requires AVX512_VNNI - implied by V4
        let r = unsafe { _mm256_dpbusd_epi32(self.0, left.0, right.0) };
        Self::from_underlying(self.arch(), r)
    }
}

impl SIMDDotProduct<i8x32, u8x32> for i32x8 {
    #[inline(always)]
    fn dot_simd(self, left: i8x32, right: u8x32) -> Self {
        self.dot_simd(right, left)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_x86_i32 {
    use super::*;
    use crate::{reference::ReferenceScalarOps, test_utils};

    #[test]
    fn miri_test_load() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_load_simd::<i32, 8, i32x8>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_store_simd::<i32, 8, i32x8>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::ops::test_splat::<i32, 8, i32x8>(arch);
        }
    }

    // Binary Ops
    test_utils::ops::test_add!(i32x8, 0xa93d54e3aab5d406, V4::new_checked_uncached());
    test_utils::ops::test_sub!(i32x8, 0x4b86c88f6958d930, V4::new_checked_uncached());
    test_utils::ops::test_mul!(i32x8, 0x0ad0524dc17b747a, V4::new_checked_uncached());
    test_utils::ops::test_fma!(i32x8, 0x277aca15e0552388, V4::new_checked_uncached());
    test_utils::ops::test_abs!(i32x8, 0x62ca26a68c1a238d, V4::new_checked_uncached());

    test_utils::ops::test_cmp!(i32x8, 0xdc88c2a44d17c78a, V4::new_checked_uncached());
    test_utils::ops::test_splitjoin!(i32x8 => i32x4, 0x475a19e80c2f3977, V4::new_checked_uncached());
    test_utils::ops::test_select!(i32x8, 0x568c7fcbf00fb03a, V4::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(i32x8, 0xc5f7d8d8df0b7b6c, V4::new_checked_uncached());

    // Dot Products
    test_utils::dot_product::test_dot_product!(
        (i16x16, i16x16) => i32x8, 0x145f89b446c03ff1, V4::new_checked_uncached()
    );
    test_utils::dot_product::test_dot_product!(
        (u8x32, i8x32) => i32x8, 0xa56e0de8ce99136c, V4::new_checked_uncached()
    );
    test_utils::dot_product::test_dot_product!(
        (i8x32, u8x32) => i32x8, 0xbcbcff932237df6d, V4::new_checked_uncached()
    );

    // Reductions
    test_utils::ops::test_sumtree!(i32x8, 0xe533708e69ca0117, V4::new_checked_uncached());
}

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
    traits::{SIMDAbs, SIMDMask, SIMDMulAdd, SIMDVector},
};

/////
///// 32-bit signed integer
/////

macros::x86_define_register!(i32x4, __m128i, BitMask<4, V4>, i32, 4, V4);
macros::x86_define_splat!(i32x4, _mm_set1_epi32, "sse2");
macros::x86_define_default!(i32x4, _mm_setzero_si128, "sse2");
macros::x86_retarget!(i32x4 => v3::i32x4);

helpers::unsafe_map_binary_op!(i32x4, std::ops::Add, add, _mm_add_epi32, "sse2");
helpers::unsafe_map_binary_op!(i32x4, std::ops::Sub, sub, _mm_sub_epi32, "sse2");
helpers::unsafe_map_binary_op!(i32x4, std::ops::Mul, mul, _mm_mullo_epi32, "sse4.1");
helpers::unsafe_map_unary_op!(i32x4, SIMDAbs, abs_simd, _mm_abs_epi32, "ssse3");

helpers::unsafe_map_binary_op!(i32x4, std::ops::BitAnd, bitand, _mm_and_si128, "sse2");
helpers::unsafe_map_binary_op!(i32x4, std::ops::BitOr, bitor, _mm_or_si128, "sse2");
helpers::unsafe_map_binary_op!(i32x4, std::ops::BitXor, bitxor, _mm_xor_si128, "sse2");
helpers::unsafe_map_binary_op!(i32x4, std::ops::Shr, shr, _mm_srav_epi32, "avx2");
helpers::unsafe_map_binary_op!(i32x4, std::ops::Shl, shl, _mm_sllv_epi32, "avx2");
helpers::scalar_shift_by_splat!(i32x4, i32);

impl std::ops::Not for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        self ^ Self::from_underlying(self.arch(), <Self as SIMDVector>::Underlying::all_ones())
    }
}

impl SIMDMulAdd for i32x4 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        self * rhs + accumulator
    }
}

macros::x86_avx512_load_store!(
    i32x4,
    _mm_loadu_epi32,
    _mm_maskz_loadu_epi32,
    _mm_storeu_epi32,
    _mm_mask_storeu_epi32,
    i32,
    "avx512f,avx512vl"
);

macros::x86_avx512_int_comparisons!(i32x4, _mm_cmp_epi32_mask, "avx512f,avx512vl");

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
            test_utils::test_load_simd::<i32, 4, i32x4>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_store_simd::<i32, 4, i32x4>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::ops::test_splat::<i32, 4, i32x4>(arch);
        }
    }

    // Binary Ops
    test_utils::ops::test_add!(i32x4, 0xf04d77df1da6aee6, V4::new_checked_uncached());
    test_utils::ops::test_sub!(i32x4, 0xd1e591845e17fc01, V4::new_checked_uncached());
    test_utils::ops::test_mul!(i32x4, 0xf0caa85d919a41a8, V4::new_checked_uncached());
    test_utils::ops::test_fma!(i32x4, 0x1f0340c2109aef6f, V4::new_checked_uncached());
    test_utils::ops::test_abs!(i32x4, 0x60710c0c88537c7d, V4::new_checked_uncached());

    test_utils::ops::test_cmp!(i32x4, 0x9a2d73b7295214c6, V4::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(i32x4, 0x763fc44f8f7cd40c, V4::new_checked_uncached());
}

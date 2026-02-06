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
        v4::{i8x16_::i8x16, i16x32_::i16x32},
    },
    constant::Const,
    helpers,
    traits::{SIMDAbs, SIMDMask, SIMDMulAdd, SIMDVector},
};

//////////////////
// 8-bit signed //
//////////////////

macros::x86_define_register!(i8x32, __m256i, BitMask<32, V4>, i8, 32, V4);
macros::x86_define_splat!(i8x32 as i8, _mm256_set1_epi8, "avx");
macros::x86_define_default!(i8x32, _mm256_setzero_si256, "avx");
macros::x86_splitjoin!(
    i8x32,
    i8x16,
    _mm256_extracti128_si256,
    _mm256_set_m128i,
    "avx2"
);

impl std::ops::Mul for i8x32 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let x: i16x32 = self.into();
        let y: i16x32 = rhs.into();
        (x * y).cast::<i8>()
    }
}

helpers::unsafe_map_binary_op!(i8x32, std::ops::Add, add, _mm256_add_epi8, "avx2");
helpers::unsafe_map_binary_op!(i8x32, std::ops::Sub, sub, _mm256_sub_epi8, "avx2");
helpers::unsafe_map_unary_op!(i8x32, SIMDAbs, abs_simd, _mm256_abs_epi8, "avx2");

helpers::unsafe_map_binary_op!(i8x32, std::ops::BitAnd, bitand, _mm256_and_si256, "avx2");
helpers::unsafe_map_binary_op!(i8x32, std::ops::BitOr, bitor, _mm256_or_si256, "avx2");
helpers::unsafe_map_binary_op!(i8x32, std::ops::BitXor, bitxor, _mm256_xor_si256, "avx2");

impl std::ops::Shr for i8x32 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self {
        let s: i16x32 = self.into();
        let rhs: i16x32 = rhs.into();
        (s.shr(rhs)).cast::<i8>()
    }
}

impl std::ops::Shl for i8x32 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self {
        let s: i16x32 = self.into();
        let rhs: i16x32 = rhs.into();
        (s.shl(rhs)).cast::<i8>()
    }
}

helpers::scalar_shift_by_splat!(i8x32, i8);

impl std::ops::Not for i8x32 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        self ^ Self::from_underlying(self.arch(), <Self as SIMDVector>::Underlying::all_ones())
    }
}

impl SIMDMulAdd for i8x32 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        self * rhs + accumulator
    }
}

macros::x86_avx512_load_store!(
    i8x32,
    _mm256_loadu_epi8,
    _mm256_maskz_loadu_epi8,
    _mm256_storeu_epi8,
    _mm256_mask_storeu_epi8,
    i8,
    "avx512bw,avx512vl"
);

macros::x86_avx512_int_comparisons!(i8x32, _mm256_cmp_epi8_mask, "avx512bw,avx512vl");

///////////
// Tests //
///////////

#[cfg(test)]
mod test_x86_i8 {
    use super::*;
    use crate::{reference::ReferenceScalarOps, test_utils};

    #[test]
    fn miri_test_load() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_load_simd::<i8, 32, i8x32>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_store_simd::<i8, 32, i8x32>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::ops::test_splat::<i8, 32, i8x32>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(i8x32, 0x3017fd73c99cc633, V4::new_checked_uncached());
    test_utils::ops::test_sub!(i8x32, 0xfc627f10b5f8db8a, V4::new_checked_uncached());
    test_utils::ops::test_mul!(i8x32, 0x0f4caa80eceaa523, V4::new_checked_uncached());
    test_utils::ops::test_fma!(i8x32, 0xb8f702ba85375041, V4::new_checked_uncached());
    test_utils::ops::test_abs!(i8x32, 0x40638a9d09522d1c, V4::new_checked_uncached());

    test_utils::ops::test_cmp!(i8x32, 0x941757bd5cc641a1, V4::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(i8x32, 0xd62d8de09f82ed4e, V4::new_checked_uncached());
}

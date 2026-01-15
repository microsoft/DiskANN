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
        v4::{i16x32_::i16x32, u8x16_::u8x16},
    },
    constant::Const,
    helpers,
    traits::{SIMDMask, SIMDMulAdd, SIMDVector},
};

////////////////////
// 8-bit unsigned //
////////////////////

macros::x86_define_register!(u8x32, __m256i, BitMask<32, V4>, u8, 32, V4);
macros::x86_define_splat!(u8x32 as i8, _mm256_set1_epi8, "avx");
macros::x86_define_default!(u8x32, _mm256_setzero_si256, "avx");
macros::x86_splitjoin!(
    u8x32,
    u8x16,
    _mm256_extracti128_si256,
    _mm256_set_m128i,
    "avx2"
);

impl std::ops::Mul for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let x: i16x32 = self.into();
        let y: i16x32 = rhs.into();
        (x * y).cast::<u8>()
    }
}

helpers::unsafe_map_binary_op!(u8x32, std::ops::Add, add, _mm256_add_epi8, "avx2");
helpers::unsafe_map_binary_op!(u8x32, std::ops::Sub, sub, _mm256_sub_epi8, "avx2");

helpers::unsafe_map_binary_op!(u8x32, std::ops::BitAnd, bitand, _mm256_and_si256, "avx2");
helpers::unsafe_map_binary_op!(u8x32, std::ops::BitOr, bitor, _mm256_or_si256, "avx2");
helpers::unsafe_map_binary_op!(u8x32, std::ops::BitXor, bitxor, _mm256_xor_si256, "avx2");

impl std::ops::Shr for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self {
        let s: i16x32 = self.into();
        let rhs: i16x32 = rhs.into();
        (s.shr(rhs)).cast::<u8>()
    }
}

impl std::ops::Shl for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self {
        let s: i16x32 = self.into();
        let rhs: i16x32 = rhs.into();
        (s.shl(rhs)).cast::<u8>()
    }
}

helpers::scalar_shift_by_splat!(u8x32, u8);

impl std::ops::Not for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        self ^ Self::from_underlying(self.arch(), <Self as SIMDVector>::Underlying::all_ones())
    }
}

impl SIMDMulAdd for u8x32 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        self * rhs + accumulator
    }
}

macros::x86_avx512_load_store!(
    u8x32,
    _mm256_loadu_epi8,
    _mm256_maskz_loadu_epi8,
    _mm256_storeu_epi8,
    _mm256_mask_storeu_epi8,
    i8,
    "avx512bw,avx512vl"
);

macros::x86_avx512_int_comparisons!(u8x32, _mm256_cmp_epu8_mask, "avx512bw,avx512vl");

///////////
// Tests //
///////////

#[cfg(test)]
mod test_x86_u8 {
    use super::*;
    use crate::{reference::ReferenceScalarOps, test_utils};

    #[test]
    fn miri_test_load() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_load_simd::<u8, 32, u8x32>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_store_simd::<u8, 32, u8x32>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::ops::test_splat::<u8, 32, u8x32>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(u8x32, 0x3017fd73c99cc633, V4::new_checked_uncached());
    test_utils::ops::test_sub!(u8x32, 0xfc627f10b5f8db8a, V4::new_checked_uncached());
    test_utils::ops::test_mul!(u8x32, 0x0f4caa80eceaa523, V4::new_checked_uncached());
    test_utils::ops::test_fma!(u8x32, 0xb8f702ba85375041, V4::new_checked_uncached());

    test_utils::ops::test_splitjoin!(u8x32 => u8x16, 0x475a19e80c2f3977, V4::new_checked_uncached());

    test_utils::ops::test_cmp!(u8x32, 0x941757bd5cc641a1, V4::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(u8x32, 0xd62d8de09f82ed4e, V4::new_checked_uncached());
}

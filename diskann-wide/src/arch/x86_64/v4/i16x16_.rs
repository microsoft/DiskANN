/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// x86 intrinsics
use std::arch::x86_64::*;

use crate::{
    SIMDMask,
    arch::x86_64::{
        V4,
        common::AllOnes,
        macros::{self, X86Default, X86LoadStore, X86Splat},
        v3,
        v4::i16x8_::i16x8, // direct import for Miri compat
    },
    bitmask::BitMask,
    constant::Const,
    helpers,
    traits::{SIMDAbs, SIMDMulAdd, SIMDVector},
};

///////////////////
// 16-bit signed //
///////////////////

macros::x86_define_register!(i16x16, __m256i, BitMask<16, V4>, i16, 16, V4);
macros::x86_define_splat!(i16x16, _mm256_set1_epi16, "avx");
macros::x86_define_default!(i16x16, _mm256_setzero_si256, "avx");
macros::x86_retarget!(i16x16 => v3::i16x16);
macros::x86_splitjoin!(
    i16x16,
    i16x8,
    _mm256_extracti128_si256,
    _mm256_set_m128i,
    "avx2"
);

helpers::unsafe_map_binary_op!(i16x16, std::ops::Add, add, _mm256_add_epi16, "avx2");
helpers::unsafe_map_binary_op!(i16x16, std::ops::Sub, sub, _mm256_sub_epi16, "avx2");
helpers::unsafe_map_binary_op!(i16x16, std::ops::Mul, mul, _mm256_mullo_epi16, "avx2");
helpers::unsafe_map_unary_op!(i16x16, SIMDAbs, abs_simd, _mm256_abs_epi16, "avx2");

helpers::unsafe_map_binary_op!(i16x16, std::ops::BitAnd, bitand, _mm256_and_si256, "avx2");
helpers::unsafe_map_binary_op!(i16x16, std::ops::BitOr, bitor, _mm256_or_si256, "avx2");
helpers::unsafe_map_binary_op!(i16x16, std::ops::BitXor, bitxor, _mm256_xor_si256, "avx2");

helpers::unsafe_map_binary_op!(i16x16, std::ops::Shr, shr, _mm256_srav_epi16, "avx2");
helpers::unsafe_map_binary_op!(i16x16, std::ops::Shl, shl, _mm256_sllv_epi16, "avx2");
helpers::scalar_shift_by_splat!(i16x16, i16);

impl std::ops::Not for i16x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        self ^ Self::from_underlying(self.arch(), <Self as SIMDVector>::Underlying::all_ones())
    }
}

impl SIMDMulAdd for i16x16 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        self * rhs + accumulator
    }
}

macros::x86_avx512_int_comparisons!(i16x16, _mm256_cmp_epi16_mask, "avx512bw,avx512vl");

macros::x86_avx512_load_store!(
    i16x16,
    _mm256_loadu_epi16,
    _mm256_maskz_loadu_epi16,
    _mm256_storeu_epi16,
    _mm256_mask_storeu_epi16,
    i16,
    "avx512bw,avx512vl"
);

///////////
// Tests //
///////////

#[cfg(test)]
mod test_x86_i16 {
    use super::*;
    use crate::{reference::ReferenceScalarOps, test_utils};

    // Test loading logic - ensure that no out of bounds accesses are made.
    // In particular, this is meant to be run under `Miri` to ensure that our guarantees
    // regarding out-of-bounds accesses are honored.
    #[test]
    fn miri_test_load() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_load_simd::<i16, 16, i16x16>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_store_simd::<i16, 16, i16x16>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::ops::test_splat::<i16, 16, i16x16>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(i16x16, 0xcd7a8ad4a3b7760c, V4::new_checked_uncached());
    test_utils::ops::test_sub!(i16x16, 0x9e5abc35dda03aa5, V4::new_checked_uncached());
    test_utils::ops::test_mul!(i16x16, 0x47159f68b972ad07, V4::new_checked_uncached());
    test_utils::ops::test_fma!(i16x16, 0xed4244971a90e2f0, V4::new_checked_uncached());
    test_utils::ops::test_abs!(i16x16, 0x9aaa0504d1598348, V4::new_checked_uncached());

    test_utils::ops::test_cmp!(i16x16, 0x242f7f6c3709920d, V4::new_checked_uncached());
    test_utils::ops::test_splitjoin!(i16x16 => i16x8, 0x05931ca2c2577fae, V4::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(i16x16, 0xba0be356b04d6427, V4::new_checked_uncached());
}

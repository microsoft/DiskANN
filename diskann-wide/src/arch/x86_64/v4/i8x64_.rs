/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// x86 intrinsics
use std::arch::x86_64::*;

use crate::{
    BitMask, SplitJoin,
    arch::x86_64::{
        V4,
        common::AllOnes,
        macros::{self, X86Default, X86LoadStore, X86Splat},
        v4::i8x32_::i8x32,
    },
    constant::Const,
    helpers,
    traits::{SIMDAbs, SIMDMask, SIMDMulAdd, SIMDVector},
};

//////////////////
// 8-bit signed //
//////////////////

macros::x86_define_register!(i8x64, __m512i, BitMask<64, V4>, i8, 64, V4);
macros::x86_define_splat!(i8x64 as i8, _mm512_set1_epi8, "avx512f");
macros::x86_define_default!(i8x64, _mm512_setzero_si512, "avx512f");
macros::x86_splitjoin!(__m512i, i8x64, i8x32);

impl std::ops::Mul for i8x64 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self.split().map_with(rhs.split(), |a, b| a * b).join()
    }
}

helpers::unsafe_map_binary_op!(i8x64, std::ops::Add, add, _mm512_add_epi8, "avx512bw");
helpers::unsafe_map_binary_op!(i8x64, std::ops::Sub, sub, _mm512_sub_epi8, "avx512bw");
helpers::unsafe_map_unary_op!(i8x64, SIMDAbs, abs_simd, _mm512_abs_epi8, "avx512bw");

helpers::unsafe_map_binary_op!(i8x64, std::ops::BitAnd, bitand, _mm512_and_si512, "avx512f");
helpers::unsafe_map_binary_op!(i8x64, std::ops::BitOr, bitor, _mm512_or_si512, "avx512f");
helpers::unsafe_map_binary_op!(i8x64, std::ops::BitXor, bitxor, _mm512_xor_si512, "avx512f");

impl std::ops::Shr for i8x64 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self {
        self.split().map_with(rhs.split(), |a, b| a.shr(b)).join()
    }
}

impl std::ops::Shl for i8x64 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self {
        self.split().map_with(rhs.split(), |a, b| a.shl(b)).join()
    }
}

helpers::scalar_shift_by_splat!(i8x64, i8);

impl std::ops::Not for i8x64 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        self ^ Self::from_underlying(self.arch(), <Self as SIMDVector>::Underlying::all_ones())
    }
}

impl SIMDMulAdd for i8x64 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        self * rhs + accumulator
    }
}

macros::x86_avx512_load_store!(
    i8x64,
    _mm512_loadu_epi8,
    _mm512_maskz_loadu_epi8,
    _mm512_storeu_epi8,
    _mm512_mask_storeu_epi8,
    i8,
    "avx512bw"
);

macros::x86_avx512_int_comparisons!(i8x64, _mm512_cmp_epi8_mask, "avx512bw");

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
            test_utils::test_load_simd::<i8, 64, i8x64>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_store_simd::<i8, 64, i8x64>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::ops::test_splat::<i8, 64, i8x64>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(i8x64, 0x3017fd73c99cc633, V4::new_checked_uncached());
    test_utils::ops::test_sub!(i8x64, 0xfc627f10b5f8db8a, V4::new_checked_uncached());
    test_utils::ops::test_mul!(i8x64, 0x0f4caa80eceaa523, V4::new_checked_uncached());
    test_utils::ops::test_fma!(i8x64, 0xb8f702ba85375041, V4::new_checked_uncached());
    test_utils::ops::test_abs!(i8x64, 0x40638a9d09522d1c, V4::new_checked_uncached());

    test_utils::ops::test_cmp!(i8x64, 0x941757bd5cc641a1, V4::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(i8x64, 0xd62d8de09f82ed4e, V4::new_checked_uncached());
}

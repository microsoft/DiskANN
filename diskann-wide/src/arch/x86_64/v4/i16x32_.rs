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
        v4::i16x16_::i16x16, // direct import for Miri compat
    },
    bitmask::BitMask,
    constant::Const,
    helpers,
    traits::{SIMDAbs, SIMDMulAdd, SIMDVector},
};

///////////////////
// 16-bit signed //
///////////////////

macros::x86_define_register!(i16x32, __m512i, BitMask<32, V4>, i16, 32, V4);
macros::x86_define_splat!(i16x32, _mm512_set1_epi16, "avx512f");
macros::x86_define_default!(i16x32, _mm512_setzero_si512, "avx512f");
macros::x86_splitjoin!(__m512i, i16x32, i16x16);

helpers::unsafe_map_binary_op!(i16x32, std::ops::Add, add, _mm512_add_epi16, "avx512bw");
helpers::unsafe_map_binary_op!(i16x32, std::ops::Sub, sub, _mm512_sub_epi16, "avx512bw");
helpers::unsafe_map_binary_op!(i16x32, std::ops::Mul, mul, _mm512_mullo_epi16, "avx512bw");
helpers::unsafe_map_unary_op!(i16x32, SIMDAbs, abs_simd, _mm512_abs_epi16, "avx512bw");

helpers::unsafe_map_binary_op!(
    i16x32,
    std::ops::BitAnd,
    bitand,
    _mm512_and_si512,
    "avx512f"
);
helpers::unsafe_map_binary_op!(i16x32, std::ops::BitOr, bitor, _mm512_or_si512, "avx512f");
helpers::unsafe_map_binary_op!(
    i16x32,
    std::ops::BitXor,
    bitxor,
    _mm512_xor_si512,
    "avx512f"
);

helpers::unsafe_map_binary_op!(i16x32, std::ops::Shr, shr, _mm512_srav_epi16, "avx512bw");
helpers::unsafe_map_binary_op!(i16x32, std::ops::Shl, shl, _mm512_sllv_epi16, "avx512bw");
helpers::scalar_shift_by_splat!(i16x32, i16);

impl std::ops::Not for i16x32 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        self ^ Self::from_underlying(self.arch(), <Self as SIMDVector>::Underlying::all_ones())
    }
}

impl SIMDMulAdd for i16x32 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        self * rhs + accumulator
    }
}

macros::x86_avx512_int_comparisons!(i16x32, _mm512_cmp_epi16_mask, "avx512bw,avx512vl");
macros::x86_avx512_load_store!(
    i16x32,
    _mm512_loadu_epi16,
    _mm512_maskz_loadu_epi16,
    _mm512_storeu_epi16,
    _mm512_mask_storeu_epi16,
    i16,
    "avx512bw"
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
            test_utils::test_load_simd::<i16, 32, i16x32>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_store_simd::<i16, 32, i16x32>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::ops::test_splat::<i16, 32, i16x32>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(i16x32, 0xcd7a8ad4a3b7760c, V4::new_checked_uncached());
    test_utils::ops::test_sub!(i16x32, 0x9e5abc35dda03aa5, V4::new_checked_uncached());
    test_utils::ops::test_mul!(i16x32, 0x47159f68b972ad07, V4::new_checked_uncached());
    test_utils::ops::test_fma!(i16x32, 0xed4244971a90e2f0, V4::new_checked_uncached());
    test_utils::ops::test_abs!(i16x32, 0x9aaa0504d1598348, V4::new_checked_uncached());

    test_utils::ops::test_cmp!(i16x32, 0x242f7f6c3709920d, V4::new_checked_uncached());
    test_utils::ops::test_splitjoin!(i16x32 => i16x16, 0x05931ca2c2577fae, V4::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(i16x32, 0xba0be356b04d6427, V4::new_checked_uncached());
}

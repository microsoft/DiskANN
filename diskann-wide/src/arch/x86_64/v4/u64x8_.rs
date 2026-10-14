/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::arch::x86_64::*;

use crate::{
    BitMask,
    arch::x86_64::{
        V4,
        common::AllOnes,
        macros::{self, X86Default, X86LoadStore, X86Splat},
        v4::u64x4_::u64x4,
    },
    constant::Const,
    helpers,
    traits::{SIMDMask, SIMDMulAdd, SIMDPopcount, SIMDSumTree, SIMDVector},
};

/////////////////////
// 64-bit unsigned //
/////////////////////

macros::x86_define_register!(u64x8, __m512i, BitMask<8, V4>, u64, 8, V4);
macros::x86_define_splat!(u64x8 as i64, _mm512_set1_epi64, "avx512f");
macros::x86_define_default!(u64x8, _mm512_setzero_si512, "avx512f");
macros::x86_splitjoin!(__m512i, u64x8, u64x4);

helpers::unsafe_map_binary_op!(u64x8, std::ops::Add, add, _mm512_add_epi64, "avx512f");
helpers::unsafe_map_binary_op!(u64x8, std::ops::Sub, sub, _mm512_sub_epi64, "avx512f");
helpers::unsafe_map_unary_op!(
    u64x8,
    SIMDPopcount,
    popcount_simd,
    _mm512_popcnt_epi64,
    "avx512vpopcntdq"
);
helpers::unsafe_map_binary_op!(u64x8, std::ops::Mul, mul, _mm512_mullo_epi64, "avx512dq");

helpers::unsafe_map_binary_op!(u64x8, std::ops::BitAnd, bitand, _mm512_and_si512, "avx512f");
helpers::unsafe_map_binary_op!(u64x8, std::ops::BitOr, bitor, _mm512_or_si512, "avx512f");
helpers::unsafe_map_binary_op!(u64x8, std::ops::BitXor, bitxor, _mm512_xor_si512, "avx512f");
helpers::unsafe_map_binary_op!(u64x8, std::ops::Shr, shr, _mm512_srlv_epi64, "avx512f");
helpers::unsafe_map_binary_op!(u64x8, std::ops::Shl, shl, _mm512_sllv_epi64, "avx512f");
helpers::scalar_shift_by_splat!(u64x8, u64);

impl std::ops::Not for u64x8 {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self {
        self ^ Self::from_underlying(self.arch(), <Self as SIMDVector>::Underlying::all_ones())
    }
}

impl SIMDMulAdd for u64x8 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        self * rhs + accumulator
    }
}

macros::x86_avx512_load_store!(
    u64x8,
    _mm512_loadu_epi64,
    _mm512_maskz_loadu_epi64,
    _mm512_storeu_epi64,
    _mm512_mask_storeu_epi64,
    i64,
    "avx512f"
);

macros::x86_avx512_int_comparisons!(u64x8, _mm512_cmp_epu64_mask, "avx512f");

impl SIMDSumTree for u64x8 {
    #[inline(always)]
    fn sum_tree(self) -> u64 {
        // SAFETY: `_mm512_reduce_add_epi64` requires AVX-512F, implied by V4.
        unsafe { _mm512_reduce_add_epi64(self.0) as u64 }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_x86_u64 {
    use super::*;
    use crate::{reference::ReferenceScalarOps, test_utils};

    #[test]
    fn miri_test_load() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_load_simd::<u64, 8, u64x8>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_store_simd::<u64, 8, u64x8>(arch);
        }
    }

    #[test]
    fn test_constructors() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::ops::test_splat::<u64, 8, u64x8>(arch);
        }
    }

    test_utils::ops::test_add!(u64x8, 0xeaee2fd0398fe357, V4::new_checked_uncached());
    test_utils::ops::test_sub!(u64x8, 0x40af040b0c2c1e28, V4::new_checked_uncached());
    test_utils::ops::test_mul!(u64x8, 0x68f68933a29c5ea9, V4::new_checked_uncached());
    test_utils::ops::test_fma!(u64x8, 0x31bc9d25e91e6744, V4::new_checked_uncached());

    test_utils::ops::test_cmp!(u64x8, 0x0beda0dd5141ec40, V4::new_checked_uncached());
    test_utils::ops::test_splitjoin!(u64x8 => u64x4, 0xb151fcd6141b10c9, V4::new_checked_uncached());

    test_utils::ops::test_sumtree!(u64x8, 0x529c27f62ea171ec, V4::new_checked_uncached());

    test_utils::ops::test_bitops!(u64x8, 0xb1ac2e16327a8d5e, V4::new_checked_uncached());
    test_utils::ops::test_popcount!(u64x8, 0xf23de3226c0141be, V4::new_checked_uncached());
}

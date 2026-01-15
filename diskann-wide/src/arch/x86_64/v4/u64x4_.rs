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
        v4::u64x2_::u64x2, // direct import for Miri compat
    },
    constant::Const,
    helpers,
    traits::{SIMDMask, SIMDMulAdd, SIMDVector},
};

/////
///// 64-bit unsigned integer
/////

macros::x86_define_register!(u64x4, __m256i, BitMask<4, V4>, u64, 4, V4);
macros::x86_define_splat!(u64x4 as i64, _mm256_set1_epi64x, "avx");
macros::x86_define_default!(u64x4, _mm256_setzero_si256, "avx");
macros::x86_retarget!(u64x4 => v3::u64x4);
macros::x86_splitjoin!(
    u64x4,
    u64x2,
    _mm256_extracti128_si256,
    _mm256_set_m128i,
    "avx2"
);

helpers::unsafe_map_binary_op!(u64x4, std::ops::Add, add, _mm256_add_epi64, "avx2");
helpers::unsafe_map_binary_op!(u64x4, std::ops::Sub, sub, _mm256_sub_epi64, "avx2");
helpers::unsafe_map_binary_op!(
    u64x4,
    std::ops::Mul,
    mul,
    _mm256_mullo_epi64,
    "avx512dq,avx512vl"
);

helpers::unsafe_map_binary_op!(u64x4, std::ops::BitAnd, bitand, _mm256_and_si256, "avx2");
helpers::unsafe_map_binary_op!(u64x4, std::ops::BitOr, bitor, _mm256_or_si256, "avx2");
helpers::unsafe_map_binary_op!(u64x4, std::ops::BitXor, bitxor, _mm256_xor_si256, "avx2");
helpers::unsafe_map_binary_op!(u64x4, std::ops::Shr, shr, _mm256_srlv_epi64, "avx2");
helpers::unsafe_map_binary_op!(u64x4, std::ops::Shl, shl, _mm256_sllv_epi64, "avx2");

helpers::scalar_shift_by_splat!(u64x4, u64);

impl std::ops::Not for u64x4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        self ^ Self::from_underlying(self.arch(), <Self as SIMDVector>::Underlying::all_ones())
    }
}

impl SIMDMulAdd for u64x4 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        self * rhs + accumulator
    }
}

macros::x86_avx512_load_store!(
    u64x4,
    _mm256_loadu_epi64,
    _mm256_maskz_loadu_epi64,
    _mm256_storeu_epi64,
    _mm256_mask_storeu_epi64,
    i64,
    "avx512f,avx512vl"
);

macros::x86_avx512_int_comparisons!(u64x4, _mm256_cmp_epu64_mask, "avx512f,avx512vl");

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
            test_utils::test_load_simd::<u64, 4, u64x4>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_store_simd::<u64, 4, u64x4>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::ops::test_splat::<u64, 4, u64x4>(arch);
        }
    }

    // Binary Ops
    test_utils::ops::test_add!(u64x4, 0xeaee2fd0398fe357, V4::new_checked_uncached());
    test_utils::ops::test_sub!(u64x4, 0x40af040b0c2c1e28, V4::new_checked_uncached());
    test_utils::ops::test_mul!(u64x4, 0x68f68933a29c5ea9, V4::new_checked_uncached());
    test_utils::ops::test_fma!(u64x4, 0x31bc9d25e91e6744, V4::new_checked_uncached());

    test_utils::ops::test_cmp!(u64x4, 0x0beda0dd5141ec40, V4::new_checked_uncached());
    test_utils::ops::test_splitjoin!(u64x4 => u64x2, 0xb151fcd6141b10c9, V4::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(u64x4, 0xb1ac2e16327a8d5e, V4::new_checked_uncached());
}

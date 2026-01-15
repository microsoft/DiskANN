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
    traits::{SIMDMask, SIMDMulAdd, SIMDVector},
};

/////
///// 64-bit unsigned integerA
/////

macros::x86_define_register!(u64x2, __m128i, BitMask<2, V4>, u64, 2, V4);
macros::x86_define_splat!(u64x2 as i64, _mm_set1_epi64x, "sse2");
macros::x86_define_default!(u64x2, _mm_setzero_si128, "sse2");
macros::x86_retarget!(u64x2 => v3::u64x2);

helpers::unsafe_map_binary_op!(u64x2, std::ops::Add, add, _mm_add_epi64, "sse2");
helpers::unsafe_map_binary_op!(u64x2, std::ops::Sub, sub, _mm_sub_epi64, "sse2");
helpers::unsafe_map_binary_op!(
    u64x2,
    std::ops::Mul,
    mul,
    _mm_mullo_epi64,
    "avx512dq,avx512vl"
);

helpers::unsafe_map_binary_op!(u64x2, std::ops::BitAnd, bitand, _mm_and_si128, "sse2");
helpers::unsafe_map_binary_op!(u64x2, std::ops::BitOr, bitor, _mm_or_si128, "sse2");
helpers::unsafe_map_binary_op!(u64x2, std::ops::BitXor, bitxor, _mm_xor_si128, "sse2");
helpers::unsafe_map_binary_op!(u64x2, std::ops::Shr, shr, _mm_srlv_epi64, "avx2");
helpers::unsafe_map_binary_op!(u64x2, std::ops::Shl, shl, _mm_sllv_epi64, "avx2");

helpers::scalar_shift_by_splat!(u64x2, u64);

impl std::ops::Not for u64x2 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        self ^ Self::from_underlying(self.arch(), <Self as SIMDVector>::Underlying::all_ones())
    }
}

impl SIMDMulAdd for u64x2 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        self * rhs + accumulator
    }
}

macros::x86_avx512_load_store!(
    u64x2,
    _mm_loadu_epi64,
    _mm_maskz_loadu_epi64,
    _mm_storeu_epi64,
    _mm_mask_storeu_epi64,
    i64,
    "avx512f,avx512vl"
);

macros::x86_avx512_int_comparisons!(u64x2, _mm_cmp_epu64_mask, "avx512f,avx512vl");

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
            test_utils::test_load_simd::<u64, 2, u64x2>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_store_simd::<u64, 2, u64x2>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::ops::test_splat::<u64, 2, u64x2>(arch);
        }
    }

    // Binary Ops
    test_utils::ops::test_add!(u64x2, 0xd1eb6bdca533ca00, V4::new_checked_uncached());
    test_utils::ops::test_sub!(u64x2, 0xde6675680ad0a65b, V4::new_checked_uncached());
    test_utils::ops::test_mul!(u64x2, 0xc78dafd8072a8868, V4::new_checked_uncached());
    test_utils::ops::test_fma!(u64x2, 0xcb45c9f29a44719f, V4::new_checked_uncached());

    test_utils::ops::test_cmp!(u64x2, 0x92486698bb7603e7, V4::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(u64x2, 0xf9566b095125ca45, V4::new_checked_uncached());
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// x86 intrinsics
use std::arch::x86_64::*;

// x86 masks
use crate::{
    BitMask,
    arch::x86_64::{
        V4,
        macros::{self, X86Default, X86LoadStore, X86Splat},
        v3,
        v4::f32x4_::f32x4, // direct import for Miri compat
    },
    constant::Const,
    helpers,
    traits::{
        SIMDAbs, SIMDMask, SIMDMinMax, SIMDMulAdd, SIMDPartialEq, SIMDPartialOrd, SIMDSelect,
        SIMDSumTree, SIMDVector,
    },
};

/////////////////////
// 32-bit floating //
/////////////////////

macros::x86_define_register!(f32x8, __m256, BitMask<8, V4>, f32, 8, V4);
macros::x86_define_splat!(f32x8, _mm256_set1_ps, "avx");
macros::x86_define_default!(f32x8, _mm256_setzero_ps, "avx");
macros::x86_splitjoin!(f32x8, f32x4, _mm256_extractf128_ps, _mm256_set_m128, "avx");
macros::x86_retarget!(f32x8 => v3::f32x8);

helpers::unsafe_map_binary_op!(f32x8, std::ops::Add, add, _mm256_add_ps, "avx");
helpers::unsafe_map_binary_op!(f32x8, std::ops::Sub, sub, _mm256_sub_ps, "avx");
helpers::unsafe_map_binary_op!(f32x8, std::ops::Mul, mul, _mm256_mul_ps, "avx");

impl f32x8 {
    #[inline(always)]
    fn is_nan(self) -> BitMask<8, V4> {
        // NOTE: `_CMP_UNORD_Q` returns `true` if either argument is NaN. Since we compare
        // `self` with `self`, this returns `true` exactly when `self` is NaN.
        BitMask::from_underlying(
            self.arch(),
            // SAFETY: `_mm256_cmp_ps_mask` requires AVX512F + AVX512VL, both of which
            // are implied by the implicit V4 architecture.
            unsafe { _mm256_cmp_ps_mask(self.to_underlying(), self.to_underlying(), _CMP_UNORD_Q) },
        )
    }
}

impl SIMDMulAdd for f32x8 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        // SAFETY: `_mm256_fmadd_ps` requires FMA - implied by V4
        Self(unsafe { _mm256_fmadd_ps(self.0, rhs.0, accumulator.0) })
    }
}

impl SIMDMinMax for f32x8 {
    #[inline(always)]
    fn min_simd(self, rhs: Self) -> Self {
        // SAFETY: `_mm256_min_ps` requires AVX, which is implied by the V4 architecture.
        Self(unsafe { _mm256_min_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn min_simd_standard(self, rhs: Self) -> Self {
        // NOTE: The behavior of the non-IEEE compliant min is order dependent and thus
        // it is important that the order of the arguments is swapped.
        let min = rhs.min_simd(self);
        self.is_nan().select(rhs, min)
    }

    #[inline(always)]
    fn max_simd(self, rhs: Self) -> Self {
        // SAFETY: `_mm256_max_ps` requires AVX, which is implied by the V4 architecture.
        Self(unsafe { _mm256_max_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn max_simd_standard(self, rhs: Self) -> Self {
        // NOTE: The behavior of the non-IEEE compliant max is order dependent.
        let max = rhs.max_simd(self);
        self.is_nan().select(rhs, max)
    }
}

impl SIMDAbs for f32x8 {
    #[inline(always)]
    fn abs_simd(self) -> Self {
        self.from(self.retarget().abs_simd())
    }
}

macros::x86_avx512_load_store!(
    f32x8,
    _mm256_loadu_ps,
    _mm256_maskz_loadu_ps,
    _mm256_storeu_ps,
    _mm256_mask_storeu_ps,
    f32,
    "avx512f,avx512vl"
);

impl SIMDPartialEq for f32x8 {
    #[inline(always)]
    fn eq_simd(self, other: Self) -> Self::Mask {
        // SAFETY: `_mm256_cmp_ps_mask` requires `AVX512F` and `AVX512VL` - implied by V4
        let m = unsafe { _mm256_cmp_ps_mask(self.0, other.0, _CMP_EQ_OQ) };
        Self::Mask::from_underlying(self.arch(), m)
    }

    #[inline(always)]
    fn ne_simd(self, other: Self) -> Self::Mask {
        // SAFETY: `_mm256_cmp_ps_mask` requires `AVX512F` and `AVX512VL` - implied by V4
        let m = unsafe { _mm256_cmp_ps_mask(self.0, other.0, _CMP_NEQ_UQ) };
        Self::Mask::from_underlying(self.arch(), m)
    }
}

impl SIMDPartialOrd for f32x8 {
    #[inline(always)]
    fn lt_simd(self, other: Self) -> Self::Mask {
        // SAFETY: `_mm256_cmp_ps_mask` requires `AVX512F` and `AVX512VL` - implied by V4
        let m = unsafe { _mm256_cmp_ps_mask(self.0, other.0, _CMP_LT_OQ) };
        Self::Mask::from_underlying(self.arch(), m)
    }

    #[inline(always)]
    fn le_simd(self, other: Self) -> Self::Mask {
        // SAFETY: `_mm256_cmp_ps_mask` requires `AVX512F` and `AVX512VL` - implied by V4
        let m = unsafe { _mm256_cmp_ps_mask(self.0, other.0, _CMP_LE_OQ) };
        Self::Mask::from_underlying(self.arch(), m)
    }

    #[inline(always)]
    fn gt_simd(self, other: Self) -> Self::Mask {
        // SAFETY: `_mm256_cmp_ps_mask` requires `AVX512F` and `AVX512VL` - implied by V4
        let m = unsafe { _mm256_cmp_ps_mask(self.0, other.0, _CMP_GT_OQ) };
        Self::Mask::from_underlying(self.arch(), m)
    }

    #[inline(always)]
    fn ge_simd(self, other: Self) -> Self::Mask {
        // SAFETY: `_mm256_cmp_ps_mask` requires `AVX512F` and `AVX512VL` - implied by V4
        let m = unsafe { _mm256_cmp_ps_mask(self.0, other.0, _CMP_GE_OQ) };
        Self::Mask::from_underlying(self.arch(), m)
    }
}

impl SIMDSumTree for f32x8 {
    #[inline(always)]
    fn sum_tree(self) -> f32 {
        self.retarget().sum_tree()
    }
}

impl SIMDSelect<f32x8> for BitMask<8, V4> {
    #[inline(always)]
    fn select(self, x: f32x8, y: f32x8) -> f32x8 {
        // SAFETY: `_mm256_mask_blend_ps` requires `AVX512F` and `AVX512VL` - implied by V4
        f32x8::from_underlying(self.arch(), unsafe {
            _mm256_mask_blend_ps(self.to_underlying(), y.to_underlying(), x.to_underlying())
        })
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_x86_f32 {
    use super::*;
    use crate::{reference::ReferenceScalarOps, test_utils};

    // Test loading logic - ensure that no out of bounds accesses are made.
    // In particular, this is meant to be run under `Miri` to ensure that our guarantees
    // regarding out-of-bounds accesses are honored.
    #[test]
    fn miri_test_load() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_load_simd::<f32, 8, f32x8>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_store_simd::<f32, 8, f32x8>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::ops::test_splat::<f32, 8, f32x8>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(f32x8, 0x3824379d4a43a416, V4::new_checked_uncached());
    test_utils::ops::test_sub!(f32x8, 0x548fc74c07ba425d, V4::new_checked_uncached());
    test_utils::ops::test_mul!(f32x8, 0x6d340672ff91b256, V4::new_checked_uncached());
    test_utils::ops::test_fma!(f32x8, 0x5f566d8968d4d201, V4::new_checked_uncached());
    test_utils::ops::test_minmax!(f32x8, 0x6d7fc8ed6d852187, V4::new_checked_uncached());
    test_utils::ops::test_abs!(f32x8, 0x2a4a9651d8ebe912, V4::new_checked_uncached());

    test_utils::ops::test_cmp!(f32x8, 0x3229520775597f50, V4::new_checked_uncached());
    test_utils::ops::test_splitjoin!(f32x8 => f32x4, 0x8fc0993e35ed899c, V4::new_checked_uncached());
    test_utils::ops::test_select!(f32x8, 0x45106cfb82bf69f3, V4::new_checked_uncached());

    // Reductions
    test_utils::ops::test_sumtree!(f32x8, 0x4e18be1451961654, V4::new_checked_uncached());
}

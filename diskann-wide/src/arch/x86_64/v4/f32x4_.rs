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
        macros::{self, X86Default, X86LoadStore, X86Splat},
        v3,
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

macros::x86_define_register!(f32x4, __m128, BitMask<4, V4>, f32, 4, V4);
macros::x86_define_splat!(f32x4, _mm_set1_ps, "sse");
macros::x86_define_default!(f32x4, _mm_setzero_ps, "sse");
macros::x86_retarget!(f32x4 => v3::f32x4);

helpers::unsafe_map_binary_op!(f32x4, std::ops::Add, add, _mm_add_ps, "sse");
helpers::unsafe_map_binary_op!(f32x4, std::ops::Sub, sub, _mm_sub_ps, "sse");
helpers::unsafe_map_binary_op!(f32x4, std::ops::Mul, mul, _mm_mul_ps, "sse");

impl f32x4 {
    #[inline(always)]
    fn is_nan(self) -> BitMask<4, V4> {
        // NOTE: `_CMP_UNORD_Q` returns `true` if either argument is NaN. Since we compare
        // `self` with `self`, this returns `true` exactly when `self` is NaN.
        BitMask::from_underlying(
            self.arch(),
            // SAFETY: `_mm_cmp_ps_mask` requires AVX512F + AVX512VL, both of which
            // are implied by the implicit V4 architecture.
            unsafe { _mm_cmp_ps_mask(self.to_underlying(), self.to_underlying(), _CMP_UNORD_Q) },
        )
    }
}

impl SIMDMulAdd for f32x4 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        // SAFETY: `_mm_fmadd_ps` requires FMA - implied by V4
        Self(unsafe { _mm_fmadd_ps(self.0, rhs.0, accumulator.0) })
    }
}

impl SIMDMinMax for f32x4 {
    #[inline(always)]
    fn min_simd(self, rhs: Self) -> Self {
        // SAFETY: `_mm_min_ps` requires SSE, which is implied by the V4 architecture.
        Self(unsafe { _mm_min_ps(self.0, rhs.0) })
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
        // SAFETY: `_mm_max_ps` requires SSE, which is implied by the V4 architecture.
        Self(unsafe { _mm_max_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn max_simd_standard(self, rhs: Self) -> Self {
        // NOTE: The behavior of the non-IEEE compliant max is order dependent and thus
        // it is important that the order of the arguments is swapped.
        let max = rhs.max_simd(self);
        self.is_nan().select(rhs, max)
    }
}

impl SIMDAbs for f32x4 {
    #[inline(always)]
    fn abs_simd(self) -> Self {
        self.from(self.retarget().abs_simd())
    }
}

macros::x86_avx512_load_store!(
    f32x4,
    _mm_loadu_ps,
    _mm_maskz_loadu_ps,
    _mm_storeu_ps,
    _mm_mask_storeu_ps,
    f32,
    "avx512f,avx512vl"
);

impl SIMDPartialEq for f32x4 {
    #[inline(always)]
    fn eq_simd(self, other: Self) -> Self::Mask {
        // SAFETY: `_mm_cmp_ps_mask` requires `AVX512F` and `AVX512VL` - implied by V4
        Self::Mask::from_underlying(self.arch(), unsafe {
            _mm_cmp_ps_mask(self.to_underlying(), other.to_underlying(), _CMP_EQ_OQ)
        })
    }

    #[inline(always)]
    fn ne_simd(self, other: Self) -> Self::Mask {
        // SAFETY: `_mm_cmp_ps_mask` requires `AVX512F` and `AVX512VL` - implied by V4
        Self::Mask::from_underlying(self.arch(), unsafe {
            _mm_cmp_ps_mask(self.to_underlying(), other.to_underlying(), _CMP_NEQ_UQ)
        })
    }
}

impl SIMDPartialOrd for f32x4 {
    #[inline(always)]
    fn lt_simd(self, other: Self) -> Self::Mask {
        // SAFETY: `_mm_cmp_ps_mask` requires `AVX512F` and `AVX512VL` - implied by V4
        Self::Mask::from_underlying(self.arch(), unsafe {
            _mm_cmp_ps_mask(self.to_underlying(), other.to_underlying(), _CMP_LT_OQ)
        })
    }

    #[inline(always)]
    fn le_simd(self, other: Self) -> Self::Mask {
        // SAFETY: `_mm_cmp_ps_mask` requires `AVX512F` and `AVX512VL` - implied by V4
        Self::Mask::from_underlying(self.arch(), unsafe {
            _mm_cmp_ps_mask(self.to_underlying(), other.to_underlying(), _CMP_LE_OQ)
        })
    }

    #[inline(always)]
    fn gt_simd(self, other: Self) -> Self::Mask {
        // SAFETY: `_mm_cmp_ps_mask` requires `AVX512F` and `AVX512VL` - implied by V4
        Self::Mask::from_underlying(self.arch(), unsafe {
            _mm_cmp_ps_mask(self.to_underlying(), other.to_underlying(), _CMP_GT_OQ)
        })
    }

    #[inline(always)]
    fn ge_simd(self, other: Self) -> Self::Mask {
        // SAFETY: `_mm_cmp_ps_mask` requires `AVX512F` and `AVX512VL` - implied by V4
        Self::Mask::from_underlying(self.arch(), unsafe {
            _mm_cmp_ps_mask(self.to_underlying(), other.to_underlying(), _CMP_GE_OQ)
        })
    }
}

impl SIMDSumTree for f32x4 {
    #[inline(always)]
    fn sum_tree(self) -> f32 {
        self.retarget().sum_tree()
    }
}

impl SIMDSelect<f32x4> for BitMask<4, V4> {
    #[inline(always)]
    fn select(self, x: f32x4, y: f32x4) -> f32x4 {
        // SAFETY: `_mm_mask_blend_ps` requires `AVX512F` and `AVX512VL` - implied by V4
        f32x4::from_underlying(self.arch(), unsafe {
            _mm_mask_blend_ps(self.to_underlying(), y.to_underlying(), x.to_underlying())
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
            test_utils::test_load_simd::<f32, 4, f32x4>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_store_simd::<f32, 4, f32x4>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::ops::test_splat::<f32, 4, f32x4>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(f32x4, 0xcd7a8fea9a3fb727, V4::new_checked_uncached());
    test_utils::ops::test_sub!(f32x4, 0x3f6562c94c923238, V4::new_checked_uncached());
    test_utils::ops::test_mul!(f32x4, 0x07e48666c0fc564c, V4::new_checked_uncached());
    test_utils::ops::test_fma!(f32x4, 0xcfde9d031302cf2c, V4::new_checked_uncached());
    test_utils::ops::test_minmax!(f32x4, 0x6d7fc8ed6d852187, V4::new_checked_uncached());
    test_utils::ops::test_abs!(f32x4, 0x8e6d9944c9c43a74, V4::new_checked_uncached());

    test_utils::ops::test_cmp!(f32x4, 0xc4f468b224622326, V4::new_checked_uncached());
    test_utils::ops::test_select!(f32x4, 0xdeb658b3e87755d0, V4::new_checked_uncached());

    // Reductions
    test_utils::ops::test_sumtree!(f32x4, 0x828bd890a470dc4d, V4::new_checked_uncached());
}

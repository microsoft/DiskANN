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
        v4::f32x8_::f32x8,
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

macros::x86_define_register!(f32x16, __m512, BitMask<16, V4>, f32, 16, V4);
macros::x86_define_splat!(f32x16, _mm512_set1_ps, "avx512f");
macros::x86_define_default!(f32x16, _mm512_setzero_ps, "avx512f");

impl crate::SplitJoin for f32x16 {
    type Halved = f32x8;

    #[inline(always)]
    fn split(self) -> crate::LoHi<Self::Halved> {
        // SAFETY: `_mm512_extractf32x8_ps` requires AVX512DQ - implied by V4
        unsafe {
            crate::LoHi::new(
                f32x8(_mm512_extractf32x8_ps(self.0, 0)),
                f32x8(_mm512_extractf32x8_ps(self.0, 1)),
            )
        }
    }

    #[inline(always)]
    fn join(lohi: crate::LoHi<Self::Halved>) -> Self {
        // SAFETY: Required by instantiator.
        let v = Self::default(lohi.lo.arch()).to_underlying();

        // SAFETY: `_mm512_insertf32x8` requires `AVX512DQ` - implied by V4.
        let v = unsafe { _mm512_insertf32x8(_mm512_insertf32x8(v, lohi.lo.0, 0), lohi.hi.0, 1) };
        Self(v)
    }
}

helpers::unsafe_map_binary_op!(f32x16, std::ops::Add, add, _mm512_add_ps, "avx");
helpers::unsafe_map_binary_op!(f32x16, std::ops::Sub, sub, _mm512_sub_ps, "avx");
helpers::unsafe_map_binary_op!(f32x16, std::ops::Mul, mul, _mm512_mul_ps, "avx");

impl f32x16 {
    #[inline(always)]
    fn is_nan(self) -> BitMask<16, V4> {
        // NOTE: `_CMP_UNORD_Q` returns `true` only if both arguments are NAN.
        BitMask::from_underlying(
            self.arch(),
            // SAFETY: `_mm512_cmp_ps_mask` requires AVX512F, which is implied by the
            // implicit V4 architecture.
            unsafe { _mm512_cmp_ps_mask(self.to_underlying(), self.to_underlying(), _CMP_UNORD_Q) },
        )
    }
}

impl SIMDMulAdd for f32x16 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        // SAFETY: `_mm512_fmadd_ps` requires `AVX512F` - implied by V4.
        Self(unsafe { _mm512_fmadd_ps(self.0, rhs.0, accumulator.0) })
    }
}

impl SIMDMinMax for f32x16 {
    #[inline(always)]
    fn min_simd(self, rhs: Self) -> Self {
        // SAFETY: `_mm512_min_ps` requires AVX512F, which is implied by the V4 architecture.
        Self(unsafe { _mm512_min_ps(self.0, rhs.0) })
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
        // SAFETY: `_mm512_max_ps` requires AVX512F, which is implied by the V4 architecture.
        Self(unsafe { _mm512_max_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn max_simd_standard(self, rhs: Self) -> Self {
        // NOTE: The behavior of the non-IEEE compliant max is order dependent and thus
        // it is important that the order of the arguments is swapped.
        let max = rhs.max_simd(self);
        self.is_nan().select(rhs, max)
    }
}

impl SIMDAbs for f32x16 {
    #[inline(always)]
    fn abs_simd(self) -> Self {
        // SAFETY: `_mm512_abs_ps` requires AVX512F - implied by V4.
        Self(unsafe { _mm512_abs_ps(self.to_underlying()) })
    }
}

macros::x86_avx512_load_store!(
    f32x16,
    _mm512_loadu_ps,
    _mm512_maskz_loadu_ps,
    _mm512_storeu_ps,
    _mm512_mask_storeu_ps,
    f32,
    "avx512f"
);

impl SIMDPartialEq for f32x16 {
    #[inline(always)]
    fn eq_simd(self, other: Self) -> Self::Mask {
        // SAFETY: `_mm512_cmp_ps_mask` requires AVX512F - implied by V4
        Self::Mask::from_underlying(self.arch(), unsafe {
            _mm512_cmp_ps_mask(self.to_underlying(), other.to_underlying(), _CMP_EQ_OQ)
        })
    }

    #[inline(always)]
    fn ne_simd(self, other: Self) -> Self::Mask {
        // SAFETY: `_mm512_cmp_ps_mask` requires AVX512F - implied by V4
        Self::Mask::from_underlying(self.arch(), unsafe {
            _mm512_cmp_ps_mask(self.to_underlying(), other.to_underlying(), _CMP_NEQ_UQ)
        })
    }
}

impl SIMDPartialOrd for f32x16 {
    #[inline(always)]
    fn lt_simd(self, other: Self) -> Self::Mask {
        Self::Mask::from_underlying(
            self.arch(),
            // SAFETY: `_mm512_cmp_ps_mask` requires AVX512F - implied by V4
            unsafe { _mm512_cmp_ps_mask(self.to_underlying(), other.to_underlying(), _CMP_LT_OQ) },
        )
    }

    #[inline(always)]
    fn le_simd(self, other: Self) -> Self::Mask {
        Self::Mask::from_underlying(
            self.arch(),
            // SAFETY: `_mm512_cmp_ps_mask` requires AVX512F - implied by V4
            unsafe { _mm512_cmp_ps_mask(self.to_underlying(), other.to_underlying(), _CMP_LE_OQ) },
        )
    }

    #[inline(always)]
    fn gt_simd(self, other: Self) -> Self::Mask {
        Self::Mask::from_underlying(
            self.arch(),
            // SAFETY: `_mm512_cmp_ps_mask` requires AVX512F - implied by V4
            unsafe { _mm512_cmp_ps_mask(self.to_underlying(), other.to_underlying(), _CMP_GT_OQ) },
        )
    }

    #[inline(always)]
    fn ge_simd(self, other: Self) -> Self::Mask {
        Self::Mask::from_underlying(
            self.arch(),
            // SAFETY: `_mm512_cmp_ps_mask` requires AVX512F - implied by V4
            unsafe { _mm512_cmp_ps_mask(self.to_underlying(), other.to_underlying(), _CMP_GE_OQ) },
        )
    }
}

impl SIMDSumTree for f32x16 {
    #[inline(always)]
    fn sum_tree(self) -> f32 {
        // SAFETY: `_mm512_reduce_add_ps` requires AVX512F - implied by V4
        unsafe { _mm512_reduce_add_ps(self.to_underlying()) }
    }
}

impl SIMDSelect<f32x16> for BitMask<16, V4> {
    #[inline(always)]
    fn select(self, x: f32x16, y: f32x16) -> f32x16 {
        // SAFETY: `_mm512_mask_blend_ps` requires AVX512F - implied by V4
        f32x16::from_underlying(self.arch(), unsafe {
            _mm512_mask_blend_ps(self.to_underlying(), y.to_underlying(), x.to_underlying())
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
            test_utils::test_load_simd::<f32, 16, f32x16>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_store_simd::<f32, 16, f32x16>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::ops::test_splat::<f32, 16, f32x16>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(f32x16, 0xa8989b97ca888d11, V4::new_checked_uncached());
    test_utils::ops::test_sub!(f32x16, 0xb2554fc13fdc1182, V4::new_checked_uncached());
    test_utils::ops::test_mul!(f32x16, 0x23becaa968b0cd71, V4::new_checked_uncached());
    test_utils::ops::test_fma!(f32x16, 0x32a814070a93df4e, V4::new_checked_uncached());
    test_utils::ops::test_minmax!(f32x16, 0x6d7fc8ed6d852187, V4::new_checked_uncached());
    test_utils::ops::test_abs!(f32x16, 0x6799e60873a2efe2, V4::new_checked_uncached());

    test_utils::ops::test_cmp!(f32x16, 0x1246278e242caecc, V4::new_checked_uncached());
    test_utils::ops::test_splitjoin!(f32x16 => f32x8, 0xde4ff375903351e3, V4::new_checked_uncached());
    test_utils::ops::test_select!(f32x16, 0xcfdfd54e2088dd90, V4::new_checked_uncached());

    // Reductions
    test_utils::ops::test_sumtree!(f32x16, 0x0180a265222e3fcf, V4::new_checked_uncached());
}

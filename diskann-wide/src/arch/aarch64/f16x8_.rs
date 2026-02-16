/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

use crate::{
    Emulated,
    arch::Scalar,
    constant::Const,
    traits::{SIMDMask, SIMDVector},
};

use half::f16;

// AArch64 masks
use super::{
    Neon, f16x4, f32x8,
    macros::{self, AArchLoadStore, AArchSplat},
    masks::mask16x8,
};

// AArch64 intrinsics
use std::arch::aarch64::*;

/////////////////////
// 16-bit floating //
/////////////////////

macros::aarch64_define_register!(f16x8, uint16x8_t, mask16x8, f16, 8, Neon);
macros::aarch64_splitjoin!(f16x8, f16x4, vget_low_u16, vget_high_u16, vcombine_u16);

impl AArchSplat for f16x8 {
    #[inline(always)]
    fn aarch_splat(_: Neon, value: f16) -> Self {
        // SAFETY: Allowed by the `Neon` architecture.
        Self(unsafe { vmovq_n_u16(value.to_bits()) })
    }

    #[inline(always)]
    fn aarch_default(arch: Neon) -> Self {
        Self::aarch_splat(arch, f16::default())
    }
}

impl AArchLoadStore for f16x8 {
    #[inline(always)]
    unsafe fn load_simd(_: Neon, ptr: *const f16) -> Self {
        // SAFETY: Pointer access safety inhereted from the caller.Allowed by the `Neon`
        // architecture.
        Self(unsafe { vld1q_u16(ptr.cast::<u16>()) })
    }

    #[inline(always)]
    unsafe fn load_simd_masked_logical(arch: Neon, ptr: *const f16, mask: Self::Mask) -> Self {
        // SAFETY: Pointer access safety inhereted from the caller.
        let e = unsafe {
            Emulated::<f16, 8>::load_simd_masked_logical(Scalar, ptr, mask.bitmask().as_scalar())
        };
        Self::from_array(arch, e.to_array())
    }

    #[inline(always)]
    unsafe fn load_simd_first(arch: Neon, ptr: *const f16, first: usize) -> Self {
        // SAFETY: Pointer access safety inhereted from the caller.
        let e = unsafe { Emulated::<f16, 8>::load_simd_first(Scalar, ptr, first) };
        Self::from_array(arch, e.to_array())
    }

    #[inline(always)]
    unsafe fn store_simd(self, ptr: *mut <Self as SIMDVector>::Scalar) {
        // SAFETY: Pointer access safety inhereted from the caller. Use of the instruction
        // is allowed by the `Neon` architecture.
        unsafe { vst1q_u16(ptr.cast::<u16>(), self.0) }
    }

    #[inline(always)]
    unsafe fn store_simd_masked_logical(self, ptr: *mut f16, mask: Self::Mask) {
        let e = Emulated::<f16, 8>::from_array(Scalar, self.to_array());
        // SAFETY: Pointer access safety inhereted from the caller.
        unsafe { e.store_simd_masked_logical(ptr, mask.bitmask().as_scalar()) }
    }

    #[inline(always)]
    unsafe fn store_simd_first(self, ptr: *mut f16, first: usize) {
        let e = Emulated::<f16, 8>::from_array(Scalar, self.to_array());
        // SAFETY: Pointer access safety inhereted from the caller.
        unsafe { e.store_simd_first(ptr, first) }
    }
}

//------------//
// Conversion //
//------------//

impl crate::SIMDCast<f32> for f16x8 {
    type Cast = f32x8;

    #[inline(always)]
    fn simd_cast(self) -> f32x8 {
        self.into()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils;

    #[test]
    fn miri_test_load() {
        test_utils::test_load_simd::<f16, 8, f16x8>(Neon::new_checked().unwrap());
    }

    #[test]
    fn miri_test_store() {
        test_utils::test_store_simd::<f16, 8, f16x8>(Neon::new_checked().unwrap());
    }

    // constructors
    #[test]
    fn test_constructors() {
        test_utils::ops::test_splat::<f16, 8, f16x8>(Neon::new_checked().unwrap());
    }

    test_utils::ops::test_splitjoin!(f16x8 => f16x4, 0xa4d00a4d04293967, Neon::new_checked());

    // Conversions
    test_utils::ops::test_cast!(f16x8 => f32x8, 0x37314659b022466a, Neon::new_checked());
}

/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

use crate::{
    Emulated,
    constant::Const,
    helpers,
    traits::{SIMDMask, SIMDMulAdd, SIMDPartialEq, SIMDPartialOrd, SIMDSumTree, SIMDVector},
};

// AArch64 masks
use super::{
    Neon, internal,
    macros::{self, AArchLoadStore, AArchSplat},
    masks::mask32x2,
};

// AArch64 intrinsics
use std::arch::aarch64::*;

/////////////////////
// 32-bit floating //
/////////////////////

macros::aarch64_define_register!(f32x2, float32x2_t, mask32x2, f32, 2, Neon);
macros::aarch64_define_splat!(f32x2, vmov_n_f32);
macros::aarch64_define_loadstore!(f32x2, vld1_f32, internal::load_first::f32x2, vst1_f32, 2);

helpers::unsafe_map_binary_op!(f32x2, std::ops::Add, add, vadd_f32, "neon");
helpers::unsafe_map_binary_op!(f32x2, std::ops::Sub, sub, vsub_f32, "neon");
helpers::unsafe_map_binary_op!(f32x2, std::ops::Mul, mul, vmul_f32, "neon");
macros::aarch64_define_fma!(f32x2, vfma_f32);

macros::aarch64_define_cmp!(
    f32x2,
    vceq_f32,
    (vmvn_u32),
    vclt_f32,
    vcle_f32,
    vcgt_f32,
    vcge_f32
);

impl SIMDSumTree for f32x2 {
    #[inline(always)]
    fn sum_tree(self) -> f32 {
        if cfg!(miri) {
            self.emulated().sum_tree()
        } else {
            // SAFETY: The presence of `Neon` enables the use of "neon" intrinsics.
            unsafe { vaddv_f32(self.to_underlying()) }
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{arch::aarch64::test_neon, reference::ReferenceScalarOps, test_utils};

    #[test]
    fn miri_test_load() {
        if let Some(arch) = test_neon() {
            test_utils::test_load_simd::<f32, 2, f32x2>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = test_neon() {
            test_utils::test_store_simd::<f32, 2, f32x2>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = test_neon() {
            test_utils::ops::test_splat::<f32, 2, f32x2>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(f32x2, 0xcd7a8fea9a3fb727, test_neon());
    test_utils::ops::test_sub!(f32x2, 0x3f6562c94c923238, test_neon());
    test_utils::ops::test_mul!(f32x2, 0x07e48666c0fc564c, test_neon());
    test_utils::ops::test_fma!(f32x2, 0xcfde9d031302cf2c, test_neon());

    test_utils::ops::test_cmp!(f32x2, 0xc4f468b224622326, test_neon());

    test_utils::ops::test_sumtree!(f32x2, 0x828bd890a470dc4d, test_neon());
}

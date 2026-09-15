/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

use crate::{
    Emulated, SIMDAbs, SIMDCast, SIMDMask, SIMDMulAdd, SIMDPartialEq, SIMDPartialOrd, SIMDSelect,
    SIMDSumTree, SIMDVector, constant::Const, helpers,
};

// AArch64 masks
use super::{
    Neon, f32x2, internal,
    macros::{self, AArchLoadStore, AArchSplat},
    masks::mask32x2,
};

// AArch64 intrinsics
use std::arch::aarch64::*;

///////////////////
// 32-bit signed //
///////////////////

macros::aarch64_define_register!(i32x2, int32x2_t, mask32x2, i32, 2, Neon);
macros::aarch64_define_splat!(i32x2, vmov_n_s32);
macros::aarch64_define_loadstore!(i32x2, vld1_s32, internal::load_first::i32x2, vst1_s32, 2);

helpers::unsafe_map_binary_op!(i32x2, std::ops::Add, add, vadd_s32, "neon");
helpers::unsafe_map_binary_op!(i32x2, std::ops::Sub, sub, vsub_s32, "neon");
helpers::unsafe_map_binary_op!(i32x2, std::ops::Mul, mul, vmul_s32, "neon");
helpers::unsafe_map_unary_op!(i32x2, SIMDAbs, abs_simd, vabs_s32, "neon");
macros::aarch64_define_fma!(i32x2, vmla_s32);

macros::aarch64_define_cmp!(
    i32x2,
    vceq_s32,
    (vmvn_u32),
    vclt_s32,
    vcle_s32,
    vcgt_s32,
    vcge_s32
);
macros::aarch64_define_bitops!(
    i32x2,
    vmvn_s32,
    vand_s32,
    vorr_s32,
    veor_s32,
    (
        vshl_s32,
        32,
        vneg_s32,
        vmin_u32,
        vreinterpret_s32_u32,
        vreinterpret_u32_s32
    ),
    (u32, i32, vmov_n_s32),
);

impl SIMDSumTree for i32x2 {
    #[inline(always)]
    fn sum_tree(self) -> i32 {
        if cfg!(miri) {
            self.emulated().sum_tree()
        } else {
            // SAFETY: Allowed by the `Neon` architecture.
            unsafe { vaddv_s32(self.0) }
        }
    }
}

impl SIMDSelect<i32x2> for mask32x2 {
    #[inline(always)]
    fn select(self, x: i32x2, y: i32x2) -> i32x2 {
        // SAFETY: Allowed by the `Neon` architecture.
        i32x2(unsafe { vbsl_s32(self.0, x.0, y.0) })
    }
}

//-------------//
// Conversions //
//-------------//

helpers::unsafe_map_cast!(
    i32x2 => (f32, f32x2),
    vcvt_f32_s32,
    "neon"
);

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
            test_utils::test_load_simd::<i32, 2, i32x2>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = test_neon() {
            test_utils::test_store_simd::<i32, 2, i32x2>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = test_neon() {
            test_utils::ops::test_splat::<i32, 2, i32x2>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(i32x2, 0x3017fd73c99cc633, test_neon());
    test_utils::ops::test_sub!(i32x2, 0xfc627f10b5f8db8a, test_neon());
    test_utils::ops::test_mul!(i32x2, 0x0f4caa80eceaa523, test_neon());
    test_utils::ops::test_fma!(i32x2, 0xb8f702ba85375041, test_neon());
    test_utils::ops::test_abs!(i32x2, 0xb8f702ba85375041, test_neon());

    test_utils::ops::test_cmp!(i32x2, 0x941757bd5cc641a1, test_neon());

    // Bit ops
    test_utils::ops::test_bitops!(i32x2, 0xd62d8de09f82ed4e, test_neon());
    test_utils::ops::test_select!(i32x2, 0xd62d8de09f82ed4e, test_neon());

    // Reductions
    test_utils::ops::test_sumtree!(i32x2, 0xb9ac82ab23a855da, test_neon());

    // Conversions
    test_utils::ops::test_cast!(i32x2 => f32x2, 0xba8fe343fc9dbeff, test_neon());
}

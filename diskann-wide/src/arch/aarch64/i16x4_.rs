/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

use crate::{
    Emulated, SIMDAbs, SIMDMask, SIMDMulAdd, SIMDPartialEq, SIMDPartialOrd, SIMDVector,
    constant::Const, helpers,
};

// AArch64 masks
use super::{
    Neon, internal,
    macros::{self, AArchLoadStore, AArchSplat},
    masks::mask16x4,
};

// AArch64 intrinsics
use std::arch::aarch64::*;

///////////////////
// 16-bit signed //
///////////////////

macros::aarch64_define_register!(i16x4, int16x4_t, mask16x4, i16, 4, Neon);
macros::aarch64_define_splat!(i16x4, vmov_n_s16);
macros::aarch64_define_loadstore!(i16x4, vld1_s16, internal::load_first::i16x4, vst1_s16, 4);

helpers::unsafe_map_binary_op!(i16x4, std::ops::Add, add, vadd_s16, "neon");
helpers::unsafe_map_binary_op!(i16x4, std::ops::Sub, sub, vsub_s16, "neon");
helpers::unsafe_map_binary_op!(i16x4, std::ops::Mul, mul, vmul_s16, "neon");
helpers::unsafe_map_unary_op!(i16x4, SIMDAbs, abs_simd, vabs_s16, "neon");
macros::aarch64_define_fma!(i16x4, vmla_s16);

macros::aarch64_define_cmp!(
    i16x4,
    vceq_s16,
    (vmvn_u16),
    vclt_s16,
    vcle_s16,
    vcgt_s16,
    vcge_s16
);
macros::aarch64_define_bitops!(
    i16x4,
    vmvn_s16,
    vand_s16,
    vorr_s16,
    veor_s16,
    (
        vshl_s16,
        16,
        vneg_s16,
        vmin_u16,
        vreinterpret_s16_u16,
        vreinterpret_u16_s16
    ),
    (u16, i16, vmov_n_s16),
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
            test_utils::test_load_simd::<i16, 4, i16x4>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = test_neon() {
            test_utils::test_store_simd::<i16, 4, i16x4>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = test_neon() {
            test_utils::ops::test_splat::<i16, 4, i16x4>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(i16x4, 0x3017fd73c99cc633, test_neon());
    test_utils::ops::test_sub!(i16x4, 0xfc627f10b5f8db8a, test_neon());
    test_utils::ops::test_mul!(i16x4, 0x0f4caa80eceaa523, test_neon());
    test_utils::ops::test_fma!(i16x4, 0xb8f702ba85375041, test_neon());
    test_utils::ops::test_abs!(i16x4, 0xb8f702ba85375041, test_neon());

    test_utils::ops::test_cmp!(i16x4, 0x941757bd5cc641a1, test_neon());

    // Bit ops
    test_utils::ops::test_bitops!(i16x4, 0xd62d8de09f82ed4e, test_neon());
}

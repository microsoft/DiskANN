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
    Neon,
    macros::{self, AArchLoadStore, AArchSplat},
    masks::mask8x8,
};

// AArch64 intrinsics
use std::arch::aarch64::*;

//////////////////
// 8-bit signed //
//////////////////

macros::aarch64_define_register!(i8x8, int8x8_t, mask8x8, i8, 8, Neon);
macros::aarch64_define_splat!(i8x8, vmov_n_s8);
macros::aarch64_define_loadstore!(i8x8, vld1_s8, vst1_s8, 8);

helpers::unsafe_map_binary_op!(i8x8, std::ops::Add, add, vadd_s8, "neon");
helpers::unsafe_map_binary_op!(i8x8, std::ops::Sub, sub, vsub_s8, "neon");
helpers::unsafe_map_binary_op!(i8x8, std::ops::Mul, mul, vmul_s8, "neon");
helpers::unsafe_map_unary_op!(i8x8, SIMDAbs, abs_simd, vabs_s8, "neon");
macros::aarch64_define_fma!(i8x8, vmla_s8);

macros::aarch64_define_cmp!(i8x8, vceq_s8, (vmvn_u8), vclt_s8, vcle_s8, vcgt_s8, vcge_s8);
macros::aarch64_define_bitops!(
    i8x8,
    vmvn_s8,
    vand_s8,
    vorr_s8,
    veor_s8,
    (
        vshl_s8,
        8,
        vneg_s8,
        vmin_u8,
        vreinterpret_s8_u8,
        vreinterpret_u8_s8
    ),
    (u8, i8, vmov_n_s8),
);

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{reference::ReferenceScalarOps, test_utils};

    #[test]
    fn miri_test_load() {
        test_utils::test_load_simd::<i8, 8, i8x8>(Neon::new_checked().unwrap());
    }

    #[test]
    fn miri_test_store() {
        test_utils::test_store_simd::<i8, 8, i8x8>(Neon::new_checked().unwrap());
    }

    // constructors
    #[test]
    fn test_constructors() {
        test_utils::ops::test_splat::<i8, 8, i8x8>(Neon::new_checked().unwrap());
    }

    // Ops
    test_utils::ops::test_add!(i8x8, 0x3017fd73c99cc633, Neon::new_checked());
    test_utils::ops::test_sub!(i8x8, 0xfc627f10b5f8db8a, Neon::new_checked());
    test_utils::ops::test_mul!(i8x8, 0x0f4caa80eceaa523, Neon::new_checked());
    test_utils::ops::test_fma!(i8x8, 0xb8f702ba85375041, Neon::new_checked());
    test_utils::ops::test_abs!(i8x8, 0xb8f702ba85375041, Neon::new_checked());

    test_utils::ops::test_cmp!(i8x8, 0x941757bd5cc641a1, Neon::new_checked());

    // Bit ops
    test_utils::ops::test_bitops!(i8x8, 0xd62d8de09f82ed4e, Neon::new_checked());
}

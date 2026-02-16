/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

use crate::{
    Emulated,
    constant::Const,
    helpers,
    traits::{SIMDMask, SIMDMulAdd, SIMDPartialEq, SIMDPartialOrd, SIMDVector},
};

// AArch64 masks
use super::{
    Neon, algorithms,
    macros::{self, AArchLoadStore, AArchSplat},
    masks::mask8x8,
};

// AArch64 intrinsics
use std::arch::aarch64::*;

////////////////////
// 8-bit unsigned //
////////////////////

macros::aarch64_define_register!(u8x8, uint8x8_t, mask8x8, u8, 8, Neon);
macros::aarch64_define_splat!(u8x8, vmov_n_u8);
macros::aarch64_define_loadstore!(u8x8, vld1_u8, algorithms::load_first::u8x8, vst1_u8, 8);

helpers::unsafe_map_binary_op!(u8x8, std::ops::Add, add, vadd_u8, "neon");
helpers::unsafe_map_binary_op!(u8x8, std::ops::Sub, sub, vsub_u8, "neon");
helpers::unsafe_map_binary_op!(u8x8, std::ops::Mul, mul, vmul_u8, "neon");
macros::aarch64_define_fma!(u8x8, vmla_u8);

macros::aarch64_define_cmp!(u8x8, vceq_u8, (vmvn_u8), vclt_u8, vcle_u8, vcgt_u8, vcge_u8);
macros::aarch64_define_bitops!(
    u8x8,
    vmvn_u8,
    vand_u8,
    vorr_u8,
    veor_u8,
    (
        vshl_u8,
        8,
        vneg_s8,
        vmin_u8,
        vreinterpret_s8_u8,
        std::convert::identity
    ),
    (u8, i8, vmov_n_s8),
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
            test_utils::test_load_simd::<u8, 8, u8x8>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = test_neon() {
            test_utils::test_store_simd::<u8, 8, u8x8>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = test_neon() {
            test_utils::ops::test_splat::<u8, 8, u8x8>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(u8x8, 0x3017fd73c99cc633, test_neon());
    test_utils::ops::test_sub!(u8x8, 0xfc627f10b5f8db8a, test_neon());
    test_utils::ops::test_mul!(u8x8, 0x0f4caa80eceaa523, test_neon());
    test_utils::ops::test_fma!(u8x8, 0xb8f702ba85375041, test_neon());

    test_utils::ops::test_cmp!(u8x8, 0x941757bd5cc641a1, test_neon());

    // Bit ops
    test_utils::ops::test_bitops!(u8x8, 0xd62d8de09f82ed4e, test_neon());
}

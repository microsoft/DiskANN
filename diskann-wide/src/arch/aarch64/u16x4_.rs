/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

use crate::{
    Emulated, SIMDMask, SIMDMulAdd, SIMDPartialEq, SIMDPartialOrd, SIMDVector, constant::Const,
    helpers,
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

macros::aarch64_define_register!(u16x4, uint16x4_t, mask16x4, u16, 4, Neon);
macros::aarch64_define_splat!(u16x4, vmov_n_u16);
macros::aarch64_define_loadstore!(u16x4, vld1_u16, internal::load_first::u16x4, vst1_u16, 4);

helpers::unsafe_map_binary_op!(u16x4, std::ops::Add, add, vadd_u16, "neon");
helpers::unsafe_map_binary_op!(u16x4, std::ops::Sub, sub, vsub_u16, "neon");
helpers::unsafe_map_binary_op!(u16x4, std::ops::Mul, mul, vmul_u16, "neon");
macros::aarch64_define_fma!(u16x4, vmla_u16);

macros::aarch64_define_cmp!(
    u16x4,
    vceq_u16,
    (vmvn_u16),
    vclt_u16,
    vcle_u16,
    vcgt_u16,
    vcge_u16
);
macros::aarch64_define_bitops!(
    u16x4,
    vmvn_u16,
    vand_u16,
    vorr_u16,
    veor_u16,
    (
        vshl_u16,
        16,
        vneg_s16,
        vmin_u16,
        vreinterpret_s16_u16,
        std::convert::identity
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
            test_utils::test_load_simd::<u16, 4, u16x4>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = test_neon() {
            test_utils::test_store_simd::<u16, 4, u16x4>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = test_neon() {
            test_utils::ops::test_splat::<u16, 4, u16x4>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(u16x4, 0x3017fd73c99cc633, test_neon());
    test_utils::ops::test_sub!(u16x4, 0xfc627f10b5f8db8a, test_neon());
    test_utils::ops::test_mul!(u16x4, 0x0f4caa80eceaa523, test_neon());
    test_utils::ops::test_fma!(u16x4, 0xb8f702ba85375041, test_neon());

    test_utils::ops::test_cmp!(u16x4, 0x941757bd5cc641a1, test_neon());

    // Bit ops
    test_utils::ops::test_bitops!(u16x4, 0xd62d8de09f82ed4e, test_neon());
}

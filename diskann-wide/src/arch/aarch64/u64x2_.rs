/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

use crate::{
    Emulated,
    arch::Scalar,
    constant::Const,
    helpers,
    traits::{SIMDMask, SIMDMulAdd, SIMDPartialEq, SIMDPartialOrd, SIMDVector},
};

// AArch64 masks
use super::{
    Neon, algorithms,
    macros::{self, AArchLoadStore, AArchSplat},
    masks::mask64x2,
};

// AArch64 intrinsics
use std::arch::aarch64::*;

/////////////////////
// 64-bit unsigned //
/////////////////////

#[inline(always)]
pub(super) unsafe fn emulated_vmvnq_u64(x: uint64x2_t) -> uint64x2_t {
    let x: [u64; 2] = u64x2(x).to_array();
    let mapped: [u64; 2] = core::array::from_fn(|i| !x[i]);
    // SAFETY: This is only called in a context where the caller guarantees `Neon` is
    // available.
    u64x2::from_array(unsafe { Neon::new() }, mapped).0
}

#[inline(always)]
pub(super) unsafe fn emulated_vminq_u64(x: uint64x2_t, y: uint64x2_t) -> uint64x2_t {
    let x = u64x2(x).to_array();
    let y = u64x2(y).to_array();
    let mapped: [u64; 2] = core::array::from_fn(|i| x[i].min(y[i]));
    // SAFETY: This is only called in a context where the caller guarantees `Neon` is
    // available.
    u64x2::from_array(unsafe { Neon::new() }, mapped).0
}

macros::aarch64_define_register!(u64x2, uint64x2_t, mask64x2, u64, 2, Neon);
macros::aarch64_define_splat!(u64x2, vmovq_n_u64);
macros::aarch64_define_loadstore!(
    u64x2,
    vld1q_u64,
    algorithms::load_first::u64x2,
    vst1q_u64,
    2
);

helpers::unsafe_map_binary_op!(u64x2, std::ops::Add, add, vaddq_u64, "neon");
helpers::unsafe_map_binary_op!(u64x2, std::ops::Sub, sub, vsubq_u64, "neon");

impl std::ops::Mul for u64x2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let x = Emulated::<u64, 2>::from_array(Scalar, self.to_array());
        let y = Emulated::<u64, 2>::from_array(Scalar, rhs.to_array());
        Self::from_array(self.arch(), (x * y).to_array())
    }
}

macros::aarch64_define_fma!(u64x2, integer);

macros::aarch64_define_cmp!(
    u64x2,
    vceqq_u64,
    (emulated_vmvnq_u64),
    vcltq_u64,
    vcleq_u64,
    vcgtq_u64,
    vcgeq_u64
);
macros::aarch64_define_bitops!(
    u64x2,
    emulated_vmvnq_u64,
    vandq_u64,
    vorrq_u64,
    veorq_u64,
    (
        vshlq_u64,
        64,
        vnegq_s64,
        emulated_vminq_u64,
        vreinterpretq_s64_u64,
        std::convert::identity
    ),
    (u64, i64, vmovq_n_s64),
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
        test_utils::test_load_simd::<u64, 2, u64x2>(Neon::new_checked().unwrap());
    }

    #[test]
    fn miri_test_store() {
        test_utils::test_store_simd::<u64, 2, u64x2>(Neon::new_checked().unwrap());
    }

    // constructors
    #[test]
    fn test_constructors() {
        test_utils::ops::test_splat::<u64, 2, u64x2>(Neon::new_checked().unwrap());
    }

    // Binary Ops
    test_utils::ops::test_add!(u64x2, 0x8d7bf28b1c6e2545, Neon::new_checked());
    test_utils::ops::test_sub!(u64x2, 0x4a1c644a1a910bed, Neon::new_checked());
    test_utils::ops::test_mul!(u64x2, 0xf42ee707a808fd10, Neon::new_checked());
    test_utils::ops::test_fma!(u64x2, 0x28540d9936a9e803, Neon::new_checked());

    test_utils::ops::test_cmp!(u64x2, 0xfae27072c6b70885, Neon::new_checked());

    // Bit ops
    test_utils::ops::test_bitops!(u64x2, 0xbe927713ea310164, Neon::new_checked());
}

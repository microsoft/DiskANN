/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The functions defined in this module are meant for the purposes of automated disassambly
//! to inspect code generation.
//!
//! These functions should not be called as a replacement for their publically exposed
//! alternatives.

use diskann_wide::{
    arch::x86_64::{V3, V4},
    Architecture,
};

use crate::{
    bits::{BitSlice, BitTranspose, Dense, Unsigned},
    distances::{self, MathematicalResult},
};

//////////
// Bits //
//////////

type USlice<'a, const N: usize, Perm = Dense> = BitSlice<'a, N, Unsigned, Perm>;
type MR<T> = MathematicalResult<T>;

//----------------//
// Regular Slices //
//----------------//

// V3 //

#[inline(never)]
pub fn bits_v3_l2_bu8_bu8(arch: V3, x: USlice<'_, 8>, y: USlice<'_, 8>) -> MR<u32> {
    arch.run2_inline(distances::SquaredL2, x, y)
}

#[inline(never)]
pub fn bits_v3_l2_bu4_bu4(arch: V3, x: USlice<'_, 4>, y: USlice<'_, 4>) -> MR<u32> {
    arch.run2_inline(distances::SquaredL2, x, y)
}

#[inline(never)]
pub fn bits_v3_l2_bu2_bu2(arch: V3, x: USlice<'_, 2>, y: USlice<'_, 2>) -> MR<u32> {
    arch.run2_inline(distances::SquaredL2, x, y)
}

#[inline(never)]
pub fn bits_v3_l2_bu1_bu1(arch: V3, x: USlice<'_, 1>, y: USlice<'_, 1>) -> MR<u32> {
    arch.run2_inline(distances::SquaredL2, x, y)
}

#[inline(never)]
pub fn bits_v3_ip_bu8_bu8(arch: V3, x: USlice<'_, 8>, y: USlice<'_, 8>) -> MR<u32> {
    arch.run2_inline(distances::InnerProduct, x, y)
}

#[inline(never)]
pub fn bits_v3_ip_bu4_bu4(arch: V3, x: USlice<'_, 4>, y: USlice<'_, 4>) -> MR<u32> {
    arch.run2_inline(distances::InnerProduct, x, y)
}

#[inline(never)]
pub fn bits_v3_ip_bu2_bu2(arch: V3, x: USlice<'_, 2>, y: USlice<'_, 2>) -> MR<u32> {
    arch.run2_inline(distances::InnerProduct, x, y)
}

#[inline(never)]
pub fn bits_v3_ip_bu1_bu1(arch: V3, x: USlice<'_, 1>, y: USlice<'_, 1>) -> MR<u32> {
    arch.run2_inline(distances::InnerProduct, x, y)
}

// V4 //

#[inline(never)]
pub fn bits_v4_l2_bu1_bu1(arch: V4, x: USlice<'_, 1>, y: USlice<'_, 1>) -> MR<u32> {
    arch.run2_inline(distances::SquaredL2, x, y)
}

#[inline(never)]
pub fn bits_v4_ip_bu1_bu1(arch: V4, x: USlice<'_, 1>, y: USlice<'_, 1>) -> MR<u32> {
    arch.run2_inline(distances::InnerProduct, x, y)
}

#[inline(never)]
pub fn bits_v4_ip_bu2_bu2(arch: V4, x: USlice<'_, 2>, y: USlice<'_, 2>) -> MR<u32> {
    arch.run2_inline(distances::InnerProduct, x, y)
}

//------------//
// Transposed //
//------------//

// V3 //

#[inline(never)]
pub fn bits_v3_ip_tu4_bu1(arch: V3, x: USlice<'_, 4, BitTranspose>, y: USlice<'_, 1>) -> MR<u32> {
    arch.run2_inline(distances::InnerProduct, x, y)
}

// V4 //

#[inline(never)]
pub fn bits_v4_ip_tu4_bu1(arch: V4, x: USlice<'_, 4, BitTranspose>, y: USlice<'_, 1>) -> MR<u32> {
    arch.run2_inline(distances::InnerProduct, x, y)
}

//----------------//
// Full Precision //
//----------------//

#[inline(never)]
pub fn bits_v3_ip_f32_bu8(arch: V3, x: &[f32], y: USlice<'_, 8>) -> MR<f32> {
    arch.run2_inline(distances::InnerProduct, x, y)
}

#[inline(never)]
pub fn bits_v3_ip_f32_bu4(arch: V3, x: &[f32], y: USlice<'_, 4>) -> MR<f32> {
    arch.run2_inline(distances::InnerProduct, x, y)
}

#[inline(never)]
pub fn bits_v3_ip_f32_bu2(arch: V3, x: &[f32], y: USlice<'_, 2>) -> MR<f32> {
    arch.run2_inline(distances::InnerProduct, x, y)
}

#[inline(never)]
pub fn bits_v3_ip_f32_bu1(arch: V3, x: &[f32], y: USlice<'_, 1>) -> MR<f32> {
    arch.run2_inline(distances::InnerProduct, x, y)
}

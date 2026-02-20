/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

use std::arch::aarch64::*;

use crate::SIMDVector;

use super::{i8x16, i32x8, u8x16, u32x8};

/// Accumulates squared Euclidean distances between 16 pairs of `u8` lanes.
///
/// Behaves as if each `u8` lane were widened to `u32`, the absolute difference between
/// lanes taken, and the results squared before being added to the accumulator.
///
/// The mapping from input lane index to accumulator lane is **opaque** — callers
/// must not rely on any particular ordering of intermediate partial sums within
/// `acc`. This function is intended for computing full squared-L2 distances
/// where the accumulator is reduced to a single scalar at the end.
#[inline(always)]
pub fn squared_euclidean_accum_u8x16(x: u8x16, y: u8x16, acc: u32x8) -> u32x8 {
    let arch = acc.arch();

    if cfg!(miri) {
        let x = x.to_array();
        let y = y.to_array();
        let acc = acc.to_array();

        let op = |x: u8, y: u8| -> u32 {
            let x: i32 = x.into();
            let y: i32 = y.into();
            let diff = x - y;
            (diff * diff) as u32
        };

        let f = |i: usize| -> u32 {
            let base = 8 * (i / 4) + (i % 4);
            acc[i]
                .wrapping_add(op(x[base], y[base]))
                .wrapping_add(op(x[base + 4], y[base + 4]))
        };

        u32x8::from_array(arch, core::array::from_fn(f))
    } else {
        // SAFETY: Neon intrinsics are allowed by the `Neon` architecture.
        let (lo, hi) = unsafe {
            let x = x.to_underlying();
            let y = y.to_underlying();
            let (acc_lo, acc_hi) = acc.to_underlying();

            // The algorithm works like this:
            //
            // * `vsubl[_high]_u8` performs subtraction and widens to `u16`
            //
            // * We reinterpret from `u16` to `s16`, which is valid because all `u8` values
            //   can be represented as an `i16`, and the subtraction is wrapping, so negative
            //   values are represented properly.
            //
            // * Then, we use `vmlal_s16` to square the results and accumulate. All of this
            //   needs to be done in the signed space so that negative values get properly
            //   sign extended and thus behave correctly.
            //
            // * At the end, we can reinterpret back. Since multiplication and addition are
            //   the same for signed and unsigned numbers (once sign-extension is accounted
            //   for), the final result is correct.
            let lo = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(x), vget_low_u8(y)));
            let hi = vreinterpretq_s16_u16(vsubl_high_u8(x, y));

            let acc_lo = vmlal_s16(
                vreinterpretq_s32_u32(acc_lo),
                vget_low_s16(lo),
                vget_low_s16(lo),
            );
            let acc_hi = vmlal_s16(
                vreinterpretq_s32_u32(acc_hi),
                vget_low_s16(hi),
                vget_low_s16(hi),
            );

            let acc_lo = vmlal_high_s16(acc_lo, lo, lo);
            let acc_hi = vmlal_high_s16(acc_hi, hi, hi);

            (vreinterpretq_u32_s32(acc_lo), vreinterpretq_u32_s32(acc_hi))
        };

        u32x8::from_underlying(arch, (lo, hi))
    }
}

/// Accumulates squared Euclidean distances between 16 pairs of `i8` lanes.
///
/// Behaves as if each `i8` lane were widened to `i32`, the corresponding lanes
/// subtracted, and the results squared before being added to the accumulator.
///
/// The mapping from input lane index to accumulator lane is **opaque** — callers
/// must not rely on any particular ordering of intermediate partial sums within
/// `acc`. This function is intended for computing full squared-L2 distances
/// where the accumulator is reduced to a single scalar at the end.
#[inline(always)]
pub fn squared_euclidean_accum_i8x16(x: i8x16, y: i8x16, acc: i32x8) -> i32x8 {
    let arch = acc.arch();

    if cfg!(miri) {
        let x = x.to_array();
        let y = y.to_array();
        let acc = acc.to_array();

        let op = |x: i8, y: i8| -> i32 {
            let x: i32 = x.into();
            let y: i32 = y.into();
            let diff = x - y;
            diff * diff
        };

        let f = |i: usize| -> i32 {
            let base = 8 * (i / 4) + (i % 4);
            acc[i]
                .wrapping_add(op(x[base], y[base]))
                .wrapping_add(op(x[base + 4], y[base + 4]))
        };

        i32x8::from_array(arch, core::array::from_fn(f))
    } else {
        // SAFETY: Neon intrinsics are allowed by the `Neon` architecture.
        let (lo, hi) = unsafe {
            let x = x.to_underlying();
            let y = y.to_underlying();
            let (acc_lo, acc_hi) = acc.to_underlying();

            // This uses an approach similar to `u8x16`, but since everything is already
            // signed, we can avoid the pesky reinterprets.
            let lo = vsubl_s8(vget_low_s8(x), vget_low_s8(y));
            let hi = vsubl_high_s8(x, y);

            let acc_lo = vmlal_s16(acc_lo, vget_low_s16(lo), vget_low_s16(lo));
            let acc_hi = vmlal_s16(acc_hi, vget_low_s16(hi), vget_low_s16(hi));

            let acc_lo = vmlal_high_s16(acc_lo, lo, lo);
            let acc_hi = vmlal_high_s16(acc_hi, hi, hi);

            (acc_lo, acc_hi)
        };

        i32x8::from_underlying(arch, (lo, hi))
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{arch::aarch64::test_neon, test_utils::driver};

    #[test]
    fn test_squared_euclidean_accum_u8x16() {
        let arch = match test_neon() {
            Some(arch) => arch,
            None => return,
        };

        let f = move |x: &[u8], y: &[u8], acc: &[u32]| {
            let x: [u8; 16] = x.try_into().unwrap();
            let y: [u8; 16] = y.try_into().unwrap();
            let acc: [u32; 8] = acc.try_into().unwrap();

            // Compute expected result
            let expected: [u32; 8] = core::array::from_fn(|i| {
                let a = acc[i];
                let block = 8 * (i / 4);
                let offset = i % 4;

                let xi: i32 = x[block + offset].into();
                let yi: i32 = y[block + offset].into();
                let diff = xi - yi;
                let prod: u32 = (diff * diff).try_into().unwrap();
                let a = a.wrapping_add(prod);

                let xi: i32 = x[block + offset + 4].into();
                let yi: i32 = y[block + offset + 4].into();
                let diff = xi - yi;
                let prod: u32 = (diff * diff).try_into().unwrap();
                a.wrapping_add(prod)
            });

            // Compute Neon result
            let got = {
                let x = u8x16::from_array(arch, x);
                let y = u8x16::from_array(arch, y);
                let acc = u32x8::from_array(arch, acc);
                squared_euclidean_accum_u8x16(x, y, acc).to_array()
            };

            assert_eq!(
                got, expected,
                "failed on input x = {:?}, y = {:?}, acc = {:?}",
                x, y, acc
            );
        };

        driver::drive_ternary(&f, (16, 16, 8), 0xc0ffee);
    }

    #[test]
    fn test_squared_euclidean_accum_i8x16() {
        let arch = match test_neon() {
            Some(arch) => arch,
            None => return,
        };

        let f = move |x: &[i8], y: &[i8], acc: &[i32]| {
            let x: [i8; 16] = x.try_into().unwrap();
            let y: [i8; 16] = y.try_into().unwrap();
            let acc: [i32; 8] = acc.try_into().unwrap();

            // Compute expected result
            let expected: [i32; 8] = core::array::from_fn(|i| {
                let a = acc[i];
                let block = 8 * (i / 4);
                let offset = i % 4;

                let xi: i32 = x[block + offset].into();
                let yi: i32 = y[block + offset].into();
                let diff = xi - yi;
                let a = a.wrapping_add(diff * diff);

                let xi: i32 = x[block + offset + 4].into();
                let yi: i32 = y[block + offset + 4].into();
                let diff = xi - yi;
                a.wrapping_add(diff * diff)
            });

            // Compute Neon result
            let got = {
                let x = i8x16::from_array(arch, x);
                let y = i8x16::from_array(arch, y);
                let acc = i32x8::from_array(arch, acc);
                squared_euclidean_accum_i8x16(x, y, acc).to_array()
            };

            assert_eq!(
                got, expected,
                "failed on input x = {:?}, y = {:?}, acc = {:?}",
                x, y, acc
            );
        };

        driver::drive_ternary(&f, (16, 16, 8), 0xc0ffee);
    }
}

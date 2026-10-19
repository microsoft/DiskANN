/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::arch::aarch64::*;

use crate::{SIMDDotProduct, SIMDVector};

use super::{i8x16, i32x8, u8x16, u32x4};

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
pub fn squared_euclidean_accum_u8x16(x: u8x16, y: u8x16, acc: u32x4) -> u32x4 {
    let arch = acc.arch();

    if cfg!(miri) {
        let x = x.to_array();
        let y = y.to_array();
        let diff = u8x16::from_array(arch, core::array::from_fn(|i| x[i].abs_diff(y[i])));
        acc.dot_simd(diff, diff)
    } else {
        // SAFETY: Neon intrinsics are allowed by the `Neon` architecture.
        let diff = u8x16::from_underlying(arch, unsafe {
            vabdq_u8(x.to_underlying(), y.to_underlying())
        });
        acc.dot_simd(diff, diff)
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
            let acc: [u32; 4] = acc.try_into().unwrap();

            // Compute expected result
            let expected: [u32; 4] = core::array::from_fn(|i| {
                (4 * i..4 * i + 4).fold(acc[i], |sum, j| {
                    let diff = x[j] as i32 - y[j] as i32;
                    sum.wrapping_add((diff * diff) as u32)
                })
            });

            // Compute Neon result
            let got = {
                let x = u8x16::from_array(arch, x);
                let y = u8x16::from_array(arch, y);
                let acc = u32x4::from_array(arch, acc);
                squared_euclidean_accum_u8x16(x, y, acc).to_array()
            };

            assert_eq!(
                got, expected,
                "failed on input x = {:?}, y = {:?}, acc = {:?}",
                x, y, acc
            );
        };

        f(&[0; 16], &[u8::MAX; 16], &[u32::MAX; 4]);
        driver::drive_ternary(&f, (16, 16, 4), 0xc0ffee);
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

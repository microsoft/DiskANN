/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::arch::aarch64::*;

use crate::{SIMDDotProduct, SIMDVector};

use super::{i8x16, i32x4, u8x16, u32x4};

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
pub fn squared_euclidean_accum_i8x16(x: i8x16, y: i8x16, acc: i32x4) -> i32x4 {
    let arch = acc.arch();
    // SAFETY: Reinterpretation preserves the accumulator bits for unsigned accumulation.
    let acc = u32x4::from_underlying(arch, unsafe { vreinterpretq_u32_s32(acc.to_underlying()) });

    if cfg!(miri) {
        let x = x.to_array();
        let y = y.to_array();
        let diff = u8x16::from_array(arch, core::array::from_fn(|i| x[i].abs_diff(y[i])));
        let result = acc.dot_simd(diff, diff);
        // SAFETY: Restore the signed accumulator type without changing its bits.
        i32x4::from_underlying(arch, unsafe { vreinterpretq_s32_u32(result.to_underlying()) })
    } else {
        // SAFETY: Absolute differences span 0..=255; reinterpret the signed intrinsic result as unsigned.
        let diff = u8x16::from_underlying(arch, unsafe {
            vreinterpretq_u8_s8(vabdq_s8(x.to_underlying(), y.to_underlying()))
        });
        let result = acc.dot_simd(diff, diff);
        // SAFETY: Restore the signed accumulator type without changing its bits.
        i32x4::from_underlying(arch, unsafe { vreinterpretq_s32_u32(result.to_underlying()) })
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
            let acc: [i32; 4] = acc.try_into().unwrap();

            // Compute expected result
            let expected: [i32; 4] = core::array::from_fn(|i| {
                (4 * i..4 * i + 4).fold(acc[i], |sum, j| {
                    let diff = x[j] as i32 - y[j] as i32;
                    sum.wrapping_add(diff * diff)
                })
            });

            // Compute Neon result
            let got = {
                let x = i8x16::from_array(arch, x);
                let y = i8x16::from_array(arch, y);
                let acc = i32x4::from_array(arch, acc);
                squared_euclidean_accum_i8x16(x, y, acc).to_array()
            };

            assert_eq!(
                got, expected,
                "failed on input x = {:?}, y = {:?}, acc = {:?}",
                x, y, acc
            );
        };

        f(&[i8::MIN; 16], &[i8::MAX; 16], &[i32::MAX; 4]);
        driver::drive_ternary(&f, (16, 16, 4), 0xc0ffee);
    }
}

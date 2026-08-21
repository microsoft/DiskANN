/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Lossy conversion between `f32` and bf16 storage.
//!
//! A bf16 value contains the upper 16 bits of an IEEE-754 `f32`. It keeps the
//! exponent and seven mantissa bits. For non-negative values, its `u16` bit order
//! matches `f32` numeric order. HashPrune applies a separate ordered-key transform
//! to signed distances.
//!
//! Conversion truncates the lower 16 bits. It does not round. It preserves sign,
//! infinity, signed zero, and the upper NaN payload bits.

/// Convert `f32` → bf16 by truncating the lower 16 mantissa bits.
#[inline(always)]
pub(super) fn f32_to_bf16(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

/// Reconstruct `f32` from a bf16 for conversion tests.
#[cfg(test)]
#[inline(always)]
fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_roundtrip_preserves_exactly_representable_values() {
        // bf16 has 7 mantissa bits. Pick f32 values whose lower 16 mantissa
        // bits are zero so the truncation is lossless.
        for &x in &[0.0_f32, 1.0, 2.0, 0.5, 0.25, 4.0, -1.0, -0.5] {
            let back = bf16_to_f32(f32_to_bf16(x));
            assert_eq!(back, x, "exact bf16 roundtrip failed for {}", x);
        }
    }

    #[test]
    fn bf16_truncation_keeps_relative_error_below_one_percent() {
        // A non-exact finite value has at most about 2^-7 relative truncation error.
        use std::f32::consts::{E, PI};
        for &x in &[1e-10_f32, 1e10, PI, E] {
            let back = bf16_to_f32(f32_to_bf16(x));
            let rel = ((back - x) / x).abs();
            assert!(rel <= 0.01, "{} → {}: rel error {} > 1%", x, back, rel);
        }
    }

    #[test]
    fn bf16_conversion_preserves_the_upper_bits_of_special_values() {
        for value in [
            0.0_f32,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::from_bits(0xFFC1_2345),
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
        ] {
            let expected_upper_bits = value.to_bits() & 0xFFFF_0000;
            assert_eq!(
                bf16_to_f32(f32_to_bf16(value)).to_bits(),
                expected_upper_bits,
                "value_bits={:08x}",
                value.to_bits()
            );
        }
    }

    #[test]
    fn bf16_truncation_moves_finite_values_toward_zero() {
        for value in [f32::MIN_POSITIVE, 0.1, 1.1, std::f32::consts::PI, f32::MAX] {
            let positive = bf16_to_f32(f32_to_bf16(value));
            let negative = bf16_to_f32(f32_to_bf16(-value));
            assert!((0.0..=value).contains(&positive), "value={value}");
            assert!((-value..=0.0).contains(&negative), "value={value}");
        }
    }

    #[test]
    fn bf16_encoding_preserves_non_negative_value_order() {
        // For non-negative f32, bf16 (as u16) preserves ordering.
        let xs: [f32; 6] = [0.0, 1e-10, 0.1, 1.0, 10.0, 1e10];
        let bs: Vec<u16> = xs.iter().map(|&x| f32_to_bf16(x)).collect();
        for w in bs.windows(2) {
            assert!(w[0] <= w[1], "bf16 ordering broken: {} > {}", w[0], w[1]);
        }
    }
}

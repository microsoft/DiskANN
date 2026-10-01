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
    use rstest::rstest;

    #[rstest]
    fn bf16_roundtrip_preserves_exactly_representable_values(
        #[values(0.0_f32, 1.0, 2.0, 0.5, 0.25, 4.0, -1.0, -0.5)] value: f32,
    ) {
        // Given: bf16 has seven mantissa bits, and these values have zero low mantissa bits.
        let expected_value = value;

        // When
        let actual_value = bf16_to_f32(f32_to_bf16(value));

        // Then
        assert_eq!(actual_value, expected_value);
    }

    #[rstest]
    fn bf16_truncation_keeps_relative_error_below_one_percent(
        #[values(1e-10_f32, 1e10, std::f32::consts::PI, std::f32::consts::E)] value: f32,
    ) {
        // Given: bf16 truncation has at most about 2^-7 relative error for finite values.
        let maximum_relative_error = 0.01;

        // When
        let truncated = bf16_to_f32(f32_to_bf16(value));
        let actual_relative_error = ((truncated - value) / value).abs();

        // Then
        assert!(actual_relative_error <= maximum_relative_error);
    }

    #[rstest]
    #[case::positive_zero(0.0)]
    #[case::negative_zero(-0.0)]
    #[case::positive_infinity(f32::INFINITY)]
    #[case::negative_infinity(f32::NEG_INFINITY)]
    #[case::positive_nan(f32::NAN)]
    #[case::negative_nan(f32::from_bits(0xFFC1_2345))]
    #[case::minimum_positive(f32::MIN_POSITIVE)]
    #[case::negative_minimum_positive(-f32::MIN_POSITIVE)]
    fn bf16_conversion_preserves_the_upper_bits_of_special_values(#[case] value: f32) {
        // Given
        let expected_upper_bits = value.to_bits() & 0xFFFF_0000;

        // When
        let actual_bits = bf16_to_f32(f32_to_bf16(value)).to_bits();

        // Then
        assert_eq!(actual_bits, expected_upper_bits);
    }

    #[rstest]
    fn bf16_truncation_moves_finite_values_toward_zero(
        #[values(f32::MIN_POSITIVE, 0.1, 1.1, std::f32::consts::PI, f32::MAX)] value: f32,
    ) {
        // When
        let positive_truncation = bf16_to_f32(f32_to_bf16(value));
        let negative_truncation = bf16_to_f32(f32_to_bf16(-value));

        // Then
        assert!((0.0..=value).contains(&positive_truncation));
        assert!((-value..=0.0).contains(&negative_truncation));
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

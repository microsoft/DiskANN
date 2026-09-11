/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Convert gathered vectors with the available CPU's FP16 instructions.

use std::any::TypeId;

use crate::utils::VectorRepr;
use diskann_vector::conversion::SliceCast;
use diskann_wide::arch;
use half::f16;

/// Preserve VectorRepr conversion errors while accelerating FP16 slices.
#[inline]
pub(super) fn as_f32_into<T: VectorRepr>(source: &[T], output: &mut [f32]) -> Result<(), T::Error> {
    if TypeId::of::<T>() == TypeId::of::<f16>() && source.len() == output.len() {
        // VectorRepr's default conversion uses the compile-time architecture.
        // Dispatch one whole slice so portable builds can use F16C or NEON too.
        let source: &[f16] = bytemuck::cast_slice(source);
        arch::dispatch2(SliceCast::<f32, f16>::new(), output, source);
        Ok(())
    } else {
        T::as_f32_into(source, output)
    }
}

#[cfg(test)]
mod as_f32_into_tests {
    use super::*;
    use rstest::rstest;

    #[test]
    fn fp16_bits_are_preserved_through_a_vector_and_tail() {
        // Given: 13 values leave a tail for both four- and eight-lane converters.
        // Normal values gain 112 in exponent bias and shift the fraction by 13;
        // the smallest FP16 subnormal becomes 2^-24. NaN keeps its payload.
        let half_bits = [
            0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x3555, 0x7bff, 0x7c00, 0xfc00,
            0x7e01, 0x8400,
        ];
        let expected = [
            0x00000000, 0x80000000, 0x33800000, 0x387fc000, 0x38800000, 0x3f800000, 0xbf800000,
            0x3eaaa000, 0x477fe000, 0x7f800000, 0xff800000, 0x7fc02000, 0xb8800000,
        ];
        let source = half_bits.map(f16::from_bits);
        let mut output = [123.0_f32; 13];

        // When
        as_f32_into(&source, &mut output).unwrap();

        // Then
        assert_eq!(output.map(f32::to_bits), expected);
    }

    #[rstest]
    #[case::short_output(7)]
    #[case::long_output(9)]
    fn mismatched_fp16_lengths_return_the_vector_repr_error(#[case] output_len: usize) {
        // Given
        let source = [f16::ONE; 8];
        let mut output = vec![0.0_f32; output_len];
        let expected = f16::as_f32_into(&source, &mut output).unwrap_err();

        // When
        let actual = as_f32_into(&source, &mut output).unwrap_err();

        // Then
        assert_eq!(actual, expected);
    }

    #[rstest]
    #[case::f32([-2.5_f32, 1.25], [-2.5, 1.25])]
    #[case::i8([-2_i8, 127], [-2.0, 127.0])]
    #[case::u8([0_u8, 255], [0.0, 255.0])]
    fn other_representations_keep_their_conversion<T: VectorRepr>(
        #[case] source: [T; 2],
        #[case] expected: [f32; 2],
    ) {
        // Given
        let mut output = [f32::NAN; 2];

        // When
        as_f32_into(&source, &mut output).unwrap();

        // Then
        assert_eq!(output, expected);
    }
}

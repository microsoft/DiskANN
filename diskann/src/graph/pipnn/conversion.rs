/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Convert gathered vectors with the available CPU's FP16 instructions.

use std::any::TypeId;

use crate::utils::VectorRepr;
use diskann_utils::views::MatrixView;
use diskann_vector::conversion::SliceCast;
use diskann_wide::{
    Architecture,
    arch::{self, Target2, Target3},
};
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

/// Gather rows into packed f32 storage, dispatching once for the whole FP16 batch.
pub(super) fn gather_as_f32<T: VectorRepr>(
    data: MatrixView<'_, T>,
    ids: &[u32],
    output: &mut [f32],
) -> Result<(), (u32, T::Error)> {
    if TypeId::of::<T>() == TypeId::of::<f16>() {
        arch::dispatch3(GatherFp16, data, ids, output);
    } else {
        for (&id, destination) in ids.iter().zip(output.chunks_exact_mut(data.ncols())) {
            T::as_f32_into(data.row(id as usize), destination).map_err(|error| (id, error))?;
        }
    }
    Ok(())
}

struct GatherFp16;

impl<A, T> Target3<A, (), MatrixView<'_, T>, &[u32], &mut [f32]> for GatherFp16
where
    A: Architecture,
    T: VectorRepr,
    for<'a, 'b> SliceCast<f32, f16>: Target2<A, (), &'a mut [f32], &'b [f16]>,
{
    #[inline(always)]
    fn run(self, arch: A, data: MatrixView<'_, T>, ids: &[u32], output: &mut [f32]) {
        for (&id, destination) in ids.iter().zip(output.chunks_exact_mut(data.ncols())) {
            let source = bytemuck::cast_slice(data.row(id as usize));
            SliceCast::<f32, f16>::new().run(arch, destination, source);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::f32([1.5_f32, -2.0, 7.0, 4.0, 0.5, 9.0], [0.5, 9.0, 1.5, -2.0, 0.5, 9.0])]
    #[case::i8([1_i8, -2, 7, 4, -5, 9], [-5.0, 9.0, 1.0, -2.0, -5.0, 9.0])]
    #[case::u8([1_u8, 2, 7, 4, 5, 9], [5.0, 9.0, 1.0, 2.0, 5.0, 9.0])]
    #[case::f16([1.5_f32, -2.0, 7.0, 4.0, 0.5, 9.0].map(f16::from_f32), [0.5, 9.0, 1.5, -2.0, 0.5, 9.0])]
    fn gathered_rows_follow_requested_ids<T: VectorRepr>(
        #[case] values: [T; 6],
        #[case] expected: [f32; 6],
    ) {
        // Given: row IDs are deliberately out of order and contain a repeat.
        let data = MatrixView::try_from(&values[..], 3, 2).unwrap();
        let mut output = [f32::NAN; 6];

        gather_as_f32(data, &[2, 0, 2], &mut output).unwrap();

        assert_eq!(output, expected);
    }

    #[rstest]
    #[case::before_vector(7)]
    #[case::complete_vector(8)]
    #[case::after_vector(9)]
    #[case::multiple_vectors(32)]
    #[case::multiple_vectors_and_tail(33)]
    #[case::embedding_and_tail(1537)]
    fn fp16_gather_converts_every_coordinate(#[case] dimensions: usize) {
        let mut values: Vec<_> = (0..3 * dimensions)
            .map(|index| f16::from_f32((index % 31) as f32 - 15.0))
            .collect();
        for (row, values) in values.chunks_exact_mut(dimensions).enumerate() {
            values[dimensions - 1] = f16::from_f32(100.0 + row as f32);
        }
        let ids = [2, 0, 2];
        let expected: Vec<_> = ids
            .iter()
            .flat_map(|&row| {
                values[row as usize * dimensions..(row as usize + 1) * dimensions]
                    .iter()
                    .map(|value| value.to_f32())
            })
            .collect();
        let mut output = vec![f32::NAN; expected.len()];

        gather_as_f32(
            MatrixView::try_from(values.as_slice(), 3, dimensions).unwrap(),
            &ids,
            &mut output,
        )
        .unwrap();

        assert_eq!(output, expected);
    }

    #[test]
    fn fp16_gather_preserves_special_values_and_zero_signs() {
        let values = [
            f16::ZERO,
            f16::NEG_ZERO,
            f16::INFINITY,
            f16::NEG_INFINITY,
            f16::from_bits(1),
            f16::NAN,
        ];
        let mut output = [42.0; 6];

        gather_as_f32(
            MatrixView::try_from(&values[..], 2, 3).unwrap(),
            &[1, 0],
            &mut output,
        )
        .unwrap();

        assert_eq!(output[0], f32::NEG_INFINITY);
        assert_eq!(output[1], 2.0_f32.powi(-24));
        assert!(output[2].is_nan());
        assert_eq!(output[3].to_bits(), 0.0_f32.to_bits());
        assert_eq!(output[4].to_bits(), (-0.0_f32).to_bits());
        assert_eq!(output[5], f32::INFINITY);
    }
}

#[cfg(test)]
mod as_f32_into_tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::empty(0)]
    #[case::shorter_than_neon(3)]
    #[case::one_neon_vector(4)]
    #[case::shorter_than_avx2(7)]
    #[case::one_avx2_vector(8)]
    #[case::avx2_tail(9)]
    #[case::one_avx512_vector(16)]
    #[case::avx512_tail(17)]
    #[case::unrolled_tail(65)]
    fn fp16_conversion_preserves_bits_at_slice_boundaries(#[case] length: usize) {
        // Given: IEEE-754 encodings for signed zeros, subnormal/normal limits,
        // finite fractions, infinities and a quiet NaN with a payload.
        let half_bits = [
            0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x3555, 0x7bff, 0x7c00, 0xfc00,
            0x7e01,
        ];
        let float_bits = [
            0x00000000, 0x80000000, 0x33800000, 0x387fc000, 0x38800000, 0x3f800000, 0xbf800000,
            0x3eaaa000, 0x477fe000, 0x7f800000, 0xff800000, 0x7fc02000,
        ];
        let source: Vec<_> = half_bits
            .into_iter()
            .cycle()
            .take(length)
            .map(f16::from_bits)
            .collect();
        let expected: Vec<_> = float_bits.into_iter().cycle().take(length).collect();
        let mut output = vec![123.0_f32; length];

        // When
        as_f32_into(&source, &mut output).unwrap();

        // Then
        assert_eq!(
            output.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn mismatched_fp16_lengths_return_the_vector_repr_error() {
        // Given
        let source = [f16::ONE; 8];
        let mut output = [0.0_f32; 7];
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

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
mod gather_as_f32_tests {
    use super::*;

    #[test]
    fn batched_fp16_gather_matches_per_row_conversion_in_requested_order() {
        // Given: repeated, noncontiguous rows of nine elements exercise tails
        // and the row stride while overwriting stale destination values.
        let points = [
            [-4.0_f32, -3.5, -2.0, -1.5, -0.0, 0.5, 1.0, 2.5, 3.0],
            [1.0, 2.5, 3.0, 0.5, -1.0, 4.0, -2.5, 0.0, -3.5],
            [8.0, 2.0, -7.0, 0.25, 0.75, 3.5, -4.5, 0.0, -2.0],
        ];
        let half_points = points.map(|row| row.map(f16::from_f32));
        let data = MatrixView::try_from(half_points.as_flattened(), 3, 9).unwrap();
        let ids = [2, 0, 2];
        let mut expected = [f32::NAN; 27];
        for (&id, row) in ids.iter().zip(expected.chunks_exact_mut(9)) {
            as_f32_into(data.row(id as usize), row).unwrap();
        }
        let mut actual = [123.0_f32; 27];

        // When
        gather_as_f32(data, &ids, &mut actual).unwrap();

        // Then
        assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
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

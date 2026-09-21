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

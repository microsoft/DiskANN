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
mod gather_as_f32_tests {
    use super::*;

    #[test]
    fn fp16_gather_preserves_bits_and_requested_row_order_with_a_tail() {
        // Given: repeated, noncontiguous rows of nine elements exercise tails
        // and the row stride while overwriting stale destination values.
        let points = [
            [-4.0_f32, -3.5, -2.0, -1.5, -0.0, 0.5, 1.0, 2.5, 3.0],
            [1.0, 2.5, 3.0, 0.5, -1.0, 4.0, -2.5, 0.0, -3.5],
            [8.0, 2.0, -7.0, 0.25, 0.75, 3.5, -4.5, 0.0, -2.0],
        ];
        let half_points = points.map(|row| row.map(f16::from_f32_const));
        let data = MatrixView::try_from(half_points.as_flattened(), 3, 9).unwrap();
        let ids = [2, 0, 2];
        // All coordinates are exactly representable in FP16, including -0.0.
        let expected = [points[2], points[0], points[2]];
        let mut actual = [123.0_f32; 27];

        // When
        gather_as_f32(data, &ids, &mut actual).unwrap();

        // Then
        assert_eq!(
            actual.map(f32::to_bits).as_slice(),
            expected
                .as_flattened()
                .iter()
                .copied()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        );
    }
}

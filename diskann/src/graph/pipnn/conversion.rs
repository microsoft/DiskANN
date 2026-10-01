/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Gather dataset rows into packed `f32` storage.
//!
//! `f16` rows use the FP16 instructions of the running CPU. The generic
//! `VectorRepr::as_f32_into` path uses the instructions of the build target
//! instead. Crates that depend on `diskann` do not inherit this repository's
//! `x86-64-v3` target flag, so for them that path converts `f16` in software.
//! On an AMD EPYC 7763, software conversion made gathers 4.6 to 22 times slower
//! (128 to 1536 dimensions). With the `x86-64-v3` target, both paths are within
//! 10% of each other.

use std::any::TypeId;

use crate::{ANNResult, utils::VectorRepr};
use diskann_utils::views::MatrixView;
use diskann_vector::conversion::SliceCast;
use diskann_wide::{
    Architecture,
    arch::{self, Target2, Target3},
};
use half::f16;

/// Convert rows `ids` of `data` into consecutive rows of `output`.
///
/// `f16` input selects the CPU's instructions once for the whole batch.
pub(super) fn gather_as_f32<T: VectorRepr>(
    data: MatrixView<'_, T>,
    ids: &[u32],
    output: &mut [f32],
) -> ANNResult<()> {
    if TypeId::of::<T>() == TypeId::of::<f16>() {
        arch::dispatch3(GatherFp16, data, ids, output);
    } else {
        for (&id, destination) in ids.iter().zip(output.chunks_exact_mut(data.ncols())) {
            T::as_f32_into(data.row(id as usize), destination).map_err(Into::into)?;
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

    #[test]
    fn gathered_rows_follow_requested_ids() {
        // Row IDs are out of order and contain a repeat.
        fn check<T: VectorRepr>(values: [T; 6], expected: [f32; 6]) {
            let data = MatrixView::try_from(&values[..], 3, 2).unwrap();
            let mut output = [f32::NAN; 6];

            gather_as_f32(data, &[2, 0, 2], &mut output).unwrap();

            assert_eq!(output, expected, "{}", std::any::type_name::<T>());
        }

        check(
            [1.5_f32, -2.0, 7.0, 4.0, 0.5, 9.0],
            [0.5, 9.0, 1.5, -2.0, 0.5, 9.0],
        );
        check([1_i8, -2, 7, 4, -5, 9], [-5.0, 9.0, 1.0, -2.0, -5.0, 9.0]);
        check([1_u8, 2, 7, 4, 5, 9], [5.0, 9.0, 1.0, 2.0, 5.0, 9.0]);
        check(
            [1.5_f32, -2.0, 7.0, 4.0, 0.5, 9.0].map(f16::from_f32),
            [0.5, 9.0, 1.5, -2.0, 0.5, 9.0],
        );
    }

    #[test]
    fn fp16_gather_converts_every_coordinate() {
        // x86 converts eight values at a time. The dimensions cover a short row,
        // full vectors with and without a tail, and an embedding with a tail.
        for dimensions in [7, 8, 9, 32, 33, 1537] {
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

            assert_eq!(output, expected, "{dimensions} dimensions");
        }
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

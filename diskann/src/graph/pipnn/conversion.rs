/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Gather dataset rows into packed `f32` storage.
//!
//! `f16` rows use the FP16 instructions of the running CPU. The generic
//! `VectorRepr::as_f32_into` path uses the instructions of the build target
//! instead. Crates that depend on `diskann` do not inherit this repository's
//! `x86-64-v3` target flag, so for them that path converts `f16` in software,
//! which is many times slower.
//!
//! Other element types use `VectorRepr::as_f32_into` for every target. `f32`
//! rows are copied. The baseline x86-64 target has SSE2 instructions that
//! convert `u8` and `i8` to `f32`, so the compiler vectorizes that loop without
//! the `x86-64-v3` flag. The baseline target has no `f16` conversion
//! instruction; F16C starts at `x86-64-v3`.

use std::any::TypeId;

use crate::{ANNResult, utils::VectorRepr};
use diskann_utils::views::{MatrixView, MutMatrixView};
use diskann_vector::conversion::SliceCast;
use diskann_wide::{
    Architecture,
    arch::{self, Target2, Target3},
};
use half::f16;

/// Convert one row to `f32`, with the FP16 instructions of the running CPU
/// for `f16` input. Errors are those of `VectorRepr::as_f32_into`.
#[inline]
pub(super) fn as_f32_into<T: VectorRepr>(source: &[T], output: &mut [f32]) -> Result<(), T::Error> {
    if TypeId::of::<T>() == TypeId::of::<f16>() && source.len() == output.len() {
        let source: &[f16] = bytemuck::cast_slice(source);
        arch::dispatch2(SliceCast::<f32, f16>::new(), output, source);
        Ok(())
    } else {
        T::as_f32_into(source, output)
    }
}

/// Return the length of a row of `data` after conversion to `f32`.
///
/// Compressed element types store a row in a different number of elements than
/// the `f32` vector it converts to. Empty data has no row to read, so it keeps
/// the stored row length.
pub(super) fn f32_dimensions<T: VectorRepr>(data: MatrixView<'_, T>) -> ANNResult<usize> {
    match data.get_row(0) {
        Some(row) => T::full_dimension(row).map_err(Into::into),
        None => Ok(data.ncols()),
    }
}

/// Convert rows `ids` of `data` into the rows of `output`, one row per ID.
///
/// `output` has [`f32_dimensions`] columns. `f16` input selects the CPU's
/// instructions once for the whole batch.
pub(super) fn gather_as_f32<T: VectorRepr>(
    data: MatrixView<'_, T>,
    ids: &[u32],
    mut output: MutMatrixView<'_, f32>,
) -> ANNResult<()> {
    debug_assert_eq!(output.nrows(), ids.len());
    if TypeId::of::<T>() == TypeId::of::<f16>() {
        arch::dispatch3(GatherFp16, data, ids, output);
    } else {
        for (&id, destination) in ids.iter().zip(output.row_iter_mut()) {
            T::as_f32_into(data.row(id as usize), destination).map_err(Into::into)?;
        }
    }
    Ok(())
}

struct GatherFp16;

impl<A, T> Target3<A, (), MatrixView<'_, T>, &[u32], MutMatrixView<'_, f32>> for GatherFp16
where
    A: Architecture,
    T: VectorRepr,
    for<'a, 'b> SliceCast<f32, f16>: Target2<A, (), &'a mut [f32], &'b [f16]>,
{
    #[inline(always)]
    fn run(
        self,
        arch: A,
        data: MatrixView<'_, T>,
        ids: &[u32],
        mut output: MutMatrixView<'_, f32>,
    ) {
        for (&id, destination) in ids.iter().zip(output.row_iter_mut()) {
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

            gather_as_f32(
                data,
                &[2, 0, 2],
                MutMatrixView::try_from(output.as_mut_slice(), 3, 2).unwrap(),
            )
            .unwrap();

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
                MutMatrixView::try_from(output.as_mut_slice(), 3, dimensions).unwrap(),
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
            MutMatrixView::try_from(output.as_mut_slice(), 2, 3).unwrap(),
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

    #[test]
    fn fp16_conversion_preserves_bits_at_slice_boundaries() {
        // IEEE-754 encodings for signed zeros, subnormal and normal limits,
        // finite fractions, infinities and a quiet NaN with a payload.
        let half_bits = [
            0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x3555, 0x7bff, 0x7c00, 0xfc00,
            0x7e01,
        ];
        let float_bits = [
            0x00000000, 0x80000000, 0x33800000, 0x387fc000, 0x38800000, 0x3f800000, 0xbf800000,
            0x3eaaa000, 0x477fe000, 0x7f800000, 0xff800000, 0x7fc02000,
        ];
        // Lengths around the Neon (4), AVX2 (8) and AVX-512 (16) widths, and an
        // unrolled tail.
        for length in [0, 3, 4, 7, 8, 9, 16, 17, 65] {
            let source: Vec<_> = half_bits
                .into_iter()
                .cycle()
                .take(length)
                .map(f16::from_bits)
                .collect();
            let expected: Vec<_> = float_bits.into_iter().cycle().take(length).collect();
            let mut output = vec![123.0_f32; length];

            as_f32_into(&source, &mut output).unwrap();

            assert_eq!(
                output.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                expected,
                "length {length}"
            );
        }
    }

    #[test]
    fn mismatched_fp16_lengths_return_the_vector_repr_error() {
        let source = [f16::ONE; 8];
        let mut output = [0.0_f32; 7];
        let expected = f16::as_f32_into(&source, &mut output).unwrap_err();

        let actual = as_f32_into(&source, &mut output).unwrap_err();

        assert_eq!(actual, expected);
    }

    #[test]
    fn other_representations_keep_their_conversion() {
        fn check<T: VectorRepr>(source: [T; 2], expected: [f32; 2]) {
            let mut output = [f32::NAN; 2];

            as_f32_into(&source, &mut output).unwrap();

            assert_eq!(output, expected, "{}", std::any::type_name::<T>());
        }

        check([-2.5_f32, 1.25], [-2.5, 1.25]);
        check([-2_i8, 127], [-2.0, 127.0]);
        check([0_u8, 255], [0.0, 255.0]);
    }
}

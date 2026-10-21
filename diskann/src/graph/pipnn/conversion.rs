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

use crate::{ANNError, ANNResult, utils::VectorRepr};
use diskann_utils::views::{MatrixView, MutMatrixView};
use diskann_vector::conversion::SliceCast;
use diskann_wide::{
    Architecture,
    arch::{self, Target2, Target3},
};
use half::f16;

/// Convert rows `ids` of `data` into the rows of `output`, one row per ID.
///
/// `output` has the `f32` row length of the build as its column count.
/// `f16` input selects the CPU's instructions once for the whole batch.
pub(super) fn gather_as_f32<T: VectorRepr>(
    data: MatrixView<'_, T>,
    ids: &[u32],
    mut output: MutMatrixView<'_, f32>,
) -> ANNResult<()> {
    debug_assert_eq!(output.nrows(), ids.len());
    if TypeId::of::<T>() == TypeId::of::<f16>() {
        if output.ncols() != data.ncols() {
            return Err(ANNError::message(format!(
                "PiPNN dimension {} does not match the f16 row length {}",
                output.ncols(),
                data.ncols()
            )));
        }
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
        // The IDs are out of order and repeat a row.
        let ids = [2, 0, 2];

        fn check<T: VectorRepr>(rows: [[T; 2]; 3], ids: [u32; 3], expected: [[f32; 2]; 3]) {
            let row_count = rows.len();
            let dimensions = rows[0].len();
            let output_rows = ids.len();
            let data = MatrixView::try_from(rows.as_flattened(), row_count, dimensions).unwrap();
            let mut output = [[f32::NAN; 2]; 3];

            gather_as_f32(
                data,
                &ids,
                MutMatrixView::try_from(output.as_flattened_mut(), output_rows, dimensions)
                    .unwrap(),
            )
            .unwrap();

            assert_eq!(output, expected, "{}", std::any::type_name::<T>());
        }

        let signed = [[1.5, -2.0], [7.0, 4.0], [0.5, 9.0]];
        let signed_expected = [[0.5, 9.0], [1.5, -2.0], [0.5, 9.0]];
        check(signed, ids, signed_expected);
        // f16 takes the FP16 path; the other types take the VectorRepr path.
        check(
            signed.map(|row| row.map(f16::from_f32)),
            ids,
            signed_expected,
        );
        check(
            [[1_i8, -2], [7, 4], [-5, 9]],
            ids,
            [[-5.0, 9.0], [1.0, -2.0], [-5.0, 9.0]],
        );
        check(
            [[1_u8, 2], [7, 4], [5, 9]],
            ids,
            [[5.0, 9.0], [1.0, 2.0], [5.0, 9.0]],
        );
    }
}

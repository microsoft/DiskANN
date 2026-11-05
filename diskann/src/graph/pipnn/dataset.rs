/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The dataset of a build, behind one object.
//!
//! [`Dataset`] hides the element type from the build steps, so they compile
//! once. One call gathers a whole stripe or leaf, so the dynamic dispatch costs
//! nothing measurable next to the conversion.
//!
//! [`Element`] selects the conversion of one element type. `f16` rows use the
//! FP16 instructions of the running CPU. The generic `VectorRepr::as_f32_into`
//! path uses the instructions of the build target instead. Crates that depend
//! on `diskann` do not inherit this repository's `x86-64-v3` target flag, so for
//! them that path converts `f16` in software, which is many times slower.
//!
//! Other element types use `VectorRepr::as_f32_into` for every target. `f32`
//! rows are copied. The baseline x86-64 target has SSE2 instructions that
//! convert `u8` and `i8` to `f32`, so the compiler vectorizes that loop without
//! the `x86-64-v3` flag. The baseline target has no `f16` conversion
//! instruction; F16C starts at `x86-64-v3`.

use crate::{ANNError, ANNResult, utils::VectorRepr};
use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use diskann_vector::{DistanceFunction, conversion::SliceCast, distance::Metric};
use diskann_wide::{
    Architecture,
    arch::{self, Target2, Target3},
};
use half::f16;

/// The dataset of one build: its metric, rows that the partition and leaf
/// steps read as `f32`, and the distance between two rows that final pruning
/// uses.
///
/// [`Rows`] implements it for a matrix of one [`Element`] type. A row source
/// of another shape implements it directly.
pub trait Dataset: Sync {
    /// Number of rows.
    fn nrows(&self) -> usize;

    /// Length of each row after conversion to `f32`.
    fn dimensions(&self) -> usize;

    /// Metric of the build. The kernels rank by it, and the occlusion rule of
    /// final pruning follows it.
    fn metric(&self) -> Metric;

    /// Convert rows `ids` into the rows of `output`, one row per ID.
    ///
    /// `output` has one row per ID and [`Self::dimensions`] columns.
    fn gather(&self, ids: &[u32], output: rowmajor::Mut<'_, f32>) -> ANNResult<()>;

    /// Return the distance between rows `source` and `target` under the metric.
    ///
    /// Final pruning ranks and occludes candidates with it, so it must agree
    /// with the distance that a search over the graph uses.
    fn distance(&self, source: u32, target: u32) -> f32;
}

/// An element type that PiPNN builds from.
///
/// Each type selects its own row conversion. The provided method converts with
/// [`VectorRepr::as_f32_into`]. `f16` replaces it with a run-time CPU dispatch.
pub trait Element: VectorRepr {
    /// Convert rows `ids` of `data` into the rows of `output`, one row per ID.
    ///
    /// `output` has one row per ID and the `f32` row length as its column count.
    fn gather(
        data: rowmajor::Ref<'_, Self>,
        ids: &[u32],
        mut output: rowmajor::Mut<'_, f32>,
    ) -> ANNResult<()> {
        for (&id, destination) in ids.iter().zip(output.rows_mut()) {
            Self::as_f32_into(data.row(id as usize), destination).map_err(Into::into)?;
        }
        Ok(())
    }
}

impl Element for f32 {}

impl Element for u8 {}

impl Element for i8 {}

impl Element for f16 {
    /// Select the CPU's FP16 instructions once for the whole batch.
    fn gather(
        data: rowmajor::Ref<'_, Self>,
        ids: &[u32],
        output: rowmajor::Mut<'_, f32>,
    ) -> ANNResult<()> {
        if output.ncols() != data.ncols() {
            return Err(ANNError::message(format!(
                "PiPNN dimension {} does not match the f16 row length {}",
                output.ncols(),
                data.ncols()
            )));
        }
        arch::dispatch3(GatherFp16, data, ids, output);
        Ok(())
    }
}

/// Converts `f16` rows with the FP16 instructions of architecture `A`.
struct GatherFp16;

impl<A> Target3<A, (), rowmajor::Ref<'_, f16>, &[u32], rowmajor::Mut<'_, f32>> for GatherFp16
where
    A: Architecture,
    for<'a, 'b> SliceCast<f32, f16>: Target2<A, (), &'a mut [f32], &'b [f16]>,
{
    #[inline(always)]
    fn run(
        self,
        arch: A,
        data: rowmajor::Ref<'_, f16>,
        ids: &[u32],
        mut output: rowmajor::Mut<'_, f32>,
    ) {
        for (&id, destination) in ids.iter().zip(output.rows_mut()) {
            SliceCast::<f32, f16>::new().run(arch, destination, data.row(id as usize));
        }
    }
}

/// A matrix of one [`Element`] type as the [`Dataset`] of a build.
pub struct Rows<'a, T: Element> {
    data: rowmajor::Ref<'a, T>,
    dimensions: usize,
    metric: Metric,
    distance: T::Distance,
}

impl<'a, T: Element> Rows<'a, T> {
    /// Wrap `data` with the distance of `metric`.
    ///
    /// The `f32` row length comes from [`VectorRepr::full_dimension`] of the
    /// first row, or is zero for empty data. Compressed element types such as
    /// MinMax store more elements per row, so it can differ from `data.ncols()`.
    pub fn new(data: rowmajor::Ref<'a, T>, metric: Metric) -> ANNResult<Self> {
        let dimensions = match data.get_row(0) {
            Some(first_row) => T::full_dimension(first_row).map_err(Into::into)?,
            None => 0,
        };
        Ok(Self {
            data,
            dimensions,
            metric,
            distance: T::distance(metric, Some(data.ncols())),
        })
    }
}

impl<T: Element> Dataset for Rows<'_, T> {
    fn nrows(&self) -> usize {
        self.data.nrows()
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn metric(&self) -> Metric {
        self.metric
    }

    fn gather(&self, ids: &[u32], output: rowmajor::Mut<'_, f32>) -> ANNResult<()> {
        T::gather(self.data, ids, output)
    }

    /// Computed with [`VectorRepr::distance`] on the stored elements, so a
    /// compressed type measures its codes, as a search over the index does.
    fn distance(&self, source: u32, target: u32) -> f32 {
        self.distance.evaluate_similarity(
            self.data.row(source as usize),
            self.data.row(target as usize),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gathered_rows_follow_requested_ids() {
        // The IDs are out of order and repeat a row.
        let ids = [2, 0, 2];

        fn check<T: Element>(rows: [[T; 2]; 3], ids: [u32; 3], expected: [[f32; 2]; 3]) {
            let row_count = rows.len();
            let dimensions = rows[0].len();
            let output_rows = ids.len();
            let data =
                rowmajor::Ref::try_from_data(rows.as_flattened(), row_count, dimensions).unwrap();
            let mut output = [[f32::NAN; 2]; 3];

            T::gather(
                data,
                &ids,
                rowmajor::Mut::try_from_data(output.as_flattened_mut(), output_rows, dimensions)
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

    #[test]
    fn an_f16_row_length_that_differs_from_the_dimension_is_an_error() {
        // A stored row has two values, but the build dimension asks for three.
        let rows = [[f16::ONE, f16::ZERO]];
        let ids = [0];
        let stored_dimensions = rows[0].len();
        let dimensions = 3;
        let data = rowmajor::Ref::try_from_data(rows.as_flattened(), rows.len(), stored_dimensions)
            .unwrap();
        let mut output = vec![f32::NAN; ids.len() * dimensions];
        let output =
            rowmajor::Mut::try_from_data(output.as_mut_slice(), ids.len(), dimensions).unwrap();

        let error = f16::gather(data, &ids, output).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("PiPNN dimension 3 does not match the f16 row length 2"),
            "{error}"
        );
    }
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Rows of a PiPNN build.
//!
//! Users describe their rows with the public [`Dataset`] trait. The build steps
//! read them through `inner::Dataset`, which a [`Backend`] holds as one trait
//! object, so the steps compile once for every row type. One call gathers a
//! whole stripe or leaf, so the dynamic dispatch costs nothing measurable next
//! to the conversion.
//!
//! The public trait describes the build by its [`Metric`]. The inner trait
//! hands out what the metric selects instead: the kernels and the occlusion
//! rule of final pruning. A future backend that no `Metric` variant describes
//! can construct a `Backend` from its own inner implementation, without a new
//! `Metric` variant.

use std::fmt::Debug;

use crate::ANNResult;
use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use diskann_vector::{
    DistanceFunction,
    conversion::SliceCast,
    distance::{Distance, DistanceProvider, Metric},
};
use diskann_wide::{
    Architecture,
    arch::{self, Target2, Target3},
};
use half::f16;

/// The rows of a PiPNN build.
///
/// [`Rows`] implements it for a row-major matrix of an [`Element`] type.
/// Implement it directly for rows that are stored another way. Pass it to
/// [`build_graph`](super::build_graph) with [`Dataset::backend`].
pub trait Dataset: Debug + Sync {
    /// Number of rows.
    fn nrows(&self) -> usize;

    /// Number of `f32` values in each row that [`Dataset::gather`] writes.
    fn dimensions(&self) -> usize;

    /// Metric of the search that the graph is built for.
    fn metric(&self) -> Metric;

    /// Write rows `ids` as `f32` values, one row of `output` for each ID.
    ///
    /// # Errors
    ///
    /// Returns an error if a row cannot be converted.
    ///
    /// # Panics
    ///
    /// May panic if an ID is not below [`Dataset::nrows`], or if `output` does
    /// not have `ids.len()` rows of [`Dataset::dimensions`] values.
    fn gather(&self, ids: &[u32], output: rowmajor::Mut<'_, f32>) -> ANNResult<()>;

    /// Distance between rows `source` and `target` under [`Dataset::metric`].
    ///
    /// It must equal the distance that a search over the built graph uses.
    ///
    /// # Panics
    ///
    /// May panic if `source` or `target` is not below [`Dataset::nrows`].
    fn distance(&self, source: u32, target: u32) -> f32;

    /// Prepare the dataset for [`build_graph`](super::build_graph).
    fn backend(&self) -> Backend<'_> {
        Backend::new(self)
    }
}

/// Types that [`build_graph`](super::build_graph) builds from.
///
/// A reference to any [`Dataset`] implements it, and so does a [`Backend`].
pub trait IntoBackend<'a> {
    /// Prepare the rows for [`build_graph`](super::build_graph).
    fn into_backend(self) -> Backend<'a>;
}

impl<'a, D> IntoBackend<'a> for &'a D
where
    D: Dataset + ?Sized,
{
    fn into_backend(self) -> Backend<'a> {
        self.backend()
    }
}

impl<'a> IntoBackend<'a> for Backend<'a> {
    fn into_backend(self) -> Backend<'a> {
        self
    }
}

/// A [`Dataset`] prepared for [`build_graph`](super::build_graph).
///
/// Create it with [`Dataset::backend`].
#[derive(Debug)]
pub struct Backend<'a> {
    inner: Box<dyn inner::Dataset + 'a>,
}

impl<'a> Backend<'a> {
    fn new<D>(data: &'a D) -> Self
    where
        D: Dataset + ?Sized,
    {
        Self {
            inner: Box::new(inner::Super(data)),
        }
    }

    /// The rows as the build steps read them.
    pub(super) fn dataset(&self) -> &(dyn inner::Dataset + 'a) {
        &*self.inner
    }
}

pub(super) mod inner {
    use std::fmt::Debug;

    use crate::{ANNResult, graph::config::PruneKind};
    use diskann_utils::views::rowmajor;

    use super::super::kernels::{self, Kernels};

    /// The rows of a build as the build steps read them.
    ///
    /// The methods mirror [`super::Dataset`], except that the metric is
    /// replaced by the parts of the build that it selects.
    pub(in crate::graph::pipnn) trait Dataset: Debug + Sync {
        fn nrows(&self) -> usize;

        fn dimensions(&self) -> usize;

        /// See [`super::Dataset::gather`].
        fn gather(&self, ids: &[u32], output: rowmajor::Mut<'_, f32>) -> ANNResult<()>;

        /// See [`super::Dataset::distance`].
        fn distance(&self, source: u32, target: u32) -> f32;

        /// Occlusion rule of final pruning.
        fn prune_kind(&self) -> PruneKind;

        /// Partition and leaf kernels for the running CPU.
        fn kernels(&self) -> Box<dyn Kernels>;
    }

    /// Bridges a public [`super::Dataset`] into [`Dataset`].
    #[derive(Debug)]
    pub(super) struct Super<'a, D: ?Sized>(pub(super) &'a D);

    impl<D> Dataset for Super<'_, D>
    where
        D: super::Dataset + ?Sized,
    {
        fn nrows(&self) -> usize {
            self.0.nrows()
        }

        fn dimensions(&self) -> usize {
            self.0.dimensions()
        }

        fn gather(&self, ids: &[u32], output: rowmajor::Mut<'_, f32>) -> ANNResult<()> {
            self.0.gather(ids, output)
        }

        fn distance(&self, source: u32, target: u32) -> f32 {
            self.0.distance(source, target)
        }

        fn prune_kind(&self) -> PruneKind {
            PruneKind::from_metric(self.0.metric())
        }

        fn kernels(&self) -> Box<dyn Kernels> {
            kernels::select(self.0.metric())
        }
    }
}

/// An element type of [`Rows`].
///
/// `f32`, `f16`, `u8` and `i8` implement it. A compressed element type
/// implements it with its own decoding.
pub trait Element: Copy + Debug + Send + Sync + 'static {
    /// Distance between two rows of `Self`.
    type Distance: for<'a, 'b> DistanceFunction<&'a [Self], &'b [Self], f32> + Debug + Send + Sync;

    /// Number of `f32` values in `row` after conversion.
    ///
    /// The default is `row.len()`. A compressed type reads it from the row.
    ///
    /// # Errors
    ///
    /// Returns an error if `row` is not a valid row of the type.
    fn dimensions(row: &[Self]) -> ANNResult<usize> {
        Ok(row.len())
    }

    /// Write `row` as `f32` values into `output`.
    ///
    /// # Errors
    ///
    /// Returns an error if `row` cannot be decoded.
    ///
    /// # Panics
    ///
    /// May panic if `output.len()` is not the dimension of `row`.
    fn convert(row: &[Self], output: &mut [f32]) -> ANNResult<()>;

    /// Distance between rows of `ncols` elements under `metric`.
    fn distance(metric: Metric, ncols: usize) -> Self::Distance;

    /// Write rows `ids` of `data` as `f32` values, one row of `output` for
    /// each ID.
    ///
    /// The default converts one row at a time with [`Element::convert`]. A
    /// type can override it to convert the whole batch at once.
    ///
    /// # Errors
    ///
    /// Returns an error if a row cannot be decoded.
    ///
    /// # Panics
    ///
    /// May panic if an ID is not below `data.nrows()`, or if `output` does not
    /// have one row of the converted dimension for each ID.
    fn gather(
        data: rowmajor::Ref<'_, Self>,
        ids: &[u32],
        mut output: rowmajor::Mut<'_, f32>,
    ) -> ANNResult<()> {
        for (&id, destination) in ids.iter().zip(output.rows_mut()) {
            Self::convert(data.row(id as usize), destination)?;
        }
        Ok(())
    }
}

/// Implement [`Element`] for a type whose values convert to `f32` one by one
/// and whose distances come from [`DistanceProvider`].
macro_rules! plain_element {
    ($T:ty) => {
        impl Element for $T {
            type Distance = Distance<$T, $T>;

            fn convert(row: &[Self], output: &mut [f32]) -> ANNResult<()> {
                for (destination, &value) in output.iter_mut().zip(row) {
                    *destination = value.into();
                }
                Ok(())
            }

            fn distance(metric: Metric, ncols: usize) -> Self::Distance {
                Self::distance_comparer(metric, Some(ncols))
            }
        }
    };
}

// The baseline x86-64 target has SSE2 instructions that convert `u8` and `i8`
// to `f32`, so the compiler vectorizes these loops for any target.
plain_element!(f32);
plain_element!(u8);
plain_element!(i8);

// `f16` conversion selects the FP16 instructions of the running CPU. The
// baseline x86-64 target has no `f16` conversion instruction; F16C starts at
// `x86-64-v3`. A crate that depends on `diskann` does not inherit this
// repository's `x86-64-v3` target flag, so a conversion compiled for the build
// target would run in software, many times slower. `gather` dispatches once for
// the whole batch, so the selection stays out of the row loop.
impl Element for f16 {
    type Distance = Distance<f16, f16>;

    fn convert(row: &[Self], output: &mut [f32]) -> ANNResult<()> {
        arch::dispatch2(SliceCast::<f32, f16>::new(), output, row);
        Ok(())
    }

    fn distance(metric: Metric, ncols: usize) -> Self::Distance {
        Self::distance_comparer(metric, Some(ncols))
    }

    fn gather(
        data: rowmajor::Ref<'_, Self>,
        ids: &[u32],
        output: rowmajor::Mut<'_, f32>,
    ) -> ANNResult<()> {
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

/// A row-major matrix of an [`Element`] type as a [`Dataset`].
#[derive(Debug)]
pub struct Rows<'a, T: Element> {
    data: rowmajor::Ref<'a, T>,
    dimensions: usize,
    metric: Metric,
    distance: T::Distance,
}

impl<'a, T: Element> Rows<'a, T> {
    /// Use `data` as the rows of a build for search under `metric`.
    ///
    /// # Errors
    ///
    /// Returns an error if [`Element::dimensions`] rejects the first row.
    pub fn new(data: rowmajor::Ref<'a, T>, metric: Metric) -> ANNResult<Self> {
        // A compressed type stores more elements per row than it decodes to, so
        // the first row gives the `f32` row length, not `data.ncols()`.
        let dimensions = match data.get_row(0) {
            Some(first_row) => T::dimensions(first_row)?,
            None => 0,
        };
        Ok(Self {
            data,
            dimensions,
            metric,
            distance: T::distance(metric, data.ncols()),
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

    fn distance(&self, source: u32, target: u32) -> f32 {
        // The distance works on the stored elements, so a compressed type
        // measures its codes, as a search over the index does.
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
        // f16 takes its batch path; the other types take the default path.
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

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Random-hyperplane sketches for HashPrune.
//!
//! A sketch is the projection of a point on `num_planes` Gaussian
//! hyperplanes. HashPrune hashes an edge by the signs of the difference of
//! the two endpoint sketches, so the sketches keep the projection values, not
//! their sign bits.

use crate::{ANNError, ANNResult};
use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
#[cfg(not(miri))]
use rayon::prelude::*;

use super::dataset::Dataset;

/// Most hyperplanes of a sketch. A direction hash has one bit per plane.
pub(super) const MAX_PLANES: usize = 16;

/// Number of points that one sketch job gathers as `f32` at a time.
const SKETCH_CHUNK_POINTS: usize = 256;

/// The sketches of every point of a dataset.
#[derive(Debug)]
pub(super) struct LshSketches {
    num_planes: usize,
    /// `num_planes` values for each point, in point order.
    sketches: Vec<f32>,
}

impl LshSketches {
    /// Project every row of `data` on `num_planes` hyperplanes drawn from `seed`.
    ///
    /// Jobs of [`SKETCH_CHUNK_POINTS`] points gather their rows as `f32` and
    /// run in parallel on the caller's Rayon pool.
    pub(super) fn try_new(data: &dyn Dataset, num_planes: usize, seed: u64) -> ANNResult<Self> {
        if num_planes == 0 {
            return Err(ANNError::message("HashPrune needs at least one hash plane"));
        }
        let npoints = data.nrows();
        let ndims = data.dimensions();
        let hyperplane_len = num_planes.checked_mul(ndims).ok_or_else(|| {
            ANNError::message(format!(
                "LSH matrix shape {num_planes} x {ndims} overflows usize"
            ))
        })?;
        let sketch_len = npoints.checked_mul(num_planes).ok_or_else(|| {
            ANNError::message(format!(
                "LSH matrix shape {npoints} x {num_planes} overflows usize"
            ))
        })?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let hyperplanes: Vec<f32> = (0..hyperplane_len)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();

        let mut sketches = vec![0.0f32; sketch_len];
        let chunk_len = SKETCH_CHUNK_POINTS * num_planes;
        #[cfg(miri)]
        {
            let mut buffer = rowmajor::Buffer::new();
            for (chunk, sketch_rows) in sketches.chunks_mut(chunk_len).enumerate() {
                let first_point = chunk * SKETCH_CHUNK_POINTS;
                fill_sketch_chunk(data, first_point, &hyperplanes, &mut buffer, sketch_rows)?;
            }
        }
        #[cfg(not(miri))]
        {
            sketches
                .par_chunks_mut(chunk_len)
                .enumerate()
                .try_for_each_init(rowmajor::Buffer::new, |buffer, (chunk, sketch_rows)| {
                    let first_point = chunk * SKETCH_CHUNK_POINTS;
                    fill_sketch_chunk(data, first_point, &hyperplanes, buffer, sketch_rows)
                })?;
        }

        Ok(Self {
            num_planes,
            sketches,
        })
    }

    #[inline]
    pub(super) fn num_planes(&self) -> usize {
        self.num_planes
    }

    #[inline]
    pub(super) fn sketches(&self) -> &[f32] {
        &self.sketches
    }
}

/// Gather the points of one chunk as `f32` and write their sketches.
///
/// `sketch_rows` holds `num_planes` values for each point of the chunk, and
/// the chunk starts at `first_point`.
fn fill_sketch_chunk(
    data: &dyn Dataset,
    first_point: usize,
    hyperplanes: &[f32],
    buffer: &mut rowmajor::Buffer<f32>,
    sketch_rows: &mut [f32],
) -> ANNResult<()> {
    let ndims = data.dimensions();
    let num_planes = hyperplanes.len() / ndims.max(1);
    let point_count = sketch_rows.len() / num_planes;
    let ids: Vec<u32> = (first_point..first_point + point_count)
        .map(|point| point as u32)
        .collect();
    buffer.resize(point_count, ndims)?;
    data.gather(&ids, buffer.as_view_mut())?;
    for (point, sketch_row) in buffer
        .as_view()
        .rows()
        .zip(sketch_rows.chunks_exact_mut(num_planes))
    {
        for (plane, destination) in hyperplanes.chunks_exact(ndims).zip(sketch_row) {
            *destination = point.iter().zip(plane).map(|(x, h)| x * h).sum();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::pipnn::dataset::Rows;
    use diskann_vector::distance::Metric;
    use rstest::rstest;

    fn run_with_threads<R: Send>(threads: usize, operation: impl FnOnce() -> R + Send) -> R {
        #[cfg(miri)]
        {
            let _ = threads;
            operation()
        }
        #[cfg(not(miri))]
        {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(operation)
        }
    }

    fn rows(data: &[f32], point_count: usize, dimensions: usize) -> Rows<'_, f32> {
        let view = rowmajor::Ref::try_from_data(data, point_count, dimensions).unwrap();
        Rows::new(view, Metric::L2).unwrap()
    }

    /// A dataset that reports a shape and holds no rows. The shape checks of
    /// [`LshSketches::try_new`] run before any row is read.
    struct Shape {
        nrows: usize,
        dimensions: usize,
    }

    impl Dataset for Shape {
        fn nrows(&self) -> usize {
            self.nrows
        }

        fn dimensions(&self) -> usize {
            self.dimensions
        }

        fn metric(&self) -> Metric {
            Metric::L2
        }

        fn gather(&self, _: &[u32], _: rowmajor::Mut<'_, f32>) -> ANNResult<()> {
            unreachable!("the shape checks fail before any row is read")
        }

        fn distance(&self, _: u32, _: u32) -> f32 {
            unreachable!("the shape checks fail before any row is read")
        }
    }

    #[test]
    fn sketch_shape_has_one_value_per_point_and_plane() {
        let point_vectors = [[1.0_f32, 0.0], [0.0, 1.0], [-1.0, 0.0]];
        let point_count = point_vectors.len();
        let dimensions = point_vectors[0].len();
        let plane_count = 4;
        let expected_sketch_value_count = point_count * plane_count;
        let data: Vec<_> = point_vectors.into_iter().flatten().collect();

        let sketches = run_with_threads(2, || {
            LshSketches::try_new(&rows(&data, point_count, dimensions), plane_count, 42)
        })
        .unwrap();

        assert_eq!(sketches.num_planes(), plane_count);
        assert_eq!(sketches.sketches().len(), expected_sketch_value_count);
    }

    #[rstest]
    #[case::first_seed(42)]
    #[case::second_seed(99)]
    fn sketches_match_seeded_serial_hyperplane_reference(#[case] seed: u64) {
        let point_vectors = [
            [-3.0_f32, -2.0, -1.0, 0.0],
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ];
        let npoints = point_vectors.len();
        let ndims = point_vectors[0].len();
        let planes = 5;
        let data: Vec<_> = point_vectors.into_iter().flatten().collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let hyperplanes: Vec<f32> = (0..planes * ndims)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        let expected_serial_sketch_values: Vec<f32> = data
            .chunks_exact(ndims)
            .flat_map(|point| {
                hyperplanes
                    .chunks_exact(ndims)
                    .map(|plane| point.iter().zip(plane).map(|(x, h)| x * h).sum())
            })
            .collect();

        let actual_sketches = run_with_threads(2, || {
            LshSketches::try_new(&rows(&data, npoints, ndims), planes, seed)
        })
        .unwrap();

        assert_eq!(actual_sketches.sketches(), expected_serial_sketch_values);
    }

    #[test]
    fn a_chunk_boundary_does_not_change_the_sketches() {
        // Two chunks and a tail: every point must get the sketch of its own row.
        let npoints = 2 * SKETCH_CHUNK_POINTS + 3;
        let ndims = 2;
        let planes = 3;
        let data: Vec<f32> = (0..npoints * ndims).map(|value| value as f32).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let hyperplanes: Vec<f32> = (0..planes * ndims)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        let expected: Vec<f32> = data
            .chunks_exact(ndims)
            .flat_map(|point| {
                hyperplanes
                    .chunks_exact(ndims)
                    .map(|plane| point.iter().zip(plane).map(|(x, h)| x * h).sum())
            })
            .collect();

        let actual = run_with_threads(3, || {
            LshSketches::try_new(&rows(&data, npoints, ndims), planes, 7)
        })
        .unwrap();

        assert_eq!(actual.sketches(), expected);
    }

    #[rstest]
    #[case::point_count_times_plane_count(usize::MAX, 0)]
    #[case::dimension_times_plane_count(0, usize::MAX)]
    fn sketch_construction_rejects_shape_overflow(#[case] nrows: usize, #[case] dimensions: usize) {
        let shape = Shape { nrows, dimensions };

        let error =
            LshSketches::try_new(&shape, 2, 42).expect_err("overflowing LSH shape must fail");

        assert!(error.to_string().contains("overflows"));
    }
}

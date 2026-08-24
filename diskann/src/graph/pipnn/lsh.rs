/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Random-hyperplane locality-sensitive hashing for dataset vectors.
//!
//! For each point `v`, the module computes
//! `Sketch(v) = [v · H_i for i in 0..num_planes]`. A seeded random generator
//! samples each hyperplane component from a standard normal distribution.
//! HashPrune compares two sketches to make a relative hash.
//!
//! `LshSketches` stores a row-major `npoints × num_planes` matrix. Each Rayon job
//! uses one `f32` conversion buffer for its source rows. `num_planes` cannot
//! exceed 16 because each relative hash is a `u16`.

use crate::{ANNError, ANNResult, utils::VectorRepr};
use diskann_utils::views::MatrixView;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

/// Maximum number of hyperplanes (the hash output is `u16`).
pub(super) const MAX_PLANES: usize = 16;

/// Precomputed LSH sketches for `npoints` vectors.
#[derive(Debug)]
pub(super) struct LshSketches {
    num_planes: usize,
    /// Row-major `npoints × num_planes`: `sketches[i*m + j] = dot(point_i, plane_j)`.
    sketches: Vec<f32>,
}

impl LshSketches {
    /// Compute random-hyperplane projections for every point in `data`.
    ///
    /// Each worker converts one source row into reusable `f32` storage. Parallel
    /// sketch work uses the currently installed Rayon pool.
    pub(super) fn try_new<T: VectorRepr>(
        data: MatrixView<'_, T>,
        num_planes: usize,
        seed: u64,
    ) -> ANNResult<Self> {
        let npoints = data.nrows();
        let ndims = data.ncols();
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

        #[allow(clippy::disallowed_methods)] // caller installs the complete build in its pool.
        sketches
            .par_chunks_mut(num_planes)
            .enumerate()
            .try_for_each_init(Vec::new, |buffer, (point, sketch_row)| {
                buffer.resize(ndims, 0.0);
                T::as_f32_into(data.row(point), &mut buffer[..ndims])
                    .map_err(Into::<ANNError>::into)
                    .map_err(|error| error.context(format!("converting LSH point {point}")))?;
                for (plane_index, destination) in sketch_row.iter_mut().enumerate() {
                    let plane = &hyperplanes[plane_index * ndims..(plane_index + 1) * ndims];
                    let mut dot = 0.0f32;
                    for dimension in 0..ndims {
                        // SAFETY: both slices have exactly `ndims` elements.
                        unsafe {
                            dot +=
                                *buffer.get_unchecked(dimension) * *plane.get_unchecked(dimension);
                        }
                    }
                    *destination = dot;
                }
                Ok::<(), ANNError>(())
            })?;

        Ok(Self {
            num_planes,
            sketches,
        })
    }

    /// Number of hyperplanes (also the number of bits in the hash).
    #[inline]
    pub(super) fn num_planes(&self) -> usize {
        self.num_planes
    }

    /// Return the row-major `npoints × num_planes` sketch buffer.
    #[inline]
    pub(super) fn sketches(&self) -> &[f32] {
        &self.sketches
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn thread_pool(threads: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
    }

    fn matrix_view<T>(data: &[T], rows: usize, columns: usize) -> MatrixView<'_, T> {
        MatrixView::try_from(data, rows, columns).unwrap()
    }

    #[test]
    fn sketch_shape_has_one_value_per_point_and_plane() {
        // Given
        let point_vectors = [[1.0_f32, 0.0], [0.0, 1.0], [-1.0, 0.0]];
        let point_count = point_vectors.len();
        let dimensions = point_vectors[0].len();
        let plane_count = 4;
        let expected_sketch_value_count = point_count * plane_count;
        let data: Vec<_> = point_vectors.into_iter().flatten().collect();

        // When
        let sketches = thread_pool(2)
            .install(|| {
                LshSketches::try_new(matrix_view(&data, point_count, dimensions), plane_count, 42)
            })
            .unwrap();

        // Then
        assert_eq!(sketches.num_planes(), plane_count);
        assert_eq!(sketches.sketches().len(), expected_sketch_value_count);
    }

    #[rstest]
    #[case::first_seed(42)]
    #[case::second_seed(99)]
    fn sketches_match_seeded_serial_hyperplane_reference(#[case] seed: u64) {
        // Given
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

        // When
        let actual_sketches = thread_pool(2)
            .install(|| LshSketches::try_new(matrix_view(&data, npoints, ndims), planes, seed))
            .unwrap();

        // Then
        assert_eq!(actual_sketches.sketches(), expected_serial_sketch_values);
    }

    #[test]
    fn zero_points_produce_an_empty_sketch() {
        let sketches = thread_pool(2)
            .install(|| LshSketches::try_new(matrix_view(&[] as &[f32], 0, 7), 4, 42))
            .unwrap();

        assert_eq!(sketches.num_planes(), 4);
        assert!(sketches.sketches().is_empty());
    }

    #[test]
    fn zero_dimensions_produce_zero_dot_products() {
        // Given
        let point_count = 3;
        let plane_count = 2;
        let expected_zero_dot_products = vec![0.0; point_count * plane_count];

        // When
        let sketches = thread_pool(2)
            .install(|| {
                LshSketches::try_new(matrix_view(&[] as &[f32], point_count, 0), plane_count, 42)
            })
            .unwrap();

        // Then
        assert_eq!(sketches.sketches(), expected_zero_dot_products);
    }

    #[rstest]
    #[case::point_count_times_plane_count(usize::MAX, 0)]
    #[case::dimension_times_plane_count(0, usize::MAX)]
    fn sketch_construction_rejects_shape_overflow(
        #[case] point_count: usize,
        #[case] dimensions: usize,
    ) {
        // Given
        let empty_data = matrix_view(&[] as &[f32], point_count, dimensions);

        // When
        let error =
            LshSketches::try_new(empty_data, 2, 42).expect_err("overflowing LSH shape must fail");

        // Then
        assert!(error.to_string().contains("overflows"));
    }
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Random-hyperplane LSH (Locality-Sensitive Hashing) for `f32` vectors.
//!
//! Computes `Sketch(v) = [v · H_i for i in 0..num_planes]` where each
//! hyperplane component is sampled from a standard normal distribution. Callers
//! use differences between two point sketches to derive relative hash bits.
//!
//! ```text
//! seeded RNG ──> hyperplanes [planes × dimensions] (immutable)
//!                                      │
//! source vector ──> worker f32 scratch ──> dot products ──> point sketch [planes]
//! ```
//!
//! | Buffer | Shape | Lifetime |
//! | --- | --- | --- |
//! | hyperplanes | `num_planes × ndims` | construction call |
//! | conversion scratch | `ndims` per Rayon job | reused across points |
//! | sketches | `npoints × num_planes` | owned by `LshSketches` |
//!
//! Sketches are computed in parallel via Rayon, with per-worker `VectorRepr`
//! conversion so f16, u8, and i8 storage does not require a full upfront f32
//! copy. `num_planes ≤ 16` keeps every relative hash in a `u16`.

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
    /// Compute LSH sketches for `npoints` points of dimension `ndims`.
    ///
    /// Each worker converts one source row into reusable `f32` scratch, avoiding
    /// a full-dataset conversion for f16, u8, and i8 inputs.
    ///
    /// Caller must be inside `rayon::ThreadPool::install(...)`; parallel work
    /// runs on the current pool.
    pub(super) fn try_new<T: VectorRepr>(
        data: MatrixView<'_, T>,
        num_planes: usize,
        seed: u64,
    ) -> ANNResult<Self> {
        if !(1..=MAX_PLANES).contains(&num_planes) {
            return Err(ANNError::message(format!(
                "num_planes ({num_planes}) must be in 1..={MAX_PLANES}"
            )));
        }
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
        let mut hyperplanes: Vec<f32> = Vec::new();
        hyperplanes
            .try_reserve_exact(hyperplane_len)
            .map_err(ANNError::new)?;
        hyperplanes.resize_with(hyperplane_len, || StandardNormal.sample(&mut rng));

        let mut sketches = Vec::new();
        sketches
            .try_reserve_exact(sketch_len)
            .map_err(ANNError::new)?;
        sketches.resize(sketch_len, 0.0f32);

        #[allow(clippy::disallowed_methods)] // caller installs the complete build in its pool.
        sketches
            .par_chunks_mut(num_planes)
            .enumerate()
            .try_for_each_init(Vec::new, |buffer, (point, sketch_row)| {
                if buffer.len() < ndims {
                    buffer
                        .try_reserve(ndims - buffer.len())
                        .map_err(ANNError::new)?;
                }
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

    /// Raw access to the row-major `npoints × num_planes` sketch buffer.
    /// Callers can scatter-gather a small per-leaf cache of sketches to avoid
    /// touching the multi-hundred-MB global buffer in tight inner loops.
    #[inline]
    pub(super) fn sketches(&self) -> &[f32] {
        &self.sketches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_pool(threads: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
    }

    fn view<T>(data: &[T], rows: usize, columns: usize) -> MatrixView<'_, T> {
        MatrixView::try_from(data, rows, columns).unwrap()
    }

    #[test]
    fn computes_expected_sketch_shape() {
        let data = [1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
        let sketches = build_pool(2)
            .install(|| LshSketches::try_new(view(&data, 3, 2), 4, 42))
            .unwrap();

        assert_eq!(sketches.num_planes(), 4);
        assert_eq!(sketches.sketches().len(), 12);
    }

    #[test]
    fn sketches_match_seeded_serial_hyperplane_reference() {
        let npoints = 3;
        let ndims = 4;
        let planes = 5;
        let data: Vec<f32> = (0..npoints * ndims)
            .map(|value| value as f32 - 3.0)
            .collect();

        for seed in [42, 99] {
            let actual = build_pool(2)
                .install(|| LshSketches::try_new(view(&data, npoints, ndims), planes, seed))
                .unwrap();

            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
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
            assert_eq!(actual.sketches(), expected, "seed={seed}");
        }
    }

    #[test]
    fn zero_points_produce_an_empty_sketch() {
        let sketches = build_pool(2)
            .install(|| LshSketches::try_new(view(&[] as &[f32], 0, 7), 4, 42))
            .unwrap();

        assert_eq!(sketches.num_planes(), 4);
        assert!(sketches.sketches().is_empty());
    }

    #[test]
    fn zero_dimensions_produce_zero_dot_products() {
        let sketches = build_pool(2)
            .install(|| LshSketches::try_new(view(&[] as &[f32], 3, 0), 2, 42))
            .unwrap();

        assert_eq!(sketches.sketches(), &[0.0; 6]);
    }

    #[test]
    fn rejects_shape_overflow() {
        for data in [
            view(&[] as &[f32], 0, usize::MAX),
            view(&[] as &[f32], usize::MAX, 0),
        ] {
            let error =
                LshSketches::try_new(data, 2, 42).expect_err("overflowing LSH shape must fail");
            assert!(error.to_string().contains("overflows"));
        }
    }

    #[test]
    fn rejects_plane_counts_outside_u16_capacity() {
        for planes in [0, MAX_PLANES + 1] {
            let error = LshSketches::try_new(view(&[0.0_f32], 1, 1), planes, 42).unwrap_err();
            assert!(error.to_string().contains("must be in 1..=16"));
        }
    }
}

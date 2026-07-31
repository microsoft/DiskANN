/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Random-hyperplane LSH (Locality-Sensitive Hashing) for `f32` vectors.
//!
//! Computes `Sketch(v) = [v · H_i for i in 0..num_planes]` where each
//! hyperplane component is sampled from a standard normal distribution. Callers
//! use differences between two sketch rows to derive relative hash bits.
//!
//! ```text
//! seeded RNG ──> hyperplanes [planes × dimensions] (immutable)
//!                                      │
//! source row ──> worker f32 scratch ──> dot products ──> sketch row [planes]
//! ```
//!
//! | Buffer | Shape | Lifetime |
//! | --- | --- | --- |
//! | hyperplanes | `num_planes × ndims` | construction call |
//! | conversion scratch | `ndims` per Rayon job | reused across point rows |
//! | sketches | `npoints × num_planes` | owned by `LshSketches` |
//!
//! Sketches are computed in parallel via Rayon, with caller-provided per-point
//! `f32` conversion so f16, u8, and i8 storage does not require a full upfront
//! f32 copy. `num_planes ≤ 16` keeps every relative hash in a `u16`.

use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

/// Maximum number of hyperplanes (the hash output is `u16`).
pub const MAX_PLANES: usize = 16;

/// Failure while constructing LSH sketches.
#[derive(Debug)]
pub enum LshSketchError<E> {
    /// The relative hash must use between one and 16 bits.
    InvalidPlaneCount { actual: usize, max: usize },
    /// A matrix shape overflowed `usize`.
    ShapeOverflow { rows: usize, columns: usize },
    /// A sketch buffer could not be allocated.
    Allocation(std::collections::TryReserveError),
    /// The caller could not materialize a point in `f32`.
    Fill(E),
}

impl<E: std::fmt::Display> std::fmt::Display for LshSketchError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPlaneCount { actual, max } => {
                write!(f, "num_planes ({actual}) must be in 1..={max}")
            }
            Self::ShapeOverflow { rows, columns } => {
                write!(f, "LSH matrix shape {rows} x {columns} overflows usize")
            }
            Self::Allocation(error) => write!(f, "LSH allocation failed: {error}"),
            Self::Fill(error) => write!(f, "point conversion failed: {error}"),
        }
    }
}

impl<E> std::error::Error for LshSketchError<E>
where
    E: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidPlaneCount { .. } => None,
            Self::ShapeOverflow { .. } => None,
            Self::Allocation(error) => Some(error),
            Self::Fill(error) => Some(error),
        }
    }
}

/// Precomputed LSH sketches for `npoints` vectors.
pub struct LshSketches {
    num_planes: usize,
    /// Row-major `npoints × num_planes`: `sketches[i*m + j] = dot(point_i, plane_j)`.
    sketches: Vec<f32>,
}

impl LshSketches {
    /// Compute LSH sketches for `npoints` points of dimension `ndims`.
    ///
    /// `fill_point` is called once per point with the point's destination
    /// buffer of length `ndims` (the buffer is allocated per worker thread
    /// and reused across calls). This avoids requiring a contiguous `&[f32]`
    /// for the whole dataset when the caller stores points as f16, u8, etc.
    ///
    /// Caller MUST be inside a `rayon::ThreadPool::install(...)` scope —
    /// parallel work runs on the current pool.
    ///
    /// Returns [`LshSketchError::InvalidPlaneCount`] unless `num_planes` fits
    /// the non-empty 16-bit relative-hash representation.
    pub fn try_new<F, E>(
        npoints: usize,
        ndims: usize,
        num_planes: usize,
        seed: u64,
        fill_point: F,
    ) -> Result<Self, LshSketchError<E>>
    where
        F: Fn(usize, &mut [f32]) -> Result<(), E> + Send + Sync,
        E: Send,
    {
        if !(1..=MAX_PLANES).contains(&num_planes) {
            return Err(LshSketchError::InvalidPlaneCount {
                actual: num_planes,
                max: MAX_PLANES,
            });
        }

        let hyperplane_len =
            num_planes
                .checked_mul(ndims)
                .ok_or(LshSketchError::ShapeOverflow {
                    rows: num_planes,
                    columns: ndims,
                })?;
        let sketch_len = npoints
            .checked_mul(num_planes)
            .ok_or(LshSketchError::ShapeOverflow {
                rows: npoints,
                columns: num_planes,
            })?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        // Standard-normal hyperplanes, row-major `num_planes × ndims`.
        let mut hyperplanes: Vec<f32> = Vec::new();
        hyperplanes
            .try_reserve_exact(hyperplane_len)
            .map_err(LshSketchError::Allocation)?;
        hyperplanes.resize_with(hyperplane_len, || StandardNormal.sample(&mut rng));

        let mut sketches = Vec::new();
        sketches
            .try_reserve_exact(sketch_len)
            .map_err(LshSketchError::Allocation)?;
        sketches.resize(sketch_len, 0.0f32);

        #[allow(clippy::disallowed_methods)] // see module docstring; caller is in pool.install().
        sketches
            .par_chunks_mut(num_planes)
            .enumerate()
            .try_for_each_init(Vec::new, |buffer, (i, sketch_row)| {
                if buffer.len() < ndims {
                    let additional = ndims - buffer.len();
                    buffer
                        .try_reserve(additional)
                        .map_err(LshSketchError::Allocation)?;
                }
                buffer.resize(ndims, 0.0);
                fill_point(i, &mut buffer[..ndims]).map_err(LshSketchError::Fill)?;
                for j in 0..num_planes {
                    let plane = &hyperplanes[j * ndims..(j + 1) * ndims];
                    let mut dot = 0.0f32;
                    for d in 0..ndims {
                        // SAFETY: `d` is in 0..ndims; both buffers have length `ndims`.
                        unsafe {
                            dot += *buffer.get_unchecked(d) * *plane.get_unchecked(d);
                        }
                    }
                    sketch_row[j] = dot;
                }
                Ok(())
            })?;

        Ok(Self {
            num_planes,
            sketches,
        })
    }

    /// Number of hyperplanes (also the number of bits in the hash).
    #[inline]
    pub fn num_planes(&self) -> usize {
        self.num_planes
    }

    /// Raw access to the row-major `npoints × num_planes` sketch buffer.
    /// Callers can scatter-gather a small per-leaf cache of sketches to avoid
    /// touching the multi-hundred-MB global buffer in tight inner loops.
    #[inline]
    pub fn sketches(&self) -> &[f32] {
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

    #[test]
    fn computes_expected_sketch_shape() {
        let pool = build_pool(2);
        let data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
        let sk = pool
            .install(|| {
                LshSketches::try_new(3, 2, 4, 42, |i, out| {
                    out.copy_from_slice(&data[i * 2..(i + 1) * 2]);
                    Ok::<_, std::convert::Infallible>(())
                })
            })
            .unwrap();
        assert_eq!(sk.num_planes(), 4);
        assert_eq!(sk.sketches().len(), 12);
    }

    #[test]
    fn sketches_match_serial_hyperplane_reference() {
        let npoints = 3;
        let ndims = 4;
        let planes = 5;
        let seed = 42;
        let data: Vec<f32> = (0..npoints * ndims)
            .map(|value| value as f32 - 3.0)
            .collect();
        let actual = build_pool(2)
            .install(|| {
                LshSketches::try_new(npoints, ndims, planes, seed, |i, out| {
                    out.copy_from_slice(&data[i * ndims..(i + 1) * ndims]);
                    Ok::<_, std::convert::Infallible>(())
                })
            })
            .unwrap();

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let hyperplanes: Vec<f32> = (0..planes * ndims)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        let mut expected = Vec::<f32>::new();
        for point in data.chunks_exact(ndims) {
            for plane in hyperplanes.chunks_exact(ndims) {
                expected.push(point.iter().zip(plane).map(|(x, h)| x * h).sum());
            }
        }
        assert_eq!(actual.sketches(), expected);
    }

    #[test]
    fn zero_points_produces_an_empty_sketch_without_filling() {
        let sketches = build_pool(2)
            .install(|| {
                LshSketches::try_new(
                    0,
                    7,
                    4,
                    42,
                    |_, _| -> Result<(), std::convert::Infallible> {
                        panic!("zero points must not invoke fill_point")
                    },
                )
            })
            .unwrap();
        assert_eq!(sketches.num_planes(), 4);
        assert!(sketches.sketches().is_empty());
    }

    #[test]
    fn zero_dimensions_produces_zero_dot_products() {
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let sketches = build_pool(2)
            .install(|| {
                LshSketches::try_new(3, 0, 2, 42, |_, out| {
                    assert!(out.is_empty());
                    calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Ok::<_, std::convert::Infallible>(())
                })
            })
            .unwrap();
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 3);
        assert_eq!(sketches.sketches(), &[0.0; 6]);
    }

    #[test]
    fn reports_shape_overflow_and_formats_errors() {
        let hyperplanes = match LshSketches::try_new(1, usize::MAX, 2, 42, |_, _| {
            Ok::<_, std::convert::Infallible>(())
        }) {
            Ok(_) => panic!("hyperplane shape overflow must fail"),
            Err(error) => error,
        };
        assert!(matches!(hyperplanes, LshSketchError::ShapeOverflow { .. }));
        assert!(hyperplanes.to_string().contains("overflows"));

        let sketches = match LshSketches::try_new(usize::MAX, 1, 2, 42, |_, _| {
            Ok::<_, std::convert::Infallible>(())
        }) {
            Ok(_) => panic!("sketch shape overflow must fail"),
            Err(error) => error,
        };
        assert!(matches!(sketches, LshSketchError::ShapeOverflow { .. }));
    }

    #[test]
    fn different_seeds_change_hashes() {
        let pool = build_pool(2);
        let data: Vec<f32> = (0..32 * 4).map(|i| (i as f32).sin()).collect();
        let build = |seed| {
            pool.install(|| {
                LshSketches::try_new(32, 4, 12, seed, |i, out| {
                    out.copy_from_slice(&data[i * 4..(i + 1) * 4]);
                    Ok::<_, std::convert::Infallible>(())
                })
            })
            .unwrap()
        };
        let sk1 = build(42);
        let sk2 = build(99);
        assert_ne!(sk1.sketches(), sk2.sketches());
    }

    #[test]
    fn try_new_propagates_fill_errors() {
        #[derive(Debug, PartialEq, thiserror::Error)]
        #[error("fill failed")]
        struct FillError;

        let pool = build_pool(1);
        let error = match pool.install(|| LshSketches::try_new(1, 2, 4, 42, |_, _| Err(FillError)))
        {
            Ok(_) => panic!("fill error should be propagated"),
            Err(error) => error,
        };

        assert!(matches!(&error, LshSketchError::Fill(FillError)));
        assert!(std::error::Error::source(&error).is_some());
    }

    #[test]
    fn try_new_reports_too_many_planes() {
        let pool = build_pool(1);
        let error = match pool.install(|| {
            LshSketches::try_new(1, 2, 17, 42, |_, _| Ok::<_, std::convert::Infallible>(()))
        }) {
            Ok(_) => panic!("too many planes should be rejected"),
            Err(error) => error,
        };

        assert!(matches!(
            error,
            LshSketchError::InvalidPlaneCount {
                actual: 17,
                max: 16
            }
        ));
    }

    #[test]
    fn try_new_rejects_zero_planes() {
        let pool = build_pool(1);
        let error = match pool.install(|| {
            LshSketches::try_new(1, 2, 0, 42, |_, _| Ok::<_, std::convert::Infallible>(()))
        }) {
            Ok(_) => panic!("zero planes should be rejected"),
            Err(error) => error,
        };

        assert!(matches!(
            error,
            LshSketchError::InvalidPlaneCount { actual: 0, max: 16 }
        ));
    }
}

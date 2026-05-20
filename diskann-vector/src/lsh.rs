/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Random-hyperplane LSH (Locality-Sensitive Hashing) for `f32` vectors.
//!
//! Computes `Sketch(v) = [v · H_i for i in 0..num_planes]` where `H_i` are
//! random unit-Gaussian hyperplanes. Two related hashes are exposed:
//!
//! - [`LshSketches::sign_hash`] — sign bits of a vector's own sketch, giving
//!   each point a static `u16` bucket.
//! - [`LshSketches::relative_hash`] — sign bits of `(Sketch(c) - Sketch(p))`,
//!   which is the relative-bucket hash PiPNN's HashPrune uses to deduplicate
//!   candidate neighbors across overlapping partitions.
//!
//! Sketches are computed in parallel via rayon, with caller-provided per-point
//! `f32` conversion (so f16, u8, etc. don't need a full upfront f32 copy).
//! `num_planes ≤ 16` so the result fits in `u16`.

use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use std::cell::RefCell;

/// Maximum number of hyperplanes (the hash output is `u16`).
pub const MAX_PLANES: usize = 16;

/// Precomputed LSH sketches for `npoints` vectors.
pub struct LshSketches {
    num_planes: usize,
    /// Row-major `npoints × num_planes`: `sketches[i*m + j] = dot(point_i, plane_j)`.
    sketches: Vec<f32>,
    npoints: usize,
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
    /// Panics if `num_planes > MAX_PLANES`.
    pub fn new<F>(npoints: usize, ndims: usize, num_planes: usize, seed: u64, fill_point: F) -> Self
    where
        F: Fn(usize, &mut [f32]) + Send + Sync,
    {
        assert!(
            num_planes <= MAX_PLANES,
            "LshSketches: num_planes ({}) exceeds MAX_PLANES ({})",
            num_planes,
            MAX_PLANES
        );

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        // Random unit-Gaussian hyperplanes, row-major `num_planes × ndims`.
        let hyperplanes: Vec<f32> = (0..num_planes * ndims)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();

        let mut sketches = vec![0.0f32; npoints * num_planes];

        #[allow(clippy::disallowed_methods)] // see module docstring; caller is in pool.install().
        sketches
            .par_chunks_mut(num_planes)
            .enumerate()
            .for_each(|(i, sketch_row)| {
                thread_local! {
                    static SKETCH_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
                }
                SKETCH_BUF.with(|cell| {
                    let mut buf = cell.borrow_mut();
                    buf.resize(ndims, 0.0);
                    fill_point(i, &mut buf[..ndims]);
                    for j in 0..num_planes {
                        let plane = &hyperplanes[j * ndims..(j + 1) * ndims];
                        let mut dot = 0.0f32;
                        for d in 0..ndims {
                            // SAFETY: `d` is in 0..ndims; both `buf` and `plane`
                            // have length `ndims`.
                            unsafe {
                                dot += *buf.get_unchecked(d) * *plane.get_unchecked(d);
                            }
                        }
                        sketch_row[j] = dot;
                    }
                });
            });

        Self {
            num_planes,
            sketches,
            npoints,
        }
    }

    /// Number of points indexed.
    #[inline]
    pub fn npoints(&self) -> usize {
        self.npoints
    }

    /// Number of hyperplanes (also the number of bits in the hash).
    #[inline]
    pub fn num_planes(&self) -> usize {
        self.num_planes
    }

    /// Hash candidate `c` relative to anchor `p`.
    ///
    /// `h_p(c) = concat(sign(Sketch(c)[j] - Sketch(p)[j]) for j in 0..num_planes)`.
    /// Asymmetric: `h_p(c) != h_c(p)` for distinct points.
    #[inline(always)]
    pub fn relative_hash(&self, p: usize, c: usize) -> u16 {
        debug_assert!(p < self.npoints);
        debug_assert!(c < self.npoints);

        let m = self.num_planes;
        let p_sketch = &self.sketches[p * m..(p + 1) * m];
        let c_sketch = &self.sketches[c * m..(c + 1) * m];

        let mut hash: u16 = 0;
        for (j, (&p_val, &c_val)) in p_sketch.iter().zip(c_sketch.iter()).enumerate() {
            if c_val - p_val >= 0.0 {
                hash |= 1u16 << j;
            }
        }
        hash
    }

    /// Static sign hash of point `p`'s own sketch (independent of any anchor).
    #[inline(always)]
    pub fn sign_hash(&self, p: usize) -> u16 {
        debug_assert!(p < self.npoints);

        let m = self.num_planes;
        let p_sketch = &self.sketches[p * m..(p + 1) * m];

        let mut hash: u16 = 0;
        for (j, &v) in p_sketch.iter().enumerate() {
            if v >= 0.0 {
                hash |= 1u16 << j;
            }
        }
        hash
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
    fn relative_hash_is_asymmetric() {
        let pool = build_pool(2);
        let data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
        let sk = pool.install(|| {
            LshSketches::new(3, 2, 4, 42, |i, out| {
                out.copy_from_slice(&data[i * 2..(i + 1) * 2])
            })
        });
        let h01 = sk.relative_hash(0, 1);
        let h10 = sk.relative_hash(1, 0);
        assert_ne!(h01, h10);
    }

    #[test]
    fn sign_hash_independent_of_anchor() {
        let pool = build_pool(2);
        let data: Vec<f32> = vec![1.0, 1.0, -1.0, -1.0];
        let sk = pool.install(|| {
            LshSketches::new(2, 2, 4, 42, |i, out| {
                out.copy_from_slice(&data[i * 2..(i + 1) * 2])
            })
        });
        // Two opposite points should hash to bitwise-inverted (or just different) signs.
        assert_ne!(sk.sign_hash(0), sk.sign_hash(1));
    }

    #[test]
    fn different_seeds_change_hashes() {
        let pool = build_pool(2);
        let data: Vec<f32> = (0..32 * 4).map(|i| (i as f32).sin()).collect();
        let sk1 = pool.install(|| {
            LshSketches::new(32, 4, 12, 42, |i, out| {
                out.copy_from_slice(&data[i * 4..(i + 1) * 4])
            })
        });
        let sk2 = pool.install(|| {
            LshSketches::new(32, 4, 12, 99, |i, out| {
                out.copy_from_slice(&data[i * 4..(i + 1) * 4])
            })
        });
        let any_diff = (0..32)
            .any(|p| (p + 1..32).any(|c| sk1.relative_hash(p, c) != sk2.relative_hash(p, c)));
        assert!(any_diff);
    }

    #[test]
    #[should_panic(expected = "num_planes")]
    fn rejects_too_many_planes() {
        let pool = build_pool(1);
        pool.install(|| LshSketches::new(1, 2, 17, 42, |_, _| {}));
    }
}

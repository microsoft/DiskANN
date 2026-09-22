/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Distance kernels for **sparse** float32/float16 vectors.
//!
//! # Storage model
//!
//! Each operand is a pair of parallel slices:
//!
//! * `idx`: `nnz` dimension indices as `u16`, **sorted ascending and unique**;
//! * `val`: `nnz` values parallel to `idx` (`f32`, or `f16` via [`Half`]).
//!
//! Indices absent from an operand are implicit zeros.
//!
//! # Kernels
//!
//! * **Inner product / cosine numerator** — intersection merge over the two sorted index
//!   arrays; only matching indices contribute.
//! * **Cosine denominator** — each operand's own L2 norm, computed by reusing
//!   [`crate::norm::FastL2Norm`] (a sparse vector's norm is just the L2 norm of its value
//!   array).
//! * **L2** — direct union merge of squared differences, `sqrt(Σ (x_i − y_i)²)`.
//!
//! Accumulation is in `f64` for numerical precision. The merge is scalar. A disjoint-range
//! fast-out skips the intersection merge when the two index ranges cannot overlap.
//!
//! These functions return the **mathematical** value of each metric. Any similarity-score
//! transform (inner product `x -> -x`, cosine `x -> 1 - x`) is applied by the caller.

use crate::{norm::FastL2Norm, Half};
use diskann_wide::arch::dispatch1;

/// Squared-norm floor below which a vector is treated as having zero norm for cosine.
const NORM_LIMIT: f32 = f32::MIN_POSITIVE;

//////////////////////////////
// Scalar merge kernels     //
//////////////////////////////

/// Intersection merge: `Σ x_val * y_val` over matching indices. `O(nnz_x + nnz_y)`.
#[inline]
fn merge_dot(
    x_idx: &[u16],
    y_idx: &[u16],
    xv: impl Fn(usize) -> f64,
    yv: impl Fn(usize) -> f64,
) -> f64 {
    // Disjoint-range fast-out: sorted arrays cannot intersect if their ranges don't touch.
    if x_idx.is_empty()
        || y_idx.is_empty()
        || x_idx[x_idx.len() - 1] < y_idx[0]
        || y_idx[y_idx.len() - 1] < x_idx[0]
    {
        return 0.0;
    }

    let mut acc = 0.0f64;
    let (mut i, mut j) = (0usize, 0usize);
    while i < x_idx.len() && j < y_idx.len() {
        let a = x_idx[i];
        let b = y_idx[j];
        if a == b {
            acc += xv(i) * yv(j);
            i += 1;
            j += 1;
        } else if a < b {
            i += 1;
        } else {
            j += 1;
        }
    }
    acc
}

/// Direct union merge for squared L2: `Σ (x_i − y_i)²`; an index present on only one side
/// contributes `v²`. `O(nnz_x + nnz_y)`.
#[inline]
fn merge_l2_sq(
    x_idx: &[u16],
    y_idx: &[u16],
    xv: impl Fn(usize) -> f64,
    yv: impl Fn(usize) -> f64,
) -> f64 {
    let mut acc = 0.0f64;
    let (mut i, mut j) = (0usize, 0usize);
    while i < x_idx.len() && j < y_idx.len() {
        let a = x_idx[i];
        let b = y_idx[j];
        if a == b {
            let d = xv(i) - yv(j);
            acc += d * d;
            i += 1;
            j += 1;
        } else if a < b {
            let v = xv(i);
            acc += v * v;
            i += 1;
        } else {
            let v = yv(j);
            acc += v * v;
            j += 1;
        }
    }
    while i < x_idx.len() {
        let v = xv(i);
        acc += v * v;
        i += 1;
    }
    while j < y_idx.len() {
        let v = yv(j);
        acc += v * v;
        j += 1;
    }
    acc
}

/// Cosine of the angle from the numerator `dot` and the two operand norms; `0` when either
/// squared norm underflows [`NORM_LIMIT`], otherwise the ratio clamped to `[-1, 1]`.
#[inline]
fn cosine_from_parts(dot: f64, nx: f32, ny: f32) -> f32 {
    if nx * nx < NORM_LIMIT || ny * ny < NORM_LIMIT {
        0.0
    } else {
        let v = dot as f32 / (nx * ny);
        (-1.0f32).max(1.0f32.min(v))
    }
}

//////////////////////////////
// f32 kernels              //
//////////////////////////////

/// `sqrt(Σ (x_i − y_i)²)` for f32 operands.
#[inline]
pub fn l2_f32(x_idx: &[u16], x_val: &[f32], y_idx: &[u16], y_val: &[f32]) -> f32 {
    merge_l2_sq(x_idx, y_idx, |i| x_val[i] as f64, |j| y_val[j] as f64).sqrt() as f32
}

/// `Σ x·y` over matching indices for f32 operands.
#[inline]
pub fn inner_product_f32(x_idx: &[u16], x_val: &[f32], y_idx: &[u16], y_val: &[f32]) -> f32 {
    merge_dot(x_idx, y_idx, |i| x_val[i] as f64, |j| y_val[j] as f64) as f32
}

/// Cosine similarity `dot / (‖x‖·‖y‖)` for f32 operands, clamped to `[-1, 1]`; `0` when
/// either norm underflows. Norms reuse [`crate::norm::FastL2Norm`].
#[inline]
pub fn cosine_f32(x_idx: &[u16], x_val: &[f32], y_idx: &[u16], y_val: &[f32]) -> f32 {
    let dot = merge_dot(x_idx, y_idx, |i| x_val[i] as f64, |j| y_val[j] as f64);
    let nx = dispatch1(FastL2Norm, x_val);
    let ny = dispatch1(FastL2Norm, y_val);
    cosine_from_parts(dot, nx, ny)
}

//////////////////////////////
// f16 kernels              //
//////////////////////////////

/// `sqrt(Σ (x_i − y_i)²)` for f16 operands (values promoted to f32).
#[inline]
pub fn l2_f16(x_idx: &[u16], x_val: &[Half], y_idx: &[u16], y_val: &[Half]) -> f32 {
    merge_l2_sq(
        x_idx,
        y_idx,
        |i| x_val[i].to_f32() as f64,
        |j| y_val[j].to_f32() as f64,
    )
    .sqrt() as f32
}

/// `Σ x·y` over matching indices for f16 operands (values promoted to f32).
#[inline]
pub fn inner_product_f16(x_idx: &[u16], x_val: &[Half], y_idx: &[u16], y_val: &[Half]) -> f32 {
    merge_dot(
        x_idx,
        y_idx,
        |i| x_val[i].to_f32() as f64,
        |j| y_val[j].to_f32() as f64,
    ) as f32
}

/// Cosine similarity for f16 operands, clamped to `[-1, 1]`; `0` when either norm
/// underflows. Norms reuse [`crate::norm::FastL2Norm`].
#[inline]
pub fn cosine_f16(x_idx: &[u16], x_val: &[Half], y_idx: &[u16], y_val: &[Half]) -> f32 {
    let dot = merge_dot(
        x_idx,
        y_idx,
        |i| x_val[i].to_f32() as f64,
        |j| y_val[j].to_f32() as f64,
    );
    let nx = dispatch1(FastL2Norm, x_val);
    let ny = dispatch1(FastL2Norm, y_val);
    cosine_from_parts(dot, nx, ny)
}

//////////////////////////////
// Tests                    //
//////////////////////////////

#[cfg(test)]
mod test {
    use super::*;

    use rand::{
        distr::{Distribution, Uniform},
        rngs::StdRng,
        SeedableRng,
    };

    // Dense f64 reference over the declared dimension, built from the sparse operands.
    fn dense(idx: &[u16], val: &[f32], dim: usize) -> Vec<f64> {
        let mut v = vec![0.0f64; dim];
        for (&i, &x) in idx.iter().zip(val.iter()) {
            v[i as usize] = x as f64;
        }
        v
    }

    fn ref_dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn ref_l2(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }

    fn ref_cos(a: &[f64], b: &[f64]) -> f64 {
        let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        // A zero-norm operand has undefined cosine; the kernel reports 0.0 for it.
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            ref_dot(a, b) / (na * nb)
        }
    }

    fn close(a: f32, b: f32, rel: f32, abs: f32) -> bool {
        (a - b).abs() <= abs + rel * a.abs().max(b.abs())
    }

    fn to_f16(v: &[f32]) -> Vec<Half> {
        v.iter().map(|&x| Half::from_f32(x)).collect()
    }

    fn to_sparse_f32(v: &[f32]) -> (Vec<u16>, Vec<f32>) {
        let mut idx = Vec::new();
        let mut val = Vec::new();
        for (i, &x) in v.iter().enumerate() {
            if x != 0.0 {
                idx.push(i as u16);
                val.push(x);
            }
        }
        (idx, val)
    }

    fn to_sparse_f16(v: &[Half]) -> (Vec<u16>, Vec<Half>) {
        let mut idx = Vec::new();
        let mut val = Vec::new();
        for (i, &x) in v.iter().enumerate() {
            if x.to_f32() != 0.0 {
                idx.push(i as u16);
                val.push(x);
            }
        }
        (idx, val)
    }

    fn fill(v: &mut [f32], dist: &Uniform<f32>, rng: &mut StdRng) {
        for x in v.iter_mut() {
            *x = dist.sample(rng);
        }
    }

    fn drop_half_to_zero(v: &mut [f32], rng: &mut StdRng) {
        let coin = Uniform::new(0.0f32, 1.0).unwrap();
        for x in v.iter_mut() {
            if coin.sample(rng) < 0.5 {
                *x = 0.0;
            }
        }
    }

    // f32 kernels agree with an f64 densified reference for dot, L2, and cosine.
    #[test]
    fn dot_l2_cosine_match_dense_reference_f32() {
        let dim = 16;
        let xi = [1u16, 3, 4, 9, 12];
        let xv = [0.5f32, -1.5, 2.0, 0.25, 3.0];
        let yi = [0u16, 3, 4, 7, 12, 15];
        let yv = [1.0f32, 2.0, -0.5, 4.0, 1.25, -2.0];

        let da = dense(&xi, &xv, dim);
        let db = dense(&yi, &yv, dim);

        let ip = inner_product_f32(&xi, &xv, &yi, &yv);
        let l2 = l2_f32(&xi, &xv, &yi, &yv);
        let cos = cosine_f32(&xi, &xv, &yi, &yv);

        assert!((ip as f64 - ref_dot(&da, &db)).abs() < 1e-5);
        assert!((l2 as f64 - ref_l2(&da, &db)).abs() < 1e-5);
        assert!((cos as f64 - ref_cos(&da, &db)).abs() < 1e-5);
    }

    // f16 kernels agree with an f64 densified reference for dot, L2, and cosine.
    #[test]
    fn dot_l2_cosine_match_dense_reference_f16() {
        let dim = 16;
        let xi = [1u16, 3, 4, 9, 12];
        let xv = to_f16(&[0.5, -1.5, 2.0, 0.25, 3.0]);
        let yi = [0u16, 3, 4, 7, 12, 15];
        let yv = to_f16(&[1.0, 2.0, -0.5, 4.0, 1.25, -2.0]);

        let xvf: Vec<f32> = xv.iter().map(|h| h.to_f32()).collect();
        let yvf: Vec<f32> = yv.iter().map(|h| h.to_f32()).collect();
        let da = dense(&xi, &xvf, dim);
        let db = dense(&yi, &yvf, dim);

        let ip = inner_product_f16(&xi, &xv, &yi, &yv);
        let l2 = l2_f16(&xi, &xv, &yi, &yv);
        let cos = cosine_f16(&xi, &xv, &yi, &yv);

        assert!((ip as f64 - ref_dot(&da, &db)).abs() < 1e-3);
        assert!((l2 as f64 - ref_l2(&da, &db)).abs() < 1e-3);
        assert!((cos as f64 - ref_cos(&da, &db)).abs() < 1e-3);
    }

    // Operands with no shared indices have zero dot and zero cosine (f32 and f16).
    #[test]
    fn disjoint_ranges_have_zero_dot_and_cosine() {
        let xi = [1u16, 2, 3];
        let xv = [1.0f32, 2.0, 3.0];
        let yi = [10u16, 11, 12];
        let yv = [1.0f32, 2.0, 3.0];
        assert_eq!(inner_product_f32(&xi, &xv, &yi, &yv), 0.0);
        assert_eq!(cosine_f32(&xi, &xv, &yi, &yv), 0.0);

        let xvh = to_f16(&xv);
        let yvh = to_f16(&yv);
        assert_eq!(inner_product_f16(&xi, &xvh, &yi, &yvh), 0.0);
        assert_eq!(cosine_f16(&xi, &xvh, &yi, &yvh), 0.0);
    }

    // An empty (zero-norm) operand yields cosine 0 and L2 equal to the other operand's norm.
    #[test]
    fn empty_operand_cosine_zero_and_l2_is_norm() {
        let yi = [0u16, 2, 4];
        let yv = [1.0f32, 2.0, 3.0];
        let empty_i: [u16; 0] = [];
        let empty_v: [f32; 0] = [];

        assert_eq!(cosine_f32(&empty_i, &empty_v, &yi, &yv), 0.0);
        assert!((l2_f32(&empty_i, &empty_v, &yi, &yv) - 14.0f32.sqrt()).abs() < 1e-5);
    }

    // f32 kernels match the f64 densified reference over random, partially-zeroed vectors.
    #[test]
    fn matches_dense_reference_over_random_f32() {
        let mut rng = StdRng::seed_from_u64(0x9e3779b97f4a7c15);
        let dist = Uniform::new(-100.0f32, 100.0f32).unwrap();
        for dim in 1..=96usize {
            for _ in 0..32 {
                let mut x = vec![0.0f32; dim];
                let mut y = vec![0.0f32; dim];
                fill(&mut x, &dist, &mut rng);
                fill(&mut y, &dist, &mut rng);
                drop_half_to_zero(&mut x, &mut rng);
                drop_half_to_zero(&mut y, &mut rng);
                let (xi, xv) = to_sparse_f32(&x);
                let (yi, yv) = to_sparse_f32(&y);
                let da: Vec<f64> = x.iter().map(|&v| v as f64).collect();
                let db: Vec<f64> = y.iter().map(|&v| v as f64).collect();

                let l2 = l2_f32(&xi, &xv, &yi, &yv);
                assert!(
                    close(l2, ref_l2(&da, &db) as f32, 1e-4, 1e-3),
                    "l2 dim={dim}"
                );

                let ip = inner_product_f32(&xi, &xv, &yi, &yv);
                assert!(
                    close(ip, ref_dot(&da, &db) as f32, 1e-4, 1e-2),
                    "ip dim={dim}"
                );

                let cos = cosine_f32(&xi, &xv, &yi, &yv);
                assert!(
                    close(cos, ref_cos(&da, &db) as f32, 1e-4, 1e-3),
                    "cos dim={dim}"
                );
            }
        }
    }

    // f16 kernels match the f64 densified reference over random, partially-zeroed vectors.
    #[test]
    fn matches_dense_reference_over_random_f16() {
        let mut rng = StdRng::seed_from_u64(0xc2b2ae3d27d4eb4f);
        let dist = Uniform::new(-10.0f32, 10.0f32).unwrap();
        for dim in 1..=96usize {
            for _ in 0..32 {
                let mut xf = vec![0.0f32; dim];
                let mut yf = vec![0.0f32; dim];
                fill(&mut xf, &dist, &mut rng);
                fill(&mut yf, &dist, &mut rng);
                drop_half_to_zero(&mut xf, &mut rng);
                drop_half_to_zero(&mut yf, &mut rng);
                let x = to_f16(&xf);
                let y = to_f16(&yf);
                let (xi, xv) = to_sparse_f16(&x);
                let (yi, yv) = to_sparse_f16(&y);
                let da: Vec<f64> = x.iter().map(|v| v.to_f32() as f64).collect();
                let db: Vec<f64> = y.iter().map(|v| v.to_f32() as f64).collect();

                let l2 = l2_f16(&xi, &xv, &yi, &yv);
                assert!(
                    close(l2, ref_l2(&da, &db) as f32, 5e-3, 5e-2),
                    "l2 dim={dim}"
                );

                let ip = inner_product_f16(&xi, &xv, &yi, &yv);
                assert!(
                    close(ip, ref_dot(&da, &db) as f32, 5e-3, 5e-2),
                    "ip dim={dim}"
                );

                let cos = cosine_f16(&xi, &xv, &yi, &yv);
                assert!(
                    close(cos, ref_cos(&da, &db) as f32, 5e-3, 5e-2),
                    "cos dim={dim}"
                );
            }
        }
    }
}

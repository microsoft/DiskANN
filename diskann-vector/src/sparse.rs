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
//! Accumulation is in `f32`. The merge is scalar. A disjoint-range fast-out skips the
//! intersection merge when the two index ranges cannot overlap.
//!
//! These functions return the **mathematical** value of each metric. Any similarity-score
//! transform (inner product `x -> -x`, cosine `x -> 1 - x`) is applied by the caller.

use std::cmp::Ordering;

use crate::conversion::CastFromSlice;
use crate::{norm::FastL2Norm, Half};
use diskann_wide::arch::dispatch1;

/// Squared-norm floor below which a vector is treated as having zero norm for cosine.
const NORM_LIMIT: f32 = f32::MIN_POSITIVE;

/// True when two sorted, unique index arrays cannot share any index (either is empty, or their
/// `[min, max]` ranges don't overlap).
#[inline]
fn disjoint_ranges(x_idx: &[u16], y_idx: &[u16]) -> bool {
    x_idx.is_empty()
        || y_idx.is_empty()
        || x_idx[x_idx.len() - 1] < y_idx[0]
        || y_idx[y_idx.len() - 1] < x_idx[0]
}

/// Widen both f16 operands into a single f32 buffer (`x` then `y`) using the dispatched SIMD
/// slice conversion. One allocation; caller splits at `x_val.len()`.
#[inline]
fn widen_pair(x_val: &[Half], y_val: &[Half]) -> Vec<f32> {
    let mut buf = vec![0.0f32; x_val.len() + y_val.len()];
    let (xf, yf) = buf.split_at_mut(x_val.len());
    xf.cast_from_slice(x_val);
    yf.cast_from_slice(y_val);
    buf
}

/// Error returned when a sparse operand's index and value slices differ in length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LengthMismatch {
    pub idx_len: usize,
    pub val_len: usize,
}

impl std::fmt::Display for LengthMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sparse operand index/value length mismatch: {} indices vs {} values",
            self.idx_len, self.val_len
        )
    }
}

impl std::error::Error for LengthMismatch {}

/// Validate that an operand's index and value slices are parallel (equal length).
#[inline]
fn check_len<T>(idx: &[u16], val: &[T]) -> Result<(), LengthMismatch> {
    if idx.len() == val.len() {
        Ok(())
    } else {
        Err(LengthMismatch {
            idx_len: idx.len(),
            val_len: val.len(),
        })
    }
}

/// Zip parallel index/value slices into `(index, value)` pairs for the merge kernels.
#[inline]
fn pairs<'a>(idx: &'a [u16], val: &'a [f32]) -> impl Iterator<Item = (u16, f32)> + 'a {
    idx.iter().copied().zip(val.iter().copied())
}

//////////////////////////////
// Scalar merge kernels     //
//////////////////////////////

/// Intersection merge: `Σ x·y` over matching indices, accumulated in `f32`. `O(nnz_x + nnz_y)`.
#[inline]
fn merge_dot<I, J>(mut x: I, mut y: J) -> f32
where
    I: Iterator<Item = (u16, f32)>,
    J: Iterator<Item = (u16, f32)>,
{
    let mut acc = 0.0f32;
    let mut a = x.next();
    let mut b = y.next();
    while let (Some((ai, av)), Some((bi, bv))) = (a, b) {
        match ai.cmp(&bi) {
            Ordering::Equal => {
                acc = av.mul_add(bv, acc);
                a = x.next();
                b = y.next();
            }
            Ordering::Less => a = x.next(),
            Ordering::Greater => b = y.next(),
        }
    }
    acc
}

/// Direct union merge for squared L2: `Σ (x_i − y_i)²`, accumulated in `f32`; an index present
/// on only one side contributes `v²`. `O(nnz_x + nnz_y)`.
#[inline]
fn merge_l2_sq<I, J>(mut x: I, mut y: J) -> f32
where
    I: Iterator<Item = (u16, f32)>,
    J: Iterator<Item = (u16, f32)>,
{
    let mut acc = 0.0f32;
    let mut a = x.next();
    let mut b = y.next();
    loop {
        match (a, b) {
            (Some((ai, av)), Some((bi, bv))) => match ai.cmp(&bi) {
                Ordering::Equal => {
                    let d = av - bv;
                    acc = d.mul_add(d, acc);
                    a = x.next();
                    b = y.next();
                }
                Ordering::Less => {
                    acc = av.mul_add(av, acc);
                    a = x.next();
                }
                Ordering::Greater => {
                    acc = bv.mul_add(bv, acc);
                    b = y.next();
                }
            },
            (Some((_, av)), None) => {
                acc = av.mul_add(av, acc);
                a = x.next();
            }
            (None, Some((_, bv))) => {
                acc = bv.mul_add(bv, acc);
                b = y.next();
            }
            (None, None) => break,
        }
    }
    acc
}

/// Cosine of the angle from the numerator `dot` and the two operand norms; `0` when either
/// squared norm underflows [`NORM_LIMIT`], otherwise the ratio clamped to `[-1, 1]`.
#[inline]
fn cosine_from_parts(dot: f32, nx: f32, ny: f32) -> f32 {
    if nx * nx < NORM_LIMIT || ny * ny < NORM_LIMIT {
        0.0
    } else {
        let v = dot / (nx * ny);
        (-1.0f32).max(1.0f32.min(v))
    }
}

//////////////////////////////
// f32 kernels              //
//////////////////////////////

/// `sqrt(Σ (x_i − y_i)²)` for f32 operands.
#[inline]
pub fn l2_f32(
    x_idx: &[u16],
    x_val: &[f32],
    y_idx: &[u16],
    y_val: &[f32],
) -> Result<f32, LengthMismatch> {
    check_len(x_idx, x_val)?;
    check_len(y_idx, y_val)?;
    let d = merge_l2_sq(pairs(x_idx, x_val), pairs(y_idx, y_val));
    Ok(d.sqrt())
}

/// `Σ x·y` over matching indices for f32 operands.
#[inline]
pub fn inner_product_f32(
    x_idx: &[u16],
    x_val: &[f32],
    y_idx: &[u16],
    y_val: &[f32],
) -> Result<f32, LengthMismatch> {
    check_len(x_idx, x_val)?;
    check_len(y_idx, y_val)?;
    if disjoint_ranges(x_idx, y_idx) {
        return Ok(0.0);
    }
    Ok(merge_dot(pairs(x_idx, x_val), pairs(y_idx, y_val)))
}

/// Cosine similarity `dot / (‖x‖·‖y‖)` for f32 operands, clamped to `[-1, 1]`; `0` when either
/// norm underflows or the operands are disjoint. Norms reuse [`crate::norm::FastL2Norm`].
#[inline]
pub fn cosine_f32(
    x_idx: &[u16],
    x_val: &[f32],
    y_idx: &[u16],
    y_val: &[f32],
) -> Result<f32, LengthMismatch> {
    check_len(x_idx, x_val)?;
    check_len(y_idx, y_val)?;
    if disjoint_ranges(x_idx, y_idx) {
        return Ok(0.0);
    }
    let dot = merge_dot(pairs(x_idx, x_val), pairs(y_idx, y_val));
    let nx = dispatch1(FastL2Norm, x_val);
    let ny = dispatch1(FastL2Norm, y_val);
    Ok(cosine_from_parts(dot, nx, ny))
}

//////////////////////////////
// f16 kernels              //
//////////////////////////////

/// `sqrt(Σ (x_i − y_i)²)` for f16 operands; values are pre-widened to f32, then reuse the f32
/// union merge.
#[inline]
pub fn l2_f16(
    x_idx: &[u16],
    x_val: &[Half],
    y_idx: &[u16],
    y_val: &[Half],
) -> Result<f32, LengthMismatch> {
    check_len(x_idx, x_val)?;
    check_len(y_idx, y_val)?;
    let buf = widen_pair(x_val, y_val);
    let (xf, yf) = buf.split_at(x_val.len());
    l2_f32(x_idx, xf, y_idx, yf)
}

/// `Σ x·y` over matching indices for f16 operands; a disjoint-range fast-out skips widening
/// when the operands cannot intersect.
#[inline]
pub fn inner_product_f16(
    x_idx: &[u16],
    x_val: &[Half],
    y_idx: &[u16],
    y_val: &[Half],
) -> Result<f32, LengthMismatch> {
    check_len(x_idx, x_val)?;
    check_len(y_idx, y_val)?;
    if disjoint_ranges(x_idx, y_idx) {
        return Ok(0.0);
    }
    let buf = widen_pair(x_val, y_val);
    let (xf, yf) = buf.split_at(x_val.len());
    inner_product_f32(x_idx, xf, y_idx, yf)
}

/// Cosine similarity for f16 operands, clamped to `[-1, 1]`; `0` when either norm underflows or
/// the operands are disjoint. Values are pre-widened once and reused for the numerator and both
/// norms.
#[inline]
pub fn cosine_f16(
    x_idx: &[u16],
    x_val: &[Half],
    y_idx: &[u16],
    y_val: &[Half],
) -> Result<f32, LengthMismatch> {
    check_len(x_idx, x_val)?;
    check_len(y_idx, y_val)?;
    if disjoint_ranges(x_idx, y_idx) {
        return Ok(0.0);
    }
    let buf = widen_pair(x_val, y_val);
    let (xf, yf) = buf.split_at(x_val.len());
    cosine_f32(x_idx, xf, y_idx, yf)
}

//////////////////////////////
// Tests                    //
//////////////////////////////

#[cfg(test)]
mod test {
    use super::*;

    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use diskann_wide::cast_f16_to_f32;
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
            if cast_f16_to_f32(x) != 0.0 {
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

        let ip = inner_product_f32(&xi, &xv, &yi, &yv).unwrap();
        let l2 = l2_f32(&xi, &xv, &yi, &yv).unwrap();
        let cos = cosine_f32(&xi, &xv, &yi, &yv).unwrap();

        assert_abs_diff_eq!(ip as f64, ref_dot(&da, &db), epsilon = 1e-5);
        assert_abs_diff_eq!(l2 as f64, ref_l2(&da, &db), epsilon = 1e-5);
        assert_abs_diff_eq!(cos as f64, ref_cos(&da, &db), epsilon = 1e-5);
    }

    // f16 kernels agree with an f64 densified reference for dot, L2, and cosine.
    #[test]
    fn dot_l2_cosine_match_dense_reference_f16() {
        let dim = 16;
        let xi = [1u16, 3, 4, 9, 12];
        let xv = to_f16(&[0.5, -1.5, 2.0, 0.25, 3.0]);
        let yi = [0u16, 3, 4, 7, 12, 15];
        let yv = to_f16(&[1.0, 2.0, -0.5, 4.0, 1.25, -2.0]);

        let xvf: Vec<f32> = xv.iter().map(|h| cast_f16_to_f32(*h)).collect();
        let yvf: Vec<f32> = yv.iter().map(|h| cast_f16_to_f32(*h)).collect();
        let da = dense(&xi, &xvf, dim);
        let db = dense(&yi, &yvf, dim);

        let ip = inner_product_f16(&xi, &xv, &yi, &yv).unwrap();
        let l2 = l2_f16(&xi, &xv, &yi, &yv).unwrap();
        let cos = cosine_f16(&xi, &xv, &yi, &yv).unwrap();

        assert_abs_diff_eq!(ip as f64, ref_dot(&da, &db), epsilon = 1e-3);
        assert_abs_diff_eq!(l2 as f64, ref_l2(&da, &db), epsilon = 1e-3);
        assert_abs_diff_eq!(cos as f64, ref_cos(&da, &db), epsilon = 1e-3);
    }

    // Operands with no shared indices have zero dot and zero cosine (f32 and f16).
    #[test]
    fn disjoint_ranges_have_zero_dot_and_cosine() {
        let xi = [1u16, 2, 3];
        let xv = [1.0f32, 2.0, 3.0];
        let yi = [10u16, 11, 12];
        let yv = [1.0f32, 2.0, 3.0];
        assert_eq!(inner_product_f32(&xi, &xv, &yi, &yv).unwrap(), 0.0);
        assert_eq!(cosine_f32(&xi, &xv, &yi, &yv).unwrap(), 0.0);

        let xvh = to_f16(&xv);
        let yvh = to_f16(&yv);
        assert_eq!(inner_product_f16(&xi, &xvh, &yi, &yvh).unwrap(), 0.0);
        assert_eq!(cosine_f16(&xi, &xvh, &yi, &yvh).unwrap(), 0.0);
    }

    // An empty (zero-norm) operand yields cosine 0 and L2 equal to the other operand's norm.
    #[test]
    fn empty_operand_cosine_zero_and_l2_is_norm() {
        let yi = [0u16, 2, 4];
        let yv = [1.0f32, 2.0, 3.0];
        let empty_i: [u16; 0] = [];
        let empty_v: [f32; 0] = [];

        assert_eq!(cosine_f32(&empty_i, &empty_v, &yi, &yv).unwrap(), 0.0);
        assert_abs_diff_eq!(
            l2_f32(&empty_i, &empty_v, &yi, &yv).unwrap(),
            14.0f32.sqrt(),
            epsilon = 1e-5
        );
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

                let l2 = l2_f32(&xi, &xv, &yi, &yv).unwrap();
                assert_relative_eq!(
                    l2 as f64,
                    ref_l2(&da, &db),
                    max_relative = 1e-4,
                    epsilon = 1e-3
                );

                let ip = inner_product_f32(&xi, &xv, &yi, &yv).unwrap();
                assert_relative_eq!(
                    ip as f64,
                    ref_dot(&da, &db),
                    max_relative = 1e-4,
                    epsilon = 1e-2
                );

                let cos = cosine_f32(&xi, &xv, &yi, &yv).unwrap();
                assert_relative_eq!(
                    cos as f64,
                    ref_cos(&da, &db),
                    max_relative = 1e-4,
                    epsilon = 1e-3
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
                let da: Vec<f64> = x.iter().map(|v| cast_f16_to_f32(*v) as f64).collect();
                let db: Vec<f64> = y.iter().map(|v| cast_f16_to_f32(*v) as f64).collect();

                let l2 = l2_f16(&xi, &xv, &yi, &yv).unwrap();
                assert_relative_eq!(
                    l2 as f64,
                    ref_l2(&da, &db),
                    max_relative = 5e-3,
                    epsilon = 5e-2
                );

                let ip = inner_product_f16(&xi, &xv, &yi, &yv).unwrap();
                assert_relative_eq!(
                    ip as f64,
                    ref_dot(&da, &db),
                    max_relative = 5e-3,
                    epsilon = 5e-2
                );

                let cos = cosine_f16(&xi, &xv, &yi, &yv).unwrap();
                assert_relative_eq!(
                    cos as f64,
                    ref_cos(&da, &db),
                    max_relative = 5e-3,
                    epsilon = 5e-2
                );
            }
        }
    }

    // Mismatched index/value lengths return a LengthMismatch error rather than panicking.
    #[test]
    fn length_mismatch_returns_error() {
        let xi = [0u16, 1, 2];
        let xv = [1.0f32, 2.0]; // one value short
        let yi = [0u16, 1];
        let yv = [1.0f32, 2.0];

        assert!(l2_f32(&xi, &xv, &yi, &yv).is_err());
        assert!(inner_product_f32(&xi, &xv, &yi, &yv).is_err());
        assert!(cosine_f32(&xi, &xv, &yi, &yv).is_err());

        let xvh = to_f16(&xv);
        let yvh = to_f16(&yv);
        assert!(l2_f16(&xi, &xvh, &yi, &yvh).is_err());

        let err = l2_f32(&xi, &xv, &yi, &yv).unwrap_err();
        assert_eq!(err.idx_len, 3);
        assert_eq!(err.val_len, 2);
    }
}

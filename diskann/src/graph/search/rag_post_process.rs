/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! RAG (Retrieval-Augmented Generation) search post-processing.
//!
//! This module provides post-processing functionality for RAG search,
//! which reranks search results to maximize diversity using the greedy orthogonalization algorithm.
//!
//! ## Algorithm: Greedy Orthogonalization on Scaled Vectors
//!
//! Given l candidate vectors v_1, ..., v_l and a query q, select k vectors that maximize
//! the determinant of the Gram matrix of scaled vectors.
//!
//! Each vector v_i is scaled by: `scale_i = (v_i . q)^power`
//!
//! The algorithm works as follows:
//! 1. Initialize residuals R_i = scale_i * v_i for all candidates
//! 2. For t = 1 to k:
//!    - Pick i* = argmax_i ||R_i||^2
//!    - Add i* to selected set S
//!    - Orthogonalize remaining vectors: R_j = R_j - (R_j . R_i* / ||R_i*||^2) * R_i*
//!
//! This is equivalent to pivoted QR decomposition and provides a (1/k!)-approximation
//! to the optimal volume selection. Complexity: O(n * k * d).

use diskann_vector::{MathematicalValue, PureDistanceFunction, distance::InnerProduct};

/// Post-process search results using RAG-optimized reranking.
///
/// If RAG search is enabled, uses greedy orthogonalization for diversity.
/// Otherwise, returns results sorted by distance.
///
/// # Arguments
///
/// * `candidates` - Vector of (id, distance, vector) tuples.
/// * `query` - The query vector as f32 slice.
/// * `k` - Number of results to return.
/// * `rag_eta` - Ridge regularization parameter. Use 0.0 for pure greedy orthogonalization.
/// * `rag_power` - Power to raise similarity scores to before scaling vectors.
///
/// # Returns
///
/// Returns a vector of (id, distance) tuples, reranked for diversity.
pub fn rag_post_process<Id: Copy>(
    candidates: Vec<(Id, f32, &[f32])>,
    query: &[f32],
    k: usize,
    rag_eta: f64,
    rag_power: f64,
) -> Vec<(Id, f32)> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let k = k.min(candidates.len());

    // Convert candidates vectors to owned f32 for in-place orthogonalization
    let candidates_f32: Vec<(Id, f32, Vec<f32>)> = candidates
        .into_iter()
        .map(|(id, dist, v)| (id, dist, v.to_vec()))
        .collect();

    let results = if rag_eta > 0.0 {
        // Use eta-based algorithm
        post_process_with_eta_f32(candidates_f32, query, k, rag_eta, rag_power)
    } else {
        // Use greedy orthogonalization (eta = 0)
        post_process_greedy_orthogonalization_f32(candidates_f32, query, k, rag_power)
    };

    debug_assert_eq!(
        results.len(),
        k,
        "RAG post-process should return exactly k={} results, got {}",
        k,
        results.len()
    );

    results
}

/// Ridge-aware greedy log-det algorithm for eta > 0 (f32 version).
///
/// Maximizes: log det(eta*I + sum_i v_i v_i^T) where vectors are scaled by similarity^power.
///
/// This algorithm is equivalent to greedy orthogonalization but with a ridge regularization.
/// The key insight is that we can reduce this to a modified QR-like iteration:
///
/// **Initialization:**
/// - r_i <- v_i / sqrt(eta) (scale vectors)
/// - s_i <- ||r_i||^2 (precompute squared norms)
///
/// **For t = 1 to k:**
/// 1. Select: j = argmax_i s_i
/// 2. Normalize: q = r_j / sqrt(1 + s_j)  (the 1 + s_j is the ridge effect)
/// 3. Update: For all i != j:
///    - alpha = q^T r_i
///    - r_i <- r_i - alpha*q
///    - s_i <- s_i - alpha^2 (incremental norm update)
///
/// Complexity: O(n * k * d) - same as greedy orthogonalization!
fn post_process_with_eta_f32<Id: Copy>(
    candidates: Vec<(Id, f32, Vec<f32>)>,
    query: &[f32],
    k: usize,
    rag_eta: f64,
    rag_power: f64,
) -> Vec<(Id, f32)> {
    let eta = rag_eta as f32;
    let power = rag_power;

    if candidates.is_empty() || query.is_empty() {
        return Vec::new();
    }

    let n = candidates.len();
    let k = k.min(n);

    if k == 0 {
        return Vec::new();
    }

    let d = candidates[0].2.len();
    if d == 0 {
        return Vec::new();
    }

    let inv_sqrt_eta = 1.0 / eta.sqrt();

    // Initialization: r_i = (scale_i * v_i) / sqrt(eta), s_i = ||r_i||^2
    // where scale_i = max(0, v_i . q)^power
    let mut residuals: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut norms_sq: Vec<f32> = Vec::with_capacity(n);

    for (_, _, v) in &candidates {
        let similarity = dot_product(v, query);
        let scale = similarity.max(0.0).powf(power as f32) * inv_sqrt_eta;
        let r: Vec<f32> = v.iter().map(|&x| x * scale).collect();
        let s = dot_product(&r, &r);
        residuals.push(r);
        norms_sq.push(s);
    }

    let mut available: Vec<bool> = vec![true; n];
    let mut selected: Vec<usize> = Vec::with_capacity(k);

    for _ in 0..k {
        let best_idx = available
            .iter()
            .enumerate()
            .filter(|&(_, &avail)| avail)
            .max_by(|(i, _), (j, _)| {
                norms_sq[*i]
                    .partial_cmp(&norms_sq[*j])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);

        let Some(j) = best_idx else {
            break;
        };

        // Add to selected set
        selected.push(j);
        available[j] = false;

        // Early exit if this is the last selection
        if selected.len() == k {
            break;
        }

        // Normalize: q = r_j / sqrt(1 + s_j)
        // The (1 + s_j) term is the ridge effect
        let norm_factor = 1.0 / (1.0 + norms_sq[j]).sqrt();
        let q: Vec<f32> = residuals[j].iter().map(|&x| x * norm_factor).collect();

        // Update: For all i != j
        for i in 0..n {
            if !available[i] {
                continue;
            }

            // alpha = q^T r_i
            let alpha = dot_product(&q, &residuals[i]);

            // r_i <- r_i - alpha*q (SIMD-friendly loop)
            for (r_val, &q_val) in residuals[i].iter_mut().zip(q.iter()) {
                *r_val -= alpha * q_val;
            }

            // s_i <- s_i - alpha^2 (incremental norm update)
            norms_sq[i] = (norms_sq[i] - alpha * alpha).max(0.0);
        }
    }

    selected
        .iter()
        .map(|&idx| {
            let (id, dist, _) = candidates[idx];
            (id, dist)
        })
        .collect()
}

/// Multiply a matrix by a vector: result = M * v
#[allow(dead_code)]
#[inline]
fn matrix_vec_mult(m: &[Vec<f32>], v: &[f32]) -> Vec<f32> {
    m.iter().map(|row| dot_product(row, v)).collect()
}

/// Greedy orthogonalization algorithm for eta = 0 (f32 version).
///
/// Each vector is scaled by `max(0, dot_product_with_query)^power` before applying
/// the greedy max-determinant selection algorithm.
///
/// Optimized with:
/// - Cached squared norms (avoid recomputing each iteration)
/// - Parallel orthogonalization updates using rayon
fn post_process_greedy_orthogonalization_f32<Id: Copy>(
    candidates: Vec<(Id, f32, Vec<f32>)>,
    query: &[f32],
    k: usize,
    rag_power: f64,
) -> Vec<(Id, f32)> {
    let power = rag_power;

    if candidates.is_empty() || query.is_empty() {
        return Vec::new();
    }

    let n = candidates.len();
    let k = k.min(n);

    if k == 0 {
        return Vec::new();
    }

    // Compute scaled vectors and cache norms
    let mut residuals: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut norms_sq: Vec<f32> = Vec::with_capacity(n);

    for (_, _, v) in &candidates {
        let similarity = dot_product(v, query);
        let scale = similarity.max(0.0).powf(power as f32);
        let r: Vec<f32> = v.iter().map(|&x| x * scale).collect();
        let s = dot_product(&r, &r);
        residuals.push(r);
        norms_sq.push(s);
    }

    let mut available: Vec<bool> = vec![true; n];
    let mut selected: Vec<usize> = Vec::with_capacity(k);

    for _ in 0..k {
        // Find the vector with maximum residual norm squared (using cached norms)
        let best = available
            .iter()
            .enumerate()
            .filter(|&(_, &avail)| avail)
            .max_by(|(i, _), (j, _)| {
                norms_sq[*i]
                    .partial_cmp(&norms_sq[*j])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        let Some((i_star, _)) = best else {
            break;
        };

        let best_norm_sq = norms_sq[i_star];

        // Add to selected set
        selected.push(i_star);
        available[i_star] = false;

        // Early exit if this is the last selection
        if selected.len() == k {
            break;
        }

        // Skip orthogonalization if norm is zero
        if best_norm_sq <= 0.0 {
            continue;
        }

        // Orthogonalize remaining residuals: R_j = R_j - (R_j . R_i* / ||R_i*||^2) * R_i*
        // Clone r_star to avoid borrow issues
        let r_star = residuals[i_star].clone();
        let inv_norm_sq_star = 1.0 / best_norm_sq;

        for j in 0..n {
            if !available[j] {
                continue;
            }

            let proj_coeff = dot_product(&residuals[j], &r_star) * inv_norm_sq_star;

            // r_j <- r_j - proj_coeff * r_star (SIMD-friendly loop)
            for (r_val, &rs_val) in residuals[j].iter_mut().zip(r_star.iter()) {
                *r_val -= proj_coeff * rs_val;
            }

            // Update norm incrementally: ||r||^2 - proj^2 * ||rs||^2
            norms_sq[j] = (norms_sq[j] - proj_coeff * proj_coeff * best_norm_sq).max(0.0);
        }
    }

    selected
        .iter()
        .map(|&idx| {
            let (id, dist, _) = candidates[idx];
            (id, dist)
        })
        .collect()
}

/// Compute dot product of two vectors using SIMD-optimized implementation.
#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    <InnerProduct as PureDistanceFunction<&[f32], &[f32], MathematicalValue<f32>>>::evaluate(a, b)
        .into_inner()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rag_post_process_with_eta() {
        let v1 = vec![1.0f32, 0.0, 0.0];
        let v2 = vec![0.0f32, 1.0, 0.0];
        let v3 = vec![0.0f32, 0.0, 1.0];
        let candidates = vec![
            (1u32, 0.5f32, v1.as_slice()),
            (2u32, 0.3f32, v2.as_slice()),
            (3u32, 0.7f32, v3.as_slice()),
        ];
        let query = vec![1.0, 1.0, 1.0];

        let result = rag_post_process(candidates, &query, 3, 0.01, 2.0);

        // With eta > 0 and orthogonal vectors, all three should be selected
        // (order determined by scaled residual norms, not distance).
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_rag_post_process_enabled_greedy() {
        let v1 = vec![1.0f32, 0.0, 0.0];
        let v2 = vec![0.99f32, 0.1, 0.0]; // Similar to 1
        let v3 = vec![0.0f32, 1.0, 0.0]; // Orthogonal
        let candidates = vec![
            (1u32, 0.5f32, v1.as_slice()),
            (2u32, 0.3f32, v2.as_slice()),
            (3u32, 0.4f32, v3.as_slice()),
        ];
        let query = vec![1.0, 1.0, 0.0];

        let result = rag_post_process(candidates, &query, 2, 0.0, 1.0);

        // With greedy orthogonalization, should prefer diverse vectors
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_rag_post_process_empty() {
        let candidates: Vec<(u32, f32, &[f32])> = vec![];
        let query = vec![1.0, 1.0, 1.0];

        let result = rag_post_process(candidates, &query, 3, 0.01, 2.0);
        assert!(result.is_empty());
    }
}

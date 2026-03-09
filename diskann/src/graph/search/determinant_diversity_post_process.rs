/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Determinant-diversity search post-processing.
//!
//! This module provides post-processing functionality for determinant-diversity search,
//! which reranks search results to maximize diversity using a greedy
//! orthogonalization algorithm.

use diskann_vector::{MathematicalValue, PureDistanceFunction, distance::InnerProduct};

/// Parameters for determinant-diversity reranking.
#[derive(Debug, Clone, Copy)]
pub struct DeterminantDiversitySearchParams {
    pub top_k: usize,
    pub determinant_diversity_eta: f64,
    pub determinant_diversity_power: f64,
}

impl DeterminantDiversitySearchParams {
    pub fn new(
        top_k: usize,
        determinant_diversity_eta: f64,
        determinant_diversity_power: f64,
    ) -> Self {
        Self {
            top_k,
            determinant_diversity_eta,
            determinant_diversity_power,
        }
    }
}

/// Post-process search results using determinant-diversity reranking.
///
/// If `determinant_diversity_eta > 0.0`, uses a ridge-aware variant.
/// Otherwise, uses greedy orthogonalization.
pub fn determinant_diversity_post_process<Id: Copy>(
    candidates: Vec<(Id, f32, &[f32])>,
    query: &[f32],
    k: usize,
    determinant_diversity_eta: f64,
    determinant_diversity_power: f64,
) -> Vec<(Id, f32)> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let k = k.min(candidates.len());

    let candidates_f32: Vec<(Id, f32, Vec<f32>)> = candidates
        .into_iter()
        .map(|(id, dist, v)| (id, dist, v.to_vec()))
        .collect();

    let results = if determinant_diversity_eta > 0.0 {
        post_process_with_eta_f32(
            candidates_f32,
            query,
            k,
            determinant_diversity_eta,
            determinant_diversity_power,
        )
    } else {
        post_process_greedy_orthogonalization_f32(
            candidates_f32,
            query,
            k,
            determinant_diversity_power,
        )
    };

    debug_assert_eq!(
        results.len(),
        k,
        "determinant-diversity post-process should return exactly k={} results, got {}",
        k,
        results.len()
    );

    results
}

fn post_process_with_eta_f32<Id: Copy>(
    candidates: Vec<(Id, f32, Vec<f32>)>,
    query: &[f32],
    k: usize,
    determinant_diversity_eta: f64,
    determinant_diversity_power: f64,
) -> Vec<(Id, f32)> {
    let eta = determinant_diversity_eta as f32;
    let power = determinant_diversity_power;

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

        selected.push(j);
        available[j] = false;

        if selected.len() == k {
            break;
        }

        let norm_factor = 1.0 / (1.0 + norms_sq[j]).sqrt();
        let q: Vec<f32> = residuals[j].iter().map(|&x| x * norm_factor).collect();

        for i in 0..n {
            if !available[i] {
                continue;
            }

            let alpha = dot_product(&q, &residuals[i]);

            for (r_val, &q_val) in residuals[i].iter_mut().zip(q.iter()) {
                *r_val -= alpha * q_val;
            }

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

fn post_process_greedy_orthogonalization_f32<Id: Copy>(
    candidates: Vec<(Id, f32, Vec<f32>)>,
    query: &[f32],
    k: usize,
    determinant_diversity_power: f64,
) -> Vec<(Id, f32)> {
    let power = determinant_diversity_power;

    if candidates.is_empty() || query.is_empty() {
        return Vec::new();
    }

    let n = candidates.len();
    let k = k.min(n);

    if k == 0 {
        return Vec::new();
    }

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

        selected.push(i_star);
        available[i_star] = false;

        if selected.len() == k {
            break;
        }

        if best_norm_sq <= 0.0 {
            continue;
        }

        let r_star = residuals[i_star].clone();
        let inv_norm_sq_star = 1.0 / best_norm_sq;

        for j in 0..n {
            if !available[j] {
                continue;
            }

            let proj_coeff = dot_product(&residuals[j], &r_star) * inv_norm_sq_star;

            for (r_val, &rs_val) in residuals[j].iter_mut().zip(r_star.iter()) {
                *r_val -= proj_coeff * rs_val;
            }

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

#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    <InnerProduct as PureDistanceFunction<&[f32], &[f32], MathematicalValue<f32>>>::evaluate(a, b)
        .into_inner()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determinant_diversity_post_process_with_eta() {
        let v1 = vec![1.0f32, 0.0, 0.0];
        let v2 = vec![0.0f32, 1.0, 0.0];
        let v3 = vec![0.0f32, 0.0, 1.0];
        let candidates = vec![
            (1u32, 0.5f32, v1.as_slice()),
            (2u32, 0.3f32, v2.as_slice()),
            (3u32, 0.7f32, v3.as_slice()),
        ];
        let query = vec![1.0, 1.0, 1.0];

        let result = determinant_diversity_post_process(candidates, &query, 3, 0.01, 2.0);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_determinant_diversity_post_process_enabled_greedy() {
        let v1 = vec![1.0f32, 0.0, 0.0];
        let v2 = vec![0.99f32, 0.1, 0.0];
        let v3 = vec![0.0f32, 1.0, 0.0];
        let candidates = vec![
            (1u32, 0.5f32, v1.as_slice()),
            (2u32, 0.3f32, v2.as_slice()),
            (3u32, 0.4f32, v3.as_slice()),
        ];
        let query = vec![1.0, 1.0, 0.0];

        let result = determinant_diversity_post_process(candidates, &query, 2, 0.0, 1.0);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_determinant_diversity_post_process_empty() {
        let candidates: Vec<(u32, f32, &[f32])> = vec![];
        let query = vec![1.0, 1.0, 1.0];

        let result = determinant_diversity_post_process(candidates, &query, 3, 0.01, 2.0);
        assert!(result.is_empty());
    }
}

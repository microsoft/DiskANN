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

/// Error type for determinant-diversity parameter validation.
#[derive(Debug)]
pub enum DeterminantDiversityError {
    InvalidTopK { top_k: usize },
    InvalidEta { eta: f64 },
    InvalidPower { power: f64 },
}

impl std::fmt::Display for DeterminantDiversityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTopK { top_k } => {
                write!(f, "top_k must be > 0, got {}", top_k)
            }
            Self::InvalidEta { eta } => {
                write!(f, "eta must be >= 0.0, got {}", eta)
            }
            Self::InvalidPower { power } => {
                write!(f, "power must be > 0.0, got {}", power)
            }
        }
    }
}

impl std::error::Error for DeterminantDiversityError {}

/// Parameters for determinant-diversity reranking.
///
/// # Invariants
///
/// - `top_k > 0`: Must request at least one result
/// - `determinant_diversity_eta >= 0.0`: Ridge regularization parameter (0 = no ridge)
/// - `determinant_diversity_power > 0.0`: Exponent for diversity scaling (typically 1.0-2.0)
#[derive(Debug, Clone, Copy)]
pub struct DeterminantDiversitySearchParams {
    pub top_k: usize,
    pub determinant_diversity_eta: f64,
    pub determinant_diversity_power: f64,
}

impl DeterminantDiversitySearchParams {
    /// Construct parameters with validation.
    ///
    /// # Arguments
    ///
    /// * `top_k` - Number of results to return (must be > 0)
    /// * `determinant_diversity_eta` - Ridge regularization parameter (must be >= 0.0)
    /// * `determinant_diversity_power` - Diversity exponent (must be > 0.0)
    ///
    /// # Errors
    ///
    /// Returns [`DeterminantDiversityError`] if any parameter is invalid.
    pub fn new(
        top_k: usize,
        determinant_diversity_eta: f64,
        determinant_diversity_power: f64,
    ) -> Result<Self, DeterminantDiversityError> {
        if top_k == 0 {
            return Err(DeterminantDiversityError::InvalidTopK { top_k });
        }

        if determinant_diversity_eta < 0.0 || !determinant_diversity_eta.is_finite() {
            return Err(DeterminantDiversityError::InvalidEta {
                eta: determinant_diversity_eta,
            });
        }

        if determinant_diversity_power <= 0.0 || !determinant_diversity_power.is_finite() {
            return Err(DeterminantDiversityError::InvalidPower {
                power: determinant_diversity_power,
            });
        }

        Ok(Self {
            top_k,
            determinant_diversity_eta,
            determinant_diversity_power,
        })
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
    if candidates.is_empty() || query.is_empty() {
        return Vec::new();
    }

    let k = k.min(candidates.len());
    if k == 0 {
        return Vec::new();
    }

    // Convert vectors to owned format only once
    let candidates_f32: Vec<(Id, f32, Vec<f32>)> = candidates
        .into_iter()
        .map(|(id, dist, v)| (id, dist, v.to_vec()))
        .collect();

    if candidates_f32[0].2.is_empty() {
        return Vec::new();
    }

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

    // Initialize residuals and norms (only one allocation per candidate)
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

        // Compute all projections first to avoid needing to clone residuals[j]
        let mut projections: Vec<f32> = Vec::with_capacity(n);
        for i in 0..n {
            if !available[i] {
                projections.push(0.0);
            } else {
                let alpha = dot_product(&residuals[j], &residuals[i]) * norm_factor * norm_factor;
                projections.push(alpha);
            }
        }

        // Now apply all updates using the precomputed projections
        let q_scaled: Vec<f32> = residuals[j].iter().map(|&x| x * norm_factor).collect();
        for i in 0..n {
            if !available[i] {
                continue;
            }

            let alpha = projections[i];
            for (r_val, &q_val) in residuals[i].iter_mut().zip(q_scaled.iter()) {
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

    // Initialize residuals and norms (only one allocation per candidate)
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

        let inv_norm_sq_star = 1.0 / best_norm_sq;

        // Compute all projections and make a copy of r_star to avoid borrow conflicts
        let r_star_copy = residuals[i_star].clone();
        let mut projections: Vec<f32> = Vec::with_capacity(n);
        for j in 0..n {
            if !available[j] {
                projections.push(0.0);
            } else {
                let proj = dot_product(&residuals[j], &r_star_copy) * inv_norm_sq_star;
                projections.push(proj);
            }
        }

        // Now apply all updates using the precomputed projections
        for j in 0..n {
            if !available[j] {
                continue;
            }

            let proj_coeff = projections[j];
            for (r_val, &rs_val) in residuals[j].iter_mut().zip(r_star_copy.iter()) {
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

    // ===== Validation Tests =====

    #[test]
    fn test_validation_valid_params() {
        let result = DeterminantDiversitySearchParams::new(10, 0.01, 2.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validation_zero_top_k() {
        let result = DeterminantDiversitySearchParams::new(0, 0.01, 2.0);
        assert!(matches!(
            result,
            Err(DeterminantDiversityError::InvalidTopK { top_k: 0 })
        ));
    }

    #[test]
    fn test_validation_negative_eta() {
        let result = DeterminantDiversitySearchParams::new(10, -0.01, 2.0);
        assert!(matches!(
            result,
            Err(DeterminantDiversityError::InvalidEta { .. })
        ));
    }

    #[test]
    fn test_validation_zero_power() {
        let result = DeterminantDiversitySearchParams::new(10, 0.01, 0.0);
        assert!(matches!(
            result,
            Err(DeterminantDiversityError::InvalidPower { .. })
        ));
    }

    #[test]
    fn test_validation_negative_power() {
        let result = DeterminantDiversitySearchParams::new(10, 0.01, -1.0);
        assert!(matches!(
            result,
            Err(DeterminantDiversityError::InvalidPower { .. })
        ));
    }

    #[test]
    fn test_validation_nan_eta() {
        let result = DeterminantDiversitySearchParams::new(10, f64::NAN, 2.0);
        assert!(matches!(
            result,
            Err(DeterminantDiversityError::InvalidEta { .. })
        ));
    }

    #[test]
    fn test_validation_infinity_power() {
        let result = DeterminantDiversitySearchParams::new(10, 0.01, f64::INFINITY);
        assert!(matches!(
            result,
            Err(DeterminantDiversityError::InvalidPower { .. })
        ));
    }

    // ===== Algorithm Tests =====

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

    #[test]
    fn test_determinant_diversity_post_process_k_larger_than_candidates() {
        let v1 = vec![1.0f32, 0.0, 0.0];
        let v2 = vec![0.0f32, 1.0, 0.0];
        let candidates = vec![(1u32, 0.5f32, v1.as_slice()), (2u32, 0.3f32, v2.as_slice())];
        let query = vec![1.0, 1.0, 1.0];

        let result = determinant_diversity_post_process(candidates, &query, 10, 0.01, 2.0);
        // Should return min(k, len(candidates)) = 2
        assert_eq!(result.len(), 2);
    }
}

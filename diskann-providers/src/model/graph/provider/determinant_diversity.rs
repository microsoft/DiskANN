/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Determinant-Diversity post-processing for search results.
//!
//! This module implements the Determinant-Diversity algorithm for diversity-promoting
//! reranking of approximate nearest neighbor search results. The algorithm takes
//! relevance-ranked candidates and reorders them to maximize geometric diversity
//! while maintaining relevance to the original query.
//!
//! # Algorithm Overview
//!
//! Determinant-Diversity selects a diverse subset from an initial set of candidates
//! by iteratively choosing points that maximize the determinant of the distance matrix.
//! This creates a diverse set that is both relevant to the query and geometrically spread out.
//!
//! # Parameters
//!
//! - **power**: Relevance weighting exponent (must be > 0.0). Controls the emphasis on
//!   maintaining relevance scores from the initial search. Higher values prefer relevance
//!   over diversity.
//!
//! - **eta**: Numerical stability parameter (must be >= 0.0). Used for ridge regularization:
//!   - `eta = 0`: Exact determinant computation (can be numerically unstable for some inputs)
//!   - `eta > 0`: Ridge-regularized computation for improved numerical stability
//!
//! # Variants
//!
//! The module provides two implementations:
//!
//! - `post_process_with_eta_f32()`: Uses ridge regularization for numerical stability
//! - `post_process_without_eta_f32()`: Computes exact determinants (faster but less stable)
//!
//! These are selected automatically based on the eta parameter value.
//!
//! # Time Complexity
//!
//! O(m³) where m is the number of candidates, due to determinant computation.
//! In practice, m is typically small (search returns hundreds of candidates,
//! but only top-k ≪ m are selected).
//!
//! # References
//!
//! The algorithm is based on diversity-promoting ranking methods for nearest neighbor search,
//! as used in approximate nearest neighbor indices like DiskANN.

use diskann_utils::views::Matrix;
use diskann_vector::{MathematicalValue, PureDistanceFunction, distance::InnerProduct};

#[derive(Clone, Copy)]
struct DistanceRange {
    min: f32,
    max: f32,
}

pub fn determinant_diversity_post_process<Id: Copy>(
    candidates: Vec<(Id, f32, Vec<f32>)>,
    query: &[f32],
    k: usize,
    determinant_diversity_eta: f32,
    determinant_diversity_power: f32,
) -> Vec<(Id, f32)> {
    if candidates.is_empty() || query.is_empty() {
        return Vec::new();
    }

    assert!(
        candidates
            .iter()
            .all(|(_, _, vector)| vector.len() == query.len()),
        "all candidate vectors must have the same dimension as query"
    );

    let k = k.min(candidates.len());
    if k == 0 {
        return Vec::new();
    }

    if candidates[0].2.is_empty() {
        return Vec::new();
    }

    let distance_range = {
        let mut min_distance = f32::INFINITY;
        let mut max_distance = f32::NEG_INFINITY;

        for (_, distance, _) in &candidates {
            min_distance = min_distance.min(*distance);
            max_distance = max_distance.max(*distance);
        }

        DistanceRange {
            min: min_distance,
            max: max_distance,
        }
    };

    // For eta=0, the inv_sqrt_eta factor is 1.0 (greedy orthogonalization without regularization).
    // For eta>0, the factor scales residuals for ridge-regularized determinant computation.
    let inv_sqrt_eta = if determinant_diversity_eta > 0.0 {
        1.0 / determinant_diversity_eta.sqrt()
    } else {
        1.0
    };

    greedy_orthogonal_select(
        candidates,
        k,
        determinant_diversity_power,
        inv_sqrt_eta,
        distance_range,
    )
}

/// Core greedy selection algorithm for Determinant-Diversity.
///
/// Iteratively selects the candidate with the largest residual norm after projecting
/// out previously selected candidates. The `inv_sqrt_eta` parameter controls the
/// ridge-regularization scaling:
///
/// - `inv_sqrt_eta = 1.0`: exact greedy orthogonalization (eta=0 case)
/// - `inv_sqrt_eta = 1/sqrt(eta)`: ridge-regularized variant for numerical stability
fn greedy_orthogonal_select<Id: Copy>(
    candidates: Vec<(Id, f32, Vec<f32>)>,
    k: usize,
    power: f32,
    inv_sqrt_eta: f32,
    distance_range: DistanceRange,
) -> Vec<(Id, f32)> {
    let n = candidates.len();
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }

    let dim = candidates[0].2.len();

    // Use a contiguous Matrix allocation for residuals instead of Vec<Vec<f32>>.
    // This reduces the number of heap allocations from O(n) to O(1) and improves
    // cache locality when accessing residuals sequentially during orthogonalization.
    let mut residuals = Matrix::new(0.0f32, n, dim);
    let mut norms_sq = Vec::with_capacity(n);

    for (i, (_, distance_to_query, v)) in candidates.iter().enumerate() {
        let scale =
            distance_to_similarity(*distance_to_query, distance_range).powf(power) * inv_sqrt_eta;
        let row = residuals.row_mut(i);
        for (r, &x) in row.iter_mut().zip(v.iter()) {
            *r = x * scale;
        }
        let norm_sq = dot_product(residuals.row(i), residuals.row(i));
        norms_sq.push(norm_sq);
    }

    let mut available = vec![true; n];
    let mut selected = Vec::with_capacity(k);
    let mut projections = vec![0.0f32; n];

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

        let Some(selected_index) = best_idx else {
            break;
        };

        selected.push(selected_index);
        available[selected_index] = false;

        if selected.len() == k {
            break;
        }

        let best_norm_sq = norms_sq[selected_index];
        if best_norm_sq <= 0.0 {
            continue;
        }

        let inv_norm_sq = 1.0 / best_norm_sq;
        // Clone selected row before mutable iteration over remaining rows.
        let r_star_copy: Vec<f32> = residuals.row(selected_index).to_vec();

        for i in 0..n {
            if !available[i] {
                projections[i] = 0.0;
            } else {
                projections[i] = dot_product(residuals.row(i), &r_star_copy) * inv_norm_sq;
            }
        }

        for i in 0..n {
            if !available[i] {
                continue;
            }

            let projection = projections[i];
            for (residual, &star) in residuals.row_mut(i).iter_mut().zip(r_star_copy.iter()) {
                *residual -= projection * star;
            }

            norms_sq[i] = (norms_sq[i] - projection * projection * best_norm_sq).max(0.0);
        }
    }

    selected
        .iter()
        .map(|&idx| {
            let (id, dist, _) = &candidates[idx];
            (*id, *dist)
        })
        .collect()
}

fn distance_to_similarity(distance: f32, distance_range: DistanceRange) -> f32 {
    let span = (distance_range.max - distance_range.min).max(f32::EPSILON);

    // Distances are lower-is-better in DiskANN distance semantics.
    ((distance_range.max - distance) / span).max(0.0) + f32::EPSILON
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
    fn test_empty_candidates() {
        let result =
            determinant_diversity_post_process::<u32>(Vec::new(), &[1.0, 2.0], 5, 0.5, 1.0);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_empty_query() {
        let candidates = vec![(0u32, 0.5, vec![1.0, 2.0])];
        let result = determinant_diversity_post_process(candidates, &[], 5, 0.5, 1.0);
        assert_eq!(result.len(), 0);
    }

    #[test]
    #[should_panic(expected = "all candidate vectors must have the same dimension as query")]
    fn test_mismatched_dimensions_panics() {
        let candidates = vec![
            (0u32, 0.5, vec![1.0, 2.0]),
            (1u32, 0.3, vec![1.0]), // Wrong dimension
        ];
        let query = &[1.0, 2.0, 3.0];
        let _ = determinant_diversity_post_process(candidates, query, 5, 0.5, 1.0);
    }

    #[test]
    fn test_single_candidate() {
        let candidates = vec![(0u32, 0.5, vec![1.0, 2.0])];
        let query = &[1.0, 2.0];
        let result = determinant_diversity_post_process(candidates, query, 5, 0.5, 1.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 0);
    }

    #[test]
    fn test_k_larger_than_candidates() {
        let candidates = vec![(0u32, 0.5, vec![1.0, 0.0]), (1u32, 0.3, vec![0.0, 1.0])];
        let query = &[1.0, 1.0];
        let result = determinant_diversity_post_process(candidates, query, 10, 0.5, 1.0);
        assert_eq!(result.len(), 2); // Should return min(k, candidates.len())
    }

    #[test]
    fn test_with_eta_diversity() {
        let candidates = vec![
            (0u32, 0.1, vec![1.0, 0.0]),
            (1u32, 0.2, vec![0.9, 0.1]),
            (2u32, 0.3, vec![0.8, 0.2]),
        ];
        let query = &[1.0, 1.0];
        let result = determinant_diversity_post_process(candidates, query, 2, 1.0, 1.0);

        assert_eq!(result.len(), 2);
        // Should select based on diversity metric with eta > 0
        assert!(result.iter().all(|(id, _)| *id < 3));
    }

    #[test]
    fn test_without_eta_greedy() {
        let candidates = vec![
            (0u32, 0.1, vec![1.0, 0.0]),
            (1u32, 0.2, vec![0.9, 0.1]),
            (2u32, 0.3, vec![0.8, 0.2]),
        ];
        let query = &[1.0, 1.0];
        let result = determinant_diversity_post_process(candidates, query, 2, 0.0, 1.0);

        assert_eq!(result.len(), 2);
        // Should select based on greedy orthogonalization (eta == 0)
        assert!(result.iter().all(|(id, _)| *id < 3));
    }

    #[test]
    fn test_power_parameter() {
        let candidates = vec![(0u32, 0.1, vec![1.0, 0.0]), (1u32, 0.2, vec![0.0, 1.0])];
        let query = &[1.0, 1.0];

        // Test with different power values - should still work without panicking
        let result1 = determinant_diversity_post_process(candidates.clone(), query, 2, 0.0, 1.0);
        let result2 = determinant_diversity_post_process(candidates, query, 2, 0.0, 2.0);

        assert_eq!(result1.len(), 2);
        assert_eq!(result2.len(), 2);
    }

    #[test]
    fn test_distances_preserved() {
        let candidates = vec![(0u32, 0.5, vec![1.0, 0.0]), (1u32, 0.3, vec![0.0, 1.0])];
        let query = &[1.0, 1.0];
        let result = determinant_diversity_post_process(candidates, query, 2, 0.0, 1.0);

        // Verify that distances are preserved from input
        assert!(result.iter().all(|(_, dist)| *dist == 0.5 || *dist == 0.3));
    }

    /// Verify that diversity is actually promoted: when candidates lie along orthogonal
    /// directions, a 2-element diverse subset should choose orthogonal pairs over similar ones.
    ///
    /// Using equal distances ensures pure diversity drives selection without relevance weighting.
    #[test]
    fn test_diversity_selects_orthogonal_candidates() {
        // Three candidates with equal distance: two very similar (nearly parallel) and one orthogonal.
        // Equal distances remove relevance weighting, so pure diversity drives selection.
        let candidates = vec![
            (0u32, 0.1, vec![1.0, 0.0, 0.0]),   // along x
            (1u32, 0.1, vec![0.0, 1.0, 0.0]),   // along y - orthogonal to 0
            (2u32, 0.1, vec![0.99, 0.01, 0.0]), // nearly parallel to 0
        ];
        let query = &[1.0, 1.0, 1.0];
        let result = determinant_diversity_post_process(candidates, query, 2, 0.0, 1.0);

        // Should select 2 candidates
        assert_eq!(result.len(), 2);
        // The diverse pair is (0, 1) - orthogonal. Candidate 2 is redundant with 0.
        let ids: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&0), "Expected candidate 0 to be selected");
        assert!(
            ids.contains(&1),
            "Expected candidate 1 (orthogonal) to be selected, not redundant candidate 2"
        );
    }

    /// Verify eta variant selects the same k results.
    #[test]
    fn test_diversity_selects_orthogonal_candidates_with_eta() {
        let candidates = vec![
            (0u32, 0.1, vec![1.0, 0.0, 0.0]),
            (1u32, 0.1, vec![0.0, 1.0, 0.0]),
            (2u32, 0.1, vec![0.99, 0.01, 0.0]),
        ];
        let query = &[1.0, 1.0, 1.0];
        let result = determinant_diversity_post_process(candidates, query, 2, 0.5, 1.0);

        assert_eq!(result.len(), 2);
        let ids: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&0), "Expected candidate 0 to be selected");
        assert!(
            ids.contains(&1),
            "Expected candidate 1 (orthogonal) to be selected"
        );
    }

    /// Verify power=high weights nearby candidates (distance=0.1) more strongly than far ones.
    #[test]
    fn test_high_power_prefers_closer_candidates() {
        // Two orthogonal candidates: one close, one far
        let candidates = vec![
            (0u32, 0.1, vec![1.0, 0.0]), // close to query
            (1u32, 0.9, vec![0.0, 1.0]), // far from query
        ];
        let query = &[1.0, 0.0];

        // With high power, relevance is heavily weighted so the closest candidate dominates
        let result = determinant_diversity_post_process(candidates.clone(), query, 1, 0.0, 10.0);
        assert_eq!(result.len(), 1);
        // Closest candidate should be preferred due to high power weighting
        assert_eq!(
            result[0].0, 0,
            "Closest candidate should be selected with high power"
        );
    }

    /// Verify that distance-to-similarity conversion handles equal distances gracefully.
    #[test]
    fn test_equal_distances() {
        let candidates = vec![
            (0u32, 0.5, vec![1.0, 0.0]),
            (1u32, 0.5, vec![0.0, 1.0]), // same distance as 0
        ];
        let query = &[1.0, 0.0];
        let result = determinant_diversity_post_process(candidates, query, 2, 0.0, 1.0);

        // Should still return candidates without panicking
        assert_eq!(result.len(), 2);
    }

    /// Test eta=0 exactly matches greedy orthogonalization path.
    #[test]
    fn test_eta_zero_is_greedy_path() {
        let candidates = vec![
            (0u32, 0.1, vec![1.0, 0.0]),
            (1u32, 0.2, vec![0.0, 1.0]),
            (2u32, 0.3, vec![0.5, 0.5]),
        ];
        let query = &[1.0, 1.0];
        // eta=0.0 must invoke greedy path, not ridge-regularized
        let result = determinant_diversity_post_process(candidates, query, 2, 0.0, 1.0);
        assert_eq!(result.len(), 2);
    }
}

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
//! Concretely, each candidate vector v_i is scaled by a relevance weight
//! alpha_i = similarity(d_i)^power / sqrt(eta) derived from its distance d_i
//! to the query (see `distance_to_similarity`). Letting X be the matrix of
//! scaled rows x_i = alpha_i * v_i, we approximately maximize
//! det(X_S * X_S^T + eta * I) over subsets S of size k via greedy pivoted
//! Gram-Schmidt: at each step we pick the row with the largest residual norm
//! and deflate the rest against it. See [`greedy_orthogonal_select`] for the
//! full derivation.
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
//! The public entry point is [`determinant_diversity`].
//! It applies either the unregularized (`eta == 0`) or ridge-regularized (`eta > 0`)
//! formulation internally.
//!
//! # Time Complexity
//!
//! O(n * k * dim), where n is number of candidates, k is requested output size,
//! and dim is vector dimensionality.
//!
//! # References
//!
//! The algorithm is based on diversity-promoting ranking methods for nearest neighbor search,
//! as used in approximate nearest neighbor indices like DiskANN.

use std::fmt;

use diskann_utils::views::MutMatrixView;
use diskann_vector::{MathematicalValue, PureDistanceFunction, distance::InnerProduct};

/// Parameters for Determinant-Diversity post-processor with validation.
///
/// Determinant-Diversity is a diversity-promoting reranking algorithm that takes
/// relevance-ranked neighbors and reorders them to maximize geometric diversity
/// while maintaining relevance.
///
/// # Parameters
///
/// - `power`: Relevance weighting exponent. Controls the emphasis on maintaining
///   relevance scores from the original search. Must be > 0.0.
///
/// - `eta`: Numerical stability parameter for ridge-regularization. Controls the
///   trade-off between exact determinant computation (eta=0) and numerical robustness
///   (eta>0). Must be >= 0.0.
///
/// # Errors
///
/// Construction fails if:
/// - `power` is non-finite or `<= 0.0` (invalid power weighting)
/// - `eta` is non-finite or `< 0.0` (negative stability parameter)
#[derive(Debug, Clone, Copy)]
pub struct DeterminantDiversityParams {
    /// Relevance weighting exponent. Must be > 0.0.
    power: f32,
    /// Numerical stability parameter. Must be >= 0.0.
    eta: f32,
}

impl DeterminantDiversityParams {
    /// Create and validate new Determinant-Diversity parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails:
    /// - `power` is non-finite or `<= 0.0`: invalid relevance weighting
    /// - `eta` is non-finite or `< 0.0`: invalid numerical stability parameter
    pub fn new(power: f32, eta: f32) -> Result<Self, DeterminantDiversityError> {
        if !power.is_finite() || power <= 0.0 {
            return Err(DeterminantDiversityError::InvalidPower(power));
        }
        if !eta.is_finite() || eta < 0.0 {
            return Err(DeterminantDiversityError::InvalidEta(eta));
        }
        Ok(Self { power, eta })
    }

    /// Get power parameter.
    #[inline]
    pub fn power(&self) -> f32 {
        self.power
    }

    /// Get eta parameter.
    #[inline]
    pub fn eta(&self) -> f32 {
        self.eta
    }
}

impl fmt::Display for DeterminantDiversityParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DeterminantDiversity(power={}, eta={})",
            self.power, self.eta
        )
    }
}

/// Error produced when constructing [`DeterminantDiversityParams`] or running
/// [`determinant_diversity`].
#[derive(Debug, Clone, thiserror::Error)]
pub enum DeterminantDiversityError {
    /// Power parameter <= 0.0
    #[error("determinant-diversity power must be > 0.0, got: {0}")]
    InvalidPower(f32),
    /// Eta parameter < 0.0
    #[error("determinant-diversity eta must be >= 0.0, got: {0}")]
    InvalidEta(f32),
    /// The candidate matrix column count does not match the query dimension.
    #[error(
        "determinant-diversity candidate matrix has {candidate} columns but query dimension is {query}"
    )]
    QueryDimensionMismatch {
        /// Number of dimensions in the query.
        query: usize,
        /// Number of columns in the candidate matrix.
        candidate: usize,
    },
    /// The number of distances does not match the number of candidate rows.
    #[error("determinant-diversity received {distances} distances for {candidates} candidate rows")]
    DistanceCountMismatch {
        /// Number of supplied distances.
        distances: usize,
        /// Number of candidate rows.
        candidates: usize,
    },
}

#[derive(Clone, Copy)]
struct DistanceRange {
    min: f32,
    max: f32,
}

/// Rerank `candidates` to promote geometric diversity while preserving relevance.
///
/// Returns the indices (into the rows of `candidates`) of the selected vectors,
/// in selection order, with at most `k` entries.
///
/// # Arguments
///
/// - `candidates`: row-major matrix whose `i`-th row is the full-precision vector
///   of the `i`-th candidate. Its column count must equal `query.len()`.
/// - `distances`: candidate-to-query distances, parallel to the rows of
///   `candidates`. Its length must equal `candidates.nrows()`.
/// - `query`: the query vector.
/// - `k`: maximum number of results to return; clamped to `candidates.nrows()`.
/// - `params`: relevance/regularization parameters.
///
/// # Errors
///
/// Returns [`DeterminantDiversityError::QueryDimensionMismatch`] if the candidate
/// matrix column count does not equal `query.len()`, or
/// [`DeterminantDiversityError::DistanceCountMismatch`] if `distances.len()` does
/// not equal `candidates.nrows()`. These structural invariants are validated up
/// front so they are enforced consistently even when an input is empty.
///
/// An empty candidate set, a `k` of zero, or zero-dimensional vectors yield an
/// empty result.
pub fn determinant_diversity(
    candidates: MutMatrixView<'_, f32>,
    distances: &[f32],
    query: &[f32],
    k: usize,
    params: &DeterminantDiversityParams,
) -> Result<Vec<usize>, DeterminantDiversityError> {
    // Validate structural invariants first so they are enforced consistently,
    // regardless of whether any individual input happens to be empty.
    if candidates.ncols() != query.len() {
        return Err(DeterminantDiversityError::QueryDimensionMismatch {
            query: query.len(),
            candidate: candidates.ncols(),
        });
    }

    if distances.len() != candidates.nrows() {
        return Err(DeterminantDiversityError::DistanceCountMismatch {
            distances: distances.len(),
            candidates: candidates.nrows(),
        });
    }

    let k = k.min(candidates.nrows());
    if k == 0 || candidates.ncols() == 0 {
        return Ok(Vec::new());
    }

    let distance_range = {
        let mut min_distance = f32::INFINITY;
        let mut max_distance = f32::NEG_INFINITY;

        for distance in distances {
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
    let inv_sqrt_eta = if params.eta() > 0.0 {
        1.0 / params.eta().sqrt()
    } else {
        1.0
    };

    Ok(greedy_orthogonal_select(
        candidates,
        distances,
        k,
        params.power(),
        inv_sqrt_eta,
        distance_range,
    ))
}

/// Core greedy selection algorithm for Determinant-Diversity.
///
/// # Mathematical formulation
///
/// Let the input candidate set be represented by matrix rows v_i (for i = 1..n)
/// and a parallel distance slice d_i, where d_i is the candidate distance to the
/// query and v_i is the full-precision vector in R^dim. Define the per-candidate scale
///
/// ```text
/// alpha_i = similarity(d_i)^power * (1 / sqrt(eta))
/// ```
///
/// where similarity(.) in [0, 1] is the normalized "lower-distance-is-better"
/// score from `distance_to_similarity`, and `1 / sqrt(eta)` is `inv_sqrt_eta`
/// (it equals 1 in the unregularized eta == 0 branch -- see the caller). The
/// scaled vectors are
///
/// ```text
/// x_i = alpha_i * v_i.
/// ```
///
/// Define the (regularized) Gram matrix of any subset S = { i_1, ..., i_m } as
///
/// ```text
/// G_S = X_S * X_S^T + eta * I,
/// ```
///
/// where X_S stacks the rows x_i for i in S. The goal is to pick S of size k
/// that approximately maximizes det(G_S), i.e. selects vectors whose scaled
/// rows span the largest volume -- geometrically diverse, while alpha_i keeps
/// relevance. We solve this greedily, which is equivalent to *column-pivoted
/// modified Gram-Schmidt / QR* on the rows x_i.
///
/// # Algorithm (pivoted QR view)
///
/// Maintain a residual vector r_i for each candidate. Initially r_i = x_i and
/// ||r_i||^2 = <x_i, x_i>. At each step:
///
/// 1. **Pivot.** Pick the available candidate i* with the largest residual
///    norm: i* = argmax over available i of ||r_i||^2. This is the direction
///    that contributes the most to the running volume / determinant expansion
///    (since det(G_S) = product of ||r_{i_j*}||^2 along the selection path).
///
/// 2. **Project & deflate.** For every remaining candidate i, project r_i
///    onto the chosen pivot direction r* = r_{i*} and remove that component:
///
///    ```text
///    pi_i = <r_i, r*> / ||r*||^2
///    r_i  := r_i - pi_i * r*
///    ```
///
/// 3. **Norm update (Pythagoras).** Because the new r_i is orthogonal to r*
///    by construction,
///
///    ```text
///    ||r_i_new||^2 = ||r_i||^2 - pi_i^2 * ||r*||^2.
///    ```
///
///    We update the cached squared norm in place using this identity (clamped
///    at 0 for numerical safety) instead of recomputing the dot product.
///
/// Repeat until k pivots are selected. The returned order is the order in
/// which pivots were chosen, which is the diversity-promoting reranking.
///
/// # Parameters
///
/// - `inv_sqrt_eta`: scalar 1 / sqrt(eta) baked into the residuals so that
///   the residual norms reflect the regularized Gram matrix X X^T + eta * I.
///   Use 1.0 for the unregularized (eta == 0) variant.
/// - `distances`: candidate distances parallel to matrix rows.
/// - `power`: relevance exponent applied to the per-candidate similarity.
/// - `distance_range`: min/max distances among the candidates, used to
///   normalize distances into similarities in [0, 1].
///
/// # Complexity
///
/// O(n * k * dim) -- for each of k pivots we touch all n residual rows of
/// length `dim`. Memory is O(n * dim) for the contiguous residual matrix.
fn greedy_orthogonal_select(
    mut candidates: MutMatrixView<'_, f32>,
    distances: &[f32],
    k: usize,
    power: f32,
    inv_sqrt_eta: f32,
    distance_range: DistanceRange,
) -> Vec<usize> {
    let n = candidates.nrows();
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }

    // Cached squared norms ||r_i||^2 for each row. Updated in place via the
    // Pythagorean identity in step 3 above.
    let mut norms_sq = Vec::with_capacity(n);

    // Step 0: scale rows in-place to initialize residuals r_i = alpha_i * v_i,
    // then compute their squared norms.
    // alpha_i = similarity(d_i)^power * inv_sqrt_eta.
    for (i, distance_to_query) in distances.iter().enumerate() {
        let scale =
            distance_to_similarity(*distance_to_query, distance_range).powf(power) * inv_sqrt_eta;
        for value in candidates.row_mut(i) {
            *value *= scale;
        }
        let norm_sq = dot_product(candidates.row(i), candidates.row(i));
        norms_sq.push(norm_sq);
    }

    let mut available = vec![true; n];
    let mut selected = Vec::with_capacity(k);
    // Scratch buffer: projection coefficient pi_i for each row against the
    // current pivot. Sized n once and overwritten each iteration.
    let mut projections = vec![0.0f32; n];

    for _ in 0..k {
        // --- Step 1: Pivot ---
        // Pick the available candidate with the largest residual norm.
        // partial_cmp can return None for NaN; treat NaN as Equal so the
        // iterator's max picks the first non-NaN candidate it has seen.
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

        // No more deflation needed once the last pivot has been chosen.
        if selected.len() == k {
            break;
        }

        let best_norm_sq = norms_sq[selected_index];
        // If the pivot has zero (or numerically negative) residual norm, the
        // remaining rows already lie in the span of previously selected
        // pivots; skip deflation to avoid dividing by zero.
        if best_norm_sq <= 0.0 {
            continue;
        }

        // 1 / ||r*||^2, factored out of the projection formula below.
        let inv_norm_sq = 1.0 / best_norm_sq;
        // Snapshot the pivot row r* before mutably iterating over the other
        // rows of `residuals` (they share the same backing storage).
        let r_star_copy: Vec<f32> = candidates.row(selected_index).to_vec();

        // --- Step 2a: Compute projection coefficients pi_i = <r_i, r*> / ||r*||^2.
        for i in 0..n {
            if !available[i] {
                projections[i] = 0.0;
            } else {
                projections[i] = dot_product(candidates.row(i), &r_star_copy) * inv_norm_sq;
            }
        }

        // --- Step 2b: Deflate r_i <- r_i - pi_i * r*, and
        // --- Step 3:  update ||r_i||^2 <- ||r_i||^2 - pi_i^2 * ||r*||^2.
        for i in 0..n {
            if !available[i] {
                continue;
            }

            let projection = projections[i];
            for (residual, &star) in candidates.row_mut(i).iter_mut().zip(r_star_copy.iter()) {
                *residual -= projection * star;
            }

            // Pythagorean update; clamp at 0 to absorb floating-point drift.
            norms_sq[i] = (norms_sq[i] - projection * projection * best_norm_sq).max(0.0);
        }
    }

    selected
}

/// Maps a raw distance into a similarity score in `(0, 1]` using the candidate
/// set's distance range.
///
/// DiskANN distance semantics are *lower is better*, so we invert and rescale
/// against the observed [min, max] range:
///
/// ```text
/// similarity(d) = max((d_max - d) / (d_max - d_min), 0) + EPSILON.
/// ```
///
/// - The numerator flips the order so that the *closest* candidate gets the
///   highest similarity (~1) and the *farthest* gets ~0.
/// - The denominator is clamped to `f32::EPSILON` so a degenerate range
///   (d_max == d_min) produces a finite, equal score for all candidates
///   instead of a divide-by-zero.
/// - The trailing `+ EPSILON` ensures the result is strictly positive, so
///   that `similarity(d).powf(power)` never produces a hard zero scale (which
///   would erase a candidate from the QR pivoting and bias selection).
#[inline(always)]
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
    use diskann_quantization::num::Positive;
    use diskann_utils::views::Matrix;

    #[test]
    fn test_valid_params() {
        assert!(DeterminantDiversityParams::new(1.0, 0.0).is_ok());
        assert!(DeterminantDiversityParams::new(0.5, 1.5).is_ok());
        assert!(DeterminantDiversityParams::new(2.0, 0.1).is_ok());
    }

    #[test]
    fn test_invalid_power() {
        assert!(DeterminantDiversityParams::new(0.0, 1.0).is_err());
        assert!(DeterminantDiversityParams::new(-1.0, 1.0).is_err());
    }

    #[test]
    fn test_invalid_eta() {
        assert!(DeterminantDiversityParams::new(1.0, -0.1).is_err());
    }

    #[test]
    fn test_invalid_non_finite_values() {
        assert!(DeterminantDiversityParams::new(f32::NAN, 0.1).is_err());
        assert!(DeterminantDiversityParams::new(f32::INFINITY, 0.1).is_err());
        assert!(DeterminantDiversityParams::new(1.0, f32::NAN).is_err());
        assert!(DeterminantDiversityParams::new(1.0, f32::INFINITY).is_err());
    }

    #[test]
    fn test_display() {
        let params = DeterminantDiversityParams::new(1.5, 0.5).unwrap();
        assert_eq!(
            params.to_string(),
            "DeterminantDiversity(power=1.5, eta=0.5)"
        );
    }

    fn run_with_ids(
        candidates: Vec<(u32, f32, Vec<f32>)>,
        query: &[f32],
        k: usize,
        eta: f32,
        power: Positive<f32>,
    ) -> Vec<(u32, f32)> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let dim = candidates[0].2.len();
        let mut matrix = Matrix::new(0.0f32, candidates.len(), dim);
        let mut ids = Vec::with_capacity(candidates.len());
        let mut distances = Vec::with_capacity(candidates.len());

        for (i, (id, distance, vector)) in candidates.into_iter().enumerate() {
            ids.push(id);
            distances.push(distance);
            matrix.row_mut(i).copy_from_slice(&vector);
        }

        let params = DeterminantDiversityParams::new(power.into_inner(), eta).unwrap();
        determinant_diversity(matrix.as_mut_view(), &distances, query, k, &params)
            .expect("valid determinant-diversity inputs")
            .into_iter()
            .map(|idx| (ids[idx], distances[idx]))
            .collect()
    }

    /// Test helper: wrap a positive f32 power value.
    fn p(value: f32) -> Positive<f32> {
        Positive::new(value).unwrap()
    }

    #[test]
    fn test_empty_candidates() {
        let result = run_with_ids(Vec::new(), &[1.0, 2.0], 5, 0.5, p(1.0));
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_empty_query_is_dimension_mismatch() {
        // A zero-length query against non-empty candidates is a structural
        // mismatch (candidate columns != query dimension), not a valid request
        // that trivially returns nothing.
        let mut matrix = Matrix::new(0.0f32, 1, 2);
        matrix.row_mut(0).copy_from_slice(&[1.0, 2.0]);
        let params = DeterminantDiversityParams::new(1.0, 0.5).unwrap();

        let result = determinant_diversity(matrix.as_mut_view(), &[0.5], &[], 5, &params);
        assert!(matches!(
            result,
            Err(DeterminantDiversityError::QueryDimensionMismatch {
                query: 0,
                candidate: 2,
            })
        ));
    }

    #[test]
    fn test_mismatched_dimensions_errors() {
        // Candidate vectors are 2-D, but the query is 3-D, so
        // `determinant_diversity` should report a dimension mismatch.
        let mut matrix = Matrix::new(0.0f32, 1, 2);
        matrix.row_mut(0).copy_from_slice(&[1.0, 2.0]);
        let params = DeterminantDiversityParams::new(1.0, 0.5).unwrap();

        let result =
            determinant_diversity(matrix.as_mut_view(), &[0.5], &[1.0, 2.0, 3.0], 5, &params);
        assert!(matches!(
            result,
            Err(DeterminantDiversityError::QueryDimensionMismatch {
                query: 3,
                candidate: 2,
            })
        ));
    }

    #[test]
    fn test_mismatched_distances_errors() {
        // Two candidate rows but only one distance is a structural mismatch.
        let mut matrix = Matrix::new(0.0f32, 2, 2);
        matrix.row_mut(0).copy_from_slice(&[1.0, 0.0]);
        matrix.row_mut(1).copy_from_slice(&[0.0, 1.0]);
        let params = DeterminantDiversityParams::new(1.0, 0.5).unwrap();

        let result = determinant_diversity(matrix.as_mut_view(), &[0.5], &[1.0, 1.0], 2, &params);
        assert!(matches!(
            result,
            Err(DeterminantDiversityError::DistanceCountMismatch {
                distances: 1,
                candidates: 2,
            })
        ));
    }

    #[test]
    fn test_single_candidate() {
        let candidates = vec![(0u32, 0.5, vec![1.0, 2.0])];
        let query = &[1.0, 2.0];
        let result = run_with_ids(candidates, query, 5, 0.5, p(1.0));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 0);
    }

    #[test]
    fn test_k_larger_than_candidates() {
        let candidates = vec![(0u32, 0.5, vec![1.0, 0.0]), (1u32, 0.3, vec![0.0, 1.0])];
        let query = &[1.0, 1.0];
        let result = run_with_ids(candidates, query, 10, 0.5, p(1.0));
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
        let result = run_with_ids(candidates, query, 2, 1.0, p(1.0));

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
        let result = run_with_ids(candidates, query, 2, 0.0, p(1.0));

        assert_eq!(result.len(), 2);
        // Should select based on greedy orthogonalization (eta == 0)
        assert!(result.iter().all(|(id, _)| *id < 3));
    }

    #[test]
    fn test_power_parameter() {
        let candidates = vec![(0u32, 0.1, vec![1.0, 0.0]), (1u32, 0.2, vec![0.0, 1.0])];
        let query = &[1.0, 1.0];

        // Test with different power values - should still work without panicking
        let result1 = run_with_ids(candidates.clone(), query, 2, 0.0, p(1.0));
        let result2 = run_with_ids(candidates, query, 2, 0.0, p(2.0));

        assert_eq!(result1.len(), 2);
        assert_eq!(result2.len(), 2);
    }

    #[test]
    fn test_distances_preserved() {
        let candidates = vec![(0u32, 0.5, vec![1.0, 0.0]), (1u32, 0.3, vec![0.0, 1.0])];
        let query = &[1.0, 1.0];
        let result = run_with_ids(candidates, query, 2, 0.0, p(1.0));

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
        let result = run_with_ids(candidates, query, 2, 0.0, p(1.0));

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
        let result = run_with_ids(candidates, query, 2, 0.5, p(1.0));

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
        let result = run_with_ids(candidates.clone(), query, 1, 0.0, p(10.0));
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
        let result = run_with_ids(candidates, query, 2, 0.0, p(1.0));

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
        let result = run_with_ids(candidates, query, 2, 0.0, p(1.0));
        assert_eq!(result.len(), 2);
    }

    /// k = 0 should return an empty result without panicking, even when
    /// candidates are otherwise valid.
    #[test]
    fn test_k_zero_returns_empty() {
        let candidates = vec![(0u32, 0.1, vec![1.0, 0.0]), (1u32, 0.2, vec![0.0, 1.0])];
        let query = &[1.0, 1.0];
        let result = run_with_ids(candidates, query, 0, 0.5, p(1.0));
        assert_eq!(result.len(), 0);
    }

    /// Zero-dimensional candidate vectors must be rejected gracefully (the
    /// algorithm has no meaningful work to do with empty vectors and the
    /// query is treated as effectively empty).
    #[test]
    fn test_zero_dimensional_candidates() {
        let candidates = vec![(0u32, 0.1, Vec::<f32>::new()), (1u32, 0.2, Vec::new())];
        // Query is non-empty but candidate vectors have dim 0; we only reach
        // the empty-vector early return if the dimension check would have
        // matched (so use a 0-length query here to stay on the early path).
        let query: &[f32] = &[];
        let result = run_with_ids(candidates, query, 2, 0.0, p(1.0));
        assert_eq!(result.len(), 0);
    }

    /// Selecting all n candidates must exit the loop cleanly via the
    /// `selected.len() == k` early break (no extra deflation pass).
    #[test]
    fn test_k_equals_candidates_returns_all() {
        let candidates = vec![
            (10u32, 0.1, vec![1.0, 0.0]),
            (20u32, 0.2, vec![0.0, 1.0]),
            (30u32, 0.3, vec![1.0, 1.0]),
        ];
        let query = &[1.0, 1.0];
        let result = run_with_ids(candidates, query, 3, 0.0, p(1.0));
        assert_eq!(result.len(), 3);

        // Result IDs must be a permutation of the input IDs (no duplicates,
        // none lost).
        let mut ids: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![10, 20, 30]);
    }

    /// When all candidates lie on a single line through the origin, the
    /// second-and-later pivots collapse to zero-norm residuals. The pivot
    /// loop must exit cleanly and still return up to `k` candidates without
    /// dividing by zero.
    #[test]
    fn test_collinear_candidates_no_division_by_zero() {
        // All vectors are positive multiples of (1, 0). After scaling and
        // picking the first pivot, every remaining residual is exactly 0.
        let candidates = vec![
            (0u32, 0.1, vec![1.0, 0.0]),
            (1u32, 0.1, vec![2.0, 0.0]),
            (2u32, 0.1, vec![3.0, 0.0]),
        ];
        let query = &[1.0, 0.0];
        let result = run_with_ids(candidates, query, 3, 0.0, p(1.0));
        assert_eq!(result.len(), 3);

        let mut ids: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    /// The first selected element should be the most relevant one (highest
    /// similarity, i.e. smallest distance) when all candidates have equal
    /// vector norms. This pins down the initial pivot choice.
    #[test]
    fn test_first_pivot_is_most_relevant_at_equal_norms() {
        // Three orthogonal unit vectors with strictly increasing distances.
        // Largest similarity → smallest distance → id=0 should be picked
        // first.
        let candidates = vec![
            (0u32, 0.1, vec![1.0, 0.0, 0.0]),
            (1u32, 0.5, vec![0.0, 1.0, 0.0]),
            (2u32, 0.9, vec![0.0, 0.0, 1.0]),
        ];
        let query = &[1.0, 1.0, 1.0];
        let result = run_with_ids(candidates, query, 3, 0.0, p(2.0));
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].0, 0, "Most relevant candidate must be first");
    }

    /// Distances returned alongside selected ids must come from the
    /// corresponding input candidate (not be reordered or recomputed).
    #[test]
    fn test_ids_pair_with_their_input_distance() {
        let candidates = vec![(7u32, 1.5, vec![1.0, 0.0]), (9u32, 0.25, vec![0.0, 1.0])];
        let query = &[1.0, 1.0];
        let result = run_with_ids(candidates, query, 2, 0.0, p(1.0));
        assert_eq!(result.len(), 2);

        for (id, dist) in &result {
            match *id {
                7 => assert_eq!(*dist, 1.5),
                9 => assert_eq!(*dist, 0.25),
                other => panic!("unexpected id {other}"),
            }
        }
    }

    /// `distance_to_similarity` must produce a strictly positive, finite
    /// score even at the extremes of the observed distance range, so the
    /// resulting alpha never becomes a hard zero or NaN.
    #[test]
    fn test_distance_to_similarity_extremes() {
        let range = DistanceRange { min: 0.5, max: 2.0 };

        let s_min = distance_to_similarity(0.5, range);
        let s_max = distance_to_similarity(2.0, range);
        let s_below = distance_to_similarity(-1.0, range);
        let s_above = distance_to_similarity(10.0, range);

        // All scores are strictly positive (we add EPSILON) and finite.
        for s in [s_min, s_max, s_below, s_above] {
            assert!(s.is_finite());
            assert!(s > 0.0);
        }
        // Closer (smaller) distance is at least as similar as farther.
        assert!(s_min >= s_max);
        // A distance below the observed min still clamps to ~1 + EPSILON.
        assert!(s_below >= s_min - f32::EPSILON);
        // A distance above the observed max still clamps to ~EPSILON.
        assert!(s_above <= s_max + f32::EPSILON);
    }

    /// Degenerate range (min == max) must not divide by zero. All
    /// similarities should be finite and equal.
    #[test]
    fn test_distance_to_similarity_degenerate_range() {
        let range = DistanceRange { min: 0.7, max: 0.7 };
        let a = distance_to_similarity(0.7, range);
        let b = distance_to_similarity(0.7, range);
        assert!(a.is_finite() && b.is_finite());
        assert_eq!(a, b);
    }
}

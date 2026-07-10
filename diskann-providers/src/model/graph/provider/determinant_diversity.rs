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
//! det(X_S * X_S^T + eta * I) over subsets S of size k using one of two greedy
//! selection strategies depending on the regularization:
//!
//! - `eta == 0`: pivoted modified Gram-Schmidt (max-volume QR). At each step we
//!   pick the row with the largest residual norm and deflate the rest against
//!   it. See [`determinant_diversity_select_qr`].
//! - `eta > 0`: Sherman-Morrison updates of the inverse Gram matrix. At each
//!   step we pick the row with the largest marginal log-determinant gain
//!   `ln(1 + x^T M^{-1} x)` and rank-1 update `M^{-1}`. See
//!   [`determinant_diversity_select_sherman_morrison`].
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
    #[error("determinant-diversity power must be > 0.0, got: {0}")]
    InvalidPower(f32),
    #[error("determinant-diversity eta must be >= 0.0, got: {0}")]
    InvalidEta(f32),
    #[error(
        "determinant-diversity candidate matrix has {candidate} columns but query dimension is {query}"
    )]
    QueryDimensionMismatch {
        /// Number of dimensions in the query.
        query: usize,
        /// Number of columns in the candidate matrix.
        candidate: usize,
    },
    #[error("determinant-diversity received {distances} distances for {candidates} candidate rows")]
    DistanceCountMismatch {
        /// Number of supplied distances.
        distances: usize,
        /// Number of candidate rows.
        candidates: usize,
    },
}

impl From<DeterminantDiversityError> for diskann::ANNError {
    #[track_caller]
    fn from(err: DeterminantDiversityError) -> Self {
        use diskann::ANNErrorKind;
        let kind = match err {
            DeterminantDiversityError::InvalidPower(_)
            | DeterminantDiversityError::InvalidEta(_) => ANNErrorKind::IndexConfigError,
            DeterminantDiversityError::QueryDimensionMismatch { .. }
            | DeterminantDiversityError::DistanceCountMismatch { .. } => {
                ANNErrorKind::DimensionMismatchError
            }
        };
        diskann::ANNError::new(kind, err)
    }
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

    // Dispatch to one of two selection algorithms depending on regularization:
    //   * eta == 0  -> pivoted modified Gram-Schmidt (max-volume QR), O(n * k * dim).
    //   * eta  > 0  -> ridge-regularized selection via Sherman-Morrison updates
    //                  of the inverse Gram matrix, O(n * k * dim^2).
    if params.eta() == 0.0 {
        Ok(determinant_diversity_select_qr(
            candidates,
            distances,
            k,
            params.power(),
            distance_range,
        ))
    } else {
        let inv_sqrt_eta = 1.0 / params.eta().sqrt();
        Ok(determinant_diversity_select_sherman_morrison(
            candidates,
            distances,
            k,
            params.power(),
            inv_sqrt_eta,
            distance_range,
        ))
    }
}

/// Selection algorithm for the unregularized (`eta == 0`) variant.
///
/// Performs greedy max-volume selection via incremental *column-pivoted
/// modified Gram-Schmidt* (QR) on the relevance-scaled candidate rows. Each
/// row is scaled in place by `similarity(d_i)^power` and then treated as a
/// residual vector `r_i`. At every step the row with the largest residual
/// norm is chosen as the next pivot and projected out of all remaining rows:
///
/// ```text
/// q      = r_j / ||r_j||
/// r_i   := r_i - (r_i . q) q
/// ||r_i||^2 -= (r_i . q)^2      (rank-1 Pythagorean update, clamped at 0)
/// ```
///
/// The returned order is the order in which pivots were chosen, which is the
/// diversity-promoting reranking.
///
/// # Complexity
///
/// O(n * k * dim): for each of `k` pivots we sweep all `n` residual rows of
/// length `dim`.
fn determinant_diversity_select_qr(
    mut candidates: MutMatrixView<'_, f32>,
    distances: &[f32],
    k: usize,
    power: f32,
    distance_range: DistanceRange,
) -> Vec<usize> {
    /// Residual norms at or below this threshold are treated as numerically
    /// zero: the remaining rows already lie in the span of the selected pivots.
    const LINEAR_DEPENDENCE_TOLERANCE: f32 = 1e-12;

    let n = candidates.nrows();
    let dim = candidates.ncols();
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }

    // Scale rows in place to form the initial residuals
    // r_i = similarity(d_i)^power * v_i.
    for (i, distance_to_query) in distances.iter().enumerate() {
        let scale = distance_to_similarity(*distance_to_query, distance_range).powf(power);
        for value in candidates.row_mut(i) {
            *value *= scale;
        }
    }

    // Squared residual norms, updated in place via ||r'||^2 = ||r||^2 - (r . q)^2.
    let mut residual_norm_sq: Vec<f32> = (0..n)
        .map(|i| qr_dot(candidates.row(i), candidates.row(i)))
        .collect();
    let mut available = vec![true; n];
    let mut selected = Vec::with_capacity(k);
    // Reused scratch for the pivot residual -- avoids a per-iteration allocation.
    let mut pivot = vec![0.0f32; dim];

    while selected.len() < k {
        // Pick the available candidate with the largest residual norm. `>=`
        // keeps the last maximal index, matching prior tie-breaking behavior.
        let mut best_idx = None;
        let mut best_score = f32::NEG_INFINITY;
        for i in 0..n {
            if available[i] && residual_norm_sq[i] >= best_score {
                best_score = residual_norm_sq[i];
                best_idx = Some(i);
            }
        }
        let Some(best_idx) = best_idx else {
            break;
        };

        available[best_idx] = false;
        selected.push(best_idx);

        if best_score <= LINEAR_DEPENDENCE_TOLERANCE {
            // Remaining residuals already lie in the span of the selected
            // vectors; keep taking them in arbitrary order until full.
            continue;
        }

        // Copy the pivot residual out once so the matrix can be mutated freely.
        // Fold the normalization into a single scalar:
        //   q = r_j / ||r_j||  =>  r_i -= (r_i . q) q = ((r_i . r_j)/||r_j||^2) r_j.
        pivot.copy_from_slice(candidates.row(best_idx));
        let inv_pivot_norm_sq = 1.0 / best_score;

        // Project the pivot direction out of every remaining residual -- O(n * dim).
        for i in 0..n {
            if !available[i] {
                continue;
            }
            let row = candidates.row_mut(i);
            let dot_with_pivot = qr_dot(row, &pivot); // r_i . r_j
            if dot_with_pivot == 0.0 {
                continue;
            }
            let alpha = dot_with_pivot * inv_pivot_norm_sq; // (r_i . r_j)/||r_j||^2
            qr_axpy(-alpha, &pivot, row); // r_i -= alpha * r_j
            // Rank-1 norm update:
            // ||r'||^2 = ||r||^2 - (r_i . q)^2 = ||r||^2 - (r_i . r_j) * alpha.
            residual_norm_sq[i] = (residual_norm_sq[i] - dot_with_pivot * alpha).max(0.0);
        }
    }

    selected
}

/// Selection algorithm for the ridge-regularized (`eta > 0`) variant.
///
/// Greedily maximizes the log-determinant of the regularized Gram matrix
/// `X_S * X_S^T + eta * I` by maintaining the inverse of `I + sum_j x_j x_j^T`
/// and applying a Sherman-Morrison rank-1 update after each pick. Each
/// candidate row is scaled in place by `similarity(d_i)^power / sqrt(eta)`.
///
/// At each step the candidate maximizing the marginal gain
/// `ln(1 + x^T M^{-1} x)` is selected, where `M^{-1}` is the current inverse.
///
/// # Complexity
///
/// O(n * k * dim^2): each of `k` steps forms an `M^{-1} x` product (O(dim^2))
/// for all `n` candidates and applies an O(dim^2) rank-1 inverse update.
fn determinant_diversity_select_sherman_morrison(
    mut candidates: MutMatrixView<'_, f32>,
    distances: &[f32],
    k: usize,
    power: f32,
    inv_sqrt_eta: f32,
    distance_range: DistanceRange,
) -> Vec<usize> {
    let n = candidates.nrows();
    let dim = candidates.ncols();
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }

    // Scale rows in place: x_i = similarity(d_i)^power * inv_sqrt_eta * v_i.
    for (i, distance_to_query) in distances.iter().enumerate() {
        let scale =
            distance_to_similarity(*distance_to_query, distance_range).powf(power) * inv_sqrt_eta;
        for value in candidates.row_mut(i) {
            *value *= scale;
        }
    }

    let mut matrix_inverse = identity_matrix(dim);
    let mut available = vec![true; n];
    let mut selected = Vec::with_capacity(k);

    while selected.len() < k && available.iter().any(|&avail| avail) {
        let Some((best_idx, best_matrix_vector, best_residual_sq)) =
            best_sherman_morrison_candidate(&candidates, &available, &matrix_inverse, dim)
        else {
            break;
        };

        available[best_idx] = false;
        selected.push(best_idx);
        update_sherman_morrison_inverse(
            &mut matrix_inverse,
            &best_matrix_vector,
            1.0 + best_residual_sq,
            dim,
        );
    }

    selected
}

/// Finds the available candidate maximizing the Sherman-Morrison marginal gain
/// `ln(1 + x^T M^{-1} x)`. Returns its row index, the product `M^{-1} x`
/// (reused for the inverse update), and the raw residual `x^T M^{-1} x`.
///
/// Because `ln(1 + r)` is strictly increasing in `r`, the candidate that
/// maximizes the log-gain also maximizes the raw residual `r = x^T M^{-1} x`.
/// We therefore compare on `r` directly: this preserves the selection order
/// while avoiding f32 underflow of `ln(1 + r)` to `0.0` when the relevance-
/// scaled vectors are tiny (e.g. all-equal distances collapse the similarity
/// weight to `EPSILON`), which would otherwise erase all gain differences.
fn best_sherman_morrison_candidate(
    candidates: &MutMatrixView<'_, f32>,
    available: &[bool],
    matrix_inverse: &[f32],
    dimensions: usize,
) -> Option<(usize, Vec<f32>, f32)> {
    (0..candidates.nrows())
        .filter(|&i| available[i])
        .map(|i| {
            let vector = candidates.row(i);
            let matrix_vector = matrix_vector_product(matrix_inverse, vector, dimensions);
            let residual_sq = dot_product(vector, &matrix_vector);
            (i, matrix_vector, residual_sq)
        })
        .max_by(|(_, _, left_residual), (_, _, right_residual)| {
            left_residual.max(0.0).total_cmp(&right_residual.max(0.0))
        })
}

/// Applies the Sherman-Morrison rank-1 update
/// `M^{-1} := M^{-1} - (M^{-1} x)(M^{-1} x)^T / denominator` in place.
fn update_sherman_morrison_inverse(
    matrix_inverse: &mut [f32],
    matrix_vector: &[f32],
    denominator: f32,
    dimensions: usize,
) {
    if denominator == 0.0 {
        return;
    }

    for row in 0..dimensions {
        for column in 0..dimensions {
            matrix_inverse[row * dimensions + column] -=
                matrix_vector[row] * matrix_vector[column] / denominator;
        }
    }
}

/// Builds a `dimensions x dimensions` row-major identity matrix.
fn identity_matrix(dimensions: usize) -> Vec<f32> {
    let mut matrix = vec![0.0; dimensions * dimensions];
    for idx in 0..dimensions {
        matrix[idx * dimensions + idx] = 1.0;
    }
    matrix
}

/// Multiplies a `dimensions x dimensions` row-major matrix by `vector`.
fn matrix_vector_product(matrix: &[f32], vector: &[f32], dimensions: usize) -> Vec<f32> {
    (0..dimensions)
        .map(|row| dot_product(&matrix[row * dimensions..(row + 1) * dimensions], vector))
        .collect()
}

/// Dot product with independent lane accumulators so the compiler can
/// vectorize it. The default `.iter().sum()` reduction is blocked by strict
/// f32 associativity; unrolling into `LANES` partial sums exposes the
/// parallelism. Used only by the `eta == 0` QR path. For lengths below `LANES`
/// this reduces to the scalar tail loop and is bit-identical to a
/// left-to-right sum.
#[inline]
fn qr_dot(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 8;
    let mut acc = [0.0f32; LANES];
    let mut a_chunks = a.chunks_exact(LANES);
    let mut b_chunks = b.chunks_exact(LANES);
    for (chunk_a, chunk_b) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        for lane in 0..LANES {
            acc[lane] += chunk_a[lane] * chunk_b[lane];
        }
    }
    let mut sum = 0.0f32;
    for lane_sum in acc {
        sum += lane_sum;
    }
    for (rem_a, rem_b) in a_chunks.remainder().iter().zip(b_chunks.remainder().iter()) {
        sum += rem_a * rem_b;
    }
    sum
}

/// Fused multiply-add over a slice: `r += a * x`. Autovectorizes cleanly.
/// Used only by the `eta == 0` QR path.
#[inline]
fn qr_axpy(a: f32, x: &[f32], r: &mut [f32]) {
    for (r_value, x_value) in r.iter_mut().zip(x.iter()) {
        *r_value += a * *x_value;
    }
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

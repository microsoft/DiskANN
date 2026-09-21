/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Hierarchical Agglomerative Clustering (HAC) for reducing late-interaction multi-vectors.
//!
//! Intended for roughly **250–2,000 vectors per document**, rather than the training
//! sets of 10k-100k vectors used with [`super::kmeans`]. Pair storage is quadratic,
//! and total distance arithmetic is O(n²d). Cached-minimum maintenance can still
//! inspect O(n³) costs in the worst case; prefer [`super::kmeans`] for large inputs.
//!
//! **Difference from classical Ward linkage:** The spherical Ward-style criterion implemented
//! here uses size-weighted cosine costs (found to be better in experiments) and normalizes the
//! centroid after every merge.

use std::{
    collections::TryReserveError,
    hash::{DefaultHasher, Hash, Hasher},
};

use diskann_utils::views::{Matrix, MatrixView};
use diskann_vector::{
    Norm,
    distance::{Distance, DistanceProvider, Metric},
    norm::FastL2NormSquared,
};
use thiserror::Error;

/// Error type for failure to reduce a multi-vector with [`spherical_ward`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SphericalWardError {
    #[error("num_clusters must be greater than zero")]
    ZeroClusters,
    #[error("invalid matrix shape: {rows} rows and {dimensions} dimensions over {len} elements")]
    InvalidShape {
        rows: usize,
        dimensions: usize,
        len: usize,
    },
    #[error("input row {row} has a zero, subnormal, or non-finite computed squared norm")]
    InvalidDirection { row: usize },
    #[error(
        "merging clusters {left} and {right} produces a zero, subnormal, or non-finite computed squared norm"
    )]
    UndefinedMerge { left: usize, right: usize },
    #[error("pair count overflows for {rows} input rows")]
    TooManyPairs { rows: usize },
    #[error("could not allocate {pairs} pair costs")]
    CostAllocation {
        pairs: usize,
        #[source]
        source: TryReserveError,
    },
}

/// Reduce the `n` rows of `data` to at most `num_clusters` unit-normalized centroids.
///
/// At each step, merge the globally cheapest pair of unit directions `ci`, `cj`:
///
/// ```text
/// cost(i, j) = ni * nj / (ni + nj) * max(0, 1 - dot(ci, cj))
/// c_merged   = normalize((ni * ci + nj * cj) / (ni + nj))
/// n_merged   = ni + nj
/// ```
/// where `ni` and `nj` are # original rows, not previous merges.
/// Note: The final centroid need not point along the mean of its original members.
///
/// Exact score ties use a priority derived from the stable pair IDs, and priorities
/// remain fixed throughout a call, independent of visitation order or parallel callers.
///
/// Throws a [`SphericalWardError`] on failure (see [`SphericalWardError`] for details).
/// Normalization rejects zero, subnormal, or non-finite computed `f32` squared norms
/// without rescaling, even when the original vector has a mathematical unit direction.
///
/// Note2: `spherical` in the name is not related to Spherical Quantizer; it just refers to
/// the unit-normalization of centroids.
///
/// # Example
///
/// ```
/// use diskann_quantization::algorithms::hac::spherical_ward;
/// use diskann_utils::views::MatrixView;
///
/// let data = MatrixView::try_from(&[3.0f32, 0.0, 2.0, 0.0, 0.0, 4.0][..], 3, 2)?;
/// let centers = spherical_ward(data, 2)?;
/// assert_eq!(centers.nrows(), 2);
/// assert_eq!(centers.row(0), &[1.0, 0.0]);
/// assert_eq!(centers.row(1), &[0.0, 1.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn spherical_ward(
    data: MatrixView<'_, f32>,
    num_clusters: usize,
) -> Result<Matrix<f32>, SphericalWardError> {
    // Initialize centroids by normalizing input vectors to unit norm
    let centers = copy_matrix_and_normalize(data, num_clusters)?;
    if centers.nrows() <= num_clusters {
        return Ok(centers);
    }

    let mut state = HacState::new(centers);
    let mut costs = HacCosts::new(state.centers.nrows())?;
    let mut nearest = vec![NO_PAIR; state.centers.nrows()];
    // Initialize nearest neighbor costs for each row, compute the initial best pair for each row.
    for (i, best) in nearest.iter_mut().enumerate() {
        for j in 0..i {
            let cost = state.cost(j, i);
            costs.values[costs.offsets[i] + j] = cost;
            let candidate = (cost, j, i);
            if precedes(candidate, *best) {
                *best = candidate;
            }
        }
    }

    // Main loop: merge clusters until the desired number of clusters is reached
    while state.active.len() > num_clusters {
        // Scanning row minima avoids an all-pairs heap and stale-entry growth.
        let mut best = NO_PAIR;
        for &i in &state.active {
            if precedes(nearest[i], best) {
                best = nearest[i];
            }
        }
        // Extract the best pair to merge next
        let (_, keep, drop) = best;
        state.merge(keep, drop)?;
        if state.active.len() == num_clusters {
            break;
        }

        // Update costs
        for &i in &state.active {
            if i != keep {
                let index = costs.index(i, keep);
                costs.values[index] = state.cost(i, keep);
            }
        }

        // Unchanged pairs retain both sizes and directions. A row whose old
        // minimum changed/disappeared must be rescanned. Otherwise only the new
        // keep edge can beat it. Row i stores only neighbors with IDs below i.
        for &i in &state.active {
            if i == keep || nearest[i].1 == keep || nearest[i].1 == drop {
                nearest[i] = costs.row_min(i, &state.sizes);
            } else if keep < i {
                let candidate = (costs.values[costs.index(i, keep)], keep, i);
                if precedes(candidate, nearest[i]) {
                    nearest[i] = candidate;
                }
            }
        }
    }
    Ok(state.finish())
}

/// Creates a "normalized" copy of the input matrix: each output row has unit norm.
fn copy_matrix_and_normalize(
    data: MatrixView<'_, f32>,
    num_clusters: usize,
) -> Result<Matrix<f32>, SphericalWardError> {
    if num_clusters == 0 {
        return Err(SphericalWardError::ZeroClusters);
    }
    if data.nrows().checked_mul(data.ncols()) != Some(data.as_slice().len()) {
        return Err(SphericalWardError::InvalidShape {
            rows: data.nrows(),
            dimensions: data.ncols(),
            len: data.as_slice().len(),
        });
    }
    let mut centers = data.to_owned();
    for i in 0..centers.nrows() {
        let row = centers.row_mut(i);
        normalize_row(row, SphericalWardError::InvalidDirection { row: i })?;
    }
    Ok(centers)
}

fn normalize_row(row: &mut [f32], error: SphericalWardError) -> Result<(), SphericalWardError> {
    let squared_norm: f32 = FastL2NormSquared.evaluate(&*row);
    if !squared_norm.is_normal() {
        return Err(error);
    }
    let norm = squared_norm.sqrt();
    for x in row {
        *x /= norm;
    }
    Ok(())
}

struct HacState {
    // Cluster centroids
    centers: Matrix<f32>,
    // Cluster sizes
    sizes: Vec<usize>,
    // Sorted stable row IDs; removed slots are never reused.
    active: Vec<usize>,
    merged: Vec<f32>,
    // N x (N - 1) distance matrix (strict lower triangle)
    distance: Distance<f32, f32>,
}

impl HacState {
    // Only constructed when merging is required; passthrough needs no scratch.
    fn new(centers: Matrix<f32>) -> Self {
        Self {
            sizes: vec![1; centers.nrows()],
            active: (0..centers.nrows()).collect(),
            merged: vec![0.0; centers.ncols()],
            distance: f32::distance_comparer(Metric::CosineNormalized, Some(centers.ncols())),
            centers,
        }
    }

    fn cost(&self, i: usize, j: usize) -> f32 {
        let ni = self.sizes[i] as f32;
        let nj = self.sizes[j] as f32;
        // Unit, finite directions ensure the distance is finite before clamping.
        let distance = self.distance.call(self.centers.row(i), self.centers.row(j));
        (ni * nj / (ni + nj)) * distance.max(0.0)
    }

    fn merge(&mut self, keep: usize, drop: usize) -> Result<(), SphericalWardError> {
        // Sizes partition the original rows, so their sum cannot overflow.
        let total = self.sizes[keep] + self.sizes[drop];
        let ni = self.sizes[keep] as f32;
        let nj = self.sizes[drop] as f32;
        for ((dst, &a), &b) in self
            .merged
            .iter_mut()
            .zip(self.centers.row(keep))
            .zip(self.centers.row(drop))
        {
            // The common divisor ni + nj cancels during normalization.
            *dst = ni * a + nj * b;
        }
        normalize_row(
            &mut self.merged,
            SphericalWardError::UndefinedMerge {
                left: keep,
                right: drop,
            },
        )?;
        self.centers.row_mut(keep).copy_from_slice(&self.merged);
        self.sizes[keep] = total;
        self.sizes[drop] = 0;
        self.active.retain(|&i| i != drop);
        Ok(())
    }

    fn finish(self) -> Matrix<f32> {
        let mut output = Matrix::new(0.0, self.active.len(), self.centers.ncols());
        for (row, &id) in self.active.iter().enumerate() {
            output.row_mut(row).copy_from_slice(self.centers.row(id));
        }
        output
    }
}

type Candidate = (f32, usize, usize);
const NO_PAIR: Candidate = (f32::INFINITY, usize::MAX, usize::MAX);

// Hash only exact ties, so ordinary distance comparisons pay no hashing cost.
// A fixed pair priority keeps cached and recomputed minima consistent. Do not
// use fresh random draws here: they would invalidate unchanged cached minima.
fn precedes(candidate: Candidate, best: Candidate) -> bool {
    candidate.0 < best.0
        || (candidate.0 == best.0 && pair_priority(candidate) < pair_priority(best))
}

fn pair_priority((_, left, right): Candidate) -> (u64, usize, usize) {
    let mut hasher = DefaultHasher::new();
    (left, right).hash(&mut hasher);
    // Stable IDs resolve the unlikely hash collision and keep the lower ID alive.
    (hasher.finish(), left, right)
}

/// Strict lower triangle. Row i holds pairs (j, i), j < i; stable slots avoid
/// moving matrix rows or columns after removal.
struct HacCosts {
    values: Vec<f32>,
    offsets: Vec<usize>,
}

impl HacCosts {
    fn new(rows: usize) -> Result<Self, SphericalWardError> {
        // Divide before multiplying so the intermediate cannot overflow when
        // the final triangular count would fit.
        let pairs = if rows.is_multiple_of(2) {
            (rows / 2).checked_mul(rows.saturating_sub(1))
        } else {
            rows.checked_mul(rows / 2)
        }
        .ok_or(SphericalWardError::TooManyPairs { rows })?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(pairs)
            .map_err(|source| SphericalWardError::CostAllocation { pairs, source })?;
        values.resize(pairs, 0.0);
        let mut offsets = Vec::with_capacity(rows);
        let mut offset = 0;
        for row in 0..rows {
            offsets.push(offset);
            offset += row;
        }
        Ok(Self { values, offsets })
    }

    fn index(&self, i: usize, j: usize) -> usize {
        debug_assert_ne!(i, j);
        self.offsets[i.max(j)] + i.min(j)
    }

    fn row_min(&self, row: usize, sizes: &[usize]) -> Candidate {
        let offset = self.offsets[row];
        let mut best = NO_PAIR;
        for (j, (&cost, &size)) in self.values[offset..offset + row]
            .iter()
            .zip(sizes)
            .enumerate()
        {
            if size > 0 {
                let candidate = (cost, j, row);
                if precedes(candidate, best) {
                    best = candidate;
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;

    /// Recompute every active pair at every merge, without a distance cache or
    /// minimum-selection state. Numeric primitives are shared deliberately so this
    /// checks cost-cache correctness even when the distance backend changes rounding.
    fn reference(
        data: MatrixView<'_, f32>,
        target: usize,
    ) -> Result<Matrix<f32>, SphericalWardError> {
        let centers = copy_matrix_and_normalize(data, target)?;
        if centers.nrows() <= target {
            return Ok(centers);
        }
        let mut state = HacState::new(centers);
        while state.active.len() > target {
            let mut best = NO_PAIR;
            for (pos, &i) in state.active.iter().enumerate() {
                for &j in &state.active[pos + 1..] {
                    let candidate = (state.cost(i, j), i, j);
                    if precedes(candidate, best) {
                        best = candidate;
                    }
                }
            }
            state.merge(best.1, best.2)?;
        }
        Ok(state.finish())
    }

    // random matrix generator
    fn synthetic(rows: usize, dim: usize, seed: u64) -> Matrix<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        Matrix::new(
            diskann_utils::views::Init(|| rng.random_range(-1.0f32..1.0)),
            rows,
            dim,
        )
    }

    // Assert that each row is a unit vector
    fn assert_unit(data: MatrixView<'_, f32>) {
        for i in 0..data.nrows() {
            let norm = data
                .row(i)
                .iter()
                .map(|&x| f64::from(x).powi(2))
                .sum::<f64>()
                .sqrt();
            // Allow f32 accumulation error, including the portable scalar norm.
            // The f64 measurement stays independent of the production kernel.
            assert!((norm - 1.0).abs() < 2e-6, "row={i}, norm={norm}");
        }
    }

    #[test]
    fn normalized_size_weighted_merge() {
        // Duplicate +x directions merge first; the final direction is (2,1), not
        // the unweighted (1,1) mean of the last two clusters.
        let input = [3.0, 0.0, 2.0, 0.0, 0.0, 4.0];
        let data = MatrixView::try_from(input.as_slice(), 3, 2).unwrap();
        let out = spherical_ward(data, 1).unwrap();
        assert_eq!((out.nrows(), out.ncols()), (1, 2));
        assert!((out.row(0)[0] - 2.0 / 5.0f32.sqrt()).abs() < 1e-7);
        assert!((out.row(0)[1] - 1.0 / 5.0f32.sqrt()).abs() < 1e-7);
        assert_eq!(input, [3.0, 0.0, 2.0, 0.0, 0.0, 4.0]);
        assert_unit(out.as_view());
    }

    #[test]
    fn empty_singleton_and_passthrough() {
        for dim in [0, 3, u32::MAX as usize, usize::MAX] {
            let data = MatrixView::try_from(&[][..], 0, dim).unwrap();
            assert_eq!(spherical_ward(data, 1).unwrap().as_view(), data);
        }
        for rows in [1, 3] {
            let input = synthetic(rows, 9, 37);
            for target in [rows, rows + 1, usize::MAX] {
                let out = spherical_ward(input.as_view(), target).unwrap();
                assert_eq!((out.nrows(), out.ncols()), (rows, 9));
                assert_unit(out.as_view());
            }
        }
    }

    #[test]
    fn rejects_invalid_and_undefined_directions() {
        for rows in [0, 2] {
            assert!(matches!(
                spherical_ward(synthetic(rows, 2, 3).as_view(), 0),
                Err(SphericalWardError::ZeroClusters)
            ));
        }
        for value in [
            0.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MAX,
            1e20,
            1e-20,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
        ] {
            let input = [1.0, 0.0, value, 0.0];
            let data = MatrixView::try_from(input.as_slice(), 2, 2).unwrap();
            for target in [1, 2, 3] {
                assert!(matches!(
                    spherical_ward(data, target),
                    Err(SphericalWardError::InvalidDirection { row: 1 })
                ));
            }
        }
        assert!(matches!(
            spherical_ward(MatrixView::try_from(&[][..], 1, 0).unwrap(), 1),
            Err(SphericalWardError::InvalidDirection { row: 0 })
        ));
        // After merging duplicate +x and -x pairs, the two size-2 clusters cancel.
        let input = [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
        let data = MatrixView::try_from(input.as_slice(), 4, 2).unwrap();
        assert!(spherical_ward(data, 2).is_ok());
        let err = spherical_ward(data, 1).unwrap_err();
        assert!(matches!(
            err,
            SphericalWardError::UndefinedMerge { left: 0, right: 1 }
        ));
        assert!(err.to_string().contains("merging clusters 0 and 1"));
    }

    #[test]
    fn finite_normal_squared_norms_normalize_safely() {
        for value in [1e19, 1.0, 1e-19] {
            let input = [value, value, -value, value];
            let data = MatrixView::try_from(input.as_slice(), 2, 2).unwrap();
            for target in [1, 2] {
                let out = spherical_ward(data, target).unwrap();
                assert_unit(out.as_view());
                if target == 1 {
                    assert_eq!(out.row(0), &[0.0, 1.0]);
                } else {
                    for (actual, expected) in out.as_slice().iter().zip([1.0, 1.0, -1.0, 1.0]) {
                        assert!((actual - expected * std::f32::consts::FRAC_1_SQRT_2).abs() < 2e-7);
                    }
                }
            }
        }
    }

    #[test]
    fn normalization_rejects_below_normal_squared_norm_boundary() {
        // Powers of two make the normal/subnormal boundary exact.
        let boundary = f32::MIN_POSITIVE.sqrt();
        let mut row = [boundary];
        normalize_row(&mut row, SphericalWardError::InvalidDirection { row: 7 }).unwrap();
        assert_eq!(row, [1.0]);

        let mut row = [boundary / 2.0];
        assert!(matches!(
            normalize_row(&mut row, SphericalWardError::InvalidDirection { row: 7 }),
            Err(SphericalWardError::InvalidDirection { row: 7 })
        ));
        assert_eq!(row, [boundary / 2.0]);
    }

    #[test]
    fn unequal_population_antipodes_have_a_defined_direction() {
        // antipode = opposite set of inputs (bad case)
        let input = [1.0, -1.0, 1.0, -1.0, 1.0];
        let data = MatrixView::try_from(input.as_slice(), 5, 1).unwrap();
        assert_eq!(spherical_ward(data, 1).unwrap().row(0), &[1.0]);
    }

    #[test]
    fn near_antipodal_merge_rejects_non_normal_squared_norm() {
        for residual in [1e-20, f32::MIN_POSITIVE, f32::from_bits(1)] {
            let input = [1.0, residual, -1.0, 0.0];
            let data = MatrixView::try_from(input.as_slice(), 2, 2).unwrap();
            // Input rows have unit norm, but the merged squared norm is
            // subnormal or underflows to zero. Do not attempt to rescale it.
            assert!(spherical_ward(data, 2).is_ok());
            assert!(matches!(
                spherical_ward(data, 1),
                Err(SphericalWardError::UndefinedMerge { left: 0, right: 1 })
            ));
        }

        // Cancellation with a normal squared norm remains valid.
        let input = [1.0, 1e-18, -1.0, 0.0];
        let data = MatrixView::try_from(input.as_slice(), 2, 2).unwrap();
        assert_eq!(spherical_ward(data, 1).unwrap().row(0), &[0.0, 1.0]);
    }

    #[test]
    fn cached_minima_match_full_recomputation_at_every_cut() {
        let dims: &[usize] = if cfg!(miri) {
            &[2, 9]
        } else {
            &[2, 7, 8, 9, 31, 128, 384, 512, 768]
        };
        let seeds = if cfg!(miri) { 1..3 } else { 1..20 };
        let rows = if cfg!(miri) { 5 } else { 20 };
        for &dim in dims {
            for seed in seeds.clone() {
                let input = synthetic(rows, dim, seed);
                for target in 1..=rows {
                    let out = spherical_ward(input.as_view(), target).unwrap();
                    assert_eq!(
                        out,
                        reference(input.as_view(), target).unwrap(),
                        "dim={dim}, seed={seed}, target={target}"
                    );
                    assert_unit(out.as_view());
                }
            }
        }
    }

    #[test]
    fn tie_priorities_never_change_score_ordering() {
        // Exercise finite costs including zero and its smallest positive neighbor.
        for score in [0.0f32, f32::MIN_POSITIVE, 0.5, 1.0, 1000.0] {
            let next = f32::from_bits(score.to_bits() + 1);
            for i in 0..32 {
                let a = (score, i, 32);
                let b = (next, 0, 33);
                assert!(precedes(a, b));
                assert!(!precedes(b, a));
                assert!(precedes(a, NO_PAIR));
                assert!(!precedes(a, a));
            }
        }
    }

    #[test]
    fn exact_ties_match_exhaustive_pair_priorities() {
        let rows = if cfg!(miri) { 8 } else { 128 };
        let costs = HacCosts::new(rows).unwrap();
        let mut first_neighbor = 0;
        for i in 1..rows {
            let expected = (0..i)
                .map(|j| (0.0, j, i))
                .min_by_key(|&pair| pair_priority(pair))
                .unwrap();
            assert_eq!(costs.row_min(i, &vec![1; rows]), expected);
            first_neighbor += usize::from(expected.1 == 0);
        }
        // Duplicate rows must not all choose row zero as their cached minimum.
        assert!(
            first_neighbor < rows - 1,
            "{first_neighbor} of {rows} rows chose zero"
        );
    }

    #[test]
    fn tied_distinct_directions_match_reference_at_every_cut() {
        // Unlike identical vectors, different tied merge choices are observable
        // in the output directions, exercising invalidation after tied merges.
        let rows = if cfg!(miri) { 5 } else { 24 };
        let mut data = Matrix::new(0.0, rows, rows);
        for i in 0..rows {
            data[(i, i)] = 1.0;
        }
        for target in 1..rows {
            let out = spherical_ward(data.as_view(), target).unwrap();
            assert_eq!(out, reference(data.as_view(), target).unwrap());
            assert_unit(out.as_view());
        }
    }

    #[test]
    fn duplicate_vectors_match_reference_at_every_cut() {
        let rows = if cfg!(miri) { 5 } else { 24 };
        for dim in [1, 3, 128] {
            let data = Matrix::new(1.0, rows, dim);
            for target in 1..rows {
                let out = spherical_ward(data.as_view(), target).unwrap();
                assert_eq!(out, reference(data.as_view(), target).unwrap());
                assert_unit(out.as_view());
            }
        }
    }

    #[test]
    fn packed_cost_storage_and_inactive_neighbors() {
        assert!(HacCosts::new(0).unwrap().values.is_empty());
        let mut costs = HacCosts::new(4).unwrap();
        assert_eq!(costs.values.len(), 6);
        assert_eq!(costs.offsets, [0, 0, 1, 3]);
        for i in 1..4 {
            for j in 0..i {
                assert_eq!(costs.index(i, j), costs.index(j, i));
            }
        }
        costs.values[3..].copy_from_slice(&[2.0, 1.0, 1.0]);
        assert_eq!(costs.row_min(0, &[1; 4]), NO_PAIR);
        let expected = [(1.0, 1, 3), (1.0, 2, 3)]
            .into_iter()
            .min_by_key(|&pair| pair_priority(pair))
            .unwrap();
        assert_eq!(costs.row_min(3, &[1; 4]), expected);
        assert_eq!(costs.row_min(3, &[1, 0, 1, 1]), (1.0, 2, 3));
        assert_eq!(costs.row_min(3, &[0, 0, 0, 1]), NO_PAIR);
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn parallel_documents_preserve_order_and_results() {
        use rayon::prelude::*;

        let count = if cfg!(miri) { 4 } else { 24 };
        let docs: Vec<_> = (1..count)
            .map(|i| synthetic(i + 2, i + 1, i as u64))
            .collect();
        let serial: Vec<_> = docs
            .iter()
            .map(|data| spherical_ward(data.as_view(), 2).unwrap())
            .collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let parallel: Vec<_> = pool.install(|| {
            docs.par_iter()
                .map(|data| spherical_ward(data.as_view(), 2).unwrap())
                .collect()
        });
        assert_eq!(parallel, serial);
    }
}

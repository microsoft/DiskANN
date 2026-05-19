/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{collections::HashSet, hash::Hash};

use diskann_utils::{
    strided::StridedView,
    views::{Matrix, MatrixView},
};
use thiserror::Error;

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RecallMetrics {
    /// The `k` value for `k-recall-at-n`.
    pub recall_k: usize,
    /// The `n` value for `k-recall-at-n`.
    pub recall_n: usize,
    /// The number of queries.
    pub num_queries: usize,
    /// The average recall across queries with non-empty groundtruth.
    /// Queries with zero groundtruth results are excluded from the average.
    pub average: f64,
}

#[derive(Debug, Error)]
pub enum ComputeRecallError {
    #[error("results matrix has {0} rows but ground truth has {1}")]
    RowsMismatch(usize, usize),
    #[error("distances matrix has {0} rows but ground truth has {1}")]
    DistanceRowsMismatch(usize, usize),
    #[error("recall k value {0} must be less than or equal to recall n {1}")]
    RecallKAndNError(usize, usize),
    #[error("number of results per query {0} must be at least the specified recall k {1}")]
    NotEnoughResults(usize, usize),
    #[error(
        "number of groundtruth values per query {0} must be at least the specified recall n {1}"
    )]
    NotEnoughGroundTruth(usize, usize),
    #[error("number of groundtruth distances {0} does not match groundtruth entries {1}")]
    NotEnoughGroundTruthDistances(usize, usize),
    #[error("number of groundtruth distances {0} for query {2} does not match groundtruth ids {1}")]
    GroundTruthDistanceMismatch(usize, usize, usize),
}

/// An abstraction over data-structures such as vector-of-vectors.
///
/// This is used in recall calculations such as [`knn`] and [`average_precision`] and
/// is purposely `dyn` compatible to reduce compilation overhead.
///
/// Implementations should ensure that if [`Self::nrows`] returns a value `N` that `row(i)`
/// returns a slice for all `i` in `0..N`. Accesses outside of that range are allowed to
/// panic.
///
/// The implementation [`Self::ncols`] is optional and can be implemented if the length of
/// all inner vectors is known and identical for all inner vectors, enabling faster error
/// paths during recall calculation.
///
/// If [`Self::ncols`] returns `Some(K)`, then the length of each slice returned from
/// [`Self::row`] should have a length equal to `K`. Note that unsafe code may **not** rely
/// on this behavior.
///
/// If [`Self::ncols`] returns `None` then no assumption can be made about the length of
/// the slices yielded from [`Self::row`].
pub trait Rows<T> {
    /// Return the number of subslices in `Self`.
    fn nrows(&self) -> usize;

    /// Return the `i`th subslice contained in self.
    fn row(&self, i: usize) -> &[T];

    /// Return `Some(K)` if all subslices are known to have length `K`. Otherwise, return
    /// `None`.
    ///
    /// # Provided Implementation
    ///
    /// The provided implementation returns `None`.
    fn ncols(&self) -> Option<usize> {
        None
    }
}

impl<T> Rows<T> for Matrix<T> {
    fn nrows(&self) -> usize {
        Matrix::<T>::nrows(self)
    }
    fn row(&self, i: usize) -> &[T] {
        Matrix::<T>::row(self, i)
    }
    fn ncols(&self) -> Option<usize> {
        Some(Matrix::<T>::ncols(self))
    }
}

impl<T> Rows<T> for MatrixView<'_, T> {
    fn nrows(&self) -> usize {
        MatrixView::<'_, T>::nrows(self)
    }
    fn row(&self, i: usize) -> &[T] {
        MatrixView::<'_, T>::row(self, i)
    }
    fn ncols(&self) -> Option<usize> {
        Some(MatrixView::<'_, T>::ncols(self))
    }
}

impl<T> Rows<T> for Vec<Vec<T>> {
    fn nrows(&self) -> usize {
        self.len()
    }
    fn row(&self, i: usize) -> &[T] {
        &self[i]
    }
}

/// Aggregate trait for required behavior when computing recall and average precision.
pub trait RecallCompatible: Eq + Hash + Clone + std::fmt::Debug {}

impl<T> RecallCompatible for T where T: Eq + Hash + Clone + std::fmt::Debug {}

/// Compute the K-nearest-neighbors recall value "K-recall-at-N".
///
/// For each entry in `groundtruth` and `results`, this computes the `recall_k` number of
/// elements of `groundtruth` that are present in the first `recall_n` entries of `results`.
///
/// If `groundtruth_distances` is provided, then it will be used to allow ties when matching
/// the last values of each entry of `results`. Values will be counted towards the recall if
/// they have the same distance as the last ordered candidate.
///
/// If `allow_insufficient_results`, an error will not be given if an entry in `results`
/// has fewer than `recall_n` candidates.
pub fn knn<T>(
    groundtruth: &dyn Rows<T>,
    groundtruth_distances: Option<StridedView<'_, f32>>,
    results: &dyn Rows<T>,
    recall_k: usize,
    recall_n: usize,
    allow_insufficient_results: bool,
) -> Result<RecallMetrics, ComputeRecallError>
where
    T: RecallCompatible,
{
    if recall_k > recall_n {
        return Err(ComputeRecallError::RecallKAndNError(recall_k, recall_n));
    }

    let nrows = results.nrows();
    if nrows != groundtruth.nrows() {
        return Err(ComputeRecallError::RowsMismatch(nrows, groundtruth.nrows()));
    }

    if let Some(cols) = results.ncols()
        && cols < recall_n
        && !allow_insufficient_results
    {
        return Err(ComputeRecallError::NotEnoughResults(cols, recall_n));
    }

    // Validate groundtruth size for fixed-size sources
    match groundtruth.ncols() {
        Some(ncols) if ncols < recall_k => {
            return Err(ComputeRecallError::NotEnoughGroundTruth(ncols, recall_k));
        }
        _ => {}
    }

    if let Some(distances) = groundtruth_distances {
        if nrows != distances.nrows() {
            return Err(ComputeRecallError::DistanceRowsMismatch(
                distances.nrows(),
                nrows,
            ));
        }

        match groundtruth.ncols() {
            Some(ncols) if distances.ncols() != ncols => {
                return Err(ComputeRecallError::NotEnoughGroundTruthDistances(
                    distances.ncols(),
                    ncols,
                ));
            }
            _ => {}
        }
    }

    // The actual recall computation for groundtruth
    let mut recall_values: Vec<f64> = Vec::new();
    let mut this_groundtruth = HashSet::new();
    let mut this_results = HashSet::new();

    for i in 0..results.nrows() {
        let result = results.row(i);
        if !allow_insufficient_results && result.len() < recall_n {
            return Err(ComputeRecallError::NotEnoughResults(result.len(), recall_n));
        }

        let gt_row = groundtruth.row(i);
        // `groundtruth` does not have to be fixed-size,
        // so we compute `recall_k` for this row based on its gt length
        let this_recall_k = gt_row.len().min(recall_k);

        if this_recall_k == 0 {
            continue;
        }

        // Populate the groundtruth using the top-k
        this_groundtruth.clear();
        this_groundtruth.extend(gt_row.iter().take(this_recall_k).cloned());

        // If we have distances, then continue to append distances as long as the distance
        // value is constant
        if let Some(distances) = groundtruth_distances {
            let distances_row = distances.row(i);
            if distances_row.len() != gt_row.len() {
                return Err(ComputeRecallError::GroundTruthDistanceMismatch(
                    distances_row.len(),
                    gt_row.len(),
                    i,
                ));
            }
            // we've already checked that `results` and `distances` have at lesat
            // `this_recall_k` entries, so it's safe to access `distances_row[this_recall_k - 1]`
            let last_distance = distances_row[this_recall_k - 1];
            for (d, g) in distances_row.iter().zip(gt_row.iter()).skip(this_recall_k) {
                if *d == last_distance {
                    this_groundtruth.insert(g.clone());
                } else {
                    break;
                }
            }
        }

        this_results.clear();
        this_results.extend(result.iter().take(recall_n).cloned());

        // Count the overlap
        let r = this_groundtruth
            .iter()
            .filter(|i| this_results.contains(i))
            .count()
            .min(this_recall_k);

        let recall = (r as f64) / (this_recall_k as f64);

        recall_values.push(recall);
    }

    // Compute the average recall
    let total: f64 = recall_values.iter().sum();
    let average = if recall_values.is_empty() {
        0.0
    } else {
        total / (recall_values.len() as f64)
    };

    Ok(RecallMetrics {
        recall_k,
        recall_n,
        num_queries: nrows,
        average,
    })
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AveragePrecisionMetrics {
    /// The number of queries.
    pub num_queries: usize,
    /// The average precision
    pub average_precision: f64,
}

#[derive(Debug, Error)]
pub enum AveragePrecisionError {
    #[error("results has {0} elements but ground truth has {1}")]
    EntriesMismatch(usize, usize),
}

/// Compute average precision of a range search result
pub fn average_precision<T>(
    results: &dyn Rows<T>,
    groundtruth: &dyn Rows<T>,
) -> Result<AveragePrecisionMetrics, AveragePrecisionError>
where
    T: RecallCompatible,
{
    let nrows = results.nrows();
    let groundtruth_nrows = groundtruth.nrows();
    if nrows != groundtruth_nrows {
        return Err(AveragePrecisionError::EntriesMismatch(
            nrows,
            groundtruth_nrows,
        ));
    }

    // The actual recall computation.
    let mut num_gt_results = 0;
    let mut num_reported_results = 0;

    let mut scratch = HashSet::new();
    let nrows = results.nrows();

    for i in 0..nrows {
        let result = results.row(i);
        let gt = groundtruth.row(i);

        scratch.clear();
        scratch.extend(result.iter().cloned());
        num_reported_results += gt.iter().filter(|i| scratch.contains(i)).count();
        num_gt_results += gt.len();
    }

    // Perform post-processing.
    let average_precision = (num_reported_results as f64) / (num_gt_results as f64);

    Ok(AveragePrecisionMetrics {
        average_precision,
        num_queries: nrows,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use diskann_utils::views::{self, Matrix};

    use super::*;

    fn test_rows_inner(rows: &dyn Rows<usize>, ncols: Option<usize>) {
        assert_eq!(rows.ncols(), ncols);
        assert_eq!(rows.nrows(), 3);
        assert_eq!(rows.row(0), &[0, 1, 2, 3]);
        assert_eq!(rows.row(1), &[4, 5, 6, 7]);
        assert_eq!(rows.row(2), &[8, 9, 10, 11]);
    }

    #[test]
    fn test_rows() {
        let mut i = 0usize;
        let mat = Matrix::new(
            views::Init(|| {
                let v = i;
                i += 1;
                v
            }),
            3,
            4,
        );

        test_rows_inner(&mat, Some(4));
        test_rows_inner(&(mat.as_view()), Some(4));

        let vecs = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11]];
        test_rows_inner(&vecs, None);
    }

    struct ExpectedRecall {
        recall_k: usize,
        recall_n: usize,
        // Recall for each component.
        components: Vec<usize>,
    }

    impl ExpectedRecall {
        fn new(recall_k: usize, recall_n: usize, components: Vec<usize>) -> Self {
            assert!(recall_k <= recall_n);
            components.iter().for_each(|x| {
                assert!(*x <= recall_k);
            });
            Self {
                recall_k,
                recall_n,
                components,
            }
        }

        fn compute_recall(&self) -> f64 {
            (self.components.iter().sum::<usize>() as f64)
                / ((self.components.len() * self.recall_k) as f64)
        }
    }

    #[test]
    fn test_happy_path() {
        let groundtruth = Matrix::try_from(
            vec![
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, // row 0
                5, 6, 7, 8, 9, 10, 11, 12, 13, 14, // row 1
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, // row 2
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, // row 3
            ]
            .into(),
            4,
            10,
        )
        .unwrap();

        let distances = Matrix::try_from(
            vec![
                0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 6.0, // row 0
                2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // row 1
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, // row 2
                0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 6.0, // row 3
            ]
            .into(),
            4,
            10,
        )
        .unwrap();

        // Shift row 0 by one and row 1 by two.
        let our_results = Matrix::try_from(
            vec![
                100, 0, 1, 2, 5, 6, // row 0
                100, 101, 7, 8, 9, 10, // row 1
                0, 1, 2, 3, 4, 5, // row 2
                0, 1, 2, 3, 4, 5, // row 3
            ]
            .into(),
            4,
            6,
        )
        .unwrap();

        //---------//
        // No Ties //
        //---------//
        let expected_no_ties = vec![
            // Equal `k` and `n`
            ExpectedRecall::new(1, 1, vec![0, 0, 1, 1]),
            ExpectedRecall::new(2, 2, vec![1, 0, 2, 2]),
            ExpectedRecall::new(3, 3, vec![2, 1, 3, 3]),
            ExpectedRecall::new(4, 4, vec![3, 2, 4, 4]),
            ExpectedRecall::new(5, 5, vec![3, 3, 5, 5]),
            ExpectedRecall::new(6, 6, vec![4, 4, 6, 6]),
            // Unequal `k` and `n`.
            ExpectedRecall::new(1, 2, vec![1, 0, 1, 1]),
            ExpectedRecall::new(1, 3, vec![1, 0, 1, 1]),
            ExpectedRecall::new(2, 3, vec![2, 0, 2, 2]),
            ExpectedRecall::new(3, 5, vec![3, 1, 3, 3]),
        ];
        let epsilon = 1e-6; // Define a small tolerance

        for (i, expected) in expected_no_ties.iter().enumerate() {
            assert_eq!(expected.components.len(), our_results.nrows());
            let recall = knn(
                &groundtruth,
                None,
                &our_results,
                expected.recall_k,
                expected.recall_n,
                false,
            )
            .unwrap();

            let left = recall.average;
            let right = expected.compute_recall();
            assert!(
                (left - right).abs() < epsilon,
                "left = {}, right = {} on input {}",
                left,
                right,
                i
            );

            assert_eq!(recall.num_queries, our_results.nrows());
            assert_eq!(recall.recall_k, expected.recall_k);
            assert_eq!(recall.recall_n, expected.recall_n);
        }

        //-----------//
        // With Ties //
        //-----------//
        let expected_with_ties = vec![
            // Equal `k` and `n`
            ExpectedRecall::new(1, 1, vec![0, 0, 1, 1]),
            ExpectedRecall::new(2, 2, vec![1, 0, 2, 2]),
            ExpectedRecall::new(3, 3, vec![2, 1, 3, 3]),
            ExpectedRecall::new(4, 4, vec![3, 2, 4, 4]),
            ExpectedRecall::new(5, 5, vec![4, 3, 5, 5]), // tie-breaker kicks in
            ExpectedRecall::new(6, 6, vec![5, 4, 6, 6]), // tie-breaker kicks in
            // Unequal `k` and `n`.
            ExpectedRecall::new(1, 2, vec![1, 0, 1, 1]),
            ExpectedRecall::new(1, 3, vec![1, 0, 1, 1]),
            ExpectedRecall::new(2, 3, vec![2, 1, 2, 2]),
            ExpectedRecall::new(4, 5, vec![4, 3, 4, 4]),
        ];

        for (i, expected) in expected_with_ties.iter().enumerate() {
            assert_eq!(expected.components.len(), our_results.nrows());
            let recall = knn(
                &groundtruth,
                Some(distances.as_view().into()),
                &our_results,
                expected.recall_k,
                expected.recall_n,
                false,
            )
            .unwrap();

            let left = recall.average;
            let right = expected.compute_recall();
            assert!(
                (left - right).abs() < epsilon,
                "left = {}, right = {} on input {}",
                left,
                right,
                i
            );

            assert_eq!(recall.num_queries, our_results.nrows());
            assert_eq!(recall.recall_k, expected.recall_k);
            assert_eq!(recall.recall_n, expected.recall_n);
        }
    }

    #[test]
    fn test_error_recall_k_and_n() {
        let groundtruth = Matrix::<u32>::new(0, 10, 10);
        let results = Matrix::<u32>::new(0, 10, 10);
        let err = knn(&groundtruth, None, &results, 11, 10, false).unwrap_err();
        assert!(matches!(err, ComputeRecallError::RecallKAndNError(..)));
    }

    #[test]
    fn test_error_rows_mismatch() {
        let groundtruth = Matrix::<u32>::new(0, 11, 10);
        let results = Matrix::<u32>::new(0, 10, 10);
        let err = knn(&groundtruth, None, &results, 10, 10, false).unwrap_err();
        assert!(matches!(err, ComputeRecallError::RowsMismatch(..)));
        let err_allow_insufficient_results =
            knn(&groundtruth, None, &results, 10, 10, true).unwrap_err();
        assert!(matches!(
            err_allow_insufficient_results,
            ComputeRecallError::RowsMismatch(..)
        ));
    }

    #[test]
    fn test_error_not_enough_results() {
        let groundtruth = Matrix::<u32>::new(0, 10, 10);
        let results = Matrix::<u32>::new(0, 10, 5);
        let err = knn(&groundtruth, None, &results, 5, 10, false).unwrap_err();
        assert!(matches!(err, ComputeRecallError::NotEnoughResults(..)));
        let _ = knn(&groundtruth, None, &results, 5, 10, true);
    }

    #[test]
    fn test_error_not_enough_results_dynamic() {
        let groundtruth = Matrix::<u32>::new(0, 10, 10);
        let results: Vec<_> = (0..10).map(|_| vec![0; 5]).collect();
        let err = knn(&groundtruth, None, &results, 5, 10, false).unwrap_err();
        assert!(matches!(err, ComputeRecallError::NotEnoughResults(..)));
        let _ = knn(&groundtruth, None, &results, 5, 10, true);
    }

    #[test]
    fn test_error_not_enough_groundtruth() {
        let groundtruth = Matrix::<u32>::new(0, 10, 5);
        let results = Matrix::<u32>::new(0, 10, 10);
        let err = knn(&groundtruth, None, &results, 10, 10, false).unwrap_err();
        assert!(matches!(err, ComputeRecallError::NotEnoughGroundTruth(..)));
        let err_allow_insufficient_results =
            knn(&groundtruth, None, &results, 10, 10, true).unwrap_err();
        assert!(matches!(
            err_allow_insufficient_results,
            ComputeRecallError::NotEnoughGroundTruth(..)
        ));
    }

    #[test]
    fn test_dynamic_groundtruth_valid() {
        let groundtruth: Vec<_> = (0..10).map(|_| vec![0u32; 5]).collect();
        let results = Matrix::<u32>::new(0, 10, 10);
        // Should succeed: each row uses this_recall_k = min(5, 10) = 5
        let recall = knn(&groundtruth, None, &results, 10, 10, false).unwrap();
        assert_eq!(recall.num_queries, 10);
    }

    #[test]
    fn test_dynamic_groundtruth_full_match() {
        let gt_row: Vec<u32> = (1..=5).collect();
        let groundtruth: Vec<_> = (0..10).map(|_| gt_row.clone()).collect();
        let mut results = Matrix::<u32>::new(0, 10, 10);
        for i in 0..10 {
            for (j, v) in (1u32..=10).enumerate() {
                results[(i, j)] = v;
            }
        }
        let recall = knn(&groundtruth, None, &results, 10, 10, false).unwrap();
        assert!((recall.average - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dynamic_groundtruth_partial_match() {
        // groundtruth: [1, 2, 3, 4, 5]; results contain [1, 2, 3, 6, 7, 8, 9, 10, 11, 12]
        let gt_row: Vec<u32> = (1..=5).collect();
        let groundtruth: Vec<_> = (0..10).map(|_| gt_row.clone()).collect();
        let mut results = Matrix::<u32>::new(0, 10, 10);
        let res_row: Vec<u32> = vec![1, 2, 3, 6, 7, 8, 9, 10, 11, 12];
        for i in 0..10 {
            for (j, &v) in res_row.iter().enumerate() {
                results[(i, j)] = v;
            }
        }
        let recall = knn(&groundtruth, None, &results, 10, 10, false).unwrap();
        assert!((recall.average - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_dynamic_groundtruth_mixed_zero_nonzero() {
        let mut groundtruth: Vec<Vec<u32>> = Vec::new();
        // First 5 rows: non-empty groundtruth
        for _ in 0..5 {
            groundtruth.push((1..=5).collect());
        }
        // Last 5 rows: empty groundtruth
        for _ in 0..5 {
            groundtruth.push(vec![]);
        }

        let mut results = Matrix::<u32>::new(0, 10, 10);
        for i in 0..10 {
            for (j, v) in (1u32..=10).enumerate() {
                results[(i, j)] = v;
            }
        }

        let recall = knn(&groundtruth, None, &results, 10, 10, false).unwrap();
        assert_eq!(recall.num_queries, 10);
        assert!((recall.average - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dynamic_groundtruth_all_zero() {
        let groundtruth: Vec<Vec<u32>> = (0..10).map(|_| vec![]).collect();
        let results = Matrix::<u32>::new(0, 10, 10);

        let recall = knn(&groundtruth, None, &results, 10, 10, false).unwrap();
        assert_eq!(recall.num_queries, 10);
        assert_eq!(recall.average, 0.0);
        assert!(!recall.average.is_nan());
        assert!(!recall.average.is_infinite());
    }

    #[test]
    fn test_error_distance_rows_mismatch() {
        let groundtruth = Matrix::<u32>::new(0, 10, 10);
        let distances = Matrix::<f32>::new(0.0, 9, 10);
        let results = Matrix::<u32>::new(0, 10, 10);
        let err = knn(
            &groundtruth,
            Some(distances.as_view().into()),
            &results,
            10,
            10,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, ComputeRecallError::DistanceRowsMismatch(..)));
    }

    #[test]
    fn test_error_distance_cols_mismatch() {
        let groundtruth = Matrix::<u32>::new(0, 10, 10);
        let distances = Matrix::<f32>::new(0.0, 10, 9);
        let results = Matrix::<u32>::new(0, 10, 10);
        let err = knn(
            &groundtruth,
            Some(distances.as_view().into()),
            &results,
            10,
            10,
            false,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ComputeRecallError::NotEnoughGroundTruthDistances(..)
        ));
    }

    #[test]
    fn test_error_result_distance_mismatch() {
        // groundtruth: 2 rows, first row has 3 elements, second row has 2
        let groundtruth: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![4, 5]];
        // distances: first row has 2 elements (should be 3), second row has 2 (matches)
        let distances: Vec<Vec<f32>> = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let distances = Matrix::try_from(
            distances.into_iter().flatten().collect::<Vec<_>>().into(),
            2,
            2,
        )
        .unwrap();
        // results: 2 rows, each with 3 elements
        let results: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let err = knn(
            &groundtruth,
            Some(distances.as_view().into()),
            &results,
            3,
            3,
            false,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ComputeRecallError::GroundTruthDistanceMismatch(2, 3, 0)
        ));
    }
}

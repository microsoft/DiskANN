/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{collections::HashSet, hash::Hash};

use diskann_utils::strided::StridedView;
use diskann_utils::views::MatrixView;

use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Serialize)]
pub(crate) struct RecallMetrics {
    /// The `k` value for `k-recall-at-n`.
    pub(crate) recall_k: usize,
    /// The `n` value for `k-recall-at-n`.
    pub(crate) recall_n: usize,
    /// The number of queries.
    pub(crate) num_queries: usize,
    /// The average recall across all queries.
    pub(crate) average: f64,
    /// The minimum observed recall (max possible value: `recall_n`).
    pub(crate) minimum: usize,
    /// The maximum observed recall (max possible value: `recall_k`).
    pub(crate) maximum: usize,
    /// Recall results by query
    pub(crate) by_query: Option<Vec<usize>>,
}

// impl RecallMetrics {
//     pub(crate) fn num_queries(&self) -> usize {
//         self.num_queries
//     }

//     pub(crate) fn average(&self) -> f64 {
//         self.average
//     }
// }

#[derive(Debug, Error)]
pub(crate) enum ComputeRecallError {
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
}

pub(crate) trait ComputeKnnRecall<T> {
    fn compute_knn_recall(
        &self,
        groundtruth_distances: Option<StridedView<'_, f32>>,
        results: StridedView<'_, T>,
        recall_k: usize,
        recall_n: usize,
        allow_insufficient_results: bool,
        enhanced_metrics: bool,
    ) -> Result<RecallMetrics, ComputeRecallError>;
}

impl<T> ComputeKnnRecall<T> for MatrixView<'_, T>
where
    T: Eq + Hash + Copy + std::fmt::Debug,
{
    fn compute_knn_recall(
        &self,
        groundtruth_distances: Option<StridedView<'_, f32>>,
        results: StridedView<'_, T>,
        recall_k: usize,
        recall_n: usize,
        allow_insufficient_results: bool,
        enhanced_metrics: bool,
    ) -> Result<RecallMetrics, ComputeRecallError> {
        compute_knn_recall(
            self,
            groundtruth_distances,
            results,
            recall_k,
            recall_n,
            allow_insufficient_results,
            enhanced_metrics,
        )
    }
}

impl<T> ComputeKnnRecall<T> for Vec<Vec<T>>
where
    T: Eq + Hash + Copy + std::fmt::Debug,
{
    fn compute_knn_recall(
        &self,
        groundtruth_distances: Option<StridedView<'_, f32>>,
        results: StridedView<'_, T>,
        recall_k: usize,
        recall_n: usize,
        allow_insufficient_results: bool,
        enhanced_metrics: bool,
    ) -> Result<RecallMetrics, ComputeRecallError> {
        compute_knn_recall(
            self,
            groundtruth_distances,
            results,
            recall_k,
            recall_n,
            allow_insufficient_results,
            enhanced_metrics,
        )
    }
}

pub(crate) trait KnnRecall {
    type Item;

    fn nrows(&self) -> usize;
    fn ncols(&self) -> Option<usize>;
    fn row(&self, i: usize) -> &[Self::Item];
}

impl<T> KnnRecall for MatrixView<'_, T> {
    type Item = T;

    fn nrows(&self) -> usize {
        MatrixView::<'_, T>::nrows(self)
    }
    fn ncols(&self) -> Option<usize> {
        Some(MatrixView::<'_, T>::ncols(self))
    }
    fn row(&self, i: usize) -> &[Self::Item] {
        MatrixView::<'_, T>::row(self, i)
    }
}

impl<T> KnnRecall for Vec<Vec<T>> {
    type Item = T;

    fn nrows(&self) -> usize {
        self.len()
    }
    fn ncols(&self) -> Option<usize> {
        None
    }
    fn row(&self, i: usize) -> &[Self::Item] {
        &self[i]
    }
}

impl<T> KnnRecall for StridedView<'_, T> {
    type Item = T;

    fn nrows(&self) -> usize {
        StridedView::<'_, T>::nrows(self)
    }
    fn ncols(&self) -> Option<usize> {
        Some(StridedView::<'_, T>::ncols(self))
    }
    fn row(&self, i: usize) -> &[Self::Item] {
        StridedView::<'_, T>::row(self, i)
    }
}

fn compute_knn_recall<T, K>(
    groundtruth: &K,
    groundtruth_distances: Option<StridedView<'_, f32>>,
    results: StridedView<'_, T>,
    recall_k: usize,
    recall_n: usize,
    allow_insufficient_results: bool,
    enhanced_metrics: bool,
) -> Result<RecallMetrics, ComputeRecallError>
where
    T: Eq + Hash + Copy + std::fmt::Debug,
    K: KnnRecall<Item = T>,
{
    if recall_k > recall_n {
        return Err(ComputeRecallError::RecallKAndNError(recall_k, recall_n));
    }

    let nrows = results.nrows();
    if nrows != groundtruth.nrows() {
        return Err(ComputeRecallError::RowsMismatch(nrows, groundtruth.nrows()));
    }

    if results.ncols() < recall_n && !allow_insufficient_results {
        return Err(ComputeRecallError::NotEnoughResults(
            results.ncols(),
            recall_n,
        ));
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

    // The actual recall computation for fixed-size groundtruth
    let mut recall_values: Vec<usize> = Vec::new();
    let mut this_groundtruth = HashSet::new();
    let mut this_results = HashSet::new();

    for (i, result) in results.row_iter().enumerate() {
        let gt_row = groundtruth.row(i);

        // Populate the groundtruth using the top-k
        this_groundtruth.clear();
        this_groundtruth.extend(gt_row.iter().copied().take(recall_k));

        // If we have distances, then continue to append distances as long as the distance
        // value is constant
        if let Some(distances) = groundtruth_distances {
            if recall_k > 0 {
                let distances_row = distances.row(i);
                if distances_row.len() > recall_k - 1 && gt_row.len() > recall_k - 1 {
                    let last_distance = distances_row[recall_k - 1];
                    for (d, g) in distances_row.iter().zip(gt_row.iter()).skip(recall_k) {
                        if *d == last_distance {
                            this_groundtruth.insert(*g);
                        } else {
                            break;
                        }
                    }
                }
            }
        }

        this_results.clear();
        this_results.extend(result.iter().copied().take(recall_n));

        // Count the overlap
        let r = this_groundtruth
            .iter()
            .filter(|i| this_results.contains(i))
            .count()
            .min(recall_k);

        recall_values.push(r);
    }

    // Perform post-processing
    let total: usize = recall_values.iter().sum();
    let minimum = recall_values.iter().min().unwrap_or(&0);
    let maximum = recall_values.iter().max().unwrap_or(&0);

    let div = if groundtruth.ncols().is_some() {
        recall_k * nrows
    } else {
        (0..groundtruth.nrows())
            .map(|i| groundtruth.row(i).len())
            .sum::<usize>()
            .max(1)
    };

    let average = (total as f64) / (div as f64);

    Ok(RecallMetrics {
        recall_k,
        recall_n,
        num_queries: nrows,
        average,
        minimum: *minimum,
        maximum: *maximum,
        by_query: if enhanced_metrics {
            Some(recall_values)
        } else {
            None
        },
    })
}

/// Compute `k-recall-at-n` for all valid combinations of values in `recall_k` and
/// `recall_n` (skipping those where `recall_k` exceeds `recall_n`).
///
/// Return all results. Currently, this is hardcoded to not allow insufficient results.
#[cfg(any(
    feature = "spherical-quantization",
    feature = "minmax-quantization",
    feature = "product-quantization"
))]
pub(crate) fn compute_multiple_recalls<T>(
    results: StridedView<'_, T>,
    groundtruth: StridedView<'_, T>,
    recall_k: &[usize],
    recall_n: &[usize],
    enhanced_metrics: bool,
) -> Result<Vec<RecallMetrics>, ComputeRecallError>
where
    T: Eq + Hash + Copy + std::fmt::Debug,
{
    let mut result = Vec::new();
    for k in recall_k {
        for n in recall_n {
            if k > n {
                continue;
            }

            result.push(compute_knn_recall(
                &groundtruth,
                None,
                results,
                *k,
                *n,
                false,
                enhanced_metrics,
            )?);
        }
    }
    Ok(result)
}

#[derive(Debug, Serialize)]
pub(crate) struct APMetrics {
    /// The number of queries.
    pub(crate) num_queries: usize,
    /// The average precision
    pub(crate) average_precision: f64,
}

#[derive(Debug, Error)]
pub(crate) enum ComputeAPError {
    #[error("results has {0} elements but ground truth has {1}")]
    EntriesMismatch(usize, usize),
}

/// Compute average precision of a range search result
pub(crate) fn compute_average_precision<T>(
    results: Vec<Vec<T>>,
    groundtruth: &[Vec<T>],
) -> Result<APMetrics, ComputeAPError>
where
    T: Eq + Hash + Copy + std::fmt::Debug,
{
    if results.len() != groundtruth.len() {
        return Err(ComputeAPError::EntriesMismatch(
            results.len(),
            groundtruth.len(),
        ));
    }

    // The actual recall computation.
    let mut num_gt_results = 0;
    let mut num_reported_results = 0;

    let mut scratch = HashSet::new();

    std::iter::zip(results.iter(), groundtruth.iter()).for_each(|(result, gt)| {
        scratch.clear();
        scratch.extend(result.iter().copied());
        num_reported_results += gt.iter().filter(|i| scratch.contains(i)).count();
        num_gt_results += gt.len();
    });

    // Perform post-processing.
    let average_precision = (num_reported_results as f64) / (num_gt_results as f64);

    Ok(APMetrics {
        average_precision,
        num_queries: results.len(),
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use diskann_utils::views::Matrix;

    use super::*;

    pub(crate) fn compute_knn_recall<G>(
        results: StridedView<'_, u32>,
        groundtruth: G, // StridedView
        groundtruth_distances: Option<StridedView<'_, f32>>,
        recall_k: usize,
        recall_n: usize,
        allow_insufficient_results: bool,
        enhanced_metrics: bool,
    ) -> Result<RecallMetrics, ComputeRecallError>
    where
        G: ComputeKnnRecall<u32> + KnnRecall<Item = u32> + Clone,
    {
        groundtruth.compute_knn_recall(
            groundtruth_distances,
            results,
            recall_k,
            recall_n,
            allow_insufficient_results,
            enhanced_metrics,
        )
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
            let recall = compute_knn_recall(
                our_results.as_view().into(),
                groundtruth.as_view(),
                None,
                expected.recall_k,
                expected.recall_n,
                false,
                true,
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
            assert_eq!(recall.minimum, *expected.components.iter().min().unwrap());
            assert_eq!(recall.maximum, *expected.components.iter().max().unwrap());
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
            let recall = compute_knn_recall(
                our_results.as_view().into(),
                groundtruth.as_view(),
                Some(distances.as_view().into()),
                expected.recall_k,
                expected.recall_n,
                false,
                true,
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
            assert_eq!(recall.minimum, *expected.components.iter().min().unwrap());
            assert_eq!(recall.maximum, *expected.components.iter().max().unwrap());
            assert_eq!(recall.by_query, Some(expected.components.clone()));
        }
    }

    #[test]
    fn test_errors() {
        // k greater than n
        {
            let groundtruth = Matrix::<u32>::new(0, 10, 10);
            let results = Matrix::<u32>::new(0, 10, 10);
            let err = compute_knn_recall(
                results.as_view().into(),
                groundtruth.as_view(),
                None,
                11,
                10,
                false,
                true,
            )
            .unwrap_err();
            assert!(matches!(err, ComputeRecallError::RecallKAndNError(..)));
        }

        // Unequal rows
        {
            let groundtruth = Matrix::<u32>::new(0, 11, 10);
            let results = Matrix::<u32>::new(0, 10, 10);
            let err = compute_knn_recall(
                results.as_view().into(),
                groundtruth.as_view(),
                None,
                10,
                10,
                false,
                true,
            )
            .unwrap_err();
            assert!(matches!(err, ComputeRecallError::RowsMismatch(..)));
            let err_allow_insufficient_results = compute_knn_recall(
                results.as_view().into(),
                groundtruth.as_view(),
                None,
                10,
                10,
                true,
                false,
            )
            .unwrap_err();
            assert!(matches!(
                err_allow_insufficient_results,
                ComputeRecallError::RowsMismatch(..)
            ));
        }

        // Not enough results
        {
            let groundtruth = Matrix::<u32>::new(0, 10, 10);
            let results = Matrix::<u32>::new(0, 10, 5);
            let err = compute_knn_recall(
                results.as_view().into(),
                groundtruth.as_view(),
                None,
                5,
                10,
                false,
                false,
            )
            .unwrap_err();
            assert!(matches!(err, ComputeRecallError::NotEnoughResults(..)));
            let _ = compute_knn_recall(
                results.as_view().into(),
                groundtruth.as_view(),
                None,
                5,
                10,
                true,
                false,
            );
        }

        // Not enough groundtruth
        {
            let groundtruth = Matrix::<u32>::new(0, 10, 5);
            let results = Matrix::<u32>::new(0, 10, 10);
            let err = compute_knn_recall(
                results.as_view().into(),
                groundtruth.as_view(),
                None,
                10,
                10,
                false,
                true,
            )
            .unwrap_err();
            assert!(matches!(err, ComputeRecallError::NotEnoughGroundTruth(..)));
            let err_allow_insufficient_results = compute_knn_recall(
                results.as_view().into(),
                groundtruth.as_view(),
                None,
                10,
                10,
                true,
                false,
            )
            .unwrap_err();
            assert!(matches!(
                err_allow_insufficient_results,
                ComputeRecallError::NotEnoughGroundTruth(..)
            ));
        }

        // Distance Row Mismatch
        {
            let groundtruth = Matrix::<u32>::new(0, 10, 10);
            let distances = Matrix::<f32>::new(0.0, 9, 10);
            let results = Matrix::<u32>::new(0, 10, 10);
            let err = compute_knn_recall(
                results.as_view().into(),
                groundtruth.as_view(),
                Some(distances.as_view().into()),
                10,
                10,
                false,
                true,
            )
            .unwrap_err();
            assert!(matches!(err, ComputeRecallError::DistanceRowsMismatch(..)));
        }

        // Distance Cols Mismatch
        {
            let groundtruth = Matrix::<u32>::new(0, 10, 10);
            let distances = Matrix::<f32>::new(0.0, 10, 9);
            let results = Matrix::<u32>::new(0, 10, 10);
            let err = compute_knn_recall(
                results.as_view().into(),
                groundtruth.as_view(),
                Some(distances.as_view().into()),
                10,
                10,
                false,
                true,
            )
            .unwrap_err();
            assert!(matches!(
                err,
                ComputeRecallError::NotEnoughGroundTruthDistances(..)
            ));
        }
    }
}

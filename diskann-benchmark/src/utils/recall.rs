/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_core as benchmark_core;
pub(crate) use benchmark_core::recall::knn;

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
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
}

impl From<&benchmark_core::recall::RecallMetrics> for RecallMetrics {
    fn from(m: &benchmark_core::recall::RecallMetrics) -> Self {
        Self {
            recall_k: m.recall_k,
            recall_n: m.recall_n,
            num_queries: m.num_queries,
            average: m.average,
            minimum: m.minimum,
            maximum: m.maximum,
        }
    }
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
    results: &dyn benchmark_core::recall::Rows<T>,
    groundtruth: &dyn benchmark_core::recall::Rows<T>,
    recall_k: &[usize],
    recall_n: &[usize],
) -> Result<Vec<RecallMetrics>, benchmark_core::recall::ComputeRecallError>
where
    T: benchmark_core::recall::RecallCompatible,
{
    let mut result = Vec::new();
    for k in recall_k {
        for n in recall_n {
            if k > n {
                continue;
            }

            let recall = benchmark_core::recall::knn(groundtruth, None, results, *k, *n, false)?;
            result.push((&recall).into());
        }
    }
    Ok(result)
}

#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub(crate) struct AveragePrecisionMetrics {
    /// The number of queries.
    pub(crate) num_queries: usize,
    /// The average precision
    pub(crate) average_precision: f64,
}

impl From<&benchmark_core::recall::AveragePrecisionMetrics> for AveragePrecisionMetrics {
    fn from(m: &benchmark_core::recall::AveragePrecisionMetrics) -> Self {
        Self {
            num_queries: m.num_queries,
            average_precision: m.average_precision,
        }
    }
}

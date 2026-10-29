/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::time::Duration;

/// Cumulative wall times and work counts for successful nonempty insertions.
/// Timings include instrumentation overhead. Initialization and failed inserts
/// are excluded. `split_points` counts point visits across splits, not unique IDs.
#[derive(Debug, Default, Clone, Copy)]
pub struct BuildMetrics {
    pub insert_time: Duration,
    /// Clustering only; excludes provider fill and list publication.
    pub split_time: Duration,
    /// Local two-means including initialization and ordered f64 accumulation.
    pub fit_time: Duration,
    /// Region-wide nearest-centroid assignment, including labels and counts.
    pub assignment_time: Duration,
    pub split_calls: usize,
    pub split_parents: usize,
    pub split_points: usize,
    /// Number of point-tile SGEMM calls, including tiles with numeric fallback.
    pub gemm_calls: usize,
    /// Coarse point jobs submitted to the caller pool for SGEMM assignment.
    /// This counts scheduled work, not the number of distinct worker threads.
    pub gemm_parallel_jobs: usize,
    /// Region assignments accepted from finite SGEMM scores.
    pub gemm_points: usize,
    /// Successful projection classifications across all Lloyd iterations.
    pub projection_points: usize,
    /// Classifications recomputed with direct L2 after non-finite arithmetic.
    pub numeric_fallbacks: usize,
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct ComputeStats {
    pub gemm_calls: usize,
    /// Coarse point-tile jobs dispatched to a caller's active Rayon pool.
    pub gemm_parallel_jobs: usize,
    /// Points whose region label came from finite GEMM scores.
    pub gemm_points: usize,
    /// Successful projection classifications across all Lloyd iterations.
    pub projection_points: usize,
    /// Non-finite classifications recomputed with direct L2.
    pub numeric_fallbacks: usize,
}

impl ComputeStats {
    pub fn merge(&mut self, other: Self) {
        self.gemm_calls += other.gemm_calls;
        self.gemm_parallel_jobs += other.gemm_parallel_jobs;
        self.gemm_points += other.gemm_points;
        self.projection_points += other.projection_points;
        self.numeric_fallbacks += other.numeric_fallbacks;
    }
}

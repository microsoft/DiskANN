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
}

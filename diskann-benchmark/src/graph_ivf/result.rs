/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::collections::BTreeMap;

use diskann_benchmark_runner::utils::percentiles::Percentiles;
use serde::Serialize;

#[derive(Serialize)]
pub(super) struct Results {
    pub implementation: &'static str,
    pub provider: &'static str,
    pub build_routing: &'static str,
    pub search_routing: &'static str,
    pub distance: &'static str,
    pub threads: usize,
    pub vectors: usize,
    pub dimensions: usize,
    pub builds: Vec<BuildResult>,
    pub searches: Vec<SearchResult>,
}

#[derive(Serialize)]
pub(super) struct BuildResult {
    pub repetition: usize,
    /// Same-binary diagnostic hash, computed outside timing.
    pub index_fingerprint: String,
    pub ivf_seconds: f64,
    pub two_means_seconds: f64,
    pub reassignment_seconds: f64,
    pub other_ivf_seconds: f64,
    pub centroid_graph_seconds: f64,
    pub total_seconds: f64,
    pub vectors_per_second: f64,
    pub inserted: usize,
    pub splits: usize,
    pub split_points: usize,
    /// Split parents per insertion -> number of insertions, including zero splits.
    pub split_parent_histogram: BTreeMap<usize, usize>,
    pub lists: usize,
    pub empty_lists: usize,
    pub max_list_size: usize,
}

#[derive(Serialize)]
pub(super) struct SearchResult {
    pub build_repetition: usize,
    pub repetition: usize,
    pub k: usize,
    pub requested_nprobe: usize,
    pub effective_nprobe: usize,
    pub queries: usize,
    pub total_seconds: f64,
    pub qps: f64,
    pub latency_ns: Percentiles<u64>,
    pub recall_at_k: f64,
    pub mean_points_scanned: f64,
    pub scanned_fraction: f64,
    pub mean_lists_scanned: f64,
    pub mean_centroid_comparisons: f64,
    pub mean_results_returned: f64,
    /// Ordered query result IDs, computed outside timing.
    pub result_fingerprint: String,
}

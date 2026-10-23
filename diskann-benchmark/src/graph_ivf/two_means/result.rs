/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::utils::percentiles::Percentiles;
use serde::Serialize;

use super::diagnostics;

#[derive(Serialize)]
pub(super) struct Results {
    pub schema_version: u32,
    pub threads: usize,
    pub timing_scope: &'static str,
    pub numerical_reference: &'static str,
    pub source: &'static str,
    pub workloads: Vec<WorkloadResults>,
}

#[derive(Serialize)]
pub(in crate::graph_ivf) struct WorkloadResults {
    pub(super) workload: Workload,
    pub(super) invocation_order: Vec<Invocation>,
    pub(super) strategies: Vec<StrategyResult>,
}

#[derive(Serialize)]
pub(super) struct Invocation {
    pub round: usize,
    pub position: usize,
    pub warmup: bool,
    pub strategy: &'static str,
}

#[derive(Serialize)]
pub(super) struct Workload {
    pub layout: &'static str,
    pub bucket_size: usize,
    pub dimensions: usize,
    pub iterations: usize,
}

#[derive(Serialize)]
pub(super) struct StrategyResult {
    pub strategy: &'static str,
    pub status: &'static str,
    pub samples: usize,
    pub total_ns: Option<Percentiles<u64>>,
    pub preparation_ns: Option<Percentiles<u64>>,
    /// Preserve measurement order for paired analysis. No slow samples are discarded.
    pub total_samples_ns: Vec<u64>,
    pub packed_bytes: Option<usize>,
    pub scratch_bytes: Option<usize>,
    pub diagnostics: Option<diagnostics::Comparison>,
    pub failures: Vec<String>,
}

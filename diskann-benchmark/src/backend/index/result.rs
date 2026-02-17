/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_core as benchmark_core;
use diskann_benchmark_runner::utils::{percentiles, MicroSeconds};
use serde::Serialize;

use crate::{
    backend::index::build::BuildStats,
    utils::{self, DisplayWrapper, MaybeDisplay},
};

//////////////////
// BuildResult  //
//////////////////
#[derive(Debug, Serialize)]
pub(super) struct BuildResult {
    pub(super) build: Option<BuildStats>,
    pub(super) search: AggregatedSearchResults,
}

impl BuildResult {
    pub(super) fn new_topk(build: Option<BuildStats>) -> Self {
        Self {
            build,
            search: AggregatedSearchResults::Topk(Vec::new()),
        }
    }

    pub(super) fn new_range(build: Option<BuildStats>) -> Self {
        Self {
            build,
            search: AggregatedSearchResults::Range(Vec::new()),
        }
    }

    pub(super) fn append(&mut self, search: AggregatedSearchResults) {
        self.search.append(search);
    }
}

impl std::fmt::Display for BuildResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ref build) = self.build {
            write!(f, "{}", build)?;
        }

        self.search.fmt(f)?;

        Ok(())
    }
}

//////////////////////
// QuantBuildResult //
//////////////////////

#[cfg(any(feature = "product-quantization", feature = "scalar-quantization",))]
#[derive(Debug, Serialize)]
pub(super) struct QuantBuildResult {
    pub(super) quant_training_time: MicroSeconds,
    pub(super) build: BuildResult,
}

#[cfg(any(feature = "product-quantization", feature = "scalar-quantization",))]
impl std::fmt::Display for QuantBuildResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Quant Training Time: {}s",
            self.quant_training_time.as_seconds()
        )?;
        self.build.fmt(f)
    }
}

///////////////////
// SearchResults //
///////////////////

#[derive(Debug, Serialize)]
pub(super) enum AggregatedSearchResults {
    Topk(Vec<SearchResults>),
    Range(Vec<RangeSearchResults>),
}

impl AggregatedSearchResults {
    pub(super) fn append(&mut self, search: AggregatedSearchResults) {
        match (self, search) {
            (Self::Topk(v), AggregatedSearchResults::Topk(s)) => v.extend(s),
            (Self::Range(v), AggregatedSearchResults::Range(s)) => v.extend(s),
            _ => panic!("Mismatched search result types"),
        }
    }
}

impl std::fmt::Display for AggregatedSearchResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Topk(v) => write!(f, "{}", DisplayWrapper(v.as_slice()))?,
            Self::Range(v) => write!(f, "{}", DisplayWrapper(v.as_slice()))?,
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub(super) struct SearchResults {
    pub(super) num_tasks: usize,
    pub(super) search_n: usize,
    pub(super) search_l: usize,
    pub(super) qps: Vec<f64>,
    pub(super) search_latencies: Vec<MicroSeconds>,
    pub(super) mean_latencies: Vec<f64>,
    pub(super) p90_latencies: Vec<MicroSeconds>,
    pub(super) p99_latencies: Vec<MicroSeconds>,
    pub(super) recall: utils::recall::RecallMetrics,
    pub(super) mean_cmps: f32,
    pub(super) mean_hops: f32,
}

impl SearchResults {
    pub fn new(summary: benchmark_core::search::graph::knn::Summary) -> Self {
        let benchmark_core::search::graph::knn::Summary {
            setup,
            parameters,
            end_to_end_latencies,
            mean_latencies,
            p90_latencies,
            p99_latencies,
            recall,
            mean_cmps,
            mean_hops,
            ..
        } = summary;

        let qps = end_to_end_latencies
            .iter()
            .map(|latency| recall.num_queries as f64 / latency.as_seconds())
            .collect();

        Self {
            num_tasks: setup.tasks.into(),
            search_n: parameters.k_value,
            search_l: parameters.l_value,
            qps,
            search_latencies: end_to_end_latencies,
            mean_latencies,
            p90_latencies,
            p99_latencies,
            recall: (&recall).into(),
            mean_cmps: mean_cmps as f32,
            mean_hops: mean_hops as f32,
        }
    }
}

fn format_search_results_table<F>(
    f: &mut std::fmt::Formatter<'_>,
    results: &[SearchResults],
    batch_formatter: Option<F>,
) -> std::fmt::Result
where
    F: Fn(usize) -> String,
{
    if results.is_empty() {
        return Ok(());
    }

    let has_batch = batch_formatter.is_some();
    let headers: &[&str] = if has_batch {
        &[
            "Batch",
            "Ls",
            "KNN",
            "Avg cmps",
            "Avg hops",
            "QPS - mean(max)",
            "Avg Latency",
            "p99 Latency",
            "Recall",
            "Threads",
        ]
    } else {
        &[
            "Ls",
            "KNN",
            "Avg cmps",
            "Avg hops",
            "QPS - mean(max)",
            "Avg Latency",
            "p99 Latency",
            "Recall",
            "Threads",
        ]
    };

    let mut table = diskann_benchmark_runner::utils::fmt::Table::new(headers, results.len());
    results.iter().enumerate().for_each(|(i, r)| {
        let mut row = table.row(i);
        let mut col_idx = 0;

        if let Some(ref formatter) = batch_formatter {
            row.insert(formatter(i), col_idx);
            col_idx += 1;
        }

        row.insert(r.search_l, col_idx);
        row.insert(r.search_n, col_idx + 1);
        row.insert(r.mean_cmps, col_idx + 2);
        row.insert(r.mean_hops, col_idx + 3);
        row.insert(
            format!(
                "{:.1} ({:.1})",
                MaybeDisplay(percentiles::mean(&r.qps), "missing"),
                MaybeDisplay(percentiles::max_f64(&r.qps), "missing"),
            ),
            col_idx + 4,
        );
        row.insert(
            format!(
                "{:.1}us ({:.1}us)",
                MaybeDisplay(percentiles::mean(&r.mean_latencies), "missing"),
                MaybeDisplay(percentiles::max_f64(&r.mean_latencies), "missing"),
            ),
            col_idx + 5,
        );
        row.insert(
            format!(
                "{:.1}us ({:.1})",
                MaybeDisplay(percentiles::mean(&r.p99_latencies), "missing"),
                MaybeDisplay(r.p99_latencies.iter().max(), "missing"),
            ),
            col_idx + 6,
        );
        row.insert(format!("{:3}", r.recall.average), col_idx + 7);
        row.insert(r.num_tasks, col_idx + 8);
    });

    write!(f, "{}", table)
}

impl std::fmt::Display for DisplayWrapper<'_, [SearchResults]> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        format_search_results_table(f, self, None::<fn(usize) -> String>)
    }
}

////////////////////////
// RangeSearchResults //
////////////////////////

#[derive(Debug, Serialize)]
pub(super) struct RangeSearchResults {
    pub(super) num_tasks: usize,
    pub(super) initial_l: usize,
    pub(super) qps: Vec<f64>,
    pub(super) search_latencies: Vec<MicroSeconds>,
    pub(super) mean_latencies: Vec<f64>,
    pub(super) p90_latencies: Vec<MicroSeconds>,
    pub(super) p99_latencies: Vec<MicroSeconds>,
    pub(super) average_precision: utils::recall::AveragePrecisionMetrics,
}

impl RangeSearchResults {
    pub fn new(summary: benchmark_core::search::graph::range::Summary) -> Self {
        let benchmark_core::search::graph::range::Summary {
            setup,
            parameters,
            end_to_end_latencies,
            mean_latencies,
            p90_latencies,
            p99_latencies,
            average_precision,
            ..
        } = summary;

        let qps = end_to_end_latencies
            .iter()
            .map(|latency| average_precision.num_queries as f64 / latency.as_seconds())
            .collect();

        Self {
            num_tasks: setup.tasks.into(),
            initial_l: parameters.starting_l_value,
            qps,
            search_latencies: end_to_end_latencies,
            mean_latencies,
            p90_latencies,
            p99_latencies,
            average_precision: (&average_precision).into(),
        }
    }
}

fn format_range_search_results_table<F>(
    f: &mut std::fmt::Formatter<'_>,
    results: &[RangeSearchResults],
    batch_formatter: Option<F>,
) -> std::fmt::Result
where
    F: Fn(usize) -> String,
{
    if results.is_empty() {
        return Ok(());
    }

    let has_batch = batch_formatter.is_some();
    let headers: &[_] = if has_batch {
        &[
            "Batch",
            "initial Ls",
            "QPS - mean(max)",
            "Avg Latency",
            "p99 Latency",
            "Average Precision",
            "Threads",
        ]
    } else {
        &[
            "initial Ls",
            "QPS - mean(max)",
            "Avg Latency",
            "p99 Latency",
            "Average Precision",
            "Threads",
        ]
    };

    let mut table = diskann_benchmark_runner::utils::fmt::Table::new(headers, results.len());
    results.iter().enumerate().for_each(|(i, r)| {
        let mut row = table.row(i);
        let mut col_idx = 0;

        if let Some(ref formatter) = batch_formatter {
            row.insert(formatter(i), col_idx);
            col_idx += 1;
        }

        row.insert(r.initial_l, col_idx);
        row.insert(
            format!(
                "{:.1} ({:.1})",
                MaybeDisplay(percentiles::mean(&r.qps), "missing"),
                MaybeDisplay(percentiles::max_f64(&r.qps), "missing"),
            ),
            col_idx + 1,
        );
        row.insert(
            format!(
                "{:.1}us ({:.1}us)",
                MaybeDisplay(percentiles::mean(&r.mean_latencies), "missing"),
                MaybeDisplay(percentiles::max_f64(&r.mean_latencies), "missing"),
            ),
            col_idx + 2,
        );
        row.insert(
            format!(
                "{:.1}us ({:.1})",
                MaybeDisplay(percentiles::mean(&r.p99_latencies), "missing"),
                MaybeDisplay(r.p99_latencies.iter().max(), "missing"),
            ),
            col_idx + 3,
        );
        row.insert(
            format!("{:3}", r.average_precision.average_precision),
            col_idx + 4,
        );
        row.insert(r.num_tasks, col_idx + 5);
    });

    write!(f, "{}", table)
}

impl std::fmt::Display for DisplayWrapper<'_, [RangeSearchResults]> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        format_range_search_results_table(f, self, None::<fn(usize) -> String>)
    }
}

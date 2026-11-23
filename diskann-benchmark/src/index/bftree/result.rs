/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_core as benchmark_core;
use diskann_benchmark_runner::utils::{percentiles, MicroSeconds};
use serde::Serialize;

use super::build::BuildStats;
use crate::utils::{self, DisplayWrapper, MaybeDisplay};

//////////////////
// BuildResult  //
//////////////////
#[derive(Debug, Serialize)]
pub(crate) struct BuildResult {
    pub(crate) build: Option<BuildStats>,
    pub(crate) search: AggregatedSearchResults,
}

impl BuildResult {
    pub(crate) fn new(build: Option<BuildStats>, search: AggregatedSearchResults) -> Self {
        Self { build, search }
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

///////////////////
// SearchResults //
///////////////////

#[derive(Debug, Serialize)]
pub(crate) enum AggregatedSearchResults {
    Topk(Vec<SearchResults>),
}

impl std::fmt::Display for AggregatedSearchResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Topk(v) => write!(f, "{}", DisplayWrapper(v.as_slice()))?,
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct SearchResults {
    pub(crate) num_tasks: usize,
    pub(crate) search_n: usize,
    pub(crate) search_l: usize,
    pub(crate) qps: Vec<f64>,
    pub(crate) search_latencies: Vec<MicroSeconds>,
    pub(crate) mean_latencies: Vec<f64>,
    pub(crate) p90_latencies: Vec<MicroSeconds>,
    pub(crate) p99_latencies: Vec<MicroSeconds>,
    pub(crate) recall: utils::recall::RecallMetrics,
    pub(crate) mean_cmps: f32,
    pub(crate) mean_hops: f32,
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
            search_n: parameters.k_value().get(),
            search_l: parameters.knn.l_value().get(),
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

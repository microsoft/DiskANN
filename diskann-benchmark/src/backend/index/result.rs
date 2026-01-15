/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann_benchmark_runner::utils::{percentiles, MicroSeconds};
use serde::Serialize;

use crate::{
    backend::index::{
        build::{MultiInsertBuildStats, SingleInsertBuildStats},
        update::{RunbookUpdateStageResults, UpdateResults},
    },
    inputs::async_::InplaceDeleteMethod,
    utils::{self, datafiles::UpdateOperationType, DisplayWrapper, MaybeDisplay},
};

//////////////////
// BuildStats   //
//////////////////
#[derive(Debug, Serialize)]
pub(super) enum BuildStats {
    SingleInsert(SingleInsertBuildStats),
    MultiInsert(MultiInsertBuildStats),
}

impl std::fmt::Display for BuildStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SingleInsert(s) => write!(f, "{}", s),
            Self::MultiInsert(s) => write!(f, "{}", s),
        }
    }
}

impl From<SingleInsertBuildStats> for BuildStats {
    fn from(value: SingleInsertBuildStats) -> Self {
        Self::SingleInsert(value)
    }
}

impl From<MultiInsertBuildStats> for BuildStats {
    fn from(value: MultiInsertBuildStats) -> Self {
        Self::MultiInsert(value)
    }
}

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

//////////////////////
// DynamicRunResult //
//////////////////////
#[derive(Debug, Serialize)]
pub(super) struct DynamicRunResult {
    pub(super) num_update_threads: NonZeroUsize,
    pub(super) insert_l: NonZeroUsize,
    pub(super) ip_delete_method: InplaceDeleteMethod,
    pub(super) ip_delete_num_to_replace: NonZeroUsize,
    pub(super) consolidate_threshold: f32,

    pub(super) update: Vec<RunbookUpdateStageResults>,
    pub(super) search: Vec<RunbookSearchStageResults>,
    // Aggregated statistics
    pub(super) mean_insert_latency: Option<f64>,
    pub(super) mean_insert_qps: Option<f64>,
    pub(super) mean_delete_latency: Option<f64>,
    pub(super) mean_delete_qps: Option<f64>,
    pub(super) mean_bg_latency: Option<f64>,
    pub(super) search_configs: Vec<(usize, usize)>, // (search_l, num_tasks) pairs
    pub(super) mean_search_latencies: Vec<f64>,     // corresponding to search_configs
    pub(super) mean_search_qps: Vec<f64>,           // corresponding to search_configs
    pub(super) mean_search_recall: Vec<f64>,        // corresponding to search_configs
}

impl DynamicRunResult {
    pub(super) fn new(
        num_update_threads: NonZeroUsize,
        insert_l: NonZeroUsize,
        ip_delete_method: InplaceDeleteMethod,
        ip_delete_num_to_replace: NonZeroUsize,
        consolidate_threshold: f32,
    ) -> Self {
        Self {
            num_update_threads,
            insert_l,
            ip_delete_method,
            ip_delete_num_to_replace,
            consolidate_threshold,
            update: Vec::new(),
            search: Vec::new(),
            mean_insert_latency: None,
            mean_insert_qps: None,
            mean_delete_latency: None,
            mean_delete_qps: None,
            mean_bg_latency: None,
            search_configs: Vec::new(),
            mean_search_latencies: Vec::new(),
            mean_search_qps: Vec::new(),
            mean_search_recall: Vec::new(),
        }
    }

    pub(super) fn append_search_results(&mut self, search: Vec<SearchResults>, stage_idx: i64) {
        for result in search {
            self.search.push(RunbookSearchStageResults {
                stage_idx,
                results: result,
            });
        }
    }

    pub(super) fn append_update_results(&mut self, update: UpdateResults, stage_idx: i64) {
        self.update.push(RunbookUpdateStageResults {
            stage_idx,
            results: update,
        });
    }

    pub fn aggregate_metrics(&mut self) {
        // Aggregate insert statistics
        let insert_stats: Vec<&UpdateResults> = self
            .update
            .iter()
            .map(|stage| &stage.results)
            .filter(|result| matches!(result.operation, UpdateOperationType::Insert))
            .collect();

        if !insert_stats.is_empty() {
            let total_qps: f64 = insert_stats.iter().map(|r| r.qps).sum();
            let total_latency: f64 = insert_stats.iter().map(|r| r.mean_latency).sum();
            self.mean_insert_qps = Some(total_qps / insert_stats.len() as f64);
            self.mean_insert_latency = Some(total_latency / insert_stats.len() as f64);
        }

        // Aggregate delete statistics
        let delete_stats: Vec<&UpdateResults> = self
            .update
            .iter()
            .map(|stage| &stage.results)
            .filter(|result| matches!(result.operation, UpdateOperationType::Delete))
            .collect();

        if !delete_stats.is_empty() {
            let total_qps: f64 = delete_stats.iter().map(|r| r.qps).sum();
            let total_latency: f64 = delete_stats.iter().map(|r| r.mean_latency).sum();
            let total_bg_latency: f64 = delete_stats.iter().map(|r| r.mean_bg_latency).sum();
            self.mean_delete_qps = Some(total_qps / delete_stats.len() as f64);
            self.mean_delete_latency = Some(total_latency / delete_stats.len() as f64);
            self.mean_bg_latency = Some(total_bg_latency / delete_stats.len() as f64);
        }

        // Aggregate search statistics grouped by configuration
        use std::collections::HashMap;
        let mut search_groups: HashMap<(usize, usize), Vec<&SearchResults>> = HashMap::new();

        for stage in &self.search {
            let key = (stage.results.search_l, stage.results.num_tasks);
            search_groups.entry(key).or_default().push(&stage.results);
        }

        // Sort configurations for consistent ordering
        let mut configs: Vec<_> = search_groups.keys().cloned().collect();
        configs.sort();

        self.search_configs = configs.clone();
        self.mean_search_latencies.clear();
        self.mean_search_qps.clear();
        self.mean_search_recall.clear();

        for (search_l, num_tasks) in configs {
            if let Some(results) = search_groups.get(&(search_l, num_tasks)) {
                // Average across all QPS values from all stagees with this config
                let all_qps: Vec<f64> = results.iter().flat_map(|r| &r.qps).cloned().collect();
                let all_mean_latencies: Vec<f64> = results
                    .iter()
                    .flat_map(|r| &r.mean_latencies)
                    .cloned()
                    .collect();
                let all_recalls: Vec<f64> = results.iter().map(|r| r.recall.average).collect();

                let mean_qps = if !all_qps.is_empty() {
                    all_qps.iter().sum::<f64>() / all_qps.len() as f64
                } else {
                    0.0
                };
                let mean_latency = if !all_mean_latencies.is_empty() {
                    all_mean_latencies.iter().sum::<f64>() / all_mean_latencies.len() as f64
                } else {
                    0.0
                };
                let mean_recall = if !all_recalls.is_empty() {
                    all_recalls.iter().sum::<f64>() / all_recalls.len() as f64
                } else {
                    0.0
                };

                self.mean_search_qps.push(mean_qps);
                self.mean_search_latencies.push(mean_latency);
                self.mean_search_recall.push(mean_recall);
            }
        }
    }
}

impl std::fmt::Display for DynamicRunResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Setup Parameters Table
        writeln!(f, "Setup Parameters:")?;
        let setup_headers = ["Parameter", "Value"];

        // Calculate the number of rows needed based on the delete method
        let num_rows = match self.ip_delete_method {
            InplaceDeleteMethod::VisitedAndTopK { .. } => 7, // Update Threads, Insert L, Method, K Value, L Value, Num to Replace
            InplaceDeleteMethod::TwoHopAndOneHop | InplaceDeleteMethod::OneHop => 5, // Update Threads, Insert L, Method, Num to Replace
        };

        let mut setup_table =
            diskann_benchmark_runner::utils::fmt::Table::new(setup_headers, num_rows);

        let mut row0 = setup_table.row(0);
        row0.insert("Update Threads", 0);
        row0.insert(self.num_update_threads, 1);

        let mut row1 = setup_table.row(1);
        row1.insert("Insert L", 0);
        row1.insert(self.insert_l, 1);

        let mut current_row = 2;
        match self.ip_delete_method {
            InplaceDeleteMethod::VisitedAndTopK { k_value, l_value } => {
                let mut row2 = setup_table.row(current_row);
                row2.insert("IP Delete Method", 0);
                row2.insert("VisitedAndTopK", 1);
                current_row += 1;

                let mut row3 = setup_table.row(current_row);
                row3.insert("IP Delete K Value", 0);
                row3.insert(k_value, 1);
                current_row += 1;

                let mut row4 = setup_table.row(current_row);
                row4.insert("IP Delete L Value", 0);
                row4.insert(l_value, 1);
                current_row += 1;
            }
            InplaceDeleteMethod::TwoHopAndOneHop => {
                let mut row2 = setup_table.row(current_row);
                row2.insert("IP Delete Method", 0);
                row2.insert("TwoHopAndOneHop", 1);
                current_row += 1;
            }
            InplaceDeleteMethod::OneHop => {
                let mut row2 = setup_table.row(current_row);
                row2.insert("IP Delete Method", 0);
                row2.insert("OneHop", 1);
                current_row += 1;
            }
        }

        let mut num_to_replace_row = setup_table.row(current_row);
        num_to_replace_row.insert("IP Delete Num to Replace", 0);
        num_to_replace_row.insert(self.ip_delete_num_to_replace, 1);
        current_row += 1;
        let mut consolidate_threshold_row = setup_table.row(current_row);
        consolidate_threshold_row.insert("Consolidate Threshold", 0);
        consolidate_threshold_row.insert(self.consolidate_threshold, 1);

        setup_table.fmt(f)?;
        writeln!(f)?;

        // Aggregated Statistics Table
        writeln!(f, "Aggregated Statistics:")?;
        let stats_headers = ["Metric", "Value"];
        let mut stats_table = diskann_benchmark_runner::utils::fmt::Table::new(stats_headers, 5);
        let mut row0 = stats_table.row(0);
        row0.insert("Mean Insert Latency (us)", 0);
        row0.insert(
            self.mean_insert_latency
                .map_or("N/A".to_string(), |v| format!("{:.2}", v)),
            1,
        );

        let mut row1 = stats_table.row(1);
        row1.insert("Mean Insert QPS", 0);
        row1.insert(
            self.mean_insert_qps
                .map_or("N/A".to_string(), |v| format!("{:.2}", v)),
            1,
        );

        let mut row2 = stats_table.row(2);
        row2.insert("Mean Delete Latency (us)", 0);
        row2.insert(
            self.mean_delete_latency
                .map_or("N/A".to_string(), |v| format!("{:.2}", v)),
            1,
        );

        let mut row3 = stats_table.row(3);
        row3.insert("Mean Delete QPS", 0);
        row3.insert(
            self.mean_delete_qps
                .map_or("N/A".to_string(), |v| format!("{:.2}", v)),
            1,
        );

        let mut row4 = stats_table.row(4);
        row4.insert("Mean BG Clean Latency (us)", 0);
        row4.insert(
            self.mean_bg_latency
                .map_or("N/A".to_string(), |v| format!("{:.2}", v)),
            1,
        );

        stats_table.fmt(f)?;
        writeln!(f)?;

        // Search Statistics by Configuration
        if !self.search_configs.is_empty() {
            writeln!(f, "Search Statistics by Configuration:")?;
            let search_headers = [
                "Search L",
                "Num Tasks",
                "Mean QPS",
                "Mean Lat(us)",
                "Mean Recall",
            ];
            let mut search_table = diskann_benchmark_runner::utils::fmt::Table::new(
                search_headers,
                self.search_configs.len(),
            );

            for (i, (search_l, num_tasks)) in self.search_configs.iter().enumerate() {
                let qps = self.mean_search_qps.get(i).unwrap_or(&0.0);
                let latency = self.mean_search_latencies.get(i).unwrap_or(&0.0);
                let recall = self.mean_search_recall.get(i).unwrap_or(&0.0);

                let mut row = search_table.row(i);
                row.insert(*search_l, 0);
                row.insert(*num_tasks, 1);
                row.insert(format!("{:.2}", qps), 2);
                row.insert(format!("{:.2}", latency), 3);
                row.insert(format!("{:.4}", recall), 4);
            }
            writeln!(f, "{}", search_table)?;
        }

        // Stage-by-stage results
        if !self.update.is_empty() {
            writeln!(f, "Update Stage-by-stage Results:")?;
            write!(f, "{}", DisplayWrapper(self.update.as_slice()))?;
            writeln!(f)?;
        }
        if !self.search.is_empty() {
            writeln!(f, "Search Stage-by-stage Results:")?;
            write!(f, "{}", DisplayWrapper(self.search.as_slice()))?;
        }
        Ok(())
    }
}

//////////////////////////////
// RunbookSearchStageResults //
//////////////////////////////
#[derive(Debug, Serialize)]
pub(super) struct RunbookSearchStageResults {
    pub(super) stage_idx: i64,
    pub(super) results: SearchResults,
}

////////////////////////
// SearchResultsSetup //
////////////////////////
pub struct SearchResultsSetup {
    pub num_tasks: usize,
    pub search_n: usize,
    pub search_l: usize,
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
            Self::Topk(ref v) => write!(f, "{}", DisplayWrapper(v.as_slice()))?,
            Self::Range(ref v) => write!(f, "{}", DisplayWrapper(v.as_slice()))?,
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
    pub fn new(
        setup: SearchResultsSetup,
        search_latencies: Vec<MicroSeconds>,
        query_latencies: Vec<Vec<MicroSeconds>>,
        cmps: Vec<Vec<u32>>,
        hops: Vec<Vec<u32>>,
        recall: utils::recall::RecallMetrics,
    ) -> Result<Self, percentiles::CannotBeEmpty> {
        // Compute QPS from `search_latencies`.
        let num_queries = recall.num_queries as f64;
        let qps = search_latencies
            .iter()
            .map(|l| num_queries / l.as_seconds())
            .collect();

        let mut mean_latencies = Vec::with_capacity(query_latencies.len());
        let mut p90_latencies = Vec::with_capacity(query_latencies.len());
        let mut p99_latencies = Vec::with_capacity(query_latencies.len());

        query_latencies.into_iter().try_for_each(
            |mut query_latencies| -> Result<(), percentiles::CannotBeEmpty> {
                let percentiles::Percentiles { mean, p90, p99, .. } =
                    percentiles::compute_percentiles(&mut query_latencies)?;
                mean_latencies.push(mean);
                p90_latencies.push(p90);
                p99_latencies.push(p99);
                Ok(())
            },
        )?;

        let avg: fn(&[u32]) -> f32 = |v| {
            if v.is_empty() {
                0.0
            } else {
                v.iter().map(|x| *x as f32).sum::<f32>() / v.len() as f32
            }
        };
        let mean_cmps = cmps.iter().map(|v| avg(v)).sum::<f32>() / cmps.len() as f32;
        let mean_hops = hops.iter().map(|v| avg(v)).sum::<f32>() / hops.len() as f32;

        let SearchResultsSetup {
            num_tasks,
            search_n,
            search_l,
        } = setup;

        Ok(Self {
            num_tasks,
            search_n,
            search_l,
            qps,
            search_latencies,
            mean_latencies,
            p90_latencies,
            p99_latencies,
            recall,
            mean_cmps,
            mean_hops,
        })
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

impl std::fmt::Display for DisplayWrapper<'_, [RunbookSearchStageResults]> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            return Ok(());
        }

        let headers = [
            "Batch",
            "Ls",
            "KNN",
            "QPS - mean(max)",
            "Avg Latency",
            "p99 Latency",
            "Recall",
            "Threads",
        ];

        let mut table = diskann_benchmark_runner::utils::fmt::Table::new(headers, self.len());
        self.iter().enumerate().for_each(|(row, isr)| {
            let mut row = table.row(row);
            let r = &isr.results;
            row.insert(isr.stage_idx, 0);
            row.insert(r.search_l, 1);
            row.insert(r.search_n, 2);
            row.insert(
                format!(
                    "{:.1} ({:.1})",
                    MaybeDisplay(percentiles::mean(&r.qps), "missing"),
                    MaybeDisplay(percentiles::max_f64(&r.qps), "missing"),
                ),
                3,
            );
            row.insert(
                format!(
                    "{:.1}us ({:.1}us)",
                    MaybeDisplay(percentiles::mean(&r.mean_latencies), "missing"),
                    MaybeDisplay(percentiles::max_f64(&r.mean_latencies), "missing"),
                ),
                4,
            );
            row.insert(
                format!(
                    "{:.1}us ({:.1})",
                    MaybeDisplay(percentiles::mean(&r.p99_latencies), "missing"),
                    MaybeDisplay(r.p99_latencies.iter().max(), "missing"),
                ),
                5,
            );
            row.insert(format!("{:3}", r.recall.average), 6);
            row.insert(r.num_tasks, 7);
        });

        table.fmt(f)
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
    pub(super) average_precision: utils::recall::APMetrics,
}

pub struct RangeSearchResultsSetup {
    pub num_tasks: usize,
    pub initial_l: usize,
}

impl RangeSearchResults {
    pub fn new(
        setup: RangeSearchResultsSetup,
        search_latencies: Vec<MicroSeconds>,
        query_latencies: Vec<Vec<MicroSeconds>>,
        average_precision: utils::recall::APMetrics,
    ) -> Result<Self, percentiles::CannotBeEmpty> {
        // Compute QPS from `search_latencies`.
        let num_queries = average_precision.num_queries as f64;
        let qps = search_latencies
            .iter()
            .map(|l| num_queries / l.as_seconds())
            .collect();

        let mut mean_latencies = Vec::with_capacity(query_latencies.len());
        let mut p90_latencies = Vec::with_capacity(query_latencies.len());
        let mut p99_latencies = Vec::with_capacity(query_latencies.len());

        query_latencies.into_iter().try_for_each(
            |mut query_latencies| -> Result<(), percentiles::CannotBeEmpty> {
                let percentiles::Percentiles { mean, p90, p99, .. } =
                    percentiles::compute_percentiles(&mut query_latencies)?;
                mean_latencies.push(mean);
                p90_latencies.push(p90);
                p99_latencies.push(p99);
                Ok(())
            },
        )?;

        let RangeSearchResultsSetup {
            num_tasks,
            initial_l,
        } = setup;

        Ok(Self {
            num_tasks,
            initial_l,
            qps,
            search_latencies,
            mean_latencies,
            p90_latencies,
            p99_latencies,
            average_precision,
        })
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
        row.insert(r.num_tasks, col_idx + 4);
    });

    write!(f, "{}", table)
}

impl std::fmt::Display for DisplayWrapper<'_, [RangeSearchResults]> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        format_range_search_results_table(f, self, None::<fn(usize) -> String>)
    }
}

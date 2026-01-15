/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use anyhow::Result;
use diskann_benchmark_runner::{
    dispatcher::{self, DispatchRule, FailureScore, MatchScore},
    output::Output,
    registry::Benchmarks,
    utils::{percentiles, MicroSeconds},
    Any, Checkpoint,
};
use diskann_label_filter::{
    kv_index::GenericIndex,
    read_and_parse_queries, read_baselabels,
    stores::bftree_store::BfTreeStore,
    traits::{
        inverted_index_trait::InvertedIndexProvider, key_codec::DefaultKeyCodec,
        posting_list_trait::RoaringPostingList,
    },
};
use serde::Serialize;
use std::{io::Write, path::Path, sync::Arc};

use crate::{
    inputs::filters::{InvertedIndexKind, MetadataIndexBuild},
    utils::filters::QueryBitmapEvaluator,
};

pub(crate) fn register_benchmarks(benchmarks: &mut Benchmarks) {
    // Register the metadata index job
    benchmarks.register::<MetadataIndexJob<'static>>(
        "metadata-index-build",
        |job, checkpoint, out| {
            let stats = job.run(checkpoint, out)?;
            Ok(serde_json::to_value(stats)?)
        },
    );
}

// Metadata-only index job wrapper
pub(super) struct MetadataIndexJob<'a> {
    input: &'a crate::inputs::filters::MetadataIndexBuild,
}

impl<'a> MetadataIndexJob<'a> {
    fn new(input: &'a crate::inputs::filters::MetadataIndexBuild) -> Self {
        Self { input }
    }
}

impl dispatcher::Map for MetadataIndexJob<'static> {
    type Type<'a> = MetadataIndexJob<'a>;
}

// Dispatch from the concrete input type
impl<'a> DispatchRule<&'a crate::inputs::filters::MetadataIndexBuild> for MetadataIndexJob<'a> {
    type Error = std::convert::Infallible;

    fn try_match(
        _from: &&'a crate::inputs::filters::MetadataIndexBuild,
    ) -> Result<MatchScore, FailureScore> {
        Ok(MatchScore(1))
    }

    fn convert(from: &'a crate::inputs::filters::MetadataIndexBuild) -> Result<Self, Self::Error> {
        Ok(MetadataIndexJob::new(from))
    }

    fn description(
        f: &mut std::fmt::Formatter<'_>,
        _from: Option<&&'a crate::inputs::filters::MetadataIndexBuild>,
    ) -> std::fmt::Result {
        writeln!(
            f,
            "tag: \"{}\"",
            crate::inputs::filters::MetadataIndexBuild::tag()
        )
    }
}

// Central dispatch mapping
impl<'a> DispatchRule<&'a Any> for MetadataIndexJob<'a> {
    type Error = anyhow::Error;

    fn try_match(from: &&'a Any) -> Result<MatchScore, FailureScore> {
        from.try_match::<MetadataIndexBuild, Self>()
    }

    fn convert(from: &'a Any) -> Result<Self, Self::Error> {
        from.convert::<MetadataIndexBuild, Self>()
    }

    fn description(f: &mut std::fmt::Formatter, from: Option<&&'a Any>) -> std::fmt::Result {
        Any::description::<MetadataIndexBuild, Self>(f, from, MetadataIndexBuild::tag())
    }
}

impl<'a> MetadataIndexJob<'a> {
    fn run(
        self,
        checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> Result<MetadataIndexBuildStats, anyhow::Error> {
        // Print the input description so the user sees the job configuration.
        writeln!(output, "{}", self.input)?;

        // Use the supplied filter parameters (required for metadata-only build)
        let filter_params = &self.input.filter_params;

        // Reuse the helper: build index, parse predicates, produce BitmapFilters and telemetry
        let (bitmap_filters_vec, filter_search_results, _label_count) =
            prepare_bitmap_filters_from_paths_with_kind(
                filter_params.data_labels.as_ref(),
                filter_params.query_predicates.as_ref(),
                self.input.inverted_index_type,
                checkpoint,
            )?;

        // Collect per-query matching counts and compute aggregates
        let counts: Vec<usize> = bitmap_filters_vec.iter().map(|bf| bf.count()).collect();
        let query_count = counts.len();
        let total_matching: usize = counts.iter().cloned().sum();

        // counts_avg will be computed below via the shared percentiles utility
        let mut sorted = counts.clone();
        // Use the shared percentiles utility when we have values.
        let (
            counts_p1,
            counts_p5,
            counts_p10,
            counts_p50,
            counts_p90,
            counts_p95,
            counts_p99,
            counts_avg,
        ) = if sorted.is_empty() {
            (
                0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0.0f64,
            )
        } else {
            sorted.sort_unstable();
            let p = percentiles::compute_percentiles(&mut sorted)?;
            // p.median is f64; round to nearest usize for display/storage
            let p50 = p.median.round() as usize;
            let p90 = p.p90;
            let p99 = p.p99;
            let n = sorted.len();
            let p1 = sorted[(n / 100).min(n - 1)];
            let p5 = sorted[((5 * n) / 100).min(n - 1)];
            let p10 = sorted[((10 * n) / 100).min(n - 1)];
            let p95 = sorted[((95 * n) / 100).min(n - 1)];
            (p1, p5, p10, p50, p90, p95, p99, p.mean)
        };

        let stats = MetadataIndexBuildStats {
            label_count: _label_count,
            query_count,
            total_matching,
            counts_avg,
            counts_p1,
            counts_p5,
            counts_p10,
            counts_p50,
            counts_p90,
            counts_p95,
            counts_p99,
            filter: filter_search_results,
        };

        // Print the human-readable summary for interactive runs.
        writeln!(output, "\n\n{}", stats)?;
        Ok(stats)
    }
}

#[derive(Debug, Serialize)]
pub struct FilterSearchResults {
    pub base_label_latencies: MicroSeconds,
    pub inverted_index_build_latencies: MicroSeconds,
    pub query_parsing_latencies: MicroSeconds,
    pub search_strategy_latencies: MicroSeconds,
}

impl Default for FilterSearchResults {
    fn default() -> Self {
        Self {
            base_label_latencies: MicroSeconds::new(0),
            inverted_index_build_latencies: MicroSeconds::new(0),
            query_parsing_latencies: MicroSeconds::new(0),
            search_strategy_latencies: MicroSeconds::new(0),
        }
    }
}

impl std::fmt::Display for FilterSearchResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // header for filters
        writeln!(f, "Filter Search Metrics:")?;
        writeln!(
            f,
            "   Base Label Latencies: {} s",
            self.base_label_latencies.as_seconds()
        )?;
        writeln!(
            f,
            "   Inverted Index Build Latencies: {} s",
            self.inverted_index_build_latencies.as_seconds()
        )?;
        writeln!(
            f,
            "   Query Parsing Latencies: {} s",
            self.query_parsing_latencies.as_seconds()
        )?;
        writeln!(
            f,
            "   Search Strategy Latencies: {} s",
            self.search_strategy_latencies.as_seconds()
        )?;
        Ok(())
    }
}

#[derive(serde::Serialize, Debug)]
pub(super) struct MetadataIndexBuildStats {
    pub(super) label_count: usize,
    pub(super) query_count: usize,
    pub(super) total_matching: usize,
    // Aggregate statistics over per-query matching counts
    pub(super) counts_avg: f64,
    pub(super) counts_p1: usize,
    pub(super) counts_p5: usize,
    pub(super) counts_p10: usize,
    pub(super) counts_p50: usize,
    pub(super) counts_p90: usize,
    pub(super) counts_p95: usize,
    pub(super) counts_p99: usize,
    pub(super) filter: FilterSearchResults,
}

impl std::fmt::Display for MetadataIndexBuildStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Metadata Index Build Stats:")?;
        writeln!(f, "   Label Count: {}", self.label_count)?;
        writeln!(f, "   Query Count: {}", self.query_count)?;
        writeln!(f, "   Total Matching: {}", self.total_matching)?;
        writeln!(f, "   Match Count Avg: {:.2}", self.counts_avg)?;
        writeln!(f, "   Match Count P1: {}", self.counts_p1)?;
        writeln!(f, "   Match Count P5: {}", self.counts_p5)?;
        writeln!(f, "   Match Count P10: {}", self.counts_p10)?;
        writeln!(f, "   Match Count P50: {}", self.counts_p50)?;
        writeln!(f, "   Match Count P90: {}", self.counts_p90)?;
        writeln!(f, "   Match Count P95: {}", self.counts_p95)?;
        writeln!(f, "   Match Count P99: {}", self.counts_p99)?;
        writeln!(f, "{}", self.filter)?;
        Ok(())
    }
}

pub fn prepare_bitmap_filters_from_paths_with_kind(
    data_labels: &Path,
    query_predicates: &Path,
    kind: InvertedIndexKind,
    _checkpoint: Checkpoint<'_>,
) -> anyhow::Result<(Vec<QueryBitmapEvaluator>, FilterSearchResults, usize)> {
    let timer = std::time::Instant::now();
    // Read base labels
    let base_labels = read_baselabels(data_labels)?;
    let label_count = base_labels.len();
    let base_label_latency = timer.elapsed();

    // Build inverted index (one-time cost)
    let timer = std::time::Instant::now();

    // Create the appropriate store backend based on the kind
    let inverted_index = match kind {
        InvertedIndexKind::BfTree => {
            // Use BfTree-based persistent store
            let store = Arc::new(BfTreeStore::memory().unwrap());
            let mut idx =
                GenericIndex::<BfTreeStore, RoaringPostingList, DefaultKeyCodec>::new(store)
                    .with_field_normalizer(|field| format!("/{}", field.replace(".", "/")));
            // Build the index
            for (doc_id, doc) in base_labels.iter().enumerate() {
                idx.insert(doc_id, &doc.flatten_metadata())?;
            }
            Arc::new(idx)
        }
    };

    let inverted_index_build_latency = timer.elapsed();

    let timer = std::time::Instant::now();
    // Parse queries and evaluate against the index
    let parsed_queries = read_and_parse_queries(query_predicates)?;

    let query_parsing_latency = timer.elapsed();

    let timer = std::time::Instant::now();

    let bitmap_filters: Vec<QueryBitmapEvaluator> = parsed_queries
        .into_iter()
        .map(|(_, query_predicate)| {
            QueryBitmapEvaluator::new(query_predicate, inverted_index.as_ref())
        })
        .collect();

    let search_strategy_latency = timer.elapsed();

    let filter_search_results = FilterSearchResults {
        base_label_latencies: base_label_latency.into(),
        inverted_index_build_latencies: inverted_index_build_latency.into(),
        query_parsing_latencies: query_parsing_latency.into(),
        search_strategy_latencies: search_strategy_latency.into(),
    };

    Ok((bitmap_filters, filter_search_results, label_count))
}

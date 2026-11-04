/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    collections::{hash_map::DefaultHasher, BTreeSet},
    hash::{Hash, Hasher},
    io::Write,
    num::NonZeroUsize,
    sync::Arc,
    time::Instant,
};

use anyhow::{ensure, Context, Result};
use diskann::{graph::SearchOutputBuffer, neighbor::Neighbor, provider::DefaultContext, ANNResult};
use diskann_benchmark_core::{
    recall::{self, GroundTruthMode},
    search,
};
use diskann_benchmark_runner::{utils::MicroSeconds, Checkpoint, Output};
use diskann_utils::views::rowmajor::{self, Matrix};
use serde::Serialize;

use super::{
    build::{BuildResult, BuiltIndex},
    data::{self, Rows},
    metric,
    search_provider::{Query, ScanStats, SearchProvider},
};
use crate::{
    inputs::ivf::SearchPhase,
    utils::{recall::RecallMetrics, SimilarityMeasure},
};

#[derive(Debug)]
pub(super) struct SearchResult {
    pub(super) neighbors: Vec<Neighbor<u32>>,
    pub(super) candidates_scanned: usize,
    pub(super) lists_scanned: usize,
}

impl BuiltIndex {
    pub(super) async fn search(
        &self,
        provider: &SearchProvider,
        vector: &[f32],
        k: NonZeroUsize,
        nprobe: usize,
    ) -> ANNResult<SearchResult> {
        let stats = ScanStats::default();
        let mut neighbors = Vec::<Neighbor<u32>>::with_capacity(k.get());
        let count = self
            .index
            .search(
                provider,
                &DefaultContext,
                Query {
                    vector,
                    stats: &stats,
                },
                k,
                NonZeroUsize::new(nprobe)
                    .ok_or_else(|| diskann::ANNError::message("zero nprobe"))?,
                &mut neighbors,
            )
            .await?;
        neighbors.truncate(count);
        Ok(SearchResult {
            neighbors,
            candidates_scanned: stats.candidates.load(std::sync::atomic::Ordering::Relaxed),
            lists_scanned: stats.lists.load(std::sync::atomic::Ordering::Relaxed),
        })
    }
}

struct Searcher {
    index: Arc<BuiltIndex>,
    provider: Arc<SearchProvider>,
    queries: Arc<Rows>,
    query_count: usize,
}

#[derive(Clone)]
struct SearchParameters {
    distance: SimilarityMeasure,
    k: NonZeroUsize,
    nprobe: usize,
    full_probe: bool,
}

impl search::Search for Searcher {
    type Id = u32;
    type Parameters = SearchParameters;
    // Keep neighbors for the independent full-probe correctness check.
    type Output = SearchResult;

    fn num_queries(&self) -> usize {
        self.query_count
    }

    fn id_count(&self, parameters: &SearchParameters) -> search::IdCount {
        search::IdCount::Fixed(parameters.k)
    }

    async fn search<O>(
        &self,
        parameters: &SearchParameters,
        buffer: &mut O,
        query: usize,
    ) -> ANNResult<Self::Output>
    where
        O: SearchOutputBuffer<u32> + Send,
    {
        let result = self
            .index
            .search(
                &self.provider,
                &self.queries[query],
                parameters.k,
                parameters.nprobe,
            )
            .await?;
        buffer.extend(result.neighbors.iter().cloned());
        Ok(result)
    }
}

#[derive(Debug, Serialize)]
pub(super) struct SearchResults {
    pub(super) distance: SimilarityMeasure,
    pub(super) nprobe: usize,
    pub(super) query_count: usize,
    pub(super) result_count: usize,
    pub(super) full_probe: bool,
    pub(super) full_probe_tie_queries: usize,
    pub(super) recall: RecallMetrics,
    pub(super) avg_candidates: f64,
    pub(super) avg_lists: f64,
    pub(super) scan_ratio: f64,
    /// Ordered query results: IDs, distance bits and per-query scan work.
    pub(super) result_fingerprint: String,
    pub(super) qps: Vec<f64>,
    pub(super) search_latencies: Vec<MicroSeconds>,
}

struct Aggregator<'a> {
    truth: &'a rowmajor::Owned<u32>,
    flat_truth: &'a rowmajor::Owned<u32>,
    base_count: usize,
    base: &'a [Vec<f32>],
    queries: &'a [Vec<f32>],
}

/// The upstream queue orders by distance, with no ID tie rule. Verify every returned
/// distance and the complete sorted top-k distance sequence, admitting only exact ties.
pub(super) fn validate_full_probe(
    result: &SearchResult,
    truth: &[u32],
    base: &[Vec<f32>],
    query: &[f32],
    distance: SimilarityMeasure,
) -> Result<bool> {
    ensure!(
        result.candidates_scanned == base.len() && result.neighbors.len() == truth.len(),
        "full-probe count mismatch"
    );
    let mut seen = BTreeSet::new();
    let mut ids_match = true;
    for (rank, (hit, &expected)) in result.neighbors.iter().zip(truth).enumerate() {
        let id = *hit.id();
        let point = base
            .get(id as usize)
            .context("full-probe returned an invalid ID")?;
        ensure!(seen.insert(id), "full-probe returned a duplicate ID");
        let actual_distance = metric::evaluate(distance, query, point.as_slice());
        let expected_distance =
            metric::evaluate(distance, query, base[expected as usize].as_slice());
        ensure!(
            hit.distance().to_bits() == actual_distance.to_bits()
                && actual_distance.to_bits() == expected_distance.to_bits(),
            "full-probe distance differs from the flat oracle at rank {rank}"
        );
        ids_match &= id == expected;
    }
    Ok(ids_match)
}

impl search::Aggregate<SearchParameters, u32, SearchResult> for Aggregator<'_> {
    type Output = SearchResults;

    fn aggregate(
        &mut self,
        run: search::Run<SearchParameters>,
        results: Vec<search::SearchResults<u32, SearchResult>>,
    ) -> Result<SearchResults> {
        // search_all has no built-in warmup. Its first complete repetition is excluded.
        let results = &results[1..];
        let first = results
            .first()
            .context("IVF search needs a measured repetition")?;
        let params = run.parameters();
        let truth = self
            .truth
            .subview(0..first.len())
            .context("ground-truth query count mismatch")?;
        let recall = recall::knn(
            &truth,
            None,
            first.ids().as_rows(),
            params.k.get(),
            params.k.get(),
            GroundTruthMode::Fixed,
        )?;
        let mut candidates = 0;
        let mut returned = 0;
        let mut lists = 0;
        let mut tie_queries = 0;
        let mut fingerprint = DefaultHasher::new();
        for (query, result) in first.output().iter().enumerate() {
            query.hash(&mut fingerprint);
            result.candidates_scanned.hash(&mut fingerprint);
            result.lists_scanned.hash(&mut fingerprint);
            result.neighbors.len().hash(&mut fingerprint);
            for hit in &result.neighbors {
                hit.id().hash(&mut fingerprint);
                hit.distance().to_bits().hash(&mut fingerprint);
            }
            candidates += result.candidates_scanned;
            lists += result.lists_scanned;
            returned += result.neighbors.len();
            if params.full_probe {
                tie_queries += usize::from(!validate_full_probe(
                    result,
                    self.flat_truth.row(query),
                    self.base,
                    &self.queries[query],
                    params.distance,
                )?);
            }
        }
        Ok(SearchResults {
            distance: params.distance,
            nprobe: params.nprobe,
            query_count: first.len(),
            result_count: returned,
            full_probe: params.full_probe,
            full_probe_tie_queries: tie_queries,
            recall: (&recall).into(),
            avg_candidates: candidates as f64 / first.len() as f64,
            avg_lists: lists as f64 / first.len() as f64,
            scan_ratio: candidates as f64 / first.len() as f64 / self.base_count as f64,
            result_fingerprint: format!("{:016x}", fingerprint.finish()),
            qps: results
                .iter()
                .map(|r| r.len() as f64 / r.end_to_end_latency().as_seconds())
                .collect(),
            search_latencies: results.iter().map(|r| r.end_to_end_latency()).collect(),
        })
    }
}

pub(super) fn run(
    index: BuiltIndex,
    base: &[Vec<f32>],
    mut queries: Rows,
    config: &SearchPhase,
    build: &BuildResult,
    checkpoint: Checkpoint<'_>,
    mut output: &mut dyn Output,
) -> Result<(Vec<SearchResults>, f64, f64)> {
    let available_queries = queries.len();
    if let Some(limit) = config.query_limit {
        queries.truncate(limit.get());
    }
    let full_queries = queries.len().min(
        config
            .full_probe_queries
            .map_or(usize::MAX, NonZeroUsize::get),
    );
    let preparation_start = Instant::now();
    let provider = Arc::new(SearchProvider::new(
        &*index.provider.read()?,
        config.distance,
    ));
    let search_preparation_ms = preparation_start.elapsed().as_secs_f64() * 1000.0;
    let gt_start = Instant::now();
    let truth = match &config.groundtruth {
        Some(path) => data::read_truth(path, available_queries, base.len(), config.k.get())?,
        None => rowmajor::Owned::try_from_data(
            queries
                .iter()
                .flat_map(|q| data::flat_top_k(base, q, config.k.get(), config.distance))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            queries.len(),
            config.k.get(),
        )?,
    };
    // Supplied truth defines recall and may use different precision or tie ordering.
    // Keep an independent canonical oracle for the full-probe correctness check.
    let canonical_truth = if config.groundtruth.is_some() {
        Some(rowmajor::Owned::try_from_data(
            queries
                .iter()
                .take(full_queries)
                .flat_map(|q| data::flat_top_k(base, q, config.k.get(), config.distance))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            full_queries,
            config.k.get(),
        )?)
    } else {
        None
    };
    let flat_truth = canonical_truth.as_ref().unwrap_or(&truth);
    let ground_truth_ms = gt_start.elapsed().as_secs_f64() * 1000.0;
    let list_count = index.provider.read()?.lists.len();
    let mut probes: Vec<_> = config
        .nprobe
        .iter()
        .map(|n| n.get().min(list_count))
        .collect();
    probes.extend(provider.calibrate(&queries, &config.target_scan_ratios)?);
    probes.push(list_count);
    probes.sort_unstable();
    probes.dedup();
    let index = Arc::new(index);
    let queries = Arc::new(queries);
    let mut results: Vec<SearchResults> = Vec::new();
    let reps = config
        .reps
        .get()
        .checked_add(1)
        .and_then(NonZeroUsize::new)
        .context("search repetition count overflow")?;
    for nprobe in probes {
        let full_probe = nprobe == list_count;
        let query_count = if full_probe {
            full_queries
        } else {
            queries.len()
        };
        writeln!(output, "Searching nprobe={nprobe}, queries={query_count}")?;
        let searcher = Arc::new(Searcher {
            index: index.clone(),
            provider: provider.clone(),
            queries: queries.clone(),
            query_count,
        });
        results.extend(search::search_all(
            searcher,
            std::iter::once(search::Run::new(
                SearchParameters {
                    distance: config.distance,
                    k: config.k,
                    nprobe,
                    full_probe,
                },
                search::Setup {
                    threads: NonZeroUsize::MIN,
                    tasks: NonZeroUsize::MIN,
                    reps,
                },
            )),
            Aggregator {
                truth: &truth,
                flat_truth,
                base_count: base.len(),
                base,
                queries: queries.as_slice(),
            },
        )?);
        #[derive(Serialize)]
        struct Progress<'a> {
            build: &'a BuildResult,
            search: &'a [SearchResults],
        }
        checkpoint.checkpoint(&Progress {
            build,
            search: &results,
        })?;
    }
    Ok((results, ground_truth_ms, search_preparation_ms))
}

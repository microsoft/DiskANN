/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{io::Write, num::NonZeroUsize, sync::Arc, time::Instant};

use anyhow::{ensure, Context, Result};
use diskann::{
    graph::SearchOutputBuffer,
    ivf::{IVFIndex, SearchResult},
    provider::DefaultContext,
    ANNResult,
};
use diskann_benchmark_core::{
    recall::{self, GroundTruthMode},
    search,
};
use diskann_benchmark_runner::{utils::MicroSeconds, Checkpoint, Output};
use diskann_utils::views::rowmajor::{self, Matrix};
use serde::Serialize;

use super::{
    build::BuildResult,
    data::{self, Rows},
    provider::InMemProvider,
};
use crate::{inputs::ivf::SearchPhase, utils::recall::RecallMetrics};

struct Searcher {
    index: Arc<IVFIndex<InMemProvider>>,
    queries: Arc<Rows>,
    query_count: usize,
}

#[derive(Clone)]
struct SearchParameters {
    k: NonZeroUsize,
    nprobe: usize,
    full_probe: bool,
}

impl search::Search for Searcher {
    type Id = u32;
    type Parameters = SearchParameters;
    // Keep neighbors for the independent full-probe correctness check.
    type Output = SearchResult<u32>;

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
                &DefaultContext,
                &self.queries[query],
                parameters.k.get(),
                parameters.nprobe,
            )
            .await?;
        buffer.extend(result.neighbors.iter().cloned());
        Ok(result)
    }
}

#[derive(Debug, Serialize)]
pub(super) struct SearchResults {
    pub(super) nprobe: usize,
    pub(super) query_count: usize,
    pub(super) result_count: usize,
    pub(super) full_probe: bool,
    pub(super) recall: RecallMetrics,
    pub(super) avg_candidates: f64,
    pub(super) qps: Vec<f64>,
    pub(super) search_latencies: Vec<MicroSeconds>,
}

struct Aggregator<'a> {
    truth: &'a rowmajor::Owned<u32>,
    flat_truth: &'a rowmajor::Owned<u32>,
    base_count: usize,
}

impl search::Aggregate<SearchParameters, u32, SearchResult<u32>> for Aggregator<'_> {
    type Output = SearchResults;

    fn aggregate(
        &mut self,
        run: search::Run<SearchParameters>,
        results: Vec<search::SearchResults<u32, SearchResult<u32>>>,
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
        for (query, result) in first.output().iter().enumerate() {
            candidates += result.candidates_scanned;
            returned += result.neighbors.len();
            if params.full_probe {
                ensure!(
                    result.neighbors.iter().map(|hit| *hit.id()).eq(self
                        .flat_truth
                        .row(query)
                        .iter()
                        .copied())
                        && result.candidates_scanned == self.base_count,
                    "full-probe search differs from flat ground truth"
                );
            }
        }
        Ok(SearchResults {
            nprobe: params.nprobe,
            query_count: first.len(),
            result_count: returned,
            full_probe: params.full_probe,
            recall: (&recall).into(),
            avg_candidates: candidates as f64 / first.len() as f64,
            qps: results
                .iter()
                .map(|r| r.len() as f64 / r.end_to_end_latency().as_seconds())
                .collect(),
            search_latencies: results.iter().map(|r| r.end_to_end_latency()).collect(),
        })
    }
}

pub(super) fn run(
    index: IVFIndex<InMemProvider>,
    base: &[Vec<f32>],
    mut queries: Rows,
    config: &SearchPhase,
    build: &BuildResult,
    checkpoint: Checkpoint<'_>,
    mut output: &mut dyn Output,
) -> Result<(Vec<SearchResults>, f64)> {
    let available_queries = queries.len();
    if let Some(limit) = config.query_limit {
        queries.truncate(limit.get());
    }
    let full_queries = queries.len().min(
        config
            .full_probe_queries
            .map_or(usize::MAX, NonZeroUsize::get),
    );
    let gt_start = Instant::now();
    let truth = match &config.groundtruth {
        Some(path) => data::read_truth(path, available_queries, base.len(), config.k.get())?,
        None => rowmajor::Owned::try_from_data(
            queries
                .iter()
                .flat_map(|q| data::flat_top_k(base, q, config.k.get()))
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
                .flat_map(|q| data::flat_top_k(base, q, config.k.get()))
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
    let list_count = index.provider().lists.len();
    let mut probes: Vec<_> = config
        .nprobe
        .iter()
        .map(|n| n.get().min(list_count))
        .collect();
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
            queries: queries.clone(),
            query_count,
        });
        results.extend(search::search_all(
            searcher,
            std::iter::once(search::Run::new(
                SearchParameters {
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
    Ok((results, ground_truth_ms))
}

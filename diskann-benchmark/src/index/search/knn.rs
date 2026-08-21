/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{num::NonZeroUsize, sync::Arc};

use diskann_benchmark_core::search::Aggregate;
use diskann_benchmark_core::{self as benchmark_core, search as core_search};
use diskann_benchmark_core::{recall::GroundTruthMode, search::graph::KnnParams};

use crate::{index::result::SearchResults, inputs::graph_index::GraphSearch};

#[derive(Debug, Clone, Copy)]
pub(crate) struct SearchSteps<'a> {
    pub reps: NonZeroUsize,
    pub num_tasks: &'a [NonZeroUsize],
    pub runs: &'a [GraphSearch],
    pub groundtruth_mode: GroundTruthMode,
}

impl<'a> SearchSteps<'a> {
    pub(crate) fn new(
        reps: NonZeroUsize,
        num_tasks: &'a [NonZeroUsize],
        runs: &'a [GraphSearch],
        groundtruth_mode: GroundTruthMode,
    ) -> Self {
        Self {
            reps,
            num_tasks,
            runs,
            groundtruth_mode,
        }
    }
}

pub(crate) fn run<I>(
    runner: &dyn Knn<I>,
    groundtruth: &dyn benchmark_core::recall::Rows<I>,
    steps: SearchSteps<'_>,
) -> anyhow::Result<Vec<SearchResults>> {
    let mut all = Vec::new();

    for threads in steps.num_tasks.iter() {
        for run in steps.runs.iter() {
            let setup = core_search::Setup {
                threads: *threads,
                tasks: *threads,
                reps: steps.reps,
            };

            let parameters: Vec<_> = run
                .search_l
                .iter()
                .map(|search_l| {
                    let search_params = KnnParams::new(run.search_n, *search_l).unwrap();

                    core_search::Run::new(search_params, setup.clone())
                })
                .collect();

            all.extend(runner.search_all(
                parameters,
                groundtruth,
                run.recall_k,
                run.search_n,
                steps.groundtruth_mode,
            )?);
        }
    }

    Ok(all)
}

/// Run multi-hop searches with a freshly constructed searcher for every repetition and parameter.
///
/// This is required for query providers with lazy per-query state: reusing one `MultiHop`
/// searcher would amortize that state across repetitions and search-L values.
pub(crate) fn run_fresh_multihop<I, S, F>(
    mut make_runner: F,
    groundtruth: &dyn benchmark_core::recall::Rows<I>,
    steps: SearchSteps<'_>,
) -> anyhow::Result<Vec<SearchResults>>
where
    I: benchmark_core::recall::RecallCompatible,
    S: core_search::Search<
        Id = I,
        Parameters = KnnParams,
        Output = core_search::graph::knn::Metrics,
    >,
    F: FnMut() -> anyhow::Result<Arc<S>>,
{
    let mut all = Vec::new();

    for threads in steps.num_tasks.iter() {
        for run in steps.runs.iter() {
            for search_l in &run.search_l {
                let parameters = KnnParams::new(run.search_n, *search_l).unwrap();
                let setup = core_search::Setup {
                    threads: *threads,
                    tasks: *threads,
                    reps: steps.reps,
                };
                let runtime = benchmark_core::tokio::runtime((*threads).into())?;
                let mut raw = Vec::with_capacity(steps.reps.get());

                for _ in 0..steps.reps.get() {
                    raw.push(core_search::search(
                        make_runner()?,
                        parameters,
                        *threads,
                        &runtime,
                    )?);
                }

                let mut aggregator = core_search::graph::knn::Aggregator::new(
                    groundtruth,
                    run.recall_k,
                    run.search_n,
                    steps.groundtruth_mode,
                );
                let summary =
                    aggregator.aggregate(core_search::Run::new(parameters, setup), raw)?;
                all.push(SearchResults::new(summary));
            }
        }
    }

    Ok(all)
}

type Run = core_search::Run<KnnParams>;
pub(crate) trait Knn<I> {
    fn search_all(
        &self,
        parameters: Vec<Run>,
        groundtruth: &dyn benchmark_core::recall::Rows<I>,
        recall_k: usize,
        recall_n: usize,
        groundtruth_mode: GroundTruthMode,
    ) -> anyhow::Result<Vec<SearchResults>>;
}

///////////
// Impls //
///////////

impl<DP, T, S, PP> Knn<DP::InternalId> for Arc<core_search::graph::KNN<DP, T, S, PP>>
where
    DP: diskann::provider::DataProvider,
    core_search::graph::KNN<DP, T, S, PP>: core_search::Search<
        Id = DP::InternalId,
        Parameters = KnnParams,
        Output = core_search::graph::knn::Metrics,
    >,
{
    fn search_all(
        &self,
        parameters: Vec<core_search::Run<KnnParams>>,
        groundtruth: &dyn benchmark_core::recall::Rows<DP::InternalId>,
        recall_k: usize,
        recall_n: usize,
        groundtruth_mode: GroundTruthMode,
    ) -> anyhow::Result<Vec<SearchResults>> {
        let results = core_search::search_all(
            self.clone(),
            parameters,
            core_search::graph::knn::Aggregator::new(
                groundtruth,
                recall_k,
                recall_n,
                groundtruth_mode,
            ),
        )?;

        Ok(results.into_iter().map(SearchResults::new).collect())
    }
}

impl<DP, T, S> Knn<DP::InternalId> for Arc<core_search::graph::MultiHop<DP, T, S>>
where
    DP: diskann::provider::DataProvider,
    core_search::graph::MultiHop<DP, T, S>: core_search::Search<
        Id = DP::InternalId,
        Parameters = KnnParams,
        Output = core_search::graph::knn::Metrics,
    >,
{
    fn search_all(
        &self,
        parameters: Vec<core_search::Run<KnnParams>>,
        groundtruth: &dyn benchmark_core::recall::Rows<DP::InternalId>,
        recall_k: usize,
        recall_n: usize,
        groundtruth_mode: GroundTruthMode,
    ) -> anyhow::Result<Vec<SearchResults>> {
        let results = core_search::search_all(
            self.clone(),
            parameters,
            core_search::graph::knn::Aggregator::new(
                groundtruth,
                recall_k,
                recall_n,
                groundtruth_mode,
            ),
        )?;

        Ok(results.into_iter().map(SearchResults::new).collect())
    }
}

impl<DP, T, S> Knn<DP::InternalId> for Arc<core_search::graph::InlineFilterSearch<DP, T, S>>
where
    DP: diskann::provider::DataProvider,
    core_search::graph::InlineFilterSearch<DP, T, S>: core_search::Search<
        Id = DP::InternalId,
        Parameters = KnnParams,
        Output = core_search::graph::knn::Metrics,
    >,
{
    fn search_all(
        &self,
        parameters: Vec<core_search::Run<KnnParams>>,
        groundtruth: &dyn benchmark_core::recall::Rows<DP::InternalId>,
        recall_k: usize,
        recall_n: usize,
        groundtruth_mode: GroundTruthMode,
    ) -> anyhow::Result<Vec<SearchResults>> {
        let results = core_search::search_all(
            self.clone(),
            parameters,
            core_search::graph::knn::Aggregator::new(
                groundtruth,
                recall_k,
                recall_n,
                groundtruth_mode,
            ),
        )?;

        Ok(results.into_iter().map(SearchResults::new).collect())
    }
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{num::NonZeroUsize, sync::Arc};

use diskann_benchmark_core::{self as benchmark_core, search as core_search};
use diskann_benchmark_core::{recall::GroundTruthMode, search::graph::KnnParams};
use diskann_utils::views::Matrix;

use crate::{
    index::result::SearchResults,
    inputs::graph_index::GraphSearch,
    utils::datafiles::{load_groundtruth, BinFile},
};

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
            // Load per-query start points if configured for this run.
            if let Some(path) = &run.start_points_file {
                let start_ids = load_groundtruth(BinFile(path.as_path()), None)
                    .map(Arc::new)
                    .map_err(|e| {
                        anyhow::anyhow!("Failed to load start_points_file {:?}: {}", path, e)
                    })?;
                runner.set_start_points(start_ids);
            }

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

    /// Apply optional per-query extra start point IDs (loaded from a groundtruth-format
    /// file). This method is called once per search run before `search_all`.
    /// Implementations that don't support extra start points may ignore this call.
    fn set_start_points(&self, start_ids: Arc<Matrix<u32>>);
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

    fn set_start_points(&self, start_ids: Arc<Matrix<u32>>) {
        (**self).set_start_points(start_ids);
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

    fn set_start_points(&self, start_ids: Arc<Matrix<u32>>) {
        (**self).set_start_points(start_ids);
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

    fn set_start_points(&self, start_ids: Arc<Matrix<u32>>) {
        (**self).set_start_points(start_ids);
    }
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{num::NonZeroUsize, sync::Arc};

use diskann_benchmark_core::{self as benchmark_core, search as core_search};

use crate::{backend::index::result::SearchResults, inputs::async_::GraphSearch};

#[derive(Debug, Clone, Copy)]
pub(crate) struct SearchSteps<'a> {
    pub reps: NonZeroUsize,
    pub num_tasks: &'a [NonZeroUsize],
    pub runs: &'a [GraphSearch],
}

impl<'a> SearchSteps<'a> {
    pub(crate) fn new(
        reps: NonZeroUsize,
        num_tasks: &'a [NonZeroUsize],
        runs: &'a [GraphSearch],
    ) -> Self {
        Self {
            reps,
            num_tasks,
            runs,
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
                    let search_params =
                        diskann::graph::SearchParams::new(run.search_n, *search_l, None).unwrap();

                    core_search::Run::new(search_params, setup.clone())
                })
                .collect();

            all.extend(runner.search_all(parameters, groundtruth, run.recall_k, run.search_n)?);
        }
    }

    Ok(all)
}

type Run = core_search::Run<diskann::graph::SearchParams>;
pub(crate) trait Knn<I> {
    fn search_all(
        &self,
        parameters: Vec<Run>,
        groundtruth: &dyn benchmark_core::recall::Rows<I>,
        recall_k: usize,
        recall_n: usize,
    ) -> anyhow::Result<Vec<SearchResults>>;
}

///////////
// Impls //
///////////

impl<DP, T, S> Knn<DP::InternalId> for Arc<core_search::graph::KNN<DP, T, S>>
where
    DP: diskann::provider::DataProvider,
    core_search::graph::KNN<DP, T, S>: core_search::Search<
        Id = DP::InternalId,
        Parameters = diskann::graph::SearchParams,
        Output = core_search::graph::knn::Metrics,
    >,
{
    fn search_all(
        &self,
        parameters: Vec<core_search::Run<diskann::graph::SearchParams>>,
        groundtruth: &dyn benchmark_core::recall::Rows<DP::InternalId>,
        recall_k: usize,
        recall_n: usize,
    ) -> anyhow::Result<Vec<SearchResults>> {
        let results = core_search::search_all(
            self.clone(),
            parameters.into_iter(),
            core_search::graph::knn::Aggregator::new(groundtruth, recall_k, recall_n),
        )?;

        Ok(results.into_iter().map(SearchResults::new).collect())
    }
}

impl<DP, T, S> Knn<DP::InternalId> for Arc<core_search::graph::MultiHop<DP, T, S>>
where
    DP: diskann::provider::DataProvider,
    core_search::graph::MultiHop<DP, T, S>: core_search::Search<
        Id = DP::InternalId,
        Parameters = diskann::graph::SearchParams,
        Output = core_search::graph::knn::Metrics,
    >,
{
    fn search_all(
        &self,
        parameters: Vec<core_search::Run<diskann::graph::SearchParams>>,
        groundtruth: &dyn benchmark_core::recall::Rows<DP::InternalId>,
        recall_k: usize,
        recall_n: usize,
    ) -> anyhow::Result<Vec<SearchResults>> {
        let results = core_search::search_all(
            self.clone(),
            parameters.into_iter(),
            core_search::graph::knn::Aggregator::new(groundtruth, recall_k, recall_n),
        )?;

        Ok(results.into_iter().map(SearchResults::new).collect())
    }
}

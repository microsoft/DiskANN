/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{num::NonZeroUsize, sync::Arc};

use diskann_benchmark_core::{self as benchmark_core, search as core_search};

use crate::{
    backend::index::result::SearchResults,
    inputs::async_::{GraphSearch, RagSearchPhase},
};

#[derive(Debug, Clone, Copy)]
pub(crate) struct RagSearchSteps<'a> {
    pub reps: NonZeroUsize,
    pub num_tasks: &'a [NonZeroUsize],
    pub runs: &'a [GraphSearch],
    pub rag_eta: f64,
    pub rag_power: f64,
}

impl<'a> RagSearchSteps<'a> {
    pub(crate) fn new(phase: &'a RagSearchPhase) -> Self {
        Self {
            reps: phase.reps,
            num_tasks: &phase.num_threads,
            runs: &phase.runs,
            rag_eta: phase.rag_eta,
            rag_power: phase.rag_power,
        }
    }
}

pub(crate) fn run<I>(
    runner: &dyn Rag<I>,
    groundtruth: &dyn benchmark_core::recall::Rows<I>,
    steps: RagSearchSteps<'_>,
) -> anyhow::Result<Vec<SearchResults>> {
    let mut all = Vec::new();

    for threads in steps.num_tasks.iter() {
        for graph_run in steps.runs.iter() {
            let setup = core_search::Setup {
                threads: *threads,
                tasks: *threads,
                reps: steps.reps,
            };

            let parameters: Vec<_> = graph_run
                .search_l
                .iter()
                .map(|search_l| {
                    let knn = diskann::graph::search::Knn::new(graph_run.search_n, *search_l, None)
                        .unwrap();
                    let rag_search =
                        diskann::graph::search::RagSearch::new(knn, steps.rag_eta, steps.rag_power);

                    core_search::Run::new(rag_search, setup.clone())
                })
                .collect();

            all.extend(runner.search_all(
                parameters,
                groundtruth,
                graph_run.recall_k,
                graph_run.search_n,
            )?);
        }
    }

    Ok(all)
}

type Run = core_search::Run<diskann::graph::search::RagSearch>;

pub(crate) trait Rag<I> {
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

impl<DP, T, S> Rag<DP::InternalId> for Arc<core_search::graph::RAG<DP, T, S>>
where
    DP: diskann::provider::DataProvider,
    core_search::graph::RAG<DP, T, S>: core_search::Search<
        Id = DP::InternalId,
        Parameters = diskann::graph::search::RagSearch,
        Output = core_search::graph::knn::Metrics,
    >,
{
    fn search_all(
        &self,
        parameters: Vec<core_search::Run<diskann::graph::search::RagSearch>>,
        groundtruth: &dyn benchmark_core::recall::Rows<DP::InternalId>,
        recall_k: usize,
        recall_n: usize,
    ) -> anyhow::Result<Vec<SearchResults>> {
        let results = core_search::search_all(
            self.clone(),
            parameters.into_iter(),
            core_search::graph::rag::Aggregator::new(groundtruth, recall_k, recall_n),
        )?;

        Ok(results.into_iter().map(SearchResults::new).collect())
    }
}

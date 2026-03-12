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
    run_search(runner, groundtruth, steps, |setup, search_l, search_n| {
        let search_params = diskann::graph::search::Knn::new(search_n, search_l, None).unwrap();
        core_search::Run::new(search_params, setup)
    })
}

pub(crate) fn run_determinant_diversity<I>(
    runner: &dyn DeterminantDiversityKnn<I>,
    groundtruth: &dyn benchmark_core::recall::Rows<I>,
    steps: SearchSteps<'_>,
    eta: f64,
    power: f64,
    results_k: Option<usize>,
) -> anyhow::Result<Vec<SearchResults>> {
    run_search_determinant_diversity(runner, groundtruth, steps, |setup, search_l, search_n| {
        let base = diskann::graph::search::Knn::new(search_n, search_l, None).unwrap();
        let processor =
                diskann_providers::model::graph::provider::async_::DeterminantDiversitySearchParams::new(
                    results_k.unwrap_or(search_n),
                    eta,
                    power,
                ).map_err(|e| anyhow::anyhow!("Invalid determinant-diversity parameters: {}", e))?;

        let search_params =
            diskann_benchmark_core::search::graph::determinant_diversity::Parameters {
                inner: base,
                processor,
            };
        Ok(core_search::Run::new(search_params, setup))
    })
}

type Run = core_search::Run<diskann::graph::search::Knn>;
pub(crate) trait Knn<I> {
    fn search_all(
        &self,
        parameters: Vec<Run>,
        groundtruth: &dyn benchmark_core::recall::Rows<I>,
        recall_k: usize,
        recall_n: usize,
    ) -> anyhow::Result<Vec<SearchResults>>;
}

type DeterminantRun =
    core_search::Run<diskann_benchmark_core::search::graph::determinant_diversity::Parameters>;

/// Generic search infrastructure that unifies `run()` and `run_determinant_diversity()`.
///
/// This helper extracts the common loop logic (iterating over threads and runs,
/// and building a setup) leaving parameter construction to a builder closure.
/// This collapses the benchmark helper infrastructure and reduces duplication.
fn run_search<I, F>(
    runner: &dyn Knn<I>,
    groundtruth: &dyn benchmark_core::recall::Rows<I>,
    steps: SearchSteps<'_>,
    builder: F,
) -> anyhow::Result<Vec<SearchResults>>
where
    F: Fn(core_search::Setup, usize, usize) -> core_search::Run<diskann::graph::search::Knn>,
{
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
                .map(|&search_l| builder(setup.clone(), search_l, run.search_n))
                .collect();

            all.extend(runner.search_all(parameters, groundtruth, run.recall_k, run.search_n)?);
        }
    }

    Ok(all)
}

/// Generic search infrastructure for determinant-diversity searches.
///
/// Mirrors the unified logic of `run_search()` but for the DeterminantDiversityKnn trait.
fn run_search_determinant_diversity<I, F>(
    runner: &dyn DeterminantDiversityKnn<I>,
    groundtruth: &dyn benchmark_core::recall::Rows<I>,
    steps: SearchSteps<'_>,
    builder: F,
) -> anyhow::Result<Vec<SearchResults>>
where
    F: Fn(
        core_search::Setup,
        usize,
        usize,
    ) -> anyhow::Result<
        core_search::Run<diskann_benchmark_core::search::graph::determinant_diversity::Parameters>,
    >,
{
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
                .map(|&search_l| builder(setup.clone(), search_l, run.search_n))
                .collect::<anyhow::Result<Vec<_>>>()?;

            all.extend(runner.search_all(parameters, groundtruth, run.recall_k, run.search_n)?);
        }
    }

    Ok(all)
}

pub(crate) trait DeterminantDiversityKnn<I> {
    fn search_all(
        &self,
        parameters: Vec<DeterminantRun>,
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
        Parameters = diskann::graph::search::Knn,
        Output = core_search::graph::knn::Metrics,
    >,
{
    fn search_all(
        &self,
        parameters: Vec<core_search::Run<diskann::graph::search::Knn>>,
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
        Parameters = diskann::graph::search::Knn,
        Output = core_search::graph::knn::Metrics,
    >,
{
    fn search_all(
        &self,
        parameters: Vec<core_search::Run<diskann::graph::search::Knn>>,
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

impl<DP, T, S> DeterminantDiversityKnn<DP::InternalId>
    for Arc<core_search::graph::determinant_diversity::KNN<DP, T, S>>
where
    DP: diskann::provider::DataProvider,
    core_search::graph::determinant_diversity::KNN<DP, T, S>: core_search::Search<
        Id = DP::InternalId,
        Parameters = diskann_benchmark_core::search::graph::determinant_diversity::Parameters,
        Output = core_search::graph::knn::Metrics,
    >,
{
    fn search_all(
        &self,
        parameters: Vec<
            core_search::Run<
                diskann_benchmark_core::search::graph::determinant_diversity::Parameters,
            >,
        >,
        groundtruth: &dyn benchmark_core::recall::Rows<DP::InternalId>,
        recall_k: usize,
        recall_n: usize,
    ) -> anyhow::Result<Vec<SearchResults>> {
        let results = core_search::search_all(
            self.clone(),
            parameters.into_iter(),
            core_search::graph::determinant_diversity::Aggregator::new(
                groundtruth,
                recall_k,
                recall_n,
            ),
        )?;

        Ok(results
            .into_iter()
            .map(SearchResults::new_determinant_diversity)
            .collect())
    }
}

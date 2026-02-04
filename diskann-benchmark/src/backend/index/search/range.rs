/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{num::NonZeroUsize, sync::Arc};

use diskann_benchmark_core::{self as benchmark_core, search as core_search};

use crate::{backend::index::result::RangeSearchResults, inputs::async_::GraphRangeSearch};

#[derive(Debug, Clone, Copy)]
pub(crate) struct RangeSearchSteps<'a> {
    pub(crate) reps: NonZeroUsize,
    pub(crate) num_tasks: &'a [NonZeroUsize],
    pub(crate) runs: &'a [GraphRangeSearch],
}

impl<'a> RangeSearchSteps<'a> {
    pub(crate) fn new(
        reps: NonZeroUsize,
        num_tasks: &'a [NonZeroUsize],
        runs: &'a [GraphRangeSearch],
    ) -> Self {
        Self {
            reps,
            num_tasks,
            runs,
        }
    }
}

type Run = core_search::Run<diskann::graph::RangeSearchParams>;

pub(crate) trait Range<I> {
    fn search_all(
        &self,
        parameters: Vec<Run>,
        groundtruth: &dyn benchmark_core::recall::Rows<I>,
    ) -> anyhow::Result<Vec<RangeSearchResults>>;
}

pub(crate) fn run<I>(
    runner: &dyn Range<I>,
    groundtruth: &dyn benchmark_core::recall::Rows<I>,
    steps: RangeSearchSteps<'_>,
) -> anyhow::Result<Vec<RangeSearchResults>> {
    let mut all = Vec::new();

    for threads in steps.num_tasks.iter() {
        for run in steps.runs.iter() {
            let setup = core_search::Setup {
                threads: *threads,
                tasks: *threads,
                reps: steps.reps,
            };

            let parameters: Vec<_> = run
                .construct_params()?
                .into_iter()
                .map(|range_search_params| {
                    core_search::Run::new(range_search_params, setup.clone())
                })
                .collect();

            all.extend(runner.search_all(parameters, groundtruth)?);
        }
    }

    Ok(all)
}

///////////
// Impls //
///////////

impl<DP, T, S> Range<DP::InternalId> for Arc<core_search::graph::Range<DP, T, S>>
where
    DP: diskann::provider::DataProvider,
    core_search::graph::Range<DP, T, S>: core_search::Search<
        Id = DP::InternalId,
        Parameters = diskann::graph::RangeSearchParams,
        Output = core_search::graph::range::Metrics,
    >,
{
    fn search_all(
        &self,
        parameters: Vec<core_search::Run<diskann::graph::RangeSearchParams>>,
        groundtruth: &dyn benchmark_core::recall::Rows<DP::InternalId>,
    ) -> anyhow::Result<Vec<RangeSearchResults>> {
        let results = core_search::search_all(
            self.clone(),
            parameters.into_iter(),
            core_search::graph::range::Aggregator::new(groundtruth),
        )?;

        Ok(results.into_iter().map(RangeSearchResults::new).collect())
    }
}

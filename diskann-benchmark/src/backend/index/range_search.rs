/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{num::NonZeroUsize, sync::Arc};

use anyhow::anyhow;
use diskann::{
    graph::{glue, DiskANNIndex, RangeSearchParams},
    provider::{DataProvider, DefaultContext},
    utils::async_tools,
};
use diskann_benchmark_runner::utils::MicroSeconds;
use diskann_utils::views::Matrix;

use crate::{
    backend::index::result::{RangeSearchResults, RangeSearchResultsSetup},
    inputs::async_::GraphRangeSearch,
    utils,
};

#[derive(Debug, Clone, Copy)]
pub(super) struct RangeSearchSteps<'a> {
    reps: NonZeroUsize,
    num_tasks: &'a [NonZeroUsize],
    runs: &'a [GraphRangeSearch],
}

impl<'a> RangeSearchSteps<'a> {
    pub(super) fn new(
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

pub(super) fn run_range_search<T, S, DP>(
    index: Arc<DiskANNIndex<DP>>,
    strategies: Arc<Vec<S>>,
    queries: Arc<Matrix<T>>,
    range_groundtruth: Vec<Vec<u32>>,
    steps: RangeSearchSteps<'_>,
) -> anyhow::Result<Vec<RangeSearchResults>>
where
    DP: DataProvider<Context = DefaultContext, ExternalId = u32, InternalId = u32>,
    S: glue::SearchStrategy<DP, [T]> + Clone + Sync + 'static,
    T: Send + Sync + 'static,
{
    let mut search_results = Vec::new();
    for num_tasks in steps.num_tasks {
        let rt = utils::tokio::runtime(num_tasks.get())?;

        for run in steps.runs {
            let params = run.construct_params()?;
            for range_param in params {
                let mut latencies = Vec::<MicroSeconds>::new();
                let mut results = Vec::new();

                // Run on search as a warm-up and to get the baseline recall results.
                for _ in 0..steps.reps.get() {
                    let start = std::time::Instant::now();
                    let r = rt.block_on(run_range_search_parallel(
                        index.clone(),
                        strategies.clone(),
                        queries.clone(),
                        *num_tasks,
                        range_param,
                    ))?;
                    latencies.push(start.elapsed().into());
                    results.push(r);
                }

                let recalls = {
                    let merged = RangeSearchLocalResults::merge(&results[0])?;
                    utils::recall::compute_average_precision(merged.ids, &range_groundtruth)?
                };

                // Prepare results
                let this_result = RangeSearchResults::new(
                    RangeSearchResultsSetup {
                        num_tasks: num_tasks.get(),
                        initial_l: range_param.l_value(),
                    },
                    latencies,
                    results
                        .into_iter()
                        .map(|i| RangeSearchLocalResults::aggregate_latencies(&i))
                        .collect(),
                    recalls,
                )?;
                search_results.push(this_result);
            }
        }
    }

    Ok(search_results)
}

async fn run_range_search_parallel<T, S, DP>(
    index: Arc<DiskANNIndex<DP>>,
    strategies: Arc<Vec<S>>,
    queries: Arc<Matrix<T>>,
    num_tasks: NonZeroUsize,
    params: RangeSearchParams,
) -> anyhow::Result<Vec<RangeSearchLocalResults>>
where
    DP: DataProvider<Context = DefaultContext, ExternalId = u32, InternalId = u32>,
    S: glue::SearchStrategy<DP, [T]> + Clone + Sync + 'static,
    T: Send + Sync + 'static,
{
    // Plan the query partitions ahead of time.
    // Consider: do we want to do parallelism differently here due to the increased
    // potential for stragglers in range search?
    let partitions: Result<Vec<_>, _> = (0..num_tasks.get())
        .map(|task_id| async_tools::partition(queries.nrows(), num_tasks, task_id))
        .collect();
    let partitions = partitions?;

    let handles: Vec<_> = partitions
        .into_iter()
        .map(|range| {
            tokio::spawn(run_range_search_local(
                index.clone(),
                strategies.clone(),
                queries.clone(),
                range,
                params,
            ))
        })
        .collect();

    let mut results = Vec::new();
    for h in handles {
        results.push(h.await??);
    }

    // NOTE: Do not merge the results here because merging involves non-trivial overhead
    // due to memory copying which could influence what we're trying to measure.
    Ok(results)
}

async fn run_range_search_local<T, S, DP>(
    index: Arc<DiskANNIndex<DP>>,
    strategies: Arc<Vec<S>>,
    queries: Arc<Matrix<T>>,
    query_range: std::ops::Range<usize>,
    params: RangeSearchParams,
) -> anyhow::Result<RangeSearchLocalResults>
where
    DP: DataProvider<Context = DefaultContext, ExternalId = u32, InternalId = u32>,
    S: glue::SearchStrategy<DP, [T]> + Clone + Sync,
    T: Send + Sync + 'static,
{
    let mut ids = Vec::with_capacity(query_range.len());
    let mut latencies = Vec::<MicroSeconds>::with_capacity(query_range.len());

    let ctx = &DefaultContext;
    for i in query_range {
        let start = std::time::Instant::now();

        let (_, ans, _) = index
            .range_search(&strategies[i], ctx, queries.row(i), &params)
            .await?;

        ids.push(ans);

        latencies.push(start.elapsed().into());
    }

    Ok(RangeSearchLocalResults { ids, latencies })
}

// similar to SearchLocalResults, but ids can't be a matrix since the
// max result size is very variable
struct RangeSearchLocalResults {
    ids: Vec<Vec<u32>>,
    latencies: Vec<MicroSeconds>,
}

impl RangeSearchLocalResults {
    /// Merge the thread-local results of a parallel run into a single result.
    ///
    /// Returns an error if the number of columns in all the `ids` matrices are not equal or
    /// if `all.is_empty()`.
    fn merge(all: &[RangeSearchLocalResults]) -> anyhow::Result<Self> {
        match all.first() {
            Some(r) => r,
            None => {
                return Err(anyhow!(
                    "internal error: search local results should not be empty"
                ));
            }
        };

        let mut ids = Vec::<Vec<u32>>::new();
        for r in all.iter() {
            ids.extend_from_slice(&r.ids);
        }

        let latencies = Self::aggregate_latencies(all);
        Ok(RangeSearchLocalResults { ids, latencies })
    }

    fn aggregate_latencies(all: &[RangeSearchLocalResults]) -> Vec<MicroSeconds> {
        let mut latencies = Vec::new();

        // This isn't the most elegant way to do this, but it is simple.
        for r in all.iter() {
            latencies.extend_from_slice(&r.latencies);
        }

        latencies
    }
}

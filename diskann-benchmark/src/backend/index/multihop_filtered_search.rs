/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{num::NonZeroUsize, sync::Arc};

use diskann::{
    graph::{glue, index::QueryLabelProvider, search_output_buffer, DiskANNIndex, SearchParams},
    provider::{DataProvider, DefaultContext},
    utils::{async_tools, IntoUsize},
};
use diskann_benchmark_runner::utils::MicroSeconds;
use diskann_utils::views::{Matrix, MutMatrixView};

use crate::{
    backend::index::{
        result::{SearchResults, SearchResultsSetup},
        search::{
            no_postprocessing, InnerSearchParams, SearchLocalResults, SearchSteps, INVALID_NEIGHBOR,
        },
    },
    utils,
};

pub(super) fn run_multihop_search<T, S, DP, G>(
    index: Arc<DiskANNIndex<DP>>,
    strategies: Arc<Vec<S>>,
    queries: Arc<Matrix<T>>,
    groundtruth: G,
    steps: SearchSteps<'_>,
    labels: Arc<Vec<Arc<dyn QueryLabelProvider<u32>>>>,
) -> anyhow::Result<Vec<SearchResults>>
where
    DP: DataProvider<Context = DefaultContext, ExternalId = u32, InternalId = u32>,
    S: glue::SearchStrategy<DP, [T]> + Clone + Sync + 'static,
    T: Send + Sync + Clone + 'static,
    G: utils::recall::ComputeKnnRecall<u32> + utils::recall::KnnRecall<Item = u32> + Clone,
{
    let mut search_results = Vec::new();

    for num_tasks in steps.num_tasks {
        for run in steps.runs {
            if let Some(search_ls) = run.search_l.as_ref() {
                for search_l in search_ls.iter() {
                    let result = run_multihop_search_inner(
                        index.clone(),
                        strategies.clone(),
                        queries.clone(),
                        labels.clone(),
                        groundtruth.clone(),
                        InnerSearchParams {
                            search_l: *search_l,
                            run: run.clone(),
                            reps: steps.reps,
                            num_tasks: *num_tasks,
                        },
                        &mut no_postprocessing,
                    )?;
                    search_results.push(result);
                }
            }
        }
    }

    Ok(search_results)
}

/// Run a parallelized search for each query over the index using its corresponding
/// position-wise strategy.
///
/// After all searches are run and the results aggregated, pass the resulting IDs to the
/// post-processing function and then compute the recall.
fn run_multihop_search_inner<T, S, DP, G>(
    index: Arc<DiskANNIndex<DP>>,
    strategies: Arc<Vec<S>>,
    queries: Arc<Matrix<T>>,
    labels: Arc<Vec<Arc<dyn QueryLabelProvider<u32>>>>,
    groundtruth: G,
    params: InnerSearchParams,
    postprocess_results: &mut dyn FnMut(MutMatrixView<u32>) -> anyhow::Result<()>,
) -> anyhow::Result<SearchResults>
where
    DP: DataProvider<Context = DefaultContext, ExternalId = u32, InternalId = u32>,
    S: glue::SearchStrategy<DP, [T]> + Clone + Sync + 'static,
    T: Send + Sync + Clone + 'static,
    G: utils::recall::ComputeKnnRecall<u32> + utils::recall::KnnRecall<Item = u32> + Clone,
{
    let rt = utils::tokio::runtime(params.num_tasks.get())?;

    let mut latencies = Vec::<MicroSeconds>::new();
    let mut results = Vec::new();

    // Run on search as a warm-up and to get the baseline recall results.
    for _ in 0..params.reps.get() {
        let start = std::time::Instant::now();
        let r = rt.block_on(run_multihop_search_parallel(
            index.clone(),
            strategies.clone(),
            queries.clone(),
            params.num_tasks,
            params.run.search_n,
            params.search_l,
            labels.clone(),
        ))?;
        latencies.push(start.elapsed().into());
        results.push(r);
    }

    // Get recalls. Measure recall of `recall_k` groundtruth at `search_n` retrieved results.
    let recalls = {
        let mut merged = SearchLocalResults::merge(&results[0])?;
        postprocess_results(merged.ids.as_mut_view())?;
        groundtruth.compute_knn_recall(
            None,
            merged.ids.as_view().into(),
            params.run.recall_k,
            params.run.search_n,
            false,
            params.run.enhanced_metrics.unwrap_or(false),
        )?
    };

    let (query_latencies, cmps, hops) = results
        .into_iter()
        .map(|r| SearchLocalResults::aggregate_metrics(&r))
        .collect::<(Vec<_>, Vec<_>, Vec<_>)>();

    // Prepare results
    Ok(SearchResults::new(
        SearchResultsSetup {
            num_tasks: params.num_tasks.get(),
            search_n: params.run.search_n,
            search_l: params.search_l,
        },
        latencies,
        query_latencies,
        cmps,
        hops,
        recalls,
    )?)
}

async fn run_multihop_search_parallel<T, S, DP>(
    index: Arc<DiskANNIndex<DP>>,
    strategies: Arc<Vec<S>>,
    queries: Arc<Matrix<T>>,
    num_tasks: NonZeroUsize,
    search_n: usize,
    search_l: usize,
    labels: Arc<Vec<Arc<dyn QueryLabelProvider<u32>>>>,
) -> anyhow::Result<Vec<SearchLocalResults>>
where
    DP: DataProvider<Context = DefaultContext, ExternalId = u32, InternalId = u32>,
    S: glue::SearchStrategy<DP, [T]> + Clone + Sync + 'static,
    T: Send + Sync + 'static,
{
    // Plan the query partitions ahead of time.
    let partitions: Result<Vec<_>, _> = (0..num_tasks.get())
        .map(|task_id| async_tools::partition(queries.nrows(), num_tasks, task_id))
        .collect();
    let partitions = partitions?;

    let handles: Vec<_> = partitions
        .into_iter()
        .map(|range| {
            let index = index.clone();
            let strategies = strategies.clone();
            let queries = queries.clone();
            let labels = labels.clone();
            tokio::spawn(async move {
                run_multihop_search_local(
                    index, strategies, queries, range, search_n, search_l, labels,
                )
                .await
            })
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

async fn run_multihop_search_local<T, S, DP>(
    index: Arc<DiskANNIndex<DP>>,
    strategies: Arc<Vec<S>>,
    queries: Arc<Matrix<T>>,
    range: std::ops::Range<usize>,
    search_n: usize,
    search_l: usize,
    labels: Arc<Vec<Arc<dyn QueryLabelProvider<u32>>>>,
) -> anyhow::Result<SearchLocalResults>
where
    DP: DataProvider<Context = DefaultContext, ExternalId = u32, InternalId = u32>,
    S: glue::SearchStrategy<DP, [T]> + Clone + Sync,
    T: Send + Sync + 'static,
{
    let mut ids = Matrix::new(0, range.len(), search_n);
    let mut distances = vec![0.0; search_n];
    let mut latencies = Vec::<MicroSeconds>::with_capacity(range.len());
    let mut comparisons = Vec::<u32>::with_capacity(range.len());
    let mut hops = Vec::<u32>::with_capacity(range.len());

    let ctx = &DefaultContext;
    for (o, i) in range.enumerate() {
        let start = std::time::Instant::now();

        let ids = ids.row_mut(o);

        let stats = {
            let mut result_output_buffer =
                search_output_buffer::IdDistance::new(ids, &mut distances);

            index
                .multihop_search(
                    &strategies[i],
                    ctx,
                    queries.row(i),
                    &SearchParams::new_default(search_n, search_l)?,
                    &mut result_output_buffer,
                    &*labels[i],
                )
                .await?
        };
        let result_count = stats.result_count.into_usize();
        ids[result_count..].fill(INVALID_NEIGHBOR);
        latencies.push(start.elapsed().into());
        comparisons.push(stats.cmps);
        hops.push(stats.hops);
    }

    Ok(SearchLocalResults {
        ids,
        latencies,
        comparisons,
        hops,
    })
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    collections::HashMap,
    num::{NonZero, NonZeroUsize},
    sync::Arc,
};

use anyhow::anyhow;
use diskann::{
    graph::{
        glue,
        search::record::{NoopSearchRecord, RecallSearchRecord, SearchRecord},
        search_output_buffer, DiskANNIndex, SearchParams,
    },
    provider::{DataProvider, DefaultContext},
    utils::{async_tools, IntoUsize},
};
use diskann_benchmark_runner::utils::MicroSeconds;
use diskann_utils::views::{Matrix, MutMatrixView};

use crate::{
    backend::index::result::{SearchResults, SearchResultsSetup},
    inputs::async_::GraphSearch,
    utils,
};

// If an insufficient number of neighbors are returned by search, we use this sentinel
// value instead to opt-out of ID translation.
pub(crate) const INVALID_NEIGHBOR: u32 = u32::MAX;

pub(crate) fn no_postprocessing(_: MutMatrixView<u32>) -> anyhow::Result<()> {
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub(super) struct SearchSteps<'a> {
    pub reps: NonZeroUsize,
    pub num_tasks: &'a [NonZeroUsize],
    pub runs: &'a [GraphSearch],
}

impl<'a> SearchSteps<'a> {
    pub(super) fn new(
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

pub(crate) struct InnerSearchParams {
    pub(crate) search_l: usize,
    pub(crate) run: GraphSearch,
    pub(crate) reps: NonZero<usize>,
    pub(crate) num_tasks: NonZero<usize>,
}

/// Run a parallelized search for each query over the index using its corresponding
/// position-wiwe strategy.
///
/// After all searches are run and the results aggregated, pass the resulting IDs to the
/// post-processing function and then compute the recall.
fn run_search_inner<T, S, DP, G>(
    index: Arc<DiskANNIndex<DP>>,
    strategies: Arc<Vec<S>>,
    queries: Arc<Matrix<T>>,
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
        let r = rt.block_on(run_search_parallel(
            index.clone(),
            strategies.clone(),
            queries.clone(),
            params.num_tasks,
            params.run.search_n,
            params.search_l,
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

/// run a search with a fixed queue length
pub(crate) fn run_search_queue_based<T, S, DP, G>(
    index: Arc<DiskANNIndex<DP>>,
    strategies: Arc<Vec<S>>,
    queries: Arc<Matrix<T>>,
    groundtruth: G,
    steps: SearchSteps<'_>,
    postprocess_results: &mut dyn FnMut(MutMatrixView<u32>) -> anyhow::Result<()>,
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
                    let result = run_search_inner(
                        index.clone(),
                        strategies.clone(),
                        queries.clone(),
                        groundtruth.clone(),
                        InnerSearchParams {
                            search_l: *search_l,
                            run: run.clone(),
                            reps: steps.reps,
                            num_tasks: *num_tasks,
                        },
                        postprocess_results,
                    )?;
                    search_results.push(result);
                }
            }
        }
    }

    Ok(search_results)
}

/// run a search using a declarative recall. This will sample some queries to determine what search_l is needed to reach the target recall.
/// The TargetRecall object inside steps.runs contains 4 values: target, percentile, max_search_l, and calibration_size. These must all be set.
/// target defines how many of the ground truths we wish to find. percentile defines what percent of the queries we want above our target threshold.
/// max_search_l is the maximum queue length we will allow, and also how long our samples will search for.
/// calibration_size is the number of queries we will use to calculate our targets.
fn run_search_declarative<T, S, DP, G>(
    index: Arc<DiskANNIndex<DP>>,
    strategies: Arc<Vec<S>>,
    queries: Arc<Matrix<T>>,
    groundtruth: G,
    steps: SearchSteps<'_>,
) -> anyhow::Result<Vec<SearchResults>>
where
    DP: DataProvider<Context = DefaultContext, ExternalId = u32, InternalId = u32>,
    S: glue::SearchStrategy<DP, [T]> + Clone + Sync + 'static,
    T: Send + Sync + Clone + 'static,
    G: utils::recall::ComputeKnnRecall<u32> + utils::recall::KnnRecall<Item = u32> + Clone,
{
    let mut search_results: Vec<SearchResults> = Vec::new();
    for num_tasks in steps.num_tasks {
        let rt = utils::tokio::runtime(num_tasks.get())?;
        for run in steps.runs {
            if let Some(target_recalls) = run.target_recall.as_ref() {
                for target_recall in target_recalls.iter() {
                    // run sample queries to determine what search_l should be for our recall targets
                    let num_queries = target_recall.calibration_size.get().min(queries.nrows());
                    let search_length = target_recall.max_search_l.get();

                    // the records we will use to track the recall
                    let records = rt.block_on(sample_recall_parallel(
                        index.clone(),
                        strategies.clone(),
                        queries.clone(),
                        groundtruth.clone(),
                        InnerSearchParams {
                            search_l: search_length,
                            run: run.clone(),
                            reps: NonZero::new(num_queries).unwrap(),
                            num_tasks: *num_tasks,
                        },
                    ))?;

                    // iterate through our recall targets
                    for target in target_recall.target.iter() {
                        for percentile in target_recall.percentile.iter() {
                            // find the number of hops it took to get our recall target for each query
                            let mut hops = records
                                .iter()
                                .map(|r: &RecallSearchRecord<u32>| {
                                    for (recall, hops) in r.recall.iter().zip(r.hops.iter()) {
                                        // hops are sorted, so we can simply take the first passing item in the history
                                        if *recall >= *target {
                                            return *hops;
                                        }
                                    }
                                    *r.hops.last().unwrap()
                                })
                                .collect::<Vec<u32>>();

                            // sort the number of hops and select the percentile. This directly defines the search length
                            hops.sort_unstable();
                            let search_l = if let Some(ncols) = groundtruth.ncols() {
                                // For fixed-size groundtruth (like StridedView), apply the max constraint
                                hops[(hops.len() as f32 * percentile) as usize].max(ncols as u32)
                            } else {
                                // For variable-size groundtruth (like Vec<Vec<u32>>), no max constraint
                                hops[(hops.len() as f32 * percentile) as usize]
                            };

                            //run the search
                            let result = run_search_inner(
                                index.clone(),
                                strategies.clone(),
                                queries.clone(),
                                groundtruth.clone(),
                                InnerSearchParams {
                                    search_l: search_l as usize,
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
        }
    }

    Ok(search_results)
}

pub(super) fn run_search<T, S, DP, G>(
    index: Arc<DiskANNIndex<DP>>,
    strategies: Arc<Vec<S>>,
    queries: Arc<Matrix<T>>,
    groundtruth: G,
    steps: SearchSteps<'_>,
) -> anyhow::Result<Vec<SearchResults>>
where
    DP: DataProvider<Context = DefaultContext, ExternalId = u32, InternalId = u32>,
    S: glue::SearchStrategy<DP, [T]> + Clone + Sync + 'static,
    T: Send + Sync + Clone + 'static,
    G: utils::recall::ComputeKnnRecall<u32> + utils::recall::KnnRecall<Item = u32> + Clone,
{
    let mut search_results = Vec::new();

    // Run search with a hard coded l parameter
    let result = run_search_queue_based(
        index.clone(),
        strategies.clone(),
        queries.clone(),
        groundtruth.clone(),
        steps,
        &mut no_postprocessing,
    );
    search_results.append(&mut result?);

    // run search defined by reaching a recall target from a sample of the queries
    let result = run_search_declarative(
        index.clone(),
        strategies.clone(),
        queries.clone(),
        groundtruth,
        steps,
    );
    search_results.append(&mut result?);

    Ok(search_results)
}

/// Sample search queries with the recall search record to record correlations between hops and recall
async fn sample_recall_parallel<T, S, DP, G>(
    index: Arc<DiskANNIndex<DP>>,
    strategies: Arc<Vec<S>>,
    queries: Arc<Matrix<T>>,
    groundtruth: G,
    params: InnerSearchParams,
) -> anyhow::Result<Vec<RecallSearchRecord<u32>>>
where
    DP: DataProvider<Context = DefaultContext, ExternalId = u32, InternalId = u32>,
    S: glue::SearchStrategy<DP, [T]> + Clone + Sync + 'static,
    T: Send + Sync + Clone + 'static,
    G: utils::recall::ComputeKnnRecall<u32> + utils::recall::KnnRecall<Item = u32> + Clone,
{
    // Plan the query partitions ahead of time.
    let partitions: Result<Vec<_>, _> = (0..params.num_tasks.get())
        .map(|task_id| {
            async_tools::partition(
                params.reps.get().min(queries.nrows()),
                params.num_tasks,
                task_id,
            )
        })
        .collect();
    let partitions = partitions?;

    let handles: Vec<_> = partitions
        .into_iter()
        .map(|range| {
            let index = index.clone();
            let strategies = strategies.clone();
            let queries = queries.clone();
            let groundtruth = groundtruth.clone();
            let mut search_records = range
                .clone()
                .map(|i| {
                    // Extract groundtruth for this query index
                    let gt_row: Vec<u32> = groundtruth.row(i).to_vec();
                    RecallSearchRecord::new(params.search_l, gt_row)
                })
                .collect::<Vec<_>>();
            tokio::spawn(async move {
                _ = run_search_local(
                    index,
                    strategies,
                    queries,
                    range,
                    params.run.search_n,
                    params.search_l,
                    &mut search_records,
                )
                .await;
                search_records
            })
        })
        .collect();

    let mut results = Vec::new();
    for h in handles {
        results.push(h.await?);
    }

    Ok(results.into_iter().flatten().collect())
}

async fn run_search_parallel<T, S, DP>(
    index: Arc<DiskANNIndex<DP>>,
    strategies: Arc<Vec<S>>,
    queries: Arc<Matrix<T>>,
    num_tasks: NonZeroUsize,
    search_n: usize,
    search_l: usize,
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
            let mut search_records = (0..range.len())
                .map(|_| NoopSearchRecord::new())
                .collect::<Vec<_>>();
            tokio::spawn(async move {
                run_search_local(
                    index,
                    strategies,
                    queries,
                    range,
                    search_n,
                    search_l,
                    &mut search_records,
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

async fn run_search_local<T, S, DP, SR>(
    index: Arc<DiskANNIndex<DP>>,
    strategies: Arc<Vec<S>>,
    queries: Arc<Matrix<T>>,
    range: std::ops::Range<usize>,
    search_n: usize,
    search_l: usize,
    search_records: &mut [SR],
) -> anyhow::Result<SearchLocalResults>
where
    DP: DataProvider<Context = DefaultContext, ExternalId = u32, InternalId = u32>,
    S: glue::SearchStrategy<DP, [T]> + Clone + Sync,
    T: Send + Sync + 'static,
    SR: SearchRecord<DP::InternalId> + Sized,
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
                .search_recorded(
                    &strategies[i],
                    ctx,
                    queries.row(i),
                    &SearchParams::new_default(search_n, search_l)?,
                    &mut result_output_buffer,
                    &mut search_records[o],
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

#[derive(Debug)]
pub(crate) struct SearchLocalResults {
    pub(crate) ids: Matrix<u32>,
    pub(crate) latencies: Vec<MicroSeconds>,
    pub(crate) comparisons: Vec<u32>,
    pub(crate) hops: Vec<u32>,
}

impl SearchLocalResults {
    /// Merge the thread-local results of a parallel run into a single result.
    ///
    /// Returns an error if the number of columns in all the `ids` matrices are not equal or
    /// if `all.is_empty()`.
    pub(crate) fn merge(all: &[SearchLocalResults]) -> anyhow::Result<Self> {
        let first = match all.first() {
            Some(r) => r,
            None => {
                return Err(anyhow!(
                    "internal error: search local results should not be empty"
                ));
            }
        };

        let num_ids = first.ids.ncols();
        let mut total_results = 0;
        for (i, r) in all.iter().enumerate() {
            if r.ids.ncols() != num_ids {
                return Err(anyhow!(
                    "internal error: result batch {} has {} cols when {} were expected",
                    i,
                    r.ids.ncols(),
                    num_ids
                ));
            }
            total_results += r.ids.nrows();
        }

        let mut ids = Matrix::new(0, total_results, num_ids);
        // This isn't the most elegant way to do this, but it is simple.
        let mut output_row = 0;
        for r in all.iter() {
            for input_row in r.ids.row_iter() {
                ids.row_mut(output_row).copy_from_slice(input_row);
                output_row += 1;
            }
        }

        let (latencies, comparisons, hops) = Self::aggregate_metrics(all);

        Ok(SearchLocalResults {
            ids,
            latencies,
            comparisons,
            hops,
        })
    }

    pub(crate) fn aggregate_metrics(
        all: &[SearchLocalResults],
    ) -> (Vec<MicroSeconds>, Vec<u32>, Vec<u32>) {
        let mut latencies = Vec::new();
        let mut comparisons = Vec::new();
        let mut hops = Vec::new();

        // This isn't the most elegant way to do this, but it is simple.
        for r in all.iter() {
            latencies.extend_from_slice(&r.latencies);
            comparisons.extend_from_slice(&r.comparisons);
            hops.extend_from_slice(&r.hops);
        }

        (latencies, comparisons, hops)
    }
}

/// Translate slot IDs to **tags** using the provided mapping.
///
/// This is currently used by the dynamic index to work-around DiskANN's lack of internal
/// support for internal/external ID mapping and is provided as the `postprocess_results`
/// argument to `run_search_inner` for the dynamic methods to do the final ID translation.
pub(crate) fn translate_ids(
    mut ids: MutMatrixView<u32>,
    mapping: &HashMap<u32, usize>,
) -> anyhow::Result<()> {
    ids.as_mut_slice().iter_mut().try_for_each(|slot_id| {
        if *slot_id == INVALID_NEIGHBOR {
            Ok(())
        } else if let Some(&tag) = mapping.get(slot_id) {
            *slot_id = tag as u32;
            Ok(())
        } else {
            Err(anyhow!(
                "Slot ID {} not found in slot-to-tag mapping",
                slot_id
            ))
        }
    })
}

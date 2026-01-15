/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{io::Write, sync::Arc};

use anyhow::anyhow;
use diskann::{
    graph::{
        glue::{self, PruneStrategy},
        DiskANNIndex, InplaceDeleteMethod, SampleableForStart,
    },
    provider::{self, DataProvider, DefaultContext, Delete},
    utils::async_tools,
};
use diskann_benchmark_runner::utils::{percentiles, MicroSeconds};
use diskann_providers::model::graph::provider::async_::inmem::DefaultProvider;
use diskann_utils::views::Matrix;
use serde::Serialize;

use crate::utils::{
    self,
    datafiles::{UpdateOperationType, UpdateStage},
    streaming::{DynamicConfig, TagSlotManager},
};

////////////
// Update //
////////////

#[derive(Debug, Serialize)]
pub(super) struct UpdateResults {
    pub(super) num_tasks: usize,
    pub(super) operation: UpdateOperationType,
    pub(super) insert_l: Option<usize>,
    pub(super) batch_latency: MicroSeconds,
    pub(super) qps: f64,
    pub(super) mean_latency: f64,
    pub(super) p90_latency: MicroSeconds,
    pub(super) p99_latency: MicroSeconds,
    // Background cleaning for deletes
    pub(super) mean_bg_latency: f64,
    pub(super) p90_bg_latency: MicroSeconds,
    pub(super) p99_bg_latency: MicroSeconds,
}

#[derive(Debug, Serialize)]
pub(super) struct RunbookUpdateStageResults {
    pub(super) stage_idx: i64,
    pub(super) results: UpdateResults,
}

impl std::fmt::Display for UpdateResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Update Results ({:?}):", self.operation)?;
        writeln!(f, "  Num tasks: {}", self.num_tasks)?;
        if let Some(insert_l) = self.insert_l {
            writeln!(f, "  Insert L: {}", insert_l)?;
        }
        writeln!(f, "  Batch latency: {}", self.batch_latency)?;
        writeln!(f, "  QPS: {:.2}", self.qps)?;
        writeln!(f, "  Mean latency: {:.2}us", self.mean_latency)?;
        writeln!(f, "  P90 latency: {}", self.p90_latency)?;
        writeln!(f, "  P99 latency: {}", self.p99_latency)?;
        writeln!(
            f,
            "  Mean BG cleaning latency: {:.2}us",
            self.mean_bg_latency
        )?;
        writeln!(f, "  P90 BG cleaning latency: {}", self.p90_bg_latency)?;
        write!(f, "  P99 BG cleaning latency: {}", self.p99_bg_latency)
    }
}

fn format_update_results_table<F>(
    f: &mut std::fmt::Formatter<'_>,
    results: &[UpdateResults],
    batch_formatter: Option<F>,
) -> std::fmt::Result
where
    F: Fn(usize) -> String,
{
    if results.is_empty() {
        return Ok(());
    }

    let has_batch = batch_formatter.is_some();
    let headers: &[_] = if has_batch {
        &[
            "Batch",
            "Operation",
            "Threads",
            "Insert L",
            "QPS",
            "Avg Latency",
            "P90 Latency",
            "P99 Latency",
            "BG Avg",
            "BG P90",
            "BG P99",
        ]
    } else {
        &[
            "Operation",
            "Threads",
            "Insert L",
            "QPS",
            "Avg Latency",
            "P90 Latency",
            "P99 Latency",
            "BG Avg",
            "BG P90",
            "BG P99",
        ]
    };

    let mut table = diskann_benchmark_runner::utils::fmt::Table::new(headers, results.len());
    results.iter().enumerate().for_each(|(i, r)| {
        let mut row = table.row(i);
        let mut col_idx = 0;

        if let Some(ref formatter) = batch_formatter {
            row.insert(formatter(i), col_idx);
            col_idx += 1;
        }

        row.insert(format!("{:?}", r.operation), col_idx);
        row.insert(r.num_tasks, col_idx + 1);
        if let Some(l) = r.insert_l {
            row.insert(l, col_idx + 2);
        } else {
            row.insert("N/A", col_idx + 2);
        };
        row.insert(format!("{:.1}", r.qps), col_idx + 3);
        row.insert(format!("{:.1}us", r.mean_latency), col_idx + 4);
        row.insert(r.p90_latency, col_idx + 5);
        row.insert(r.p99_latency, col_idx + 6);
        row.insert(format!("{:.1}us", r.mean_bg_latency), col_idx + 7);
        row.insert(r.p90_bg_latency, col_idx + 8);
        row.insert(r.p99_bg_latency, col_idx + 9);
    });

    write!(f, "{}", table)
}

impl std::fmt::Display for utils::DisplayWrapper<'_, [UpdateResults]> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        format_update_results_table(f, self, None::<fn(usize) -> String>)
    }
}

impl std::fmt::Display for utils::DisplayWrapper<'_, [RunbookUpdateStageResults]> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            return Ok(());
        }

        let headers = [
            "Batch",
            "Operation",
            "Threads",
            "Insert L",
            "QPS",
            "Avg Latency",
            "P90 Latency",
            "P99 Latency",
            "BG Avg",
            "BG P90",
            "BG P99",
        ];

        let mut table = diskann_benchmark_runner::utils::fmt::Table::new(headers, self.len());
        self.iter().enumerate().for_each(|(row, iur)| {
            let mut row = table.row(row);
            let r = &iur.results;
            row.insert(iur.stage_idx, 0);
            row.insert(format!("{:?}", r.operation), 1);
            row.insert(r.num_tasks, 2);
            if let Some(l) = r.insert_l {
                row.insert(l, 3)
            } else {
                row.insert("N/A", 3)
            };
            row.insert(format!("{:.1}", r.qps), 4);
            row.insert(format!("{:.1}us", r.mean_latency), 5);
            row.insert(r.p90_latency, 6);
            row.insert(r.p99_latency, 7);
            row.insert(format!("{:.1}us", r.mean_bg_latency), 8);
            row.insert(r.p90_bg_latency, 9);
            row.insert(r.p99_bg_latency, 10);
        });

        table.fmt(f)
    }
}

struct UpdateResultsSetup {
    num_tasks: usize,
    operation: UpdateOperationType,
    insert_l: Option<usize>,
}

/// Helper functions to get start and end id values based on operation type
/// These ids are the sequential IDs of the corresponding vector data in
/// the entire input data matrix (watch out for Replace operations)
fn get_operation_tag_range(update_stage: &UpdateStage) -> anyhow::Result<(usize, usize)> {
    match update_stage.operation {
        UpdateOperationType::Insert | UpdateOperationType::Delete => {
            match (update_stage.start, update_stage.end) {
                (Some(start), Some(end)) => {
                    if end <= start {
                        return Err(anyhow!(
                            "Invalid update range: end (non-inclusive) ({}) must be > start ({})",
                            end,
                            start
                        ));
                    }
                    Ok((start, end))
                }
                _ => Err(anyhow!(
                    "Insert and Delete operations must have start and end fields defined"
                )),
            }
        }
        UpdateOperationType::Replace => match (update_stage.tags_start, update_stage.tags_end) {
            (Some(tags_start), Some(tags_end)) => {
                if tags_end <= tags_start {
                    return Err(anyhow!(
                        "Invalid update range: end (non-inclusive) ({}) must be > start ({})",
                        tags_end,
                        tags_start
                    ));
                }
                Ok((tags_start, tags_end))
            }
            _ => Err(anyhow!(
                "Replace operations must have tags_start/tags_end defined"
            )),
        },
        UpdateOperationType::Search => {
            Err(anyhow!("Search operations should be handled separately"))
        }
    }
}

impl UpdateResults {
    fn new(
        setup: UpdateResultsSetup,
        batch_latency: MicroSeconds,
        mut update_query_latencies: Vec<MicroSeconds>,
        mut background_cleaning_latencies: Vec<MicroSeconds>,
    ) -> Result<Self, percentiles::CannotBeEmpty> {
        // Compute QPS from `update_query_latencies`.
        let num_queries = update_query_latencies.len() as f64;
        let qps = num_queries / batch_latency.as_seconds();

        let update_query_stats = percentiles::compute_percentiles(&mut update_query_latencies)?;

        let (mean_bg_latency, p90_bg_latency, p99_bg_latency): (f64, MicroSeconds, MicroSeconds);
        if !background_cleaning_latencies.is_empty() {
            let background_cleaning_stats =
                percentiles::compute_percentiles(&mut background_cleaning_latencies)?;

            mean_bg_latency = background_cleaning_stats.mean;
            p90_bg_latency = background_cleaning_stats.p90;
            p99_bg_latency = background_cleaning_stats.p99;
        } else {
            (mean_bg_latency, p90_bg_latency, p99_bg_latency) = (
                0.0,
                MicroSeconds::from(std::time::Duration::ZERO),
                MicroSeconds::from(std::time::Duration::ZERO),
            );
        }
        let UpdateResultsSetup {
            num_tasks,
            operation,
            insert_l,
        } = setup;

        Ok(Self {
            num_tasks,
            operation,
            insert_l,
            batch_latency,
            qps,
            mean_latency: update_query_stats.mean,
            p90_latency: update_query_stats.p90,
            p99_latency: update_query_stats.p99,
            mean_bg_latency,
            p90_bg_latency,
            p99_bg_latency,
        })
    }
}

/// num_tasks is num_update_threads specified in the worload json file.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_update<T, S, SI, D, F, Q, C>(
    index: Arc<DiskANNIndex<DefaultProvider<F, Q, C>>>,
    config: &DynamicConfig<'_, S, SI, D>,
    update_stage: UpdateStage,
    insert_queries: Option<Arc<Matrix<T>>>,
    max_capacity: usize,
    bookkeeping: &mut TagSlotManager,
    mut output: &mut dyn diskann_benchmark_runner::output::Output,
) -> anyhow::Result<UpdateResults>
where
    DefaultProvider<F, Q, C>: DataProvider<Context = DefaultContext, ExternalId = u32, InternalId = u32>
        + provider::SetElement<[T]>
        + provider::Delete,
    S: glue::SearchStrategy<DefaultProvider<F, Q, C>, [T]> + Clone + Sync,
    SI: glue::InsertStrategy<DefaultProvider<F, Q, C>, [T]> + Clone + Sync,
    D: glue::InplaceDeleteStrategy<DefaultProvider<F, Q, C>> + Clone + Sync,
    T: Send + Sync + 'static + SampleableForStart + std::fmt::Debug + Clone,
{
    let num_tasks = config.num_update_threads.get();
    let insert_l = config.insert_l.get();
    let inplace_delete_method = config.inplace_delete_method();
    let num_to_replace = config.num_to_replace();
    let consolidate_threshold = config.consolidate_threshold();
    let rt = utils::tokio::runtime(num_tasks)?;

    let mut n_active_pts = max_capacity - bookkeeping.empty_slots.len();
    if update_stage.operation == UpdateOperationType::Insert {
        let (tags_start, tags_end) = get_operation_tag_range(&update_stage)?;
        n_active_pts += tags_end - tags_start;
    }

    // Track cleaning results
    let mut cleaning_results = Vec::new();

    writeln!(
        output,
        "Checking for background cleaning: deleted_slots len {}, n active points {}, threshold {}, empty_slots len {}, capacity {}",
        bookkeeping.deleted_slots.len(),
        n_active_pts,
        consolidate_threshold,
        bookkeeping.empty_slots.len(),
        max_capacity
    )?;

    if bookkeeping.deleted_slots.len()
        > (n_active_pts as f32 * consolidate_threshold).ceil() as usize
        || n_active_pts > max_capacity
    {
        // Print cleaning information
        writeln!(output, "Triggering background cleaning:")?;
        writeln!(output, "  Max capacity: {}", max_capacity)?;
        writeln!(output, "  Empty slots: {}", bookkeeping.empty_slots.len())?;
        writeln!(
            output,
            "  Deleted slots: {}",
            bookkeeping.deleted_slots.len()
        )?;
        writeln!(output, "  Active points: {}", n_active_pts)?;
        writeln!(output, "  Consolidate threshold: {}", consolidate_threshold)?;

        // Call the parallel cleaning procedure
        let cleaning_start = std::time::Instant::now();
        cleaning_results = rt.block_on(run_cleaning_parallel(
            index.clone(),
            config.delete_strategy.clone(),
            num_tasks,
        ))?;
        let cleaning_elapsed = cleaning_start.elapsed();

        writeln!(
            output,
            "  Background cleaning completed in {:?}",
            cleaning_elapsed
        )?;

        // Clear the deleted_slots set after cleaning
        bookkeeping.consolidate();
    }

    // Generate target slot IDs based on operation type (before timing measurement)
    let slots: Vec<u32> = match update_stage.operation {
        UpdateOperationType::Insert => {
            // For inserts, allocate new slots from empty_slots but don't update mappings yet
            let (start_tag, end_tag) = get_operation_tag_range(&update_stage)?;
            let total_size = end_tag - start_tag;
            bookkeeping.get_n_empty_slots(total_size)?
        }
        UpdateOperationType::Delete | UpdateOperationType::Replace => {
            // For deletes, use existing slots from tag_to_slot mapping but don't update mappings yet
            let (start_tag, end_tag) = get_operation_tag_range(&update_stage)?;
            bookkeeping.find_slots_by_tags(start_tag..end_tag)?
        }
        UpdateOperationType::Search => {
            return Err(anyhow!("Search operations should be handled separately"));
        }
    };

    // Run the update operations (timing starts here)
    let start = std::time::Instant::now();
    let results = rt.block_on(run_update_parallel(
        index.clone(),
        config.insert_strategy.clone(),
        config.delete_strategy.clone(),
        insert_queries,
        update_stage.clone(),
        slots.clone(),
        num_tasks,
        inplace_delete_method,
        num_to_replace,
    ))?;
    let batch_latency = start.elapsed().into();

    // Update slot bookkeeping only after successful completion
    match update_stage.operation {
        UpdateOperationType::Insert => {
            let (start_tag, end_tag) = get_operation_tag_range(&update_stage)?;
            bookkeeping.assign_slots_to_tags(start_tag..end_tag, slots)?;
        }
        UpdateOperationType::Delete => {
            let (start_tag, end_tag) = get_operation_tag_range(&update_stage)?;
            bookkeeping.mark_tags_deleted(start_tag..end_tag)?;
        }
        UpdateOperationType::Replace => {
            // The compute groundtruth script in diskann rust uses the **tags**.
            // tags stay the same after replace; nothing needs to happen.
        }
        UpdateOperationType::Search => {
            return Err(anyhow!("Search operations should be handled separately"));
        }
    }

    // Prepare results
    let merged_result = UpdateResults::new(
        UpdateResultsSetup {
            num_tasks,
            operation: update_stage.operation,
            insert_l: Some(insert_l),
        },
        batch_latency,
        UpdateLocalResults::aggregate_latencies(&results),
        UpdateLocalResults::aggregate_background_cleaning_latencies(&cleaning_results),
    )?;

    Ok(merged_result)
}

/// Run background cleaning operations in parallel across multiple threads collaboratively.
async fn run_cleaning_parallel<S, U, V, D>(
    index: Arc<DiskANNIndex<DefaultProvider<U, V, D>>>,
    delete_strategy: S,
    num_tasks: usize,
) -> anyhow::Result<Vec<UpdateLocalResults>>
where
    DefaultProvider<U, V, D>: provider::Delete
        + DataProvider<Context = DefaultContext, ExternalId = u32, InternalId = u32>,
    S: glue::InplaceDeleteStrategy<DefaultProvider<U, V, D>> + Clone + Sync,
{
    let num_tasks = match std::num::NonZeroUsize::new(num_tasks) {
        Some(tasks) => tasks,
        None => anyhow::bail!("number of tasks must be nonzero"),
    };

    let iter = async_tools::PartitionIter::new(index.provider().total_points(), num_tasks);
    let handles: Vec<_> = iter
        .clone()
        .map(|cleaning_range| {
            tokio::spawn(run_cleaning_local(
                index.clone(),
                delete_strategy.clone(),
                cleaning_range.clone(),
            ))
        })
        .collect();

    let mut results = Vec::new();
    for h in handles {
        results.push(h.await??);
    }

    let release_handles: Vec<_> = iter
        .map(|cleaning_range| {
            let index_clone = index.clone();
            tokio::spawn(async move {
                for id in cleaning_range {
                    let id: u32 = id.try_into().unwrap();
                    if index_clone
                        .provider()
                        .status_by_external_id(&DefaultContext, &id)
                        .await
                        .expect("in mem-provider should always succeed in translation")
                        .is_deleted()
                    {
                        index_clone
                            .provider()
                            .release(&DefaultContext, id)
                            .await
                            .expect("Releasing a deleted node should succeed.");
                    }
                }
                Ok::<(), anyhow::Error>(())
            })
        })
        .collect();
    for h in release_handles {
        h.await??;
    }

    Ok(results)
}

/// The caller of insert/replace (via insert)/inplace_delete directly specifies the target slot IDs (i.e. internal IDs)
#[allow(clippy::too_many_arguments)]
async fn run_update_parallel<T, S, D, DP>(
    index: Arc<DiskANNIndex<DP>>,
    insert_strategy: S,
    delete_strategy: D,
    insert_queries: Option<Arc<Matrix<T>>>,
    update_stage: UpdateStage,
    target_slots: Vec<u32>,
    num_tasks: usize,
    inplace_delete_method: InplaceDeleteMethod,
    num_to_replace: usize,
) -> anyhow::Result<Vec<UpdateLocalResults>>
where
    DP: DataProvider<Context = DefaultContext, ExternalId = u32, InternalId = u32>
        + provider::SetElement<[T]>
        + provider::Delete,
    S: glue::InsertStrategy<DP, [T]> + Clone + Sync,
    D: glue::InplaceDeleteStrategy<DP> + Clone + Sync,
    T: Send + Sync + 'static + SampleableForStart + std::fmt::Debug,
{
    // Validate that non-search operations have appropriate range fields defined
    let (ids_start, ids_end) = match update_stage.operation {
        // For Delete, it is not actually the id here (but tag), but we will not need to index into `insert_queries`
        // with it
        UpdateOperationType::Insert | UpdateOperationType::Delete => {
            get_operation_tag_range(&update_stage)?
        }
        UpdateOperationType::Replace => match (update_stage.ids_start, update_stage.ids_end) {
            (Some(ids_start), Some(ids_end)) => (ids_start, ids_end),
            _ => {
                return Err(anyhow!(
                    "Replace operations must have ids_start/ids_end fields defined"
                ));
            }
        },
        UpdateOperationType::Search => {
            return Err(anyhow!("Search operations should be handled separately"));
        }
    };

    // Plan the query partitions ahead of time
    let total_size = ids_end - ids_start;
    let partitions: Result<Vec<_>, _> = (0..num_tasks)
        .map(|task_id| async_tools::partition(total_size, num_tasks.try_into().unwrap(), task_id))
        .map(|partition| partition.map(|range| (range.start + ids_start)..(range.end + ids_start)))
        .collect();
    let partitions = partitions?;
    let i_queries = insert_queries.unwrap();

    let handles: Vec<_> = partitions
        .into_iter()
        .map(|worker_ids_range| {
            let start_idx = worker_ids_range.start - ids_start;
            let end_idx = worker_ids_range.end - ids_start;
            let worker_slots = target_slots[start_idx..end_idx].to_vec();

            match update_stage.operation {
                UpdateOperationType::Insert | UpdateOperationType::Replace => {
                    tokio::spawn(run_insert_local(
                        index.clone(),
                        insert_strategy.clone(),
                        i_queries.clone(),
                        worker_ids_range,
                        worker_slots,
                    ))
                }
                UpdateOperationType::Delete => tokio::spawn(run_delete_local::<D, DP>(
                    index.clone(),
                    delete_strategy.clone(),
                    worker_slots,
                    inplace_delete_method,
                    num_to_replace,
                )),
                UpdateOperationType::Search => tokio::spawn(async {
                    Err(anyhow!("Search operations should be handled separately"))
                }),
            }
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

/// Run cleaning operations for a local range of nodes, dropping deleted neighbors.
async fn run_cleaning_local<D, DP>(
    index: Arc<DiskANNIndex<DP>>,
    delete_strategy: D,
    cleaning_range: std::ops::Range<usize>,
) -> anyhow::Result<UpdateLocalResults>
where
    DP: DataProvider<Context = DefaultContext, ExternalId = u32, InternalId = u32>
        + provider::Delete,
    D: glue::InplaceDeleteStrategy<DP> + Clone + Sync,
{
    let mut cleaning_latencies = Vec::<MicroSeconds>::new();
    let ctx = &DefaultContext;

    let bg_start = std::time::Instant::now();
    // Get prune accessor from the delete strategy - this implements AsNeighborMut
    let prune_strategy = delete_strategy.prune_strategy();
    let mut accessor = prune_strategy.prune_accessor(index.provider(), ctx)?;

    for id in cleaning_range {
        index
            .drop_deleted_neighbors(
                ctx,
                &mut accessor,
                id.try_into().unwrap(),
                false, /* only_orphans */
            )
            .await
            .unwrap();
    }
    cleaning_latencies.push(bg_start.elapsed().into());

    Ok(UpdateLocalResults {
        latencies: Vec::new(), // No update latencies for cleaning-only operations
        background_cleaning_latencies: Some(cleaning_latencies),
    })
}

/// Run insert operations for a local batch.
/// REQUIRES: range is within the update_stage of caller and slots.len() == range.len()
async fn run_insert_local<T, S, DP>(
    index: Arc<DiskANNIndex<DP>>,
    insert_strategy: S,
    insert_queries: Arc<Matrix<T>>,
    range: std::ops::Range<usize>,
    slots: Vec<u32>,
) -> anyhow::Result<UpdateLocalResults>
where
    DP: DataProvider<Context = DefaultContext, ExternalId = u32, InternalId = u32>
        + provider::SetElement<[T]>,
    S: glue::InsertStrategy<DP, [T]> + Clone + Sync,
    T: Send + Sync + 'static,
{
    let mut latencies = Vec::<MicroSeconds>::with_capacity(range.len());

    let ctx = &DefaultContext;
    for (id, slot) in range.clone().zip(slots.iter()) {
        let start = std::time::Instant::now();
        index
            .insert(insert_strategy.clone(), ctx, slot, insert_queries.row(id))
            .await?;
        latencies.push(start.elapsed().into());
    }
    Ok(UpdateLocalResults {
        latencies,
        background_cleaning_latencies: None,
    })
}

/// Run delete operations for a local batch.
async fn run_delete_local<D, DP>(
    index: Arc<DiskANNIndex<DP>>,
    delete_strategy: D,
    slots: Vec<u32>,
    inplace_delete_method: InplaceDeleteMethod,
    num_to_replace: usize,
) -> anyhow::Result<UpdateLocalResults>
where
    DP: DataProvider<Context = DefaultContext, ExternalId = u32, InternalId = u32>
        + provider::Delete,
    D: glue::InplaceDeleteStrategy<DP> + Clone + Sync,
{
    let mut latencies = Vec::<MicroSeconds>::with_capacity(slots.len());
    let ctx = &DefaultContext;

    for slot in &slots {
        let start = std::time::Instant::now();
        index
            .inplace_delete(
                delete_strategy.clone(),
                ctx,
                slot,
                num_to_replace,
                inplace_delete_method,
            )
            .await?;
        latencies.push(start.elapsed().into());
    }

    Ok(UpdateLocalResults {
        latencies,
        background_cleaning_latencies: None, // No cleaning done in delete operations
    })
}

struct UpdateLocalResults {
    latencies: Vec<MicroSeconds>,
    background_cleaning_latencies: Option<Vec<MicroSeconds>>,
}
impl UpdateLocalResults {
    fn aggregate_latencies(all: &[UpdateLocalResults]) -> Vec<MicroSeconds> {
        let mut latencies = Vec::new();
        for result in all {
            latencies.extend(result.latencies.clone());
        }
        latencies
    }

    fn aggregate_background_cleaning_latencies(all: &[UpdateLocalResults]) -> Vec<MicroSeconds> {
        let mut latencies = Vec::new();
        for result in all {
            if let Some(bg_latencies) = &result.background_cleaning_latencies {
                latencies.extend(bg_latencies.clone());
            }
        }
        latencies
    }
}

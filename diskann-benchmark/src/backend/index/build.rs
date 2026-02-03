/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

use diskann::{
    error::DiskANNError::StartPointComputeError,
    graph::{glue, DiskANNIndex, SampleableForStart, StartPointStrategy},
    provider::{self, DataProvider, DefaultContext},
    utils::async_tools,
    ANNError, ANNResult,
};
use diskann_benchmark_runner::{
    output::Output,
    utils::{percentiles, MicroSeconds},
};
use diskann_providers::{
    self,
    model::{
        configuration::IndexConfiguration,
    },
    storage::{AsyncIndexMetadata, LoadWith, SaveWith},
};
use diskann_inmem::{DefaultProvider, SetStartPoints};
use diskann_utils::{
    future::AsyncFriendly,
    views::{Matrix, MatrixView},
};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;

///////////////////////////////
// Start Point Configuration //
///////////////////////////////

pub(super) fn set_start_points<U, V, D, T>(
    provider: &DefaultProvider<U, V, D>,
    data: MatrixView<'_, T>,
    start_strategy: StartPointStrategy,
) -> ANNResult<()>
where
    DefaultProvider<U, V, D>:
        DataProvider<Context = DefaultContext, ExternalId = u32> + SetStartPoints<[T]>,
    T: diskann::graph::SampleableForStart
        + std::fmt::Debug
        + AsyncFriendly
        + diskann_utils::sampling::WithApproximateNorm,
{
    let start_points = start_strategy
        .compute(data)
        .map_err(|e| ANNError::new(diskann::ANNErrorKind::DiskANN(StartPointComputeError), e))?;
    debug_assert!(start_points.nrows() == provider.num_start_points());
    provider.set_start_points(start_points.row_iter())
}

///////////////////////////////////////////
// Build via the Single Insert Interface //
///////////////////////////////////////////

#[derive(Debug, Serialize)]
pub(super) struct SingleInsertBuildStats {
    total_time: MicroSeconds,
    vectors_inserted: usize,
    insert_latencies: percentiles::Percentiles<MicroSeconds>,
}

impl std::fmt::Display for SingleInsertBuildStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Index Build Time: {}s", self.total_time.as_seconds())?;
        write!(
            f,
            "Insert Latencies:\n  average: {}us\n      p90: {}\n      p99: {}\n\n",
            self.insert_latencies.mean, self.insert_latencies.p90, self.insert_latencies.p99,
        )
    }
}

/// Run async index build to completion.
pub(super) async fn build_single_insert<T, S, DP>(
    index: Arc<DiskANNIndex<DP>>,
    strategy: S,
    data: Arc<Matrix<T>>,
    num_tasks: usize,
    output: &mut dyn Output,
) -> anyhow::Result<SingleInsertBuildStats>
where
    DP: DataProvider<Context = DefaultContext, ExternalId = u32> + provider::SetElement<[T]>,
    S: glue::InsertStrategy<DP, [T]> + Clone + Sync,
    T: Send + Sync + 'static + SampleableForStart + std::fmt::Debug,
{
    // A cancellation flag to shut down all tasks if one fails.
    let block = ControlBlock::new(data, output.draw_target())?;
    let start = std::time::Instant::now();
    let tasks: Vec<_> = (0..num_tasks)
        .map(|_| {
            let block_clone = block.clone();
            let index_clone = index.clone();
            let strategy_clone = strategy.clone();
            tokio::spawn(async move {
                let mut latencies = Vec::<MicroSeconds>::new();
                loop {
                    let ctx = &DefaultContext;
                    let result = match block_clone.next() {
                        Some((id, data)) => {
                            let start = std::time::Instant::now();
                            let result = index_clone
                                .insert(
                                    strategy_clone.clone(),
                                    ctx,
                                    &(id.try_into().unwrap()),
                                    data,
                                )
                                .await;

                            // Note on the primitive cast: it is *highly* unlikely that an
                            // insert runs long enough to overfloa a `u64`.
                            latencies.push(start.elapsed().into());
                            result
                        }
                        None => return Ok(latencies),
                    };
                    match result {
                        Ok(()) => {}
                        Err(err) => {
                            block_clone.cancel();
                            return Err(err);
                        }
                    };
                }
            })
        })
        .collect();

    // Collect all the tasks.
    let mut task_latencies = Vec::new();
    for task in tasks {
        task_latencies.push(task.await??);
    }
    let total_time: MicroSeconds = start.elapsed().into();

    let mut insert_latencies = Vec::<MicroSeconds>::new();
    task_latencies
        .into_iter()
        .for_each(|l| insert_latencies.extend_from_slice(&l));

    let insert_latencies = percentiles::compute_percentiles(&mut insert_latencies)?;

    Ok(SingleInsertBuildStats {
        total_time,
        vectors_inserted: block.data.nrows(),
        insert_latencies,
    })
}

fn make_progress_bar(
    nrows: usize,
    draw_target: indicatif::ProgressDrawTarget,
) -> anyhow::Result<ProgressBar> {
    let progress = ProgressBar::with_draw_target(Some(nrows as u64), draw_target);
    progress.set_style(ProgressStyle::with_template(
        "Building [{elapsed_precise}] {wide_bar} {percent}",
    )?);
    Ok(progress)
}

#[derive(Debug)]
struct ControlBlock<T> {
    data: Arc<Matrix<T>>,
    position: AtomicUsize,
    cancel: AtomicBool,
    progress: ProgressBar,
}

impl<T> ControlBlock<T> {
    fn new(
        data: Arc<Matrix<T>>,
        draw_target: indicatif::ProgressDrawTarget,
    ) -> anyhow::Result<Arc<Self>> {
        let nrows = data.nrows();
        Ok(Arc::new(Self {
            data,
            position: AtomicUsize::new(0),
            cancel: AtomicBool::new(false),
            progress: make_progress_bar(nrows, draw_target)?,
        }))
    }

    /// Return the next data to insert.
    fn next(&self) -> Option<(usize, &[T])> {
        let cancel = self.cancel.load(Ordering::Relaxed);
        if cancel {
            None
        } else {
            let i = self.position.fetch_add(1, Ordering::Relaxed);
            let result = self.data.get_row(i).map(move |row| (i, row));

            // Increment the progress bar if appropriate.
            if result.is_some() {
                self.progress.inc(1);
            }
            result
        }
    }

    /// Tell all users of the `ControlBlock` to cancel and return early.
    fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl<T> Drop for ControlBlock<T> {
    fn drop(&mut self) {
        self.progress.finish();
    }
}

////////////////////////////
// Build via Multi Insert //
////////////////////////////

#[derive(Debug, Serialize)]
pub(super) struct MultiInsertBuildStats {
    total_time: MicroSeconds,
    vectors_inserted: usize,
    multi_insert_latencies: percentiles::Percentiles<MicroSeconds>,
}

impl std::fmt::Display for MultiInsertBuildStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Index Build Time: {}s\n", self.total_time.as_seconds())?;
        write!(
            f,
            "Multi Insert Latencies:\n  average: {}us\n      p90: {}\n      p99: {}\n\n",
            self.multi_insert_latencies.mean,
            self.multi_insert_latencies.p90,
            self.multi_insert_latencies.p99,
        )
    }
}

pub(super) async fn build_multi_insert<T, S, DP>(
    index: Arc<DiskANNIndex<DP>>,
    strategy: S,
    data: Arc<Matrix<T>>,
    batch_size: usize,
    output: &mut dyn Output,
) -> anyhow::Result<MultiInsertBuildStats>
where
    DP: DataProvider<Context = DefaultContext, ExternalId = u32> + provider::SetElement<[T]>,
    S: glue::InsertStrategy<DP, [T]> + Clone + Send + Sync,
    S::PruneStrategy: Clone,
    for<'a> glue::aliases::InsertPruneAccessor<'a, S, DP, [T]>: glue::AsElement<&'a [T]>,
    T: Send + Sync + 'static + SampleableForStart + std::fmt::Debug + Clone,
{
    let start = std::time::Instant::now();

    // Only launch one top-level task.
    //
    // Parallelism is attained within the algorithm.
    let nrows = data.nrows();
    let progress = make_progress_bar(nrows, output.draw_target())?;
    let task: tokio::task::JoinHandle<anyhow::Result<Vec<MicroSeconds>>> =
        tokio::spawn(async move {
            let mut latencies = Vec::<MicroSeconds>::new();
            let mut start = 0;
            loop {
                let ctx = &DefaultContext;
                let stop = (start + batch_size).min(data.nrows());
                let batch: Box<[_]> = (start..stop)
                    .map(|i| {
                        async_tools::VectorIdBoxSlice::<u32, T>::new(i as u32, data.row(i).into())
                    })
                    .collect();

                let timer = std::time::Instant::now();
                index.multi_insert(strategy.clone(), ctx, batch).await?;
                latencies.push(timer.elapsed().into());
                progress.inc((stop - start) as u64);

                // Update and exit if done.
                start = stop;
                if start == data.nrows() {
                    break;
                }
            }
            progress.finish();
            Ok(latencies)
        });

    // Wait for completion.
    let mut latencies = task.await??;

    let total_time: MicroSeconds = start.elapsed().into();

    let result = MultiInsertBuildStats {
        total_time,
        vectors_inserted: nrows,
        multi_insert_latencies: percentiles::compute_percentiles(&mut latencies)?,
    };

    Ok(result)
}

////////////////////////
// Save and Load API ///
////////////////////////

pub(super) async fn save_index<DP, T>(
    index: Arc<DiskANNIndex<DP>>,
    save_path: &str,
) -> anyhow::Result<()>
where
    DP: DataProvider<Context = DefaultContext, ExternalId = u32> + provider::SetElement<[T]>,
    DiskANNIndex<DP>: SaveWith<AsyncIndexMetadata, Error = ANNError>,
{
    index
        .save_with(
            &diskann_providers::storage::FileStorageProvider,
            &AsyncIndexMetadata::new(save_path),
        )
        .await?;

    Ok(())
}

// for now, this only works with full-precision indices
pub(super) async fn load_index<'a, DP>(
    load_path: &'a str,
    index_config: &IndexConfiguration,
) -> anyhow::Result<DiskANNIndex<DP>>
where
    DP: DataProvider<Context = DefaultContext, ExternalId = u32>,
    DiskANNIndex<DP>:
        diskann_providers::storage::LoadWith<(&'a str, IndexConfiguration), Error = ANNError>,
{
    let index = DiskANNIndex::<DP>::load_with(
        &diskann_providers::storage::FileStorageProvider,
        &(load_path, index_config.clone()),
    )
    .await?;

    Ok(index)
}

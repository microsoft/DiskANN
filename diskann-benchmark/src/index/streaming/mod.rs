/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{io::Write, sync::Arc};

use diskann::utils::VectorRepr;
use diskann_benchmark_core::streaming::{self, executors::bigann, Executor};
use diskann_benchmark_runner::output::Output;
use diskann_utils::views::Matrix;

pub(crate) mod managed;
pub(crate) mod runner;
pub(crate) mod stats;

pub(crate) use managed::{Managed, ManagedStream};
#[cfg(feature = "bftree")]
pub(crate) use runner::BfTreeMaintainer;
pub(crate) use runner::StreamRunner;

use crate::{
    inputs::graph_index::{StreamingRunbookParams, StreamingSearchParams},
    utils::datafiles,
};

/// Trait for streaming benchmark outputs that wrap or produce [`stats::StreamStats`].
///
/// This allows [`run_streaming`] to work with both direct streams (which produce
/// `StreamStats` directly) and managed streams (which produce `managed::Stats<StreamStats>`).
pub(crate) trait StreamingOutput: std::fmt::Display + 'static {
    fn stream_stats(&self) -> &stats::StreamStats;
}

impl StreamingOutput for stats::StreamStats {
    fn stream_stats(&self) -> &stats::StreamStats {
        self
    }
}

impl StreamingOutput for managed::Stats<stats::StreamStats> {
    fn stream_stats(&self) -> &stats::StreamStats {
        self.inner()
    }
}

/// Construct the streaming stack: load data/queries, create the managed stream via the
/// closure, then wrap in [`Managed`] and [`bigann::WithData`].
///
/// `capacity` is the pre-computed slot count passed to [`Managed::new`]. Each backend
/// computes this differently (inmem applies headroom, bf_tree adds start points).
///
/// The closure receives `(&data, capacity)` so it can use `data.ncols()` for provider params.
pub(crate) fn build_managed_streamer<T, M, F>(
    data_path: &diskann_benchmark_runner::files::InputFile,
    search: &StreamingSearchParams,
    reclaim: managed::SlotReclaim,
    capacity: usize,
    make_stream: F,
) -> anyhow::Result<bigann::WithData<T, u32, Managed<T, stats::StreamStats>>>
where
    T: bytemuck::Pod + VectorRepr + 'static,
    M: ManagedStream<T, Output = stats::StreamStats> + 'static,
    F: FnOnce(&Matrix<T>, usize) -> anyhow::Result<M>,
{
    let data = datafiles::load_dataset::<T>(datafiles::BinFile(data_path))?;
    let queries = Arc::new(datafiles::load_dataset::<T>(datafiles::BinFile(
        &search.queries,
    ))?);

    let managed_stream = make_stream(&data, capacity)?;
    let managed = Managed::new(capacity, reclaim, managed_stream);

    let max_k = search.max_k();
    let layered = bigann::WithData::new(managed, data, queries, move |path| {
        Ok(Box::new(datafiles::load_groundtruth(
            datafiles::BinFile(path),
            Some(max_k),
        )?))
    });

    Ok(layered)
}

/// Construct a direct streaming stack (no ID management layer).
///
/// For providers where external IDs match internal slots (e.g., bf-tree), the
/// [`Managed`] layer is unnecessary. This function creates the stack without it.
///
/// The closure receives `(&data, capacity)` so it can use `data.ncols()` for provider params.
#[cfg(feature = "bftree")]
pub(crate) fn build_direct_streamer<T, S, F>(
    data_path: &diskann_benchmark_runner::files::InputFile,
    search: &StreamingSearchParams,
    capacity: usize,
    make_stream: F,
) -> anyhow::Result<bigann::WithData<T, u32, S>>
where
    T: bytemuck::Pod + VectorRepr + 'static,
    S: streaming::Stream<bigann::DataArgs<T, u32>> + 'static,
    F: FnOnce(&Matrix<T>, usize) -> anyhow::Result<S>,
{
    let data = datafiles::load_dataset::<T>(datafiles::BinFile(data_path))?;
    let queries = Arc::new(datafiles::load_dataset::<T>(datafiles::BinFile(
        &search.queries,
    ))?);

    let stream = make_stream(&data, capacity)?;

    let max_k = search.max_k();
    let layered = bigann::WithData::new(stream, data, queries, move |path| {
        Ok(Box::new(datafiles::load_groundtruth(
            datafiles::BinFile(path),
            Some(max_k),
        )?))
    });

    Ok(layered)
}

/// Run a streaming benchmark using the given runbook parameters.
///
/// `make_streamer` receives `max_points` from the loaded runbook and returns the
/// constructed streamer. This is shared across all streaming benchmarks (inmem, bftree)
/// to avoid duplicating the runbook load → run_with → stage banner → summary logic.
pub(crate) fn run_streaming<T, S, F>(
    runbook_params: &StreamingRunbookParams,
    make_streamer: F,
    mut output: &mut dyn Output,
) -> anyhow::Result<Vec<S::Output>>
where
    T: 'static,
    S: streaming::Stream<bigann::DataArgs<T, u32>>,
    S::Output: StreamingOutput,
    F: FnOnce(usize) -> anyhow::Result<bigann::WithData<T, u32, S>>,
{
    let groundtruth_directory = runbook_params
        .resolved_gt_directory
        .as_ref()
        .ok_or_else(|| {
            anyhow::anyhow!("Ground truth directory path was not resolved during validation")
        })?;

    let mut runbook = bigann::RunBook::load(
        &runbook_params.runbook_path,
        &runbook_params.dataset_name,
        &mut bigann::ScanDirectory::new(groundtruth_directory)?,
    )?;

    let mut streamer = make_streamer(runbook.max_points())?;

    let mut results = Vec::new();
    let stages = runbook.len();
    let mut i = 1;

    runbook.run_with(&mut streamer, |o: S::Output| -> anyhow::Result<()> {
        if o.stream_stats().is_maintain() {
            let message = format!("Ran maintenance before stage {}", i);
            write!(output, "{}", crate::utils::SmallBanner(&message))?;
        } else {
            let message = format!(
                "Finished stage {} of {}: {}",
                i,
                stages,
                o.stream_stats().kind()
            );
            write!(output, "{}", crate::utils::SmallBanner(&message))?;
            i += 1;
        }
        writeln!(output, "{}", o)?;
        results.push(o);
        Ok(())
    })?;

    write!(
        output,
        "{}",
        crate::utils::SmallBanner("End of Run Summary")
    )?;

    writeln!(
        output,
        "{}",
        stats::Summary::new(results.iter().map(|r| r.stream_stats()))
    )?;

    Ok(results)
}

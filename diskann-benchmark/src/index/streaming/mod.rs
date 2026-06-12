/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{io::Write, sync::Arc};

use diskann::utils::VectorRepr;
use diskann_benchmark_core::streaming::{executors::bigann, Executor};
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
    inputs::graph_index::{StreamingRunbookParams, TopkSearchPhase},
    utils::datafiles,
};

/// Construct the streaming stack: load data/queries, create the managed stream via the
/// closure, then wrap in [`Managed`] and [`bigann::WithData`].
///
/// `capacity` is the pre-computed slot count passed to [`Managed::new`]. Each backend
/// computes this differently (inmem applies headroom, bf_tree adds start points).
///
/// The closure receives `(&data, capacity)` so it can use `data.ncols()` for provider params.
pub(crate) fn build_streamer<T, M, F>(
    data_path: &diskann_benchmark_runner::files::InputFile,
    topk: &TopkSearchPhase,
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
        &topk.queries,
    ))?);

    let managed_stream = make_stream(&data, capacity)?;
    let managed = Managed::new(capacity, reclaim, managed_stream);

    let max_k = topk.max_k();
    let layered = bigann::WithData::new(managed, data, queries, move |path| {
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
pub(crate) fn run_streaming<T, F>(
    runbook_params: &StreamingRunbookParams,
    make_streamer: F,
    mut output: &mut dyn Output,
) -> anyhow::Result<Vec<managed::Stats<stats::StreamStats>>>
where
    T: 'static,
    F: FnOnce(usize) -> anyhow::Result<bigann::WithData<T, u32, Managed<T, stats::StreamStats>>>,
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

    runbook.run_with(
        &mut streamer,
        |o: managed::Stats<stats::StreamStats>| -> anyhow::Result<()> {
            if o.inner().is_maintain() {
                let message = format!("Ran maintenance before stage {}", i);
                write!(output, "{}", crate::utils::SmallBanner(&message))?;
            } else {
                let message = format!("Finished stage {} of {}: {}", i, stages, o.inner().kind());
                write!(output, "{}", crate::utils::SmallBanner(&message))?;
                i += 1;
            }
            writeln!(output, "{}", o)?;
            results.push(o);
            Ok(())
        },
    )?;

    write!(
        output,
        "{}",
        crate::utils::SmallBanner("End of Run Summary")
    )?;

    writeln!(
        output,
        "{}",
        stats::Summary::new(results.iter().map(|r| r.inner()))
    )?;

    Ok(results)
}

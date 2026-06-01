/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::Write;

use diskann_benchmark_core::streaming::{executors::bigann, Executor};
use diskann_benchmark_runner::output::Output;

pub(crate) mod full_precision;
pub(crate) mod managed;
pub(crate) mod stats;

pub(crate) use full_precision::FullPrecisionStream;
pub(crate) use managed::{Managed, ManagedStream};

use crate::inputs::graph_index::DynamicRunbookParams;

/// Run a streaming benchmark using the given runbook parameters.
///
/// `make_streamer` receives `max_points` from the loaded runbook and returns the
/// constructed streamer. This is shared across all streaming benchmarks (inmem, bftree)
/// to avoid duplicating the runbook load → run_with → stage banner → summary logic.
pub(crate) fn run_streaming<T, F>(
    runbook_params: &DynamicRunbookParams,
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

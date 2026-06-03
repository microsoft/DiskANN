/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
//! Bf_tree-backed benchmarks for DiskANN graph indices.
//!
//! Unlike the inmem provider, bf_tree uses a log-structured merge tree for storage,
//! allowing datasets larger than memory and supporting persistent save/load. This module
//! provides both full-precision and spherical-quantized variants, each with static (build
//! once, search) and streaming (insert/delete/search interleaved) modes.
//!
//! Registered tags:
//! - `graph-index-bftree-full-precision-f32` — static FP build + search
//! - `graph-index-bftree-stream-full-precision-f32` — streaming FP
//! - `graph-index-build-bftree-spherical-quantization` — static spherical (1/2/4-bit)
//! - `graph-index-stream-bftree-spherical-quantization` — streaming spherical (1/2/4-bit)

use std::io::Write;

use diskann_benchmark_core::streaming::{executors::bigann, Executor};
use diskann_benchmark_runner::{output::Output, Registry};

use super::{
    search::plugins::Topk,
    streaming::{
        managed::{self, Managed},
        stats::StreamStats,
    },
};
use crate::inputs::graph_index::DynamicRunbookParams;

mod full_precision;
mod full_precision_streaming;
mod spherical;
mod spherical_streaming;

/// Run a streaming benchmark using the given runbook parameters.
///
/// `make_streamer` receives `max_points` from the loaded runbook and returns the
/// constructed streamer. This is shared between full-precision and spherical streaming
/// benchmarks to avoid duplicating the runbook load → run_with → stage banner → summary logic.
fn run_streaming<T, F>(
    runbook_params: &DynamicRunbookParams,
    make_streamer: F,
    mut output: &mut dyn Output,
) -> anyhow::Result<Vec<managed::Stats<StreamStats>>>
where
    T: 'static,
    F: FnOnce(usize) -> anyhow::Result<bigann::WithData<T, u32, Managed<T, StreamStats>>>,
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
        |o: managed::Stats<StreamStats>| -> anyhow::Result<()> {
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
        crate::backend::index::streaming::stats::Summary::new(results.iter().map(|r| r.inner()))
    )?;

    Ok(results)
}

pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    registry.register(
        "graph-index-bftree-full-precision-f32",
        full_precision::BfTreeFullPrecision::<f32>::new().search(Topk),
    )?;

    registry.register(
        "graph-index-bftree-stream-full-precision-f32",
        full_precision_streaming::StreamingFullPrecision::<f32>::new(),
    )?;

    registry.register(
        "graph-index-bftree-spherical-quantization",
        spherical::BfTreeSpherical::new().search(Topk),
    )?;

    registry.register(
        "graph-index-stream-bftree-spherical-quantization",
        spherical_streaming::StreamingSpherical::new(),
    )?;

    Ok(())
}

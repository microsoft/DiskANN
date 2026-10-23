/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Complete-fit measurements and independent numerical diagnostics.

use diskann_benchmark_runner::Registry;

use crate::inputs::graph_ivf_two_means::TwoMeansStrategy;

mod benchmarks;
mod data;
mod diagnostics;
mod measurement;
mod result;

pub(super) use benchmarks::run_workload;
pub(super) use data::Data;
pub(super) use result::WorkloadResults;

const BASELINE: TwoMeansStrategy = TwoMeansStrategy::PairedPoint;

pub(super) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    registry.register("graph-ivf-two-means-strategies", benchmarks::Strategies)?;
    Ok(())
}

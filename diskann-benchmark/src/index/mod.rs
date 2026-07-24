/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Registry;

mod build;
mod search;
mod streaming;

mod benchmarks;
mod inmem;
mod result;

mod bftree;

pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    benchmarks::register_benchmarks(registry)?;
    bftree::register_benchmarks(registry)?;

    Ok(())
}

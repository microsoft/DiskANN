/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Registry;

pub(crate) mod build;
mod search;
mod streaming;

mod benchmarks;
mod inmem;
mod result;
mod inmem2;

#[cfg(feature = "bftree")]
mod bftree;

pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    benchmarks::register_benchmarks(registry)?;
    inmem2::register_benchmarks(registry)?;

    #[cfg(feature = "bftree")]
    bftree::register_benchmarks(registry)?;

    Ok(())
}

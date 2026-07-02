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

#[cfg(feature = "bftree")]
mod bftree;

#[cfg(feature = "inmem2")]
mod inmem2;

pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    benchmarks::register_benchmarks(registry)?;

    #[cfg(feature = "bftree")]
    bftree::register_benchmarks(registry)?;

    #[cfg(feature = "inmem2")]
    inmem2::register_benchmarks(registry)?;

    Ok(())
}

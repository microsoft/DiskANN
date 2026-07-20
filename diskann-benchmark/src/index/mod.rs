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

#[cfg(feature = "inmem2")]
mod inmem2;

pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    benchmarks::register_benchmarks(registry)?;
    bftree::register_benchmarks(registry)?;

    #[cfg(feature = "inmem2")]
    inmem2::register_benchmarks(registry)?;

    #[cfg(not(feature = "inmem2"))]
    {
        registry.register_gated(
            "inmem2",
            "inmem2",
            diskann_benchmark_runner::Features::new("inmem2"),
            "Index build-and-search using version 2 of the inmem index",
        )?;

        registry.register_gated(
            "inmem2-streaming",
            "inmem2-stream",
            diskann_benchmark_runner::Features::new("inmem2"),
            "Streaming runs using version 2 of the inmem index",
        )?;
    }

    Ok(())
}

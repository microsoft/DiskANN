/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Registry;

mod build;
mod search;
mod streaming;

// mod benchmarks;
mod inmem;
mod result;

mod bftree;

#[cfg(feature = "inmem2")]
mod inmem2;

pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    // benchmarks::register_benchmarks(registry)?;
    bftree::register_benchmarks(registry)?;

    #[cfg(feature = "inmem2")]
    inmem2::register_benchmarks(registry)?;

    #[cfg(not(feature = "inmem2"))]
    {
        let features = diskann_benchmark_runner::Features::new("inmem2");

        let benches = [
            (
                "inmem2-fp-bigann-stream",
                "Streaming runs using `diskann-inmem`",
            ),
            (
                "inmem2-fp-static",
                "Index build and search using `diskann-inmem` (full precison)",
            ),
            (
                "inmem2-pq-static",
                "Index build and search using `diskann-inmem` (product quantization)",
            ),
            (
                "inmem2-sphq-static",
                "Index build and search using `diskann-inmem` (spherical quantization)",
            ),
            (
                "inmem2-sq-static",
                "Index build and search using `diskann-inmem` (scalar quantization)",
            ),
        ];

        for (tag, bench) in benches {
            registry.register_gated(tag, "inmem2", features.clone(), bench)?;
        }
    }

    Ok(())
}

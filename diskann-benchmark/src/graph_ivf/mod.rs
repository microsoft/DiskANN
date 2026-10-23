/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Static build + search benchmark for the new generic IVF implementation.
//! Search composition and storage are benchmark-local, pending the library APIs.

use diskann_benchmark_runner::Registry;

mod benchmarks;
mod build;
mod data;
mod provider;
#[cfg(feature = "ivf-two-means")]
mod replay;
mod result;
mod search;
#[cfg(feature = "ivf-two-means")]
mod two_means;

pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    benchmarks::register_benchmarks(registry)?;
    #[cfg(feature = "ivf-two-means")]
    {
        two_means::register_benchmarks(registry)?;
        replay::register_benchmarks(registry)?;
    }
    #[cfg(not(feature = "ivf-two-means"))]
    registry.register_gated(
        "graph-ivf-two-means",
        "graph-ivf-two-means-strategies",
        diskann_benchmark_runner::Features::new("ivf-two-means"),
        "Complete two-means strategy comparison",
    )?;
    Ok(())
}

#[cfg(test)]
mod tests;

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
mod result;
mod search;
#[cfg(feature = "ivf-two-means")]
mod two_means;

pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    benchmarks::register_benchmarks(registry)
}

#[cfg(test)]
mod tests;

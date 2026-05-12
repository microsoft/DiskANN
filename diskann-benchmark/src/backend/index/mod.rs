/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod build;
mod search;
mod streaming;

mod benchmarks;
mod result;

// Feature based backends.
mod product;
mod scalar;
mod spherical;

pub(crate) fn register_benchmarks(
    registry: &mut diskann_benchmark_runner::registry::Registry,
) -> anyhow::Result<()> {
    benchmarks::register_benchmarks(registry)
}

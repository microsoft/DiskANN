/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Registry;

mod build;
mod search;
mod streaming;

mod benchmarks;
mod result;

// Feature based backends.
mod product;
mod scalar;
mod spherical;

pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    benchmarks::register_benchmarks(registry)
}

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

pub(crate) fn register_benchmarks(benchmarks: &mut diskann_benchmark_runner::registry::Benchmarks) {
    benchmarks::register_benchmarks(benchmarks)
}

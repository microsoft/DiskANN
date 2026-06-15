/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Registry;

mod benchmarks;
mod build;
mod search;

pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    benchmarks::register_benchmarks(registry)
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod benchmark;

// Public registration function
pub(crate) fn register_benchmarks(
    registry: &mut diskann_benchmark_runner::registry::Registry,
) -> anyhow::Result<()> {
    benchmark::register_benchmarks(registry)
}

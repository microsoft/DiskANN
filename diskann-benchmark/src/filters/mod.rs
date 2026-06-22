/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Registry;

mod benchmark;

// Public registration function
pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    benchmark::register_benchmarks(registry)
}

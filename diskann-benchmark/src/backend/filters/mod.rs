/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod benchmark;

// Public registration function
pub(crate) fn register_benchmarks(benchmarks: &mut diskann_benchmark_runner::registry::Benchmarks) {
    benchmark::register_benchmarks(benchmarks);
}

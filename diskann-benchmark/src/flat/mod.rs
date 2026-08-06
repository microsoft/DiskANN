/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Registry;

mod search;

pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    search::register_benchmarks(registry)
}

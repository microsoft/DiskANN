/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Registry;

mod disk_index;
mod exhaustive;
mod filters;
mod index;
mod inmem2;
mod multi_vector;

pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    exhaustive::register_benchmarks(registry)?;
    disk_index::register_benchmarks(registry)?;
    index::register_benchmarks(registry)?;
    filters::register_benchmarks(registry)?;
    multi_vector::register_benchmarks(registry)?;
    inmem2::register_benchmarks(registry)?;
    Ok(())
}

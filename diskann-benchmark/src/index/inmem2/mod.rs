/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::{self as dbr};

mod benchmark;

mod full;
mod quantized;

mod common;

/// The [`diskann_inmem::Provider`] used throughout inmem benchmarks.
type Provider<T> = diskann_inmem::Provider<T, u32>;

/// The [`diskann::graph::DiskANNIndex`] used throughout inmem benchmarks.
type Index<T> = diskann::graph::DiskANNIndex<Provider<T>>;

//--------------//
// Registration //
//--------------//

pub(crate) fn register_benchmarks(registry: &mut dbr::Registry) -> anyhow::Result<()> {
    full::register_benchmarks(registry)?;
    quantized::register_benchmarks(registry)?;
    Ok(())
}

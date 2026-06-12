/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Registry;

pub(crate) mod build;
mod search;
mod streaming;

mod benchmarks;
mod result;

// Feature based backends.
mod product;
mod scalar;
mod spherical;

#[cfg(feature = "bftree")]
mod bftree;

pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    benchmarks::register_benchmarks(registry)?;

    #[cfg(feature = "bftree")]
    bftree::register_benchmarks(registry)?;

    Ok(())
}

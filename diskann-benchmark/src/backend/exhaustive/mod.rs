/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#[cfg(any(
    feature = "spherical-quantization",
    feature = "minmax-quantization",
    feature = "product-quantization"
))]
mod algos;

mod minmax;
mod product;
mod spherical;

use diskann_benchmark_runner::Registry;

pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    spherical::register_benchmarks(registry)?;
    minmax::register_benchmarks(registry)?;
    product::register_benchmarks(registry)?;
    Ok(())
}

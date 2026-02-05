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
mod multi;
mod product;
mod spherical;

use diskann_benchmark_runner::registry::Benchmarks;

pub(crate) fn register_benchmarks(benchmarks: &mut Benchmarks) {
    spherical::register_benchmarks(benchmarks);
    minmax::register_benchmarks(benchmarks);
    product::register_benchmarks(benchmarks);
    multi::register_benchmarks(benchmarks);
}

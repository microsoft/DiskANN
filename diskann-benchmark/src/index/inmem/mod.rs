/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#[cfg(feature = "product-quantization")]
pub(crate) mod product;

#[cfg(feature = "scalar-quantization")]
pub(crate) mod scalar;

#[cfg(feature = "spherical-quantization")]
pub(crate) mod spherical;

pub(crate) fn register_benchmarks(
    registry: &mut diskann_benchmark_runner::Registry,
) -> anyhow::Result<()> {
    #[cfg(feature = "product-quantization")]
    product::register_benchmarks(registry)?;

    #[cfg(not(feature = "product-quantization"))]
    registry.register_partially_gated::<crate::inputs::graph_index::IndexPQOperation>(
        "graph-index-pq",
        "product-quantization",
        "PQ based graph index build/search",
    )?;

    #[cfg(feature = "scalar-quantization")]
    scalar::register_benchmarks(registry)?;

    #[cfg(not(feature = "scalar-quantization"))]
    registry.register_partially_gated::<crate::inputs::graph_index::IndexSQOperation>(
        "graph-index-sq",
        "scalar-quantization",
        "Scalar-quantization based graph index build/search",
    )?;

    #[cfg(feature = "spherical-quantization")]
    spherical::register_benchmarks(registry)?;

    #[cfg(not(feature = "spherical-quantization"))]
    registry.register_partially_gated::<crate::inputs::graph_index::SphericalQuantBuild>(
        "graph-index-spherical-quantization",
        "spherical-quantization",
        "Spherical-quantization (RabitQ) based graph index build/search",
    )?;

    Ok(())
}

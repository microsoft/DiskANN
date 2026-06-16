/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Registry;
use diskann_vector::distance;
use diskann_vector::PureDistanceFunction;

use crate::utils::SimilarityMeasure;

mod benchmarks;
mod build;
mod search;

pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    benchmarks::register_benchmarks(registry)
}

/// Returns a SIMD-accelerated distance function for the given metric.
/// All returned functions follow the convention: lower = more similar.
pub(super) fn distance_fn(metric: SimilarityMeasure) -> fn(&[f32], &[f32]) -> f32 {
    match metric {
        SimilarityMeasure::SquaredL2 => |a, b| distance::SquaredL2::evaluate(a, b),
        SimilarityMeasure::InnerProduct
        | SimilarityMeasure::Cosine
        | SimilarityMeasure::CosineNormalized => |a, b| distance::InnerProduct::evaluate(a, b),
    }
}

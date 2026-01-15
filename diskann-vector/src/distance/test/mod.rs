/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod cosine;
mod distance_test;
mod l2;

use crate::{distance, distance::Metric, Half, PureDistanceFunction};

/// Distance contract for full-precision vertex
pub(crate) trait FullPrecisionDistance<T, const N: usize> {
    /// Get the distance between vertex a and vertex b
    fn distance_compare(a: &[T; N], b: &[T; N], vec_type: Metric) -> f32;
}

// reason = "Not supported Metric type Metric::Cosine"
#[allow(clippy::panic)]
impl<const N: usize> FullPrecisionDistance<f32, N> for [f32; N] {
    /// Calculate distance between two f32 Vertex
    #[inline(always)]
    fn distance_compare(a: &[f32; N], b: &[f32; N], metric: Metric) -> f32 {
        match metric {
            Metric::L2 => distance::SquaredL2::evaluate(a, b),
            Metric::Cosine => distance::Cosine::evaluate(a, b),
            Metric::CosineNormalized => distance::CosineNormalized::evaluate(a, b),
            Metric::InnerProduct => distance::InnerProduct::evaluate(a, b),
        }
    }
}

// reason = "Not supported Metric type Metric::Cosine"
#[allow(clippy::panic)]
impl<const N: usize> FullPrecisionDistance<Half, N> for [Half; N] {
    fn distance_compare(a: &[Half; N], b: &[Half; N], metric: Metric) -> f32 {
        match metric {
            Metric::L2 => distance::SquaredL2::evaluate(a, b),
            Metric::Cosine => distance::Cosine::evaluate(a, b),
            Metric::CosineNormalized => distance::CosineNormalized::evaluate(a, b),
            Metric::InnerProduct => distance::InnerProduct::evaluate(a, b),
        }
    }
}

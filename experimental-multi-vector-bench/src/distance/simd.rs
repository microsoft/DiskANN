// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! SIMD-accelerated implementation of multi-vector distance computation.

use diskann_vector::distance::InnerProduct;
use diskann_vector::{DistanceFunction, PureDistanceFunction};

use super::Chamfer;
use crate::MultiVector;

/// SIMD-accelerated approach using vectorized distance computations.
///
/// This approach leverages SIMD instructions (e.g., AVX2 on x86_64, NEON on ARM)
/// for faster distance calculations between vectors.
///
/// # Performance
///
/// This implementation provides significant speedups over [`super::NaiveApproach`]
/// by using hardware-accelerated vector operations for the inner distance
/// computations. Typical speedups are 5-10x depending on vector dimensions
/// and hardware capabilities.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimdApproach;

impl DistanceFunction<&MultiVector, &MultiVector> for Chamfer<SimdApproach> {
    fn evaluate_similarity(&self, query: &MultiVector, doc: &MultiVector) -> f32 {
        let mut score = 0.0;
        for q_vec in query.rows() {
            // InnerProduct::evaluate returns negated inner product (-dot),
            // so we find the minimum (most similar = highest dot = lowest -dot)
            let min_dist = doc
                .rows()
                .map(|d_vec| InnerProduct::evaluate(q_vec, d_vec))
                .fold(f32::MAX, f32::min);

            score += min_dist;
        }
        score
    }
}

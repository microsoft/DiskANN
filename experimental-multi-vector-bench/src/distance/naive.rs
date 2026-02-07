// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Naive scalar implementation of multi-vector distance computation.

use diskann_vector::DistanceFunction;

use super::Chamfer;
use crate::MultiVector;

/// Naive O(n²) approach using scalar operations for multi-vector distance computation.
///
/// This approach iterates through vectors using standard scalar arithmetic.
/// Use [`super::SimdApproach`] for better performance on supported hardware.
///
/// # Performance
///
/// This implementation is useful for:
/// - Baseline performance comparisons
/// - Debugging and verification
/// - Platforms without SIMD support
#[derive(Debug, Clone, Copy, Default)]
pub struct NaiveApproach;

impl DistanceFunction<&MultiVector, &MultiVector> for Chamfer<NaiveApproach> {
    fn evaluate_similarity(&self, query: &MultiVector, doc: &MultiVector) -> f32 {
        let mut score = 0.0;
        for q_vec in query.rows() {
            // Find max similarity (highest inner product) for this query vector
            let mut max_similarity = f32::MIN;
            for d_vec in doc.rows() {
                let similarity: f32 = q_vec.iter().zip(d_vec.iter()).map(|(x, y)| x * y).sum();
                max_similarity = max_similarity.max(similarity);
            }
            // Negate to convert similarity to distance (for min-heap compatibility)
            score += -max_similarity;
        }
        score
    }
}

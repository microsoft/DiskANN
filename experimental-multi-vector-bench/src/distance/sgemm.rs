// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! SGEMM-based Chamfer distance computation using BLAS matrix multiplication.
//!
//! This module provides a high-performance implementation that computes Chamfer distance
//! by first computing the full Q×D similarity matrix via SGEMM, then performing
//! SIMD-accelerated row-wise max reduction.
//!
//! # Backend
//!
//! This uses **faer** (pure Rust BLAS with AVX2/AVX-512 optimizations), which provides
//! excellent performance without any external dependencies.
//!
//! # Algorithm
//!
//! The Chamfer distance computation is expressed as:
//!
//! ```text
//! 1. Compute similarity matrix: S = Q × Dᵀ  (dimensions: [num_query × num_doc])
//! 2. For each query i: max_sim[i] = max_j(S[i, j])  (SIMD row-wise max)
//! 3. Chamfer distance = -Σ max_sim[i]
//! ```
//!
//! # Performance Characteristics
//!
//! - **Best for large Q×D configurations** (e.g., 32 query tokens × 128 doc tokens)
//! - **1.4x–4.3x faster** than baseline SIMD depending on configuration
//! - Peak gains (3.7x–4.3x) at large Q×D (32×128) with dim ≥ 256
//! - Leverages highly-optimized BLAS SGEMM kernels (faer)
//! - SIMD-accelerated row reduction using f32x8 vectors
//! - Pre-allocated scratch buffer avoids allocation overhead
//!
//! # Scratch Buffer
//!
//! The [`SgemmScratch`] type provides a pre-allocated buffer for the similarity matrix,
//! avoiding allocation on the hot path. This is critical for fair benchmarking.

use diskann_linalg::Transpose;
use diskann_vector::DistanceFunction;
use diskann_wide::{SIMDMinMax, SIMDVector};

use super::Chamfer;
use crate::MultiVector;

diskann_wide::alias!(f32s = f32x8);

/// Pre-allocated scratch buffer for SGEMM-based Chamfer distance computation.
///
/// This struct holds a reusable buffer for the Q×D similarity matrix, avoiding
/// allocation overhead during distance computation. The buffer is automatically
/// resized when needed.
///
/// # Example
///
/// ```
/// use experimental_multi_vector_bench::{
///     Chamfer, SgemmApproach, SgemmScratch, MultiVector, Standard,
/// };
///
/// let query = MultiVector::new(Standard::new(8, 128), 0.0f32).unwrap();
/// let doc = MultiVector::new(Standard::new(32, 128), 0.0f32).unwrap();
///
/// let mut scratch = SgemmScratch::new();
/// let chamfer = Chamfer::<SgemmApproach>::new();
///
/// // The scratch buffer is reused across multiple distance computations
/// let distance = chamfer.evaluate_similarity_with_scratch(&query, &doc, &mut scratch);
/// ```
#[derive(Debug, Default)]
pub struct SgemmScratch {
    /// Buffer for the Q×D similarity matrix (row-major).
    similarity_matrix: Vec<f32>,
}

impl SgemmScratch {
    /// Creates a new empty scratch buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a scratch buffer with pre-allocated capacity.
    ///
    /// # Arguments
    ///
    /// * `num_query` - Expected number of query tokens
    /// * `num_doc` - Expected number of document tokens
    pub fn with_capacity(num_query: usize, num_doc: usize) -> Self {
        Self {
            similarity_matrix: vec![0.0; num_query * num_doc],
        }
    }

    /// Ensures the buffer has sufficient capacity for the given dimensions.
    ///
    /// This only reallocates if the current capacity is insufficient.
    #[inline]
    fn ensure_capacity(&mut self, num_query: usize, num_doc: usize) {
        let required = num_query * num_doc;
        if self.similarity_matrix.len() < required {
            self.similarity_matrix.resize(required, 0.0);
        }
    }

    /// Returns a mutable slice of the similarity matrix with the given dimensions.
    #[inline]
    fn as_mut_slice(&mut self, num_query: usize, num_doc: usize) -> &mut [f32] {
        self.ensure_capacity(num_query, num_doc);
        &mut self.similarity_matrix[..num_query * num_doc]
    }
}

/// SGEMM-based approach for Chamfer distance computation.
///
/// This approach computes the full similarity matrix via BLAS SGEMM, then performs
/// row-wise max reduction. It serves as a baseline to compare against custom SIMD
/// implementations.
///
/// # Algorithm
///
/// 1. Compute `S = Q × Dᵀ` using SGEMM (similarity matrix)
/// 2. For each row i in S, find `max_j(S[i, j])`
/// 3. Sum and negate the max values to get Chamfer distance
///
/// # Usage
///
/// ```
/// use experimental_multi_vector_bench::{
///     Chamfer, SgemmApproach, SgemmScratch, MultiVector, Standard,
/// };
///
/// let query = MultiVector::new(Standard::new(8, 128), 0.0f32).unwrap();
/// let doc = MultiVector::new(Standard::new(32, 128), 0.0f32).unwrap();
///
/// let chamfer = Chamfer::<SgemmApproach>::new();
/// let mut scratch = SgemmScratch::new();
///
/// let distance = chamfer.evaluate_similarity_with_scratch(&query, &doc, &mut scratch);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct SgemmApproach;

impl Chamfer<SgemmApproach> {
    /// Computes Chamfer distance using SGEMM with a pre-allocated scratch buffer.
    ///
    /// This is the recommended entry point for benchmarking, as it avoids allocation
    /// overhead on the hot path.
    ///
    /// # Arguments
    ///
    /// * `query` - The query multi-vector (Q tokens × D dimensions)
    /// * `doc` - The document multi-vector (N tokens × D dimensions)
    /// * `scratch` - Pre-allocated scratch buffer for the similarity matrix
    ///
    /// # Returns
    ///
    /// The Chamfer distance (negated sum of max similarities).
    pub fn evaluate_similarity_with_scratch(
        &self,
        query: &MultiVector,
        doc: &MultiVector,
        scratch: &mut SgemmScratch,
    ) -> f32 {
        let num_query = query.num_vectors();
        let num_doc = doc.num_vectors();
        let dim = query.vector_dim();

        debug_assert_eq!(
            dim,
            doc.vector_dim(),
            "Query and document must have the same embedding dimension"
        );

        // Early return for empty inputs
        if num_query == 0 || num_doc == 0 {
            return 0.0;
        }

        // Get similarity matrix buffer
        let similarity = scratch.as_mut_slice(num_query, num_doc);

        // Compute S = Q × Dᵀ using SGEMM
        // Q is [num_query × dim], D is [num_doc × dim]
        // S = Q × Dᵀ = [num_query × dim] × [dim × num_doc] = [num_query × num_doc]
        diskann_linalg::sgemm(
            Transpose::None,     // Q is not transposed
            Transpose::Ordinary, // D is transposed to get Dᵀ
            num_query,           // m = rows in output (and Q)
            num_doc,             // n = cols in output (and rows in D, cols in Dᵀ)
            dim,                 // k = cols in Q = rows in Dᵀ = cols in D
            1.0,                 // alpha = 1.0
            query.as_slice(),
            doc.as_slice(),
            None, // beta = None means overwrite C entirely
            similarity,
        );

        // Row-wise max reduction, then negate and sum
        // Chamfer(Q, D) = Σᵢ -maxⱼ IP(qᵢ, dⱼ)
        let mut total = 0.0f32;
        for i in 0..num_query {
            let row_start = i * num_doc;
            let row_end = row_start + num_doc;
            let row = &similarity[row_start..row_end];

            // Find max in this row using SIMD
            let max_sim = simd_row_max(row);
            total += max_sim;
        }

        -total
    }
}

/// SIMD-accelerated row-wise maximum using f32x8 vectors.
///
/// Processes 8 elements at a time, then reduces the SIMD vector to a scalar max.
#[inline]
fn simd_row_max(row: &[f32]) -> f32 {
    let len = row.len();
    if len == 0 {
        return f32::NEG_INFINITY;
    }

    // Process full SIMD lanes (8 elements at a time)
    let simd_lanes = 8;
    let simd_chunks = len / simd_lanes;
    let remainder = len % simd_lanes;

    let mut max_vec = f32s::splat(diskann_wide::ARCH, f32::NEG_INFINITY);

    // Main SIMD loop
    let ptr = row.as_ptr();
    for i in 0..simd_chunks {
        // SAFETY: i * simd_lanes + simd_lanes <= simd_chunks * simd_lanes <= len
        let chunk = unsafe { f32s::load_simd(diskann_wide::ARCH, ptr.add(i * simd_lanes)) };
        max_vec = max_vec.max_simd(chunk);
    }

    // Reduce SIMD vector to scalar using to_array() pattern from transposed_tiling.rs
    let mut scalar_max = max_vec
        .to_array()
        .into_iter()
        .fold(f32::NEG_INFINITY, f32::max);

    // Handle remainder elements
    if remainder > 0 {
        let remainder_start = simd_chunks * simd_lanes;
        for j in remainder_start..len {
            // SAFETY: j < len
            let val = unsafe { *row.get_unchecked(j) };
            scalar_max = scalar_max.max(val);
        }
    }

    scalar_max
}

// Note: We implement DistanceFunction for API compatibility, but it allocates internally.
// For benchmarking, use `evaluate_similarity_with_scratch` instead.
impl DistanceFunction<&MultiVector, &MultiVector> for Chamfer<SgemmApproach> {
    fn evaluate_similarity(&self, query: &MultiVector, doc: &MultiVector) -> f32 {
        let mut scratch = SgemmScratch::with_capacity(query.num_vectors(), doc.num_vectors());
        self.evaluate_similarity_with_scratch(query, doc, &mut scratch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::NaiveApproach;
    use crate::Standard;

    fn make_multi_vector(data: &[f32], rows: usize, cols: usize) -> MultiVector {
        let mut mat = MultiVector::new(Standard::new(rows, cols), 0.0f32).unwrap();
        for (i, chunk) in data.chunks(cols).enumerate() {
            if let Some(row) = mat.get_row_mut(i) {
                row.copy_from_slice(chunk);
            }
        }
        mat
    }

    #[test]
    fn test_sgemm_matches_naive() {
        // Create test data
        let query = make_multi_vector(
            &[
                1.0, 0.0, 0.0, // q0
                0.0, 1.0, 0.0, // q1
            ],
            2,
            3,
        );
        let doc = make_multi_vector(
            &[
                1.0, 0.0, 0.0, // d0
                0.0, 1.0, 0.0, // d1
                0.5, 0.5, 0.0, // d2
            ],
            3,
            3,
        );

        let naive = Chamfer::<NaiveApproach>::new();
        let sgemm = Chamfer::<SgemmApproach>::new();

        let naive_result = naive.evaluate_similarity(&query, &doc);
        let sgemm_result = sgemm.evaluate_similarity(&query, &doc);

        assert!(
            (naive_result - sgemm_result).abs() < 1e-5,
            "SGEMM result {} should match naive result {}",
            sgemm_result,
            naive_result
        );
    }

    #[test]
    fn test_sgemm_with_scratch_reuse() {
        let query = make_multi_vector(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let doc1 = make_multi_vector(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2, 3);
        let doc2 = make_multi_vector(&[0.0, 0.0, 1.0, 1.0, 1.0, 1.0], 2, 3);

        let chamfer = Chamfer::<SgemmApproach>::new();
        let mut scratch = SgemmScratch::new();

        // Compute distances reusing scratch buffer
        let d1 = chamfer.evaluate_similarity_with_scratch(&query, &doc1, &mut scratch);
        let d2 = chamfer.evaluate_similarity_with_scratch(&query, &doc2, &mut scratch);

        // Verify against DistanceFunction trait implementation
        assert!((d1 - chamfer.evaluate_similarity(&query, &doc1)).abs() < 1e-5);
        assert!((d2 - chamfer.evaluate_similarity(&query, &doc2)).abs() < 1e-5);
    }

    #[test]
    fn test_empty_inputs() {
        let empty_query = make_multi_vector(&[], 0, 3);
        let doc = make_multi_vector(&[1.0, 2.0, 3.0], 1, 3);

        let chamfer = Chamfer::<SgemmApproach>::new();
        let mut scratch = SgemmScratch::new();

        assert_eq!(
            chamfer.evaluate_similarity_with_scratch(&empty_query, &doc, &mut scratch),
            0.0
        );
    }
}

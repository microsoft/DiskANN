// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Multi-vector representation for DiskANN.
//!
//! This module provides the [`TransposedMultiVector`] type alias and
//! [`transpose_multi_vector`] conversion function for block-transposed
//! SIMD-optimized multi-vector representations.
//!
//! For row-major multi-vectors, use [`MultiVector`](crate::MultiVector) (re-exported from
//! `diskann_quantization::multi_vector::Mat<Standard<f32>>`).
//!
//! # Background
//!
//! Traditional vector search represents each document as a single embedding vector.
//! Multi-vector representations instead encode each document (or query) as a *bag of embeddings*,
//! typically one per token or segment. This approach enables:
//!
//! - **Fine-grained matching**: Token-level similarity captures nuanced semantic relationships
//!   that single-vector representations may miss.
//! - **Late interaction**: Document embeddings can be pre-computed offline, with only the
//!   lightweight similarity aggregation performed at query time.
//! - **Better recall**: Chamfer aggregation ensures that if any part of a query matches
//!   any part of a document well, the document receives a high score.

use diskann_quantization::multi_vector::BlockTransposed;

use crate::MultiVector;

/// A multi-vector with block-transposed layout for SIMD operations.
///
/// This is an alias for [`BlockTransposed<f32, 16>`] from `diskann-quantization`.
/// The block-transposed memory layout groups 16 vectors together and stores their
/// dimensions contiguously, enabling efficient SIMD operations by loading 8 values
/// at once (f32x8) and computing 16 inner products simultaneously.
///
/// # Memory Layout
///
/// ```text
/// Standard:    [v0_d0, v0_d1, ...], [v1_d0, v1_d1, ...], ...
/// Transposed:  [v0_d0..v15_d0], [v0_d1..v15_d1], ...
/// ```
///
/// # Usage
///
/// Documents are transposed because in the Chamfer distance computation:
/// - We iterate over each query vector (row-major = sequential access)
/// - For each query vector, we compute inner products with all document vectors
/// - The transposed layout enables SIMD-parallel inner product computation
///
/// # Examples
///
/// ```
/// use experimental_multi_vector_bench::{
///     transpose_multi_vector, TransposedMultiVector, MultiVector, Standard,
/// };
///
/// // Create a multi-vector with 32 token embeddings of dimension 128
/// let mv = MultiVector::new(Standard::new(32, 128).unwrap(), 0.0f32).unwrap();
/// let transposed = transpose_multi_vector(&mv);
///
/// assert_eq!(transposed.nrows(), 32);   // 32 tokens
/// assert_eq!(transposed.ncols(), 128);  // 128-dim embeddings
/// ```
pub type TransposedMultiVector = BlockTransposed<f32, 16>;

/// Creates a [`TransposedMultiVector`] from a row-major [`MultiVector`].
///
/// This computes the block-transposed layout for SIMD-optimized distance
/// computations.
///
/// # Examples
///
/// ```
/// use experimental_multi_vector_bench::{
///     transpose_multi_vector, TransposedMultiVector, MultiVector, Standard,
/// };
///
/// let mv = MultiVector::new(Standard::new(32, 128).unwrap(), 0.0f32).unwrap();
/// let transposed = transpose_multi_vector(&mv);
/// assert_eq!(transposed.nrows(), 32);
/// ```
pub fn transpose_multi_vector(mv: &MultiVector) -> TransposedMultiVector {
    let nrows = mv.num_vectors();
    let ncols = mv.vector_dim();
    let slice = mv.as_slice();

    let matrix =
        diskann_utils::views::MatrixView::try_from(slice, nrows, ncols).expect("valid dimensions");

    BlockTransposed::from_matrix_view(matrix)
}

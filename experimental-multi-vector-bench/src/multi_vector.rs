// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Multi-vector representation for DiskANN.
//!
//! This module provides the [`TransposedMultiVector`] type for block-transposed
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

use diskann_quantization::algorithms::kmeans::BlockTranspose;
use diskann_quantization::multi_vector::{MatRef, Standard};

use crate::MultiVector;

/// A document multi-vector with block-transposed layout for SIMD operations.
///
/// This structure provides a block-transposed memory layout optimized for SIMD
/// distance computations. It groups 16 document vectors together and stores their
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
/// use experimental_multi_vector_bench::{TransposedMultiVector, MultiVector, Standard};
///
/// // Create a multi-vector with 32 token embeddings of dimension 128
/// let mv = MultiVector::new(Standard::new(32, 128), 0.0f32).unwrap();
/// let transposed = TransposedMultiVector::from(&mv);
///
/// assert_eq!(transposed.num_vectors(), 32);   // 32 tokens
/// assert_eq!(transposed.vector_dim(), 128);   // 128-dim embeddings
/// ```
#[derive(Debug)]
pub struct TransposedMultiVector {
    /// Block-transposed layout for SIMD-friendly access patterns.
    block_transposed: BlockTranspose<16>,
}

impl TransposedMultiVector {
    /// Creates a new `TransposedMultiVector` from a [`MatRef`] view.
    ///
    /// This computes the block-transposed layout for SIMD-optimized distance
    /// computations.
    pub fn from_view(view: MatRef<'_, Standard<f32>>) -> Self {
        // Build a matrix view compatible with BlockTranspose
        let nrows = view.num_vectors();
        let ncols = view.vector_dim();

        // Collect rows into a flat buffer for BlockTranspose
        let mut data = vec![0.0f32; nrows * ncols];
        for (i, row) in view.rows().enumerate() {
            data[i * ncols..(i + 1) * ncols].copy_from_slice(row);
        }

        let matrix = diskann_utils::views::Matrix::try_from(data.into_boxed_slice(), nrows, ncols)
            .expect("valid dimensions");

        let block_transposed = BlockTranspose::from_matrix_view(matrix.as_view());
        Self { block_transposed }
    }

    /// Returns a reference to the block-transposed representation.
    #[inline]
    pub fn block_transposed(&self) -> &BlockTranspose<16> {
        &self.block_transposed
    }

    /// Returns the number of token embeddings in this multi-vector.
    #[inline]
    pub fn num_vectors(&self) -> usize {
        self.block_transposed.nrows()
    }

    /// Returns the dimensionality of each token embedding.
    #[inline]
    pub fn vector_dim(&self) -> usize {
        self.block_transposed.ncols()
    }
}

impl From<&MultiVector> for TransposedMultiVector {
    fn from(mv: &MultiVector) -> Self {
        Self::from_view(mv.as_view())
    }
}

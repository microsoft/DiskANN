// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Experimental multi-vector benchmarking support for DiskANN.
//!
//! This crate provides high-performance distance functions for multi-vector representations,
//! where a single entity (document, image, etc.) is represented by multiple embedding vectors.
//!
//! # Overview
//!
//! Multi-vector representations are used in advanced retrieval systems that employ
//! "late interaction" - instead of encoding an entity into a single vector, each token
//! or segment produces its own embedding. This enables more fine-grained semantic matching
//! by computing similarity at the token/segment level rather than aggregating into a
//! single representation upfront.
//!
//! # Use Cases
//!
//! - **Token-level retrieval**: Each token in a document/query has its own embedding,
//!   and relevance is computed via MaxSim (maximum similarity) aggregation.
//! - **Multi-aspect embeddings**: A single entity represented by embeddings from different
//!   views, modalities, or chunked segments.
//! - **Passage chunking**: Long documents split into chunks, each with its own embedding,
//!   where the final score aggregates similarities across all chunks.
//!
//! # Distance Computation
//!
//! For multi-vector search, the typical approach is:
//! 1. Compute pairwise Inner Product similarities between all vectors in the query and document.
//! 2. For each query vector, find the maximum similarity to any document vector.
//! 3. Negate and sum these values to get the final Chamfer distance (lower = more similar).
//!
//! This "late interaction" pattern preserves fine-grained token-level semantics while
//! enabling efficient pre-computation of document representations.
//!
//! # Available Approaches
//!
//! | Approach | Best For | Speedup vs SIMD |
//! |----------|----------|-----------------|
//! | [`NaiveApproach`] | Reference/debugging | 0.1x (baseline) |
//! | [`SimdApproach`] | General purpose | 1.0x |
//! | [`TransposedApproach`] | Medium Q×D | 1.4–1.7x |
//! | [`TransposedWithTilingApproach`] | Small D (≤32 docs) | 1.8–2.6x |
//! | [`QueryTransposedWithTilingApproach`] | Many queries (≥16) | 1.8–2.2x |
//! | [`SgemmApproach`] | Large Q×D (≥16×64) | 1.9–4.2x |
//!
//! # Type Aliases
//!
//! This crate uses types from `diskann-quantization` for multi-vector representation:
//!
//! - [`MultiVector`] = `Mat<Standard<f32>>` - Owning row-major matrix
//! - [`MultiVectorRef`] = `MatRef<Standard<f32>>` - Borrowed view
//!
//! # Example
//!
//! ```
//! use experimental_multi_vector_bench::{
//!     Chamfer, SimdApproach, TransposedWithTilingApproach,
//!     MultiVector, TransposedMultiVector, Standard,
//! };
//! use diskann_vector::DistanceFunction;
//!
//! // Create multi-vectors
//! let query = MultiVector::new(Standard::new(8, 128), 0.0f32).unwrap();
//! let doc = MultiVector::new(Standard::new(32, 128), 0.0f32).unwrap();
//!
//! // Basic: SIMD-accelerated
//! let chamfer = Chamfer::<SimdApproach>::new();
//! let distance = chamfer.evaluate_similarity(&query, &doc);
//!
//! // Optimized: transpose documents for better cache utilization
//! let chamfer = Chamfer::<TransposedWithTilingApproach>::new();
//! let transposed_doc = TransposedMultiVector::from(&doc);
//! let distance = chamfer.evaluate_similarity(&query, &transposed_doc);
//!
//! // For large Q×D: use SGEMM (best for ≥16 queries × ≥64 docs)
//! use experimental_multi_vector_bench::{SgemmApproach, SgemmScratch};
//! let chamfer = Chamfer::<SgemmApproach>::new();
//! let mut scratch = SgemmScratch::new();
//! let distance = chamfer.evaluate_similarity_with_scratch(&query, &doc, &mut scratch);
//! ```

#![warn(missing_docs)]

pub mod bench;
pub mod distance;
mod multi_vector;

pub use distance::{
    Chamfer, NaiveApproach, QueryTransposedWithTilingApproach, SgemmApproach, SgemmScratch,
    SimdApproach, TransposedApproach, TransposedWithTilingApproach,
};
pub use multi_vector::TransposedMultiVector;

// Re-export types from diskann-quantization for unified multi-vector representation
pub use diskann_quantization::multi_vector::{distance::QueryMatRef, Mat, MatRef, Standard};

/// A multi-vector representation using standard f32 row-major format.
///
/// This is an alias for `Mat<Standard<f32>>` from diskann-quantization.
pub type MultiVector = Mat<Standard<f32>>;

/// An immutable view of a multi-vector.
///
/// This is an alias for `MatRef<Standard<f32>>` from diskann-quantization.
pub type MultiVectorRef<'a> = MatRef<'a, Standard<f32>>;

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Distance computation for multi-vector representations.
//!
//! This module provides implementations for computing distances
//! between multi-vector representations using various approaches.
//!
//! # Approaches
//!
//! | Approach | Query Type | Doc Type | Description |
//! |----------|------------|----------|-------------|
//! | [`NaiveApproach`] | `MultiVector` | `MultiVector` | Scalar O(n²) baseline |
//! | [`SimdApproach`] | `MultiVector` | `MultiVector` | SIMD inner products |
//! | [`TransposedApproach`] | `MultiVector` | `TransposedMultiVector` | Block-transposed docs |
//! | [`TransposedWithTilingApproach`] | `MultiVector` | `TransposedMultiVector` | + Query pair tiling |
//! | [`QueryTransposedWithTilingApproach`] | `TransposedMultiVector` | `MultiVector` | Transposed query + doc pair tiling |
//! | [`SgemmApproach`] | `MultiVector` | `MultiVector` | BLAS SGEMM + SIMD row-max |
//!
//! # Choosing an Approach
//!
//! - **Few query tokens (≤8)**: Use [`TransposedWithTilingApproach`] - transposes documents
//! - **Many query tokens (≥16)**: Use [`QueryTransposedWithTilingApproach`] - transposes query
//! - **Large Q×D (≥32×128)**: Use [`SgemmApproach`] - BLAS dominates at scale (up to 4.3x vs SIMD)
//! - **Baseline/debugging**: Use [`NaiveApproach`] or [`SimdApproach`]
//!
//! # Distance Calculator
//!
//! - [`Chamfer`]: Computes asymmetric Chamfer distance using Inner Product similarity.
//!
//! # Example
//!
//! ```
//! use experimental_multi_vector_bench::{
//!     Chamfer, TransposedWithTilingApproach, QueryTransposedWithTilingApproach,
//!     MultiVector, TransposedMultiVector, Standard,
//! };
//! use diskann_vector::DistanceFunction;
//!
//! // Create multi-vectors
//! let query = MultiVector::new(Standard::new(8, 128), 0.0f32).unwrap();
//! let doc = MultiVector::new(Standard::new(32, 128), 0.0f32).unwrap();
//!
//! // For queries with few tokens: transpose documents
//! let chamfer = Chamfer::<TransposedWithTilingApproach>::new();
//! let transposed_doc = TransposedMultiVector::from(&doc);
//! let distance = chamfer.evaluate_similarity(&query, &transposed_doc);
//!
//! // For queries with many tokens: transpose query
//! let chamfer = Chamfer::<QueryTransposedWithTilingApproach>::new();
//! let transposed_query = TransposedMultiVector::from(&query);
//! let distance = chamfer.evaluate_similarity(&transposed_query, &doc);
//! ```

mod naive;
mod query_transposed_tiling;
mod sgemm;
mod simd;
mod transposed;
mod transposed_tiling;

pub use naive::NaiveApproach;
pub use query_transposed_tiling::QueryTransposedWithTilingApproach;
pub use sgemm::{SgemmApproach, SgemmScratch};
pub use simd::SimdApproach;
pub use transposed::TransposedApproach;
pub use transposed_tiling::TransposedWithTilingApproach;

/// Chamfer aggregation strategy for multi-vector similarity using Inner Product.
///
/// Computes the sum of maximum similarities from each vector in `a` to vectors in `b`,
/// negated to produce a distance metric:
///
/// ```text
/// Chamfer(Q, D) = Σᵢ -maxⱼ IP(qᵢ, dⱼ)
/// ```
///
/// This uses Inner Product similarity (higher = more similar), negated for
/// compatibility with min-heap operations. Also known as asymmetric Chamfer distance.
///
/// # Type Parameters
///
/// * `Approach` - The computation approach to use:
///   - `NaiveApproach`: Scalar baseline
///   - `SimdApproach`: SIMD-accelerated (recommended)
///   - `TransposedApproach`: Block-transposed SIMD for large datasets
///   - `TransposedWithTilingApproach`: Block-transposed SIMD with tiling
///   - `QueryTransposedWithTilingApproach`: Query-transposed SIMD with tiling and scratch buffer
#[derive(Debug)]
pub struct Chamfer<Approach> {
    approach: Approach,
}

impl<Approach: Default> Default for Chamfer<Approach> {
    fn default() -> Self {
        Self {
            approach: Approach::default(),
        }
    }
}

impl<Approach: Default> Chamfer<Approach> {
    /// Creates a new Chamfer distance calculator.
    pub fn new() -> Self {
        Self::default()
    }
}

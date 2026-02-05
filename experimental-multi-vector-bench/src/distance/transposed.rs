// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Block-transposed SIMD implementation of multi-vector distance computation.
//!
//! This module provides a SIMD-accelerated implementation that uses a block-transposed
//! memory layout for the document vectors, enabling efficient SIMD computation of
//! inner products between a row-major query and transposed document vectors.
//!
//! # Memory Layout
//!
//! The block-transposed layout groups 16 document vectors together and stores their
//! dimensions contiguously. For vectors with dimensions `[d0, d1, d2, ...]`, the
//! transposed layout stores all 16 `d0` values together, then all 16 `d1` values, etc:
//!
//! ```text
//! Standard:    [v0_d0, v0_d1, ...], [v1_d0, v1_d1, ...], ...
//! Transposed:  [v0_d0..v15_d0], [v0_d1..v15_d1], ...
//! ```
//!
//! This layout enables efficient SIMD operations by loading 8 document values at once
//! (f32x8) and computing 16 inner products simultaneously using two SIMD registers.

use diskann_quantization::algorithms::kmeans::BlockTranspose;
use diskann_vector::DistanceFunction;
use diskann_wide::{SIMDMask, SIMDMinMax, SIMDMulAdd, SIMDSelect, SIMDVector};

use super::Chamfer;
use crate::{MultiVector, TransposedMultiVector};

diskann_wide::alias!(f32s = f32x8);
diskann_wide::alias!(m32s = mask_f32x8);

/// Block-transposed SIMD approach for Chamfer distance computation.
///
/// This approach uses a block-transposed memory layout for document vectors,
/// enabling efficient SIMD computation. The query remains in row-major format
/// for sequential iteration.
///
/// # Algorithm
///
/// Computes the asymmetric Chamfer distance: `Σ_q -max_d IP(q, d)`
///
/// For each query vector `q` (row-major, sequential access):
/// 1. Process document vectors in blocks of 16 (transposed layout)
/// 2. For each dimension, broadcast `q[dim]` and multiply-add with 16 doc values
/// 3. Track maximum similarity across all 16 lanes using SIMD max operations
/// 4. Handle remainder vectors (< 16) with masked operations
/// 5. Reduce to scalar max and negate for distance
///
/// # Performance
///
/// This implementation is optimized for scenarios with many document tokens where
/// the block-transposed layout improves cache utilization and enables SIMD parallelism.
///
/// # Example
///
/// ```
/// use experimental_multi_vector_bench::{
///     Chamfer, TransposedApproach, TransposedMultiVector, MultiVector, Standard,
/// };
/// use diskann_vector::DistanceFunction;
///
/// let query = MultiVector::new(Standard::new(8, 128), 0.0f32).unwrap();
/// let doc = MultiVector::new(Standard::new(32, 128), 0.0f32).unwrap();
/// let transposed_doc = TransposedMultiVector::from(&doc);
///
/// let chamfer = Chamfer::<TransposedApproach>::new();
/// let distance = chamfer.evaluate_similarity(&query, &transposed_doc);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct TransposedApproach;

/// Block size for transposed layout (number of vectors per block).
const N: usize = 16;
/// Half block size for dual-register processing.
const N2: usize = N / 2;

impl DistanceFunction<&MultiVector, &TransposedMultiVector> for Chamfer<TransposedApproach> {
    fn evaluate_similarity(&self, query: &MultiVector, doc: &TransposedMultiVector) -> f32 {
        let mut score = 0.0;
        // For each query vector, find max similarity to any document vector
        for query_vec in query.rows() {
            score += max_inner_product_to_transposed_doc(query_vec, doc.block_transposed());
        }
        score
    }
}

/// Finds the maximum inner product between `query_vec` and any vector in the transposed document.
///
/// Returns the negated max similarity (distance = -similarity for inner product).
///
/// # Algorithm
///
/// 1. Process full blocks of 16 document vectors using SIMD
/// 2. For each block, compute inner products for all dimensions using FMA
/// 3. Track running maximum across all document vectors using SIMD max operations
/// 4. Handle partial remainder block with lane masking
/// 5. Reduce SIMD max to scalar and negate
#[inline(always)]
fn max_inner_product_to_transposed_doc(query_vec: &[f32], doc: &BlockTranspose<N>) -> f32 {
    let min_val = f32s::splat(diskann_wide::ARCH, f32::MIN);
    let mut max_similarity = min_val;

    // Process full blocks (each contains exactly 16 document vectors)
    for block in 0..doc.full_blocks() {
        let (sim1, sim2) = compute_block_inner_products(query_vec, doc, block);
        max_similarity = max_similarity.max_simd(sim1);
        max_similarity = max_similarity.max_simd(sim2);
    }

    // Process remainder block if present (< 16 document vectors)
    let remainder = doc.remainder();
    if remainder != 0 {
        let (mut sim1, mut sim2) = compute_block_inner_products(query_vec, doc, doc.full_blocks());

        // Compute how many valid lanes in each register
        let lo = remainder.min(N2); // Valid lanes in sim1 (0-8)
        let hi = remainder.saturating_sub(N2); // Valid lanes in sim2 (0-8)

        // Mask invalid lanes to MIN so they never win the max comparison
        sim1 = m32s::keep_first(diskann_wide::ARCH, lo).select(sim1, min_val);
        sim2 = m32s::keep_first(diskann_wide::ARCH, hi).select(sim2, min_val);

        max_similarity = max_similarity.max_simd(sim1);
        max_similarity = max_similarity.max_simd(sim2);
    }

    // Horizontal max reduction and negate (distance = -similarity)
    -max_similarity
        .to_array()
        .into_iter()
        .fold(f32::MIN, f32::max)
}

/// Computes inner products between `query_vec` and 16 document vectors in the specified block.
///
/// Returns two f32x8 vectors containing similarities for document vectors 0-7 and 8-15.
#[inline(always)]
fn compute_block_inner_products(
    query_vec: &[f32],
    doc: &BlockTranspose<N>,
    block: usize,
) -> (f32s, f32s) {
    debug_assert!(block < doc.num_blocks());

    // Use 4 accumulator registers to reduce FMA dependency chains
    let mut sim1_a = f32s::default(diskann_wide::ARCH);
    let mut sim2_a = f32s::default(diskann_wide::ARCH);
    let mut sim1_b = f32s::default(diskann_wide::ARCH);
    let mut sim2_b = f32s::default(diskann_wide::ARCH);

    // SAFETY: block < num_blocks() ensures this access is in-bounds.
    let block_ptr = unsafe { doc.block_ptr_unchecked(block) };

    let ncols = doc.ncols();

    // Process 4 dimensions at a time, alternating accumulators
    // Register count: 4 acc + 8 doc loads + 4 query = 16 registers (at AVX2 limit)
    for dim in (0..ncols.saturating_sub(3)).step_by(4) {
        // SAFETY: For all rows in this block, 16 reads are valid per dimension.
        // dim + 3 < ncols ensures all dimension accesses are in-bounds.
        // dim + 3 < ncols <= query_vec.len() by caller contract ensures query accesses are valid.
        let (d0_0, d1_0, d0_1, d1_1, d0_2, d1_2, d0_3, d1_3, q0, q1, q2, q3) = unsafe {
            (
                f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * dim)),
                f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * dim + N2)),
                f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * (dim + 1))),
                f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * (dim + 1) + N2)),
                f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * (dim + 2))),
                f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * (dim + 2) + N2)),
                f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * (dim + 3))),
                f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * (dim + 3) + N2)),
                f32s::splat(diskann_wide::ARCH, *query_vec.get_unchecked(dim)),
                f32s::splat(diskann_wide::ARCH, *query_vec.get_unchecked(dim + 1)),
                f32s::splat(diskann_wide::ARCH, *query_vec.get_unchecked(dim + 2)),
                f32s::splat(diskann_wide::ARCH, *query_vec.get_unchecked(dim + 3)),
            )
        };

        // Fused multiply-add into alternating accumulators (dims 0,2 -> _a, dims 1,3 -> _b)
        sim1_a = q0.mul_add_simd(d0_0, sim1_a);
        sim2_a = q0.mul_add_simd(d1_0, sim2_a);
        sim1_b = q1.mul_add_simd(d0_1, sim1_b);
        sim2_b = q1.mul_add_simd(d1_1, sim2_b);
        sim1_a = q2.mul_add_simd(d0_2, sim1_a);
        sim2_a = q2.mul_add_simd(d1_2, sim2_a);
        sim1_b = q3.mul_add_simd(d0_3, sim1_b);
        sim2_b = q3.mul_add_simd(d1_3, sim2_b);
    }

    // Handle remaining dimensions (0-3)
    for dim in (ncols - (ncols % 4))..ncols {
        // SAFETY: dim < ncols ensures valid block access; dim < query_vec.len() by caller contract.
        let (d0, d1, q) = unsafe {
            (
                f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * dim)),
                f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * dim + N2)),
                f32s::splat(diskann_wide::ARCH, *query_vec.get_unchecked(dim)),
            )
        };
        sim1_a = q.mul_add_simd(d0, sim1_a);
        sim2_a = q.mul_add_simd(d1, sim2_a);
    }

    // Combine accumulators
    (sim1_a + sim1_b, sim2_a + sim2_b)
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Block-transposed SIMD implementation with query tiling for multi-vector distance computation.
//!
//! This module provides a SIMD-accelerated implementation that uses block-transposed
//! memory layout for **query** vectors (instead of documents), with documents remaining
//! in row-major format.
//!
//! # Tiling Strategy
//!
//! The key optimization is processing **pairs of document vectors together** against each
//! query block. This amortizes the cost of loading query data from memory by reusing it
//! for both document vectors simultaneously.
//!
//! # Use Case
//!
//! This approach is beneficial when:
//! - Queries are reused across multiple documents (batch scoring)
//! - Query transposition can be amortized over many document comparisons
//! - Documents are received in streaming/row-major format
//!
//! # Memory Layout
//!
//! - **Query**: Block-transposed (16 vectors per block, dimensions contiguous)
//! - **Document**: Row-major (standard MultiVector format)

use std::cell::UnsafeCell;

use diskann_quantization::algorithms::kmeans::BlockTranspose;
use diskann_vector::DistanceFunction;
use diskann_wide::{SIMDMinMax, SIMDMulAdd, SIMDVector};

use super::Chamfer;
use crate::{MultiVector, TransposedMultiVector};

diskann_wide::alias!(f32s = f32x8);
diskann_wide::alias!(m32s = mask_f32x8);

/// Block-transposed SIMD approach with query tiling for Chamfer distance computation.
///
/// This approach uses a block-transposed memory layout for **query** vectors and
/// row-major format for documents. It processes pairs of document vectors together
/// to amortize query memory loads.
///
/// The approach holds a pre-allocated scratch buffer for storing per-query max similarities,
/// avoiding allocation on each `evaluate_similarity` call.
///
/// # Algorithm
///
/// Computes the asymmetric Chamfer distance: `Σ_q -max_d IP(q, d)`
///
/// 1. Reset scratch buffer (stored in this approach) to f32::MIN
/// 2. Process document vectors in pairs (d1, d2)
/// 3. For each document pair, compute inner products with all query vectors
///    using the transposed query layout
/// 4. Update max similarities for each query vector
/// 5. Sum and negate the max similarities
///
/// # Performance Characteristics
///
/// - **Best for**: Scenarios where queries are reused across multiple documents
/// - **Trade-off**: Requires mutable reference to self (via UnsafeCell) for scratch buffer
/// - **Register usage**: Optimized for AVX2 (16 YMM registers in hot loops)
///
/// # Example
///
/// ```
/// use experimental_multi_vector_bench::{
///     Chamfer, QueryTransposedWithTilingApproach, TransposedMultiVector, MultiVector, Standard,
/// };
/// use diskann_vector::DistanceFunction;
///
/// let query = MultiVector::new(Standard::new(16, 128), 0.0f32).unwrap();
/// let doc = MultiVector::new(Standard::new(32, 128), 0.0f32).unwrap();
/// let query_transposed = TransposedMultiVector::from(&query);
///
/// let chamfer = Chamfer::<QueryTransposedWithTilingApproach>::new();
/// let distance = chamfer.evaluate_similarity(&query_transposed, &doc);
/// ```
pub struct QueryTransposedWithTilingApproach {
    /// Pre-allocated scratch buffer for per-query max similarities.
    /// Uses UnsafeCell for interior mutability with zero overhead.
    scratch: UnsafeCell<Vec<f32>>,
}

impl std::fmt::Debug for QueryTransposedWithTilingApproach {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryTransposedWithTilingApproach")
            .field(
                "scratch_capacity",
                // SAFETY: Read-only access to get capacity. Single-threaded access assumed.
                &unsafe { &*self.scratch.get() }.capacity(),
            )
            .finish()
    }
}

impl Default for QueryTransposedWithTilingApproach {
    fn default() -> Self {
        Self {
            scratch: UnsafeCell::new(Vec::new()),
        }
    }
}

impl QueryTransposedWithTilingApproach {
    /// Returns a mutable reference to the scratch buffer, resized and reset to f32::MIN.
    ///
    /// # Safety
    ///
    /// This uses UnsafeCell for interior mutability. The caller must ensure
    /// single-threaded access (this type is !Sync).
    #[inline(always)]
    #[allow(clippy::mut_from_ref)] // Intentional: UnsafeCell provides interior mutability
    fn scratch_reset(&self, num_queries: usize) -> &mut [f32] {
        // SAFETY: Single-threaded access assumed. This type is !Sync.
        let scratch = unsafe { &mut *self.scratch.get() };

        // Resize if needed, then reset to f32::MIN
        scratch.resize(num_queries, f32::MIN);
        scratch[..num_queries].fill(f32::MIN);

        &mut scratch[..num_queries]
    }
}

/// Block size for transposed layout (number of vectors per block).
const N: usize = 16;
/// Half block size for dual-register processing.
const N2: usize = N / 2;

impl DistanceFunction<&TransposedMultiVector, &MultiVector>
    for Chamfer<QueryTransposedWithTilingApproach>
{
    fn evaluate_similarity(&self, query: &TransposedMultiVector, doc: &MultiVector) -> f32 {
        let query_transposed = query.block_transposed();
        let num_queries = query.num_vectors();
        let num_docs = doc.num_vectors();

        // Use pre-allocated scratch buffer from the approach (resets to f32::MIN)
        let max_similarities = self.approach.scratch_reset(num_queries);

        // Process pairs of document vectors together to amortize query load costs
        for i in (0..num_docs.saturating_sub(1)).step_by(2) {
            // SAFETY: i + 1 < num_docs ensures both indices are valid.
            let (d1, d2) = unsafe { (doc.get_row_unchecked(i), doc.get_row_unchecked(i + 1)) };
            update_max_similarities_pair(d1, d2, query_transposed, max_similarities);
        }

        // Handle odd remainder document vector
        if !num_docs.is_multiple_of(2) {
            // SAFETY: num_docs - 1 < num_docs ensures index is valid
            update_max_similarities_single(
                unsafe { doc.get_row_unchecked(num_docs - 1) },
                query_transposed,
                max_similarities,
            );
        }

        // Sum negated max similarities to get Chamfer distance
        max_similarities.iter().map(|&s| -s).sum()
    }
}

/// Process two document vectors against all query blocks simultaneously.
/// Updates max_similarities in-place for each query vector.
///
/// This amortizes query memory loads by reusing them for both documents.
#[inline(always)]
fn update_max_similarities_pair(
    d1: &[f32],
    d2: &[f32],
    query: &BlockTranspose<N>,
    max_similarities: &mut [f32],
) {
    // Process full blocks of 16 query vectors
    for block in 0..query.full_blocks() {
        let (sim_d1_lo, sim_d1_hi, sim_d2_lo, sim_d2_hi) =
            compute_block_inner_products_pair(d1, d2, query, block);

        // Update max similarities for this block's query vectors
        let base_idx = block * N;
        update_max_from_simd_pair(
            &sim_d1_lo,
            &sim_d1_hi,
            &sim_d2_lo,
            &sim_d2_hi,
            &mut max_similarities[base_idx..base_idx + N],
        );
    }

    // Process remainder block if present
    let remainder = query.remainder();
    if remainder != 0 {
        let (sim_d1_lo, sim_d1_hi, sim_d2_lo, sim_d2_hi) =
            compute_block_inner_products_pair(d1, d2, query, query.full_blocks());

        let base_idx = query.full_blocks() * N;
        update_max_from_simd_pair_masked(
            &sim_d1_lo,
            &sim_d1_hi,
            &sim_d2_lo,
            &sim_d2_hi,
            &mut max_similarities[base_idx..],
            remainder,
        );
    }
}

/// Compute inner products between two document vectors and 16 query vectors in a block.
/// Returns (sim_d1_lo, sim_d1_hi, sim_d2_lo, sim_d2_hi) for query vectors 0-7 and 8-15.
#[inline(always)]
fn compute_block_inner_products_pair(
    d1: &[f32],
    d2: &[f32],
    query: &BlockTranspose<N>,
    block: usize,
) -> (f32s, f32s, f32s, f32s) {
    debug_assert!(block < query.num_blocks());

    // 8 accumulators total (4 per document)
    let mut sim_d1_lo_a = f32s::default(diskann_wide::ARCH);
    let mut sim_d1_hi_a = f32s::default(diskann_wide::ARCH);
    let mut sim_d1_lo_b = f32s::default(diskann_wide::ARCH);
    let mut sim_d1_hi_b = f32s::default(diskann_wide::ARCH);
    let mut sim_d2_lo_a = f32s::default(diskann_wide::ARCH);
    let mut sim_d2_hi_a = f32s::default(diskann_wide::ARCH);
    let mut sim_d2_lo_b = f32s::default(diskann_wide::ARCH);
    let mut sim_d2_hi_b = f32s::default(diskann_wide::ARCH);

    // SAFETY: block < num_blocks() ensures this access is in-bounds.
    let block_ptr = unsafe { query.block_ptr_unchecked(block) };

    let ncols = query.ncols();

    // Process 2 dimensions at a time
    for dim in (0..ncols.saturating_sub(1)).step_by(2) {
        // SAFETY: For all rows in this block, 16 reads are valid.
        // dim + 1 < ncols ensures all dimension accesses are in-bounds.
        let (q_lo_0, q_hi_0, q_lo_1, q_hi_1, d1_0, d1_1, d2_0, d2_1) = unsafe {
            (
                f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * dim)),
                f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * dim + N2)),
                f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * (dim + 1))),
                f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * (dim + 1) + N2)),
                f32s::splat(diskann_wide::ARCH, *d1.get_unchecked(dim)),
                f32s::splat(diskann_wide::ARCH, *d1.get_unchecked(dim + 1)),
                f32s::splat(diskann_wide::ARCH, *d2.get_unchecked(dim)),
                f32s::splat(diskann_wide::ARCH, *d2.get_unchecked(dim + 1)),
            )
        };

        // FMA for document 1
        sim_d1_lo_a = d1_0.mul_add_simd(q_lo_0, sim_d1_lo_a);
        sim_d1_hi_a = d1_0.mul_add_simd(q_hi_0, sim_d1_hi_a);
        sim_d1_lo_b = d1_1.mul_add_simd(q_lo_1, sim_d1_lo_b);
        sim_d1_hi_b = d1_1.mul_add_simd(q_hi_1, sim_d1_hi_b);

        // FMA for document 2
        sim_d2_lo_a = d2_0.mul_add_simd(q_lo_0, sim_d2_lo_a);
        sim_d2_hi_a = d2_0.mul_add_simd(q_hi_0, sim_d2_hi_a);
        sim_d2_lo_b = d2_1.mul_add_simd(q_lo_1, sim_d2_lo_b);
        sim_d2_hi_b = d2_1.mul_add_simd(q_hi_1, sim_d2_hi_b);
    }

    // Handle remaining dimension
    if !ncols.is_multiple_of(2) {
        let dim = ncols - 1;
        // SAFETY: dim < ncols ensures all dimension accesses are in-bounds.
        // block_ptr is valid for N * ncols elements.
        let (q_lo, q_hi, d1_val, d2_val) = unsafe {
            (
                f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * dim)),
                f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * dim + N2)),
                f32s::splat(diskann_wide::ARCH, *d1.get_unchecked(dim)),
                f32s::splat(diskann_wide::ARCH, *d2.get_unchecked(dim)),
            )
        };
        sim_d1_lo_a = d1_val.mul_add_simd(q_lo, sim_d1_lo_a);
        sim_d1_hi_a = d1_val.mul_add_simd(q_hi, sim_d1_hi_a);
        sim_d2_lo_a = d2_val.mul_add_simd(q_lo, sim_d2_lo_a);
        sim_d2_hi_a = d2_val.mul_add_simd(q_hi, sim_d2_hi_a);
    }

    (
        sim_d1_lo_a + sim_d1_lo_b,
        sim_d1_hi_a + sim_d1_hi_b,
        sim_d2_lo_a + sim_d2_lo_b,
        sim_d2_hi_a + sim_d2_hi_b,
    )
}

/// Update max similarities from SIMD results for a pair of documents.
#[inline(always)]
fn update_max_from_simd_pair(
    sim_d1_lo: &f32s,
    sim_d1_hi: &f32s,
    sim_d2_lo: &f32s,
    sim_d2_hi: &f32s,
    max_sims: &mut [f32],
) {
    debug_assert!(max_sims.len() >= N);

    // SAFETY: max_sims.len() >= N ensures we can read/write 16 f32s (2x f32x8)
    unsafe {
        // Load current max values as SIMD
        let current_max_lo = f32s::load_simd(diskann_wide::ARCH, max_sims.as_ptr());
        let current_max_hi = f32s::load_simd(diskann_wide::ARCH, max_sims.as_ptr().add(N2));

        // SIMD max: max(d1, d2) then max with current
        let max_from_docs_lo = sim_d1_lo.max_simd(*sim_d2_lo);
        let max_from_docs_hi = sim_d1_hi.max_simd(*sim_d2_hi);

        let new_max_lo = current_max_lo.max_simd(max_from_docs_lo);
        let new_max_hi = current_max_hi.max_simd(max_from_docs_hi);

        // Store back
        new_max_lo.store_simd(max_sims.as_mut_ptr());
        new_max_hi.store_simd(max_sims.as_mut_ptr().add(N2));
    }
}

/// Update max similarities from SIMD results with masking for remainder block.
#[inline(always)]
fn update_max_from_simd_pair_masked(
    sim_d1_lo: &f32s,
    sim_d1_hi: &f32s,
    sim_d2_lo: &f32s,
    sim_d2_hi: &f32s,
    max_sims: &mut [f32],
    valid_count: usize,
) {
    if valid_count >= N2 {
        // SIMD for full lo portion (8 elements)
        // SAFETY: valid_count >= N2 ensures we have at least 8 elements
        unsafe {
            let current_max_lo = f32s::load_simd(diskann_wide::ARCH, max_sims.as_ptr());
            let max_from_docs_lo = sim_d1_lo.max_simd(*sim_d2_lo);
            let new_max_lo = current_max_lo.max_simd(max_from_docs_lo);
            new_max_lo.store_simd(max_sims.as_mut_ptr());
        }

        // Scalar for hi remainder (0-7 elements)
        let hi_count = valid_count - N2;
        let arr_d1_hi = sim_d1_hi.to_array();
        let arr_d2_hi = sim_d2_hi.to_array();
        for i in 0..hi_count {
            let max_from_docs = arr_d1_hi[i].max(arr_d2_hi[i]);
            max_sims[N2 + i] = max_sims[N2 + i].max(max_from_docs);
        }
    } else {
        // Scalar for partial lo (1-7 elements)
        let arr_d1_lo = sim_d1_lo.to_array();
        let arr_d2_lo = sim_d2_lo.to_array();
        for i in 0..valid_count {
            let max_from_docs = arr_d1_lo[i].max(arr_d2_lo[i]);
            max_sims[i] = max_sims[i].max(max_from_docs);
        }
    }
}

/// Fallback for single document vector (odd remainder).
#[inline(always)]
fn update_max_similarities_single(
    doc_vec: &[f32],
    query: &BlockTranspose<N>,
    max_similarities: &mut [f32],
) {
    // Process full blocks of 16 query vectors
    for block in 0..query.full_blocks() {
        let (sim_lo, sim_hi) = compute_block_inner_products_single(doc_vec, query, block);

        let base_idx = block * N;
        update_max_from_simd_single(
            &sim_lo,
            &sim_hi,
            &mut max_similarities[base_idx..base_idx + N],
        );
    }

    // Process remainder block if present
    let remainder = query.remainder();
    if remainder != 0 {
        let (sim_lo, sim_hi) =
            compute_block_inner_products_single(doc_vec, query, query.full_blocks());

        let base_idx = query.full_blocks() * N;
        update_max_from_simd_single_masked(
            &sim_lo,
            &sim_hi,
            &mut max_similarities[base_idx..],
            remainder,
        );
    }
}

/// Compute inner products between one document vector and 16 query vectors in a block.
#[inline(always)]
fn compute_block_inner_products_single(
    doc_vec: &[f32],
    query: &BlockTranspose<N>,
    block: usize,
) -> (f32s, f32s) {
    debug_assert!(block < query.num_blocks());

    // Use 4 accumulator registers to reduce FMA dependency chains
    let mut sim_lo_a = f32s::default(diskann_wide::ARCH);
    let mut sim_hi_a = f32s::default(diskann_wide::ARCH);
    let mut sim_lo_b = f32s::default(diskann_wide::ARCH);
    let mut sim_hi_b = f32s::default(diskann_wide::ARCH);

    // SAFETY: block < num_blocks() ensures this access is in-bounds.
    let block_ptr = unsafe { query.block_ptr_unchecked(block) };

    let ncols = query.ncols();

    // Process 4 dimensions at a time
    for dim in (0..ncols.saturating_sub(3)).step_by(4) {
        // SAFETY: dim + 3 < ncols ensures all dimension accesses are in-bounds.
        // block_ptr is valid for N * ncols elements.
        let (q_lo_0, q_hi_0, q_lo_1, q_hi_1, q_lo_2, q_hi_2, q_lo_3, q_hi_3, d0, d1, d2, d3) = unsafe {
            (
                f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * dim)),
                f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * dim + N2)),
                f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * (dim + 1))),
                f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * (dim + 1) + N2)),
                f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * (dim + 2))),
                f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * (dim + 2) + N2)),
                f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * (dim + 3))),
                f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * (dim + 3) + N2)),
                f32s::splat(diskann_wide::ARCH, *doc_vec.get_unchecked(dim)),
                f32s::splat(diskann_wide::ARCH, *doc_vec.get_unchecked(dim + 1)),
                f32s::splat(diskann_wide::ARCH, *doc_vec.get_unchecked(dim + 2)),
                f32s::splat(diskann_wide::ARCH, *doc_vec.get_unchecked(dim + 3)),
            )
        };

        // Fused multiply-add into alternating accumulators
        sim_lo_a = d0.mul_add_simd(q_lo_0, sim_lo_a);
        sim_hi_a = d0.mul_add_simd(q_hi_0, sim_hi_a);
        sim_lo_b = d1.mul_add_simd(q_lo_1, sim_lo_b);
        sim_hi_b = d1.mul_add_simd(q_hi_1, sim_hi_b);
        sim_lo_a = d2.mul_add_simd(q_lo_2, sim_lo_a);
        sim_hi_a = d2.mul_add_simd(q_hi_2, sim_hi_a);
        sim_lo_b = d3.mul_add_simd(q_lo_3, sim_lo_b);
        sim_hi_b = d3.mul_add_simd(q_hi_3, sim_hi_b);
    }

    // Handle remaining dimensions
    for dim in (ncols - (ncols % 4))..ncols {
        // SAFETY: dim < ncols ensures all dimension accesses are in-bounds.
        // block_ptr is valid for N * ncols elements.
        let (q_lo, q_hi, d) = unsafe {
            (
                f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * dim)),
                f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * dim + N2)),
                f32s::splat(diskann_wide::ARCH, *doc_vec.get_unchecked(dim)),
            )
        };
        sim_lo_a = d.mul_add_simd(q_lo, sim_lo_a);
        sim_hi_a = d.mul_add_simd(q_hi, sim_hi_a);
    }

    (sim_lo_a + sim_lo_b, sim_hi_a + sim_hi_b)
}

/// Update max similarities from SIMD results for a single document.
#[inline(always)]
fn update_max_from_simd_single(sim_lo: &f32s, sim_hi: &f32s, max_sims: &mut [f32]) {
    debug_assert!(max_sims.len() >= N);

    // SAFETY: max_sims.len() >= N ensures we can read/write 16 f32s (2x f32x8)
    unsafe {
        // Load current max values as SIMD
        let current_max_lo = f32s::load_simd(diskann_wide::ARCH, max_sims.as_ptr());
        let current_max_hi = f32s::load_simd(diskann_wide::ARCH, max_sims.as_ptr().add(N2));

        // SIMD max with current
        let new_max_lo = current_max_lo.max_simd(*sim_lo);
        let new_max_hi = current_max_hi.max_simd(*sim_hi);

        // Store back
        new_max_lo.store_simd(max_sims.as_mut_ptr());
        new_max_hi.store_simd(max_sims.as_mut_ptr().add(N2));
    }
}

/// Update max similarities from SIMD results with masking for remainder block.
#[inline(always)]
fn update_max_from_simd_single_masked(
    sim_lo: &f32s,
    sim_hi: &f32s,
    max_sims: &mut [f32],
    valid_count: usize,
) {
    if valid_count >= N2 {
        // SIMD for full lo portion (8 elements)
        // SAFETY: valid_count >= N2 ensures we have at least 8 elements
        unsafe {
            let current_max_lo = f32s::load_simd(diskann_wide::ARCH, max_sims.as_ptr());
            let new_max_lo = current_max_lo.max_simd(*sim_lo);
            new_max_lo.store_simd(max_sims.as_mut_ptr());
        }

        // Scalar for hi remainder (0-7 elements)
        let hi_count = valid_count - N2;
        let arr_hi = sim_hi.to_array();
        for i in 0..hi_count {
            max_sims[N2 + i] = max_sims[N2 + i].max(arr_hi[i]);
        }
    } else {
        // Scalar for partial lo (1-7 elements)
        let arr_lo = sim_lo.to_array();
        for i in 0..valid_count {
            max_sims[i] = max_sims[i].max(arr_lo[i]);
        }
    }
}

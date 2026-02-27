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

use diskann_quantization::{multi_vector::{MatRef, Standard}, algorithms::kmeans::BlockTranspose};
use diskann_vector::DistanceFunction;
use diskann_wide::{SIMDMinMax, SIMDMulAdd, SIMDVector, arch::x86_64::V3};

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

// /// Block size for transposed layout (number of vectors per block).
// const N: usize = 16;
// /// Half block size for dual-register processing.
// const N2: usize = N / 2;

impl DistanceFunction<&TransposedMultiVector, &MultiVector>
    for Chamfer<QueryTransposedWithTilingApproach>
{
    fn evaluate_similarity(&self, query: &TransposedMultiVector, doc: &MultiVector) -> f32 {
        let query_transposed = query.block_transposed();
        let num_queries = query.num_vectors();
        let scratch_size = query_transposed.available_rows();

        // Use pre-allocated scratch buffer from the approach (resets to f32::MIN)
        let mut max_similarities = self.approach.scratch_reset(scratch_size);

        test_function(
            diskann_wide::ARCH,
            query_transposed,
            doc.as_view(),
            &mut max_similarities,
        );

        // Sum negated max similarities to get Chamfer distance
        max_similarities.iter().take(num_queries).map(|&s| -s).sum()
    }
}

diskann_wide::alias!(f32x8 = <V3>::f32x8);


#[inline(never)]
#[cold]
fn test_function_panic() {
    panic!("test_function: precondition failed (scratch.len != available_rows or dimension mismatch)");
}

#[derive(Debug, Clone)]
struct Budgets {
    l1: usize,
    l2: usize,
}

impl Budgets {
    const fn new() -> Self {
        Self {
            l1: L1_B_TILE_BUDGET,
            l2: L2_A_TILE_BUDGET,
        }
    }
}

/// Approximate usable L1 data cache in bytes (conservative estimate).
const L1_CACHE: usize = 48_000;

/// Approximate usable L2 cache in bytes (conservative estimate).
const L2_CACHE: usize = 1_250_000;

/// Fraction of L2 reserved for the A tile. The remainder accommodates B streaming
/// traffic, partial_c (Strategy B), and incidental cache pollution.
const L2_A_TILE_BUDGET: usize = L2_CACHE / 2;

/// Fraction of L1 available for the B tile. The A micro-panel is subtracted at
/// runtime since it depends on K; this is the total L1 budget before that subtraction.
const L1_B_TILE_BUDGET: usize = L1_CACHE * 3 / 4;

/// Number of A-rows processed per micro-kernel invocation (SIMD width: 2 × f32x8 = 16 lanes).
const A_PANEL: usize = 16;

/// Number of B-rows processed per micro-kernel invocation (broadcast unroll factor).
const B_PANEL: usize = 4;

/// A plan for performing the reducing GEMM that is used when the contraction dimension `k`
/// is small enough that a micro-panel of `A` fits comfortably in L1 cache with room for
/// multiple micro-panels of `B`.
#[derive(Debug, Clone, Copy)]
struct FullReduce {
    /// The number of micro panels of `A` that make up a tile.
    ///
    /// NOTE: This value must be multiplied by the `A` micro-panel size to obtain the
    /// number of rows!
    a_panels: usize,

    /// The number of micro panels of `B` that make up a tile.
    ///
    /// NOTE: This value must be multiplied by the `B` micro-panel size to obtain the
    /// number of rows!
    b_panels: usize,
}

impl FullReduce {
    // Construct a new `Plan` for the given contraction dimension.
    //
    // The tile sizes will be such that:
    //
    // * One `A` tile fits withing the L2 budget.
    // * One tile of `B` plus one micro-panel of `A` fit within the L1 budget.
    // * The tile sizes computed are guaranteed to be multiples of the
    //
    // Budget overruns may occur for sufficiently large `k`. In these situations, a micro
    // panel of `A` on its own can overrun the L1 cache and a different strategy should
    // be used.
    fn new<T>(k: usize, budgets: Budgets) -> Self {
        let row_bytes = (k * std::mem::size_of::<T>()).max(std::mem::size_of::<T>());

        let a_panels = (budgets.l2 / (row_bytes * A_PANEL)).max(1);

        let a_panel_bytes = A_PANEL * row_bytes;
        let b_tile_budget = budgets.l1.saturating_sub(a_panel_bytes);
        let b_panels = (b_tile_budget / (row_bytes * B_PANEL)).max(1);

        Self {
            a_panels,
            b_panels,
        }
    }
}

pub fn test_function(
    arch: V3,
    a: &BlockTranspose<A_PANEL>,
    b: MatRef<'_, Standard<f32>>,
    scratch: &mut [f32],
) {
    // Let's get this out of the way.
    if scratch.len() != a.available_rows()
        || a.ncols() != b.ncols()
    {
        test_function_panic();
    }

    let k = a.ncols();
    let plan = FullReduce::new::<f32>(k, Budgets::new());

    let op = |x: f32x8, y: f32x8| x.max_simd(y);

    // Precompute strides (in elements, not bytes).
    let a_panel_stride = A_PANEL * k;
    let a_tile_stride = a_panel_stride * plan.a_panels;
    let b_panel_stride = B_PANEL * k;
    let b_tile_stride = b_panel_stride * plan.b_panels;

    // SAFETY: We trust the caller.
    let pa_end = unsafe { a.as_ptr().add(a.nrows() * k) };
    // SAFETY: We trust the caller.
    let pb_end = unsafe { b.as_ptr().add(b.nrows() * k) };

    // Compute once how many remainder B-rows there are after all full panels.
    let remainder = b.nrows() % B_PANEL;
    // Pointer to the start of remainder B-rows (past all full panels).
    let pb_full_end = unsafe { pb_end.sub(remainder * k) };

    unsafe {
        let mut pa_tile = a.as_ptr();
        let mut pr_tile = scratch.as_mut_ptr();

        // Loop 1: Tiles of `A`.
        while pa_tile < pa_end {
            let pa_tile_end = pa_tile.add(a_tile_stride.min(pa_end.offset_from_unsigned(pa_tile)));

            let mut pb_tile = b.as_ptr();

            // Loop 2: Full B-tiles (every panel in the tile is complete).
            while pb_tile.wrapping_add(b_tile_stride) <= pb_full_end {
                let pb_tile_end = pb_tile.add(b_tile_stride);

                let mut pa_panel = pa_tile;
                let mut pr_panel = pr_tile;

                // Loop 3: Micro-panels of `A`.
                while pa_panel < pa_tile_end {
                    let mut pb_panel = pb_tile;

                    // Loop 4: Micro-panels of `B` (all full, no remainder check).
                    while pb_panel < pb_tile_end {
                        microkernel::<B_PANEL, _>(arch, pa_panel, pb_panel, k, pr_panel, op);
                        pb_panel = pb_panel.add(b_panel_stride);
                    }

                    pa_panel = pa_panel.add(a_panel_stride);
                    pr_panel = pr_panel.add(A_PANEL);
                }
                pb_tile = pb_tile.add(b_tile_stride);
            }

            // Peeled last B-tile: contains remaining full panels + remainder rows.
            if pb_tile < pb_end {
                let mut pa_panel = pa_tile;
                let mut pr_panel = pr_tile;

                // Loop 3 (peeled): Micro-panels of `A`.
                while pa_panel < pa_tile_end {
                    let mut pb_panel = pb_tile;

                    // Loop 4 (peeled): Full B-panels in the last tile.
                    while pb_panel < pb_full_end {
                        microkernel::<B_PANEL, _>(arch, pa_panel, pb_panel, k, pr_panel, op);
                        pb_panel = pb_panel.add(b_panel_stride);
                    }

                    // Remainder dispatch: 1–3 leftover B-rows.
                    if remainder == 1 {
                        microkernel::<1, _>(arch, pa_panel, pb_panel, k, pr_panel, op);
                    } else if remainder == 2 {
                        microkernel::<2, _>(arch, pa_panel, pb_panel, k, pr_panel, op);
                    } else if remainder == 3 {
                        microkernel::<3, _>(arch, pa_panel, pb_panel, k, pr_panel, op);
                    }

                    pa_panel = pa_panel.add(a_panel_stride);
                    pr_panel = pr_panel.add(A_PANEL);
                }
            }

            // NOTE: Use `wrapping_add` so we can still do this on the last iteration.
            pa_tile = pa_tile.wrapping_add(a_tile_stride);
            pr_tile = pr_tile.wrapping_add(A_PANEL * plan.a_panels);
        }
    }
}

// TODO: Unroll loops.
#[inline(always)]
unsafe fn microkernel<const UNROLL: usize, Op>(
    arch: V3,
    a_packed: *const f32,
    b: *const f32,
    k: usize,
    r: *mut f32,
    reduce: Op,
) where
    Op: Fn(f32x8, f32x8) -> f32x8,
    [f32x8; UNROLL]: Reduce<Element = f32x8>,
{
    let mut p0 = [f32x8::default(arch); UNROLL];
    let mut p1 = [f32x8::default(arch); UNROLL];
    let offsets: [usize; UNROLL] = core::array::from_fn(|i| k * i);

    let a_stride = 2 * f32x8::LANES;
    let a_stride_half = f32x8::LANES;

    for i in 0..k {
        unsafe {
            let a0 = f32x8::load_simd(arch, a_packed.add(a_stride * i));
            let a1 = f32x8::load_simd(arch, a_packed.add(a_stride * i + a_stride_half));

            for j in 0..UNROLL {
                let bj = f32x8::splat(arch, b.add(i + offsets[j]).read_unaligned());
                p0[j] = a0.mul_add_simd(bj, p0[j]);
                p1[j] = a1.mul_add_simd(bj, p1[j]);
            }
        }
    }

    let mut r0 = unsafe { f32x8::load_simd(arch, r) };
    let mut r1 = unsafe { f32x8::load_simd(arch, r.add(f32x8::LANES)) };

    r0 = reduce(r0, p0.reduce(&reduce));
    r1 = reduce(r1, p1.reduce(&reduce));

    unsafe { r0.store_simd(r) };
    unsafe { r1.store_simd(r.add(f32x8::LANES)) };
}

trait Reduce {
    type Element;
    fn reduce<F>(self, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element;
}

impl<T> Reduce for [T; 1]
where
    T: Copy
{
    type Element = T;

    #[inline(always)]
    fn reduce<F>(self, _f: F) -> T
    where
        F: Fn(T, T) ->  T{
        self[0]
    }
}

impl<T> Reduce for [T; 2]
where
    T: Copy
{
    type Element = T;

    #[inline(always)]
    fn reduce<F>(self, f: F) -> T
    where
        F: Fn(T, T) ->  T{
        f(self[0], self[1])
    }
}

impl<T> Reduce for [T; 3]
where
    T: Copy
{
    type Element = T;

    #[inline(always)]
    fn reduce<F>(self, f: F) -> T
    where
        F: Fn(T, T) ->  T{
        f(f(self[0], self[1]), self[2])
    }
}

impl<T> Reduce for [T; 4]
where
    T: Copy
{
    type Element = T;

    #[inline(always)]
    fn reduce<F>(self, f: F) -> T
    where
        F: Fn(T, T) ->  T{
        f(f(self[0], self[1]), f(self[2], self[3]))
    }
}

// /// Process two document vectors against all query blocks simultaneously.
// /// Updates max_similarities in-place for each query vector.
// ///
// /// This amortizes query memory loads by reusing them for both documents.
// #[inline(always)]
// fn update_max_similarities_pair(
//     d1: &[f32],
//     d2: &[f32],
//     query: &BlockTranspose<N>,
//     max_similarities: &mut [f32],
// ) {
//     // Process full blocks of 16 query vectors
//     for block in 0..query.full_blocks() {
//         let (sim_d1_lo, sim_d1_hi, sim_d2_lo, sim_d2_hi) =
//             compute_block_inner_products_pair(d1, d2, query, block);
//
//         // Update max similarities for this block's query vectors
//         let base_idx = block * N;
//         update_max_from_simd_pair(
//             &sim_d1_lo,
//             &sim_d1_hi,
//             &sim_d2_lo,
//             &sim_d2_hi,
//             &mut max_similarities[base_idx..base_idx + N],
//         );
//     }
//
//     // Process remainder block if present
//     let remainder = query.remainder();
//     if remainder != 0 {
//         let (sim_d1_lo, sim_d1_hi, sim_d2_lo, sim_d2_hi) =
//             compute_block_inner_products_pair(d1, d2, query, query.full_blocks());
//
//         let base_idx = query.full_blocks() * N;
//         update_max_from_simd_pair_masked(
//             &sim_d1_lo,
//             &sim_d1_hi,
//             &sim_d2_lo,
//             &sim_d2_hi,
//             &mut max_similarities[base_idx..],
//             remainder,
//         );
//     }
// }
//
// /// Compute inner products between two document vectors and 16 query vectors in a block.
// /// Returns (sim_d1_lo, sim_d1_hi, sim_d2_lo, sim_d2_hi) for query vectors 0-7 and 8-15.
// #[inline(always)]
// fn compute_block_inner_products_pair(
//     d1: &[f32],
//     d2: &[f32],
//     query: &BlockTranspose<N>,
//     block: usize,
// ) -> (f32s, f32s, f32s, f32s) {
//     debug_assert!(block < query.num_blocks());
//
//     // 8 accumulators total (4 per document)
//     let mut sim_d1_lo_a = f32s::default(diskann_wide::ARCH);
//     let mut sim_d1_hi_a = f32s::default(diskann_wide::ARCH);
//     let mut sim_d1_lo_b = f32s::default(diskann_wide::ARCH);
//     let mut sim_d1_hi_b = f32s::default(diskann_wide::ARCH);
//     let mut sim_d2_lo_a = f32s::default(diskann_wide::ARCH);
//     let mut sim_d2_hi_a = f32s::default(diskann_wide::ARCH);
//     let mut sim_d2_lo_b = f32s::default(diskann_wide::ARCH);
//     let mut sim_d2_hi_b = f32s::default(diskann_wide::ARCH);
//
//     // SAFETY: block < num_blocks() ensures this access is in-bounds.
//     let block_ptr = unsafe { query.block_ptr_unchecked(block) };
//
//     let ncols = query.ncols();
//
//     // Process 2 dimensions at a time
//     for dim in (0..ncols.saturating_sub(1)).step_by(2) {
//         // SAFETY: For all rows in this block, 16 reads are valid.
//         // dim + 1 < ncols ensures all dimension accesses are in-bounds.
//         let (q_lo_0, q_hi_0, q_lo_1, q_hi_1, d1_0, d1_1, d2_0, d2_1) = unsafe {
//             (
//                 f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * dim)),
//                 f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * dim + N2)),
//                 f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * (dim + 1))),
//                 f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * (dim + 1) + N2)),
//                 f32s::splat(diskann_wide::ARCH, *d1.get_unchecked(dim)),
//                 f32s::splat(diskann_wide::ARCH, *d1.get_unchecked(dim + 1)),
//                 f32s::splat(diskann_wide::ARCH, *d2.get_unchecked(dim)),
//                 f32s::splat(diskann_wide::ARCH, *d2.get_unchecked(dim + 1)),
//             )
//         };
//
//         // FMA for document 1
//         sim_d1_lo_a = d1_0.mul_add_simd(q_lo_0, sim_d1_lo_a);
//         sim_d1_hi_a = d1_0.mul_add_simd(q_hi_0, sim_d1_hi_a);
//         sim_d1_lo_b = d1_1.mul_add_simd(q_lo_1, sim_d1_lo_b);
//         sim_d1_hi_b = d1_1.mul_add_simd(q_hi_1, sim_d1_hi_b);
//
//         // FMA for document 2
//         sim_d2_lo_a = d2_0.mul_add_simd(q_lo_0, sim_d2_lo_a);
//         sim_d2_hi_a = d2_0.mul_add_simd(q_hi_0, sim_d2_hi_a);
//         sim_d2_lo_b = d2_1.mul_add_simd(q_lo_1, sim_d2_lo_b);
//         sim_d2_hi_b = d2_1.mul_add_simd(q_hi_1, sim_d2_hi_b);
//     }
//
//     // Handle remaining dimension
//     if !ncols.is_multiple_of(2) {
//         let dim = ncols - 1;
//         // SAFETY: dim < ncols ensures all dimension accesses are in-bounds.
//         // block_ptr is valid for N * ncols elements.
//         let (q_lo, q_hi, d1_val, d2_val) = unsafe {
//             (
//                 f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * dim)),
//                 f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * dim + N2)),
//                 f32s::splat(diskann_wide::ARCH, *d1.get_unchecked(dim)),
//                 f32s::splat(diskann_wide::ARCH, *d2.get_unchecked(dim)),
//             )
//         };
//         sim_d1_lo_a = d1_val.mul_add_simd(q_lo, sim_d1_lo_a);
//         sim_d1_hi_a = d1_val.mul_add_simd(q_hi, sim_d1_hi_a);
//         sim_d2_lo_a = d2_val.mul_add_simd(q_lo, sim_d2_lo_a);
//         sim_d2_hi_a = d2_val.mul_add_simd(q_hi, sim_d2_hi_a);
//     }
//
//     (
//         sim_d1_lo_a + sim_d1_lo_b,
//         sim_d1_hi_a + sim_d1_hi_b,
//         sim_d2_lo_a + sim_d2_lo_b,
//         sim_d2_hi_a + sim_d2_hi_b,
//     )
// }
//
// /// Update max similarities from SIMD results for a pair of documents.
// #[inline(always)]
// fn update_max_from_simd_pair(
//     sim_d1_lo: &f32s,
//     sim_d1_hi: &f32s,
//     sim_d2_lo: &f32s,
//     sim_d2_hi: &f32s,
//     max_sims: &mut [f32],
// ) {
//     debug_assert!(max_sims.len() >= N);
//
//     // SAFETY: max_sims.len() >= N ensures we can read/write 16 f32s (2x f32x8)
//     unsafe {
//         // Load current max values as SIMD
//         let current_max_lo = f32s::load_simd(diskann_wide::ARCH, max_sims.as_ptr());
//         let current_max_hi = f32s::load_simd(diskann_wide::ARCH, max_sims.as_ptr().add(N2));
//
//         // SIMD max: max(d1, d2) then max with current
//         let max_from_docs_lo = sim_d1_lo.max_simd(*sim_d2_lo);
//         let max_from_docs_hi = sim_d1_hi.max_simd(*sim_d2_hi);
//
//         let new_max_lo = current_max_lo.max_simd(max_from_docs_lo);
//         let new_max_hi = current_max_hi.max_simd(max_from_docs_hi);
//
//         // Store back
//         new_max_lo.store_simd(max_sims.as_mut_ptr());
//         new_max_hi.store_simd(max_sims.as_mut_ptr().add(N2));
//     }
// }
//
// /// Update max similarities from SIMD results with masking for remainder block.
// #[inline(always)]
// fn update_max_from_simd_pair_masked(
//     sim_d1_lo: &f32s,
//     sim_d1_hi: &f32s,
//     sim_d2_lo: &f32s,
//     sim_d2_hi: &f32s,
//     max_sims: &mut [f32],
//     valid_count: usize,
// ) {
//     if valid_count >= N2 {
//         // SIMD for full lo portion (8 elements)
//         // SAFETY: valid_count >= N2 ensures we have at least 8 elements
//         unsafe {
//             let current_max_lo = f32s::load_simd(diskann_wide::ARCH, max_sims.as_ptr());
//             let max_from_docs_lo = sim_d1_lo.max_simd(*sim_d2_lo);
//             let new_max_lo = current_max_lo.max_simd(max_from_docs_lo);
//             new_max_lo.store_simd(max_sims.as_mut_ptr());
//         }
//
//         // Scalar for hi remainder (0-7 elements)
//         let hi_count = valid_count - N2;
//         let arr_d1_hi = sim_d1_hi.to_array();
//         let arr_d2_hi = sim_d2_hi.to_array();
//         for i in 0..hi_count {
//             let max_from_docs = arr_d1_hi[i].max(arr_d2_hi[i]);
//             max_sims[N2 + i] = max_sims[N2 + i].max(max_from_docs);
//         }
//     } else {
//         // Scalar for partial lo (1-7 elements)
//         let arr_d1_lo = sim_d1_lo.to_array();
//         let arr_d2_lo = sim_d2_lo.to_array();
//         for i in 0..valid_count {
//             let max_from_docs = arr_d1_lo[i].max(arr_d2_lo[i]);
//             max_sims[i] = max_sims[i].max(max_from_docs);
//         }
//     }
// }
//
// /// Fallback for single document vector (odd remainder).
// #[inline(always)]
// fn update_max_similarities_single(
//     doc_vec: &[f32],
//     query: &BlockTranspose<N>,
//     max_similarities: &mut [f32],
// ) {
//     // Process full blocks of 16 query vectors
//     for block in 0..query.full_blocks() {
//         let (sim_lo, sim_hi) = compute_block_inner_products_single(doc_vec, query, block);
//
//         let base_idx = block * N;
//         update_max_from_simd_single(
//             &sim_lo,
//             &sim_hi,
//             &mut max_similarities[base_idx..base_idx + N],
//         );
//     }
//
//     // Process remainder block if present
//     let remainder = query.remainder();
//     if remainder != 0 {
//         let (sim_lo, sim_hi) =
//             compute_block_inner_products_single(doc_vec, query, query.full_blocks());
//
//         let base_idx = query.full_blocks() * N;
//         update_max_from_simd_single_masked(
//             &sim_lo,
//             &sim_hi,
//             &mut max_similarities[base_idx..],
//             remainder,
//         );
//     }
// }
//
// /// Compute inner products between one document vector and 16 query vectors in a block.
// #[inline(always)]
// fn compute_block_inner_products_single(
//     doc_vec: &[f32],
//     query: &BlockTranspose<N>,
//     block: usize,
// ) -> (f32s, f32s) {
//     debug_assert!(block < query.num_blocks());
//
//     // Use 4 accumulator registers to reduce FMA dependency chains
//     let mut sim_lo_a = f32s::default(diskann_wide::ARCH);
//     let mut sim_hi_a = f32s::default(diskann_wide::ARCH);
//     let mut sim_lo_b = f32s::default(diskann_wide::ARCH);
//     let mut sim_hi_b = f32s::default(diskann_wide::ARCH);
//
//     // SAFETY: block < num_blocks() ensures this access is in-bounds.
//     let block_ptr = unsafe { query.block_ptr_unchecked(block) };
//
//     let ncols = query.ncols();
//
//     // Process 4 dimensions at a time
//     for dim in (0..ncols.saturating_sub(3)).step_by(4) {
//         // SAFETY: dim + 3 < ncols ensures all dimension accesses are in-bounds.
//         // block_ptr is valid for N * ncols elements.
//         let (q_lo_0, q_hi_0, q_lo_1, q_hi_1, q_lo_2, q_hi_2, q_lo_3, q_hi_3, d0, d1, d2, d3) = unsafe {
//             (
//                 f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * dim)),
//                 f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * dim + N2)),
//                 f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * (dim + 1))),
//                 f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * (dim + 1) + N2)),
//                 f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * (dim + 2))),
//                 f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * (dim + 2) + N2)),
//                 f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * (dim + 3))),
//                 f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * (dim + 3) + N2)),
//                 f32s::splat(diskann_wide::ARCH, *doc_vec.get_unchecked(dim)),
//                 f32s::splat(diskann_wide::ARCH, *doc_vec.get_unchecked(dim + 1)),
//                 f32s::splat(diskann_wide::ARCH, *doc_vec.get_unchecked(dim + 2)),
//                 f32s::splat(diskann_wide::ARCH, *doc_vec.get_unchecked(dim + 3)),
//             )
//         };
//
//         // Fused multiply-add into alternating accumulators
//         sim_lo_a = d0.mul_add_simd(q_lo_0, sim_lo_a);
//         sim_hi_a = d0.mul_add_simd(q_hi_0, sim_hi_a);
//         sim_lo_b = d1.mul_add_simd(q_lo_1, sim_lo_b);
//         sim_hi_b = d1.mul_add_simd(q_hi_1, sim_hi_b);
//         sim_lo_a = d2.mul_add_simd(q_lo_2, sim_lo_a);
//         sim_hi_a = d2.mul_add_simd(q_hi_2, sim_hi_a);
//         sim_lo_b = d3.mul_add_simd(q_lo_3, sim_lo_b);
//         sim_hi_b = d3.mul_add_simd(q_hi_3, sim_hi_b);
//     }
//
//     // Handle remaining dimensions
//     for dim in (ncols - (ncols % 4))..ncols {
//         // SAFETY: dim < ncols ensures all dimension accesses are in-bounds.
//         // block_ptr is valid for N * ncols elements.
//         let (q_lo, q_hi, d) = unsafe {
//             (
//                 f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * dim)),
//                 f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * dim + N2)),
//                 f32s::splat(diskann_wide::ARCH, *doc_vec.get_unchecked(dim)),
//             )
//         };
//         sim_lo_a = d.mul_add_simd(q_lo, sim_lo_a);
//         sim_hi_a = d.mul_add_simd(q_hi, sim_hi_a);
//     }
//
//     (sim_lo_a + sim_lo_b, sim_hi_a + sim_hi_b)
// }
//
// /// Update max similarities from SIMD results for a single document.
// #[inline(always)]
// fn update_max_from_simd_single(sim_lo: &f32s, sim_hi: &f32s, max_sims: &mut [f32]) {
//     debug_assert!(max_sims.len() >= N);
//
//     // SAFETY: max_sims.len() >= N ensures we can read/write 16 f32s (2x f32x8)
//     unsafe {
//         // Load current max values as SIMD
//         let current_max_lo = f32s::load_simd(diskann_wide::ARCH, max_sims.as_ptr());
//         let current_max_hi = f32s::load_simd(diskann_wide::ARCH, max_sims.as_ptr().add(N2));
//
//         // SIMD max with current
//         let new_max_lo = current_max_lo.max_simd(*sim_lo);
//         let new_max_hi = current_max_hi.max_simd(*sim_hi);
//
//         // Store back
//         new_max_lo.store_simd(max_sims.as_mut_ptr());
//         new_max_hi.store_simd(max_sims.as_mut_ptr().add(N2));
//     }
// }
//
// /// Update max similarities from SIMD results with masking for remainder block.
// #[inline(always)]
// fn update_max_from_simd_single_masked(
//     sim_lo: &f32s,
//     sim_hi: &f32s,
//     max_sims: &mut [f32],
//     valid_count: usize,
// ) {
//     if valid_count >= N2 {
//         // SIMD for full lo portion (8 elements)
//         // SAFETY: valid_count >= N2 ensures we have at least 8 elements
//         unsafe {
//             let current_max_lo = f32s::load_simd(diskann_wide::ARCH, max_sims.as_ptr());
//             let new_max_lo = current_max_lo.max_simd(*sim_lo);
//             new_max_lo.store_simd(max_sims.as_mut_ptr());
//         }
//
//         // Scalar for hi remainder (0-7 elements)
//         let hi_count = valid_count - N2;
//         let arr_hi = sim_hi.to_array();
//         for i in 0..hi_count {
//             max_sims[N2 + i] = max_sims[N2 + i].max(arr_hi[i]);
//         }
//     } else {
//         // Scalar for partial lo (1-7 elements)
//         let arr_lo = sim_lo.to_array();
//         for i in 0..valid_count {
//             max_sims[i] = max_sims[i].max(arr_lo[i]);
//         }
//     }
// }

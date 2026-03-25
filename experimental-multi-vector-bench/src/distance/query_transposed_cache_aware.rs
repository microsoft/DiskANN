// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Cache-aware block-transposed SIMD implementation with tiling for multi-vector distance
//! computation.
//!
//! This module provides a SIMD-accelerated implementation that uses block-transposed
//! memory layout for **query** vectors (instead of documents), with documents remaining
//! in row-major format.
//!
//! # Cache-Aware Tiling Strategy
//!
//! Unlike the simpler [`QueryTransposedWithTilingApproach`](super::QueryTransposedWithTilingApproach)
//! which processes document pairs, this approach uses a reducing-GEMM pattern modeled
//! after high-performance BLAS implementations:
//!
//! - **L2 cache**: Tiles of the transposed query ("A") are sized to fit in L2.
//! - **L1 cache**: Tiles of the document ("B") plus one micro-panel of A are sized to fit in L1.
//! - **Micro-kernel**: A `16×4` micro-kernel (A_PANEL × B_PANEL) processes 16 query vectors
//!   against 4 document vectors per invocation, accumulating max-IP into a scratch buffer.
//!
//! # Memory Layout
//!
//! - **Query**: Block-transposed (16 vectors per block, dimensions contiguous)
//! - **Document**: Row-major (standard MultiVector format)

use std::cell::UnsafeCell;

use diskann_quantization::multi_vector::{BlockTransposedRef, MatRef, Standard};
use diskann_vector::DistanceFunction;
use diskann_wide::{SIMDMinMax, SIMDMulAdd, SIMDVector};

use super::Chamfer;
use crate::{MultiVector, TransposedMultiVector};

diskann_wide::alias!(f32s = f32x8);

/// Cache-aware block-transposed SIMD approach with tiling for Chamfer distance computation.
///
/// This approach uses a block-transposed memory layout for **query** vectors and
/// row-major format for documents. It tiles loop iterations to respect L1/L2 cache
/// budgets, and uses a `16×4` micro-kernel for the inner computation.
///
/// The approach holds a pre-allocated scratch buffer for storing per-query max similarities,
/// avoiding allocation on each `evaluate_similarity` call.
pub struct QueryTransposedCacheAwareApproach {
    /// Pre-allocated scratch buffer for per-query max similarities.
    /// Uses UnsafeCell for interior mutability with zero overhead.
    scratch: UnsafeCell<Vec<f32>>,
}

impl std::fmt::Debug for QueryTransposedCacheAwareApproach {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryTransposedCacheAwareApproach")
            .field(
                "scratch_capacity",
                // SAFETY: Read-only access to get capacity. Single-threaded access assumed.
                &unsafe { &*self.scratch.get() }.capacity(),
            )
            .finish()
    }
}

impl Default for QueryTransposedCacheAwareApproach {
    fn default() -> Self {
        Self {
            scratch: UnsafeCell::new(Vec::new()),
        }
    }
}

impl QueryTransposedCacheAwareApproach {
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

impl DistanceFunction<&TransposedMultiVector, &MultiVector>
    for Chamfer<QueryTransposedCacheAwareApproach>
{
    fn evaluate_similarity(&self, query: &TransposedMultiVector, doc: &MultiVector) -> f32 {
        let query_view = query.as_view();
        let num_queries = query.nrows();
        let scratch_size = query.available_rows();

        // Use pre-allocated scratch buffer from the approach (resets to f32::MIN)
        let max_similarities = self.approach.scratch_reset(scratch_size);

        cache_aware_chamfer(
            diskann_wide::ARCH,
            query_view,
            doc.as_view(),
            max_similarities,
        );

        // Sum negated max similarities to get Chamfer distance
        max_similarities.iter().take(num_queries).map(|&s| -s).sum()
    }
}

// ── Cache budget constants ───────────────────────────────────────

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

// ── Tile planner ─────────────────────────────────────────────────

/// A plan for performing the reducing GEMM when the contraction dimension `k`
/// is small enough that a micro-panel of `A` fits comfortably in L1 cache with room for
/// multiple micro-panels of `B`.
#[derive(Debug, Clone, Copy)]
struct FullReduce {
    /// The number of micro panels of `A` that make up a tile.
    a_panels: usize,

    /// The number of micro panels of `B` that make up a tile.
    b_panels: usize,
}

impl FullReduce {
    fn new<T>(k: usize, l2_budget: usize, l1_budget: usize) -> Self {
        let row_bytes = (k * std::mem::size_of::<T>()).max(std::mem::size_of::<T>());

        let a_panels = (l2_budget / (row_bytes * A_PANEL)).max(1);

        let a_panel_bytes = A_PANEL * row_bytes;
        let b_tile_budget = l1_budget.saturating_sub(a_panel_bytes);
        let b_panels = (b_tile_budget / (row_bytes * B_PANEL)).max(1);

        Self {
            a_panels,
            b_panels,
        }
    }
}

// ── Core computation ─────────────────────────────────────────────

#[inline(never)]
#[cold]
fn cache_aware_chamfer_panic() {
    panic!("cache_aware_chamfer: precondition failed (scratch.len != available_rows or dimension mismatch)");
}

/// Compute the reducing max-IP GEMM between a block-transposed query (`a`) and
/// a row-major document matrix (`b`), writing per-query max similarities into `scratch`.
pub fn cache_aware_chamfer(
    arch: diskann_wide::arch::Current,
    a: BlockTransposedRef<'_, f32, 16>,
    b: MatRef<'_, Standard<f32>>,
    scratch: &mut [f32],
) {
    if scratch.len() != a.available_rows() || a.ncols() != b.vector_dim() {
        cache_aware_chamfer_panic();
    }

    let k = a.ncols();
    let plan = FullReduce::new::<f32>(k, L2_A_TILE_BUDGET, L1_B_TILE_BUDGET);

    let op = |x: f32s, y: f32s| x.max_simd(y);

    // Precompute strides (in elements, not bytes).
    let a_panel_stride = A_PANEL * k;
    let a_tile_stride = a_panel_stride * plan.a_panels;
    let b_panel_stride = B_PANEL * k;
    let b_tile_stride = b_panel_stride * plan.b_panels;

    let b_slice = b.as_slice();
    let b_nrows = b.num_vectors();

    // SAFETY: a.as_ptr() is valid for a.available_rows() * k elements.
    let pa_end = unsafe { a.as_ptr().add(a.available_rows() * k) };
    // SAFETY: b_slice.as_ptr() is valid for b_nrows * k elements.
    let pb_end = unsafe { b_slice.as_ptr().add(b_nrows * k) };

    // Compute how many remainder B-rows there are after all full panels.
    let remainder = b_nrows % B_PANEL;
    // Pointer past all full B-panels.
    // SAFETY: remainder < B_PANEL and pb_end points past b_nrows * k elements,
    // so subtracting remainder * k stays within the allocation.
    let pb_full_end = unsafe { pb_end.sub(remainder * k) };

    // SAFETY: All pointer arithmetic stays within the respective allocations.
    // a.as_ptr() is valid for available_rows * k elements.
    // b_slice.as_ptr() is valid for b_nrows * k elements.
    // scratch.as_mut_ptr() is valid for available_rows elements.
    unsafe {
        let mut pa_tile = a.as_ptr();
        let mut pr_tile = scratch.as_mut_ptr();

        // Loop 1: Tiles of `A`.
        while pa_tile < pa_end {
            let remaining_a = (pa_end as usize - pa_tile as usize) / std::mem::size_of::<f32>();
            let pa_tile_end = pa_tile.add(a_tile_stride.min(remaining_a));

            let mut pb_tile = b_slice.as_ptr();

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
                        microkernel::<B_PANEL, _>(arch, pa_panel, pb_panel, k, pr_panel, &op);
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
                        microkernel::<B_PANEL, _>(arch, pa_panel, pb_panel, k, pr_panel, &op);
                        pb_panel = pb_panel.add(b_panel_stride);
                    }

                    // Remainder dispatch: 1–3 leftover B-rows.
                    if remainder == 1 {
                        microkernel::<1, _>(arch, pa_panel, pb_panel, k, pr_panel, &op);
                    } else if remainder == 2 {
                        microkernel::<2, _>(arch, pa_panel, pb_panel, k, pr_panel, &op);
                    } else if remainder == 3 {
                        microkernel::<3, _>(arch, pa_panel, pb_panel, k, pr_panel, &op);
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

// ── Micro-kernel ─────────────────────────────────────────────────

#[inline(always)]
unsafe fn microkernel<const UNROLL: usize, Op>(
    arch: diskann_wide::arch::Current,
    a_packed: *const f32,
    b: *const f32,
    k: usize,
    r: *mut f32,
    reduce: &Op,
) where
    Op: Fn(f32s, f32s) -> f32s,
    [f32s; UNROLL]: Reduce<Element = f32s>,
{
    let mut p0 = [f32s::default(arch); UNROLL];
    let mut p1 = [f32s::default(arch); UNROLL];
    let offsets: [usize; UNROLL] = core::array::from_fn(|i| k * i);

    let a_stride = 2 * f32s::LANES;
    let a_stride_half = f32s::LANES;

    for i in 0..k {
        // SAFETY: a_packed points to A_PANEL * k contiguous f32s (one micro-panel).
        // b points to UNROLL rows of k contiguous f32s each. All reads are in-bounds.
        unsafe {
            let a0 = f32s::load_simd(arch, a_packed.add(a_stride * i));
            let a1 = f32s::load_simd(arch, a_packed.add(a_stride * i + a_stride_half));

            for j in 0..UNROLL {
                let bj = f32s::splat(arch, b.add(i + offsets[j]).read_unaligned());
                p0[j] = a0.mul_add_simd(bj, p0[j]);
                p1[j] = a1.mul_add_simd(bj, p1[j]);
            }
        }
    }

    // SAFETY: r points to at least A_PANEL = 16 writable f32s (2 × f32x8).
    let mut r0 = unsafe { f32s::load_simd(arch, r) };
    // SAFETY: r + f32s::LANES is within the same A_PANEL-sized scratch region.
    let mut r1 = unsafe { f32s::load_simd(arch, r.add(f32s::LANES)) };

    r0 = reduce(r0, p0.reduce(reduce));
    r1 = reduce(r1, p1.reduce(reduce));

    // SAFETY: r points to at least A_PANEL = 16 writable f32s (2 × f32x8).
    unsafe { r0.store_simd(r) };
    // SAFETY: r + f32s::LANES is within the same A_PANEL-sized scratch region.
    unsafe { r1.store_simd(r.add(f32s::LANES)) };
}

// ── Reduce trait for compile-time unroll reduction ───────────────

trait Reduce {
    type Element;
    fn reduce<F>(&self, f: &F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element;
}

impl<T: Copy> Reduce for [T; 1] {
    type Element = T;

    #[inline(always)]
    fn reduce<F>(&self, _f: &F) -> T
    where
        F: Fn(T, T) -> T,
    {
        self[0]
    }
}

impl<T: Copy> Reduce for [T; 2] {
    type Element = T;

    #[inline(always)]
    fn reduce<F>(&self, f: &F) -> T
    where
        F: Fn(T, T) -> T,
    {
        f(self[0], self[1])
    }
}

impl<T: Copy> Reduce for [T; 3] {
    type Element = T;

    #[inline(always)]
    fn reduce<F>(&self, f: &F) -> T
    where
        F: Fn(T, T) -> T,
    {
        f(f(self[0], self[1]), self[2])
    }
}

impl<T: Copy> Reduce for [T; 4] {
    type Element = T;

    #[inline(always)]
    fn reduce<F>(&self, f: &F) -> T
    where
        F: Fn(T, T) -> T,
    {
        f(f(self[0], self[1]), f(self[2], self[3]))
    }
}

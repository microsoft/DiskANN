// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Generic tiling loop and shared reduction utilities.
//!
//! # Tiling Strategy
//!
//! This approach uses a reducing-GEMM pattern modeled after high-performance BLAS
//! implementations:
//!
//! - **L2 cache**: Tiles of A (the transposed query) are sized to fit in L2.
//! - **L1 cache**: Tiles of B (the document) plus one micro-panel of A are sized
//!   to fit in L1.
//! - **Micro-kernel**: An `A_PANEL × B_PANEL` micro-kernel (e.g. 16×4 for f32 on V3)
//!   processes a panel of A rows against a panel of B rows per invocation,
//!   accumulating max-IP into a scratch buffer. The panel sizes are determined
//!   by the `Kernel<A>` implementation for each element type.

use diskann_wide::Architecture;

use super::layouts::{ConvertTo, Layout};
use super::{Kernel, TileBudget};

// ── Tile planner ─────────────────────────────────────────────────

/// Tile-panel counts derived from cache budgets.
#[derive(Debug, Clone, Copy)]
struct FullReduce {
    a_panels_per_tile: usize,

    b_panels_per_tile: usize,
}

impl FullReduce {
    /// Compute A-tile and B-tile panel counts from cache budgets.
    ///
    /// The L1 budget is reduced by one A micro-panel before splitting it into
    /// B panels, since both must coexist in L1 during the inner loop.
    fn new(
        a_row_bytes: usize,
        b_row_bytes: usize,
        a_panel: usize,
        b_panel: usize,
        budget: TileBudget,
    ) -> Self {
        let a_row_bytes = a_row_bytes.max(1);
        let b_row_bytes = b_row_bytes.max(1);

        let a_panels_per_tile = (budget.l2_a / (a_row_bytes * a_panel)).max(1);

        let a_panel_bytes = a_panel * a_row_bytes;
        let b_tile_budget = budget.l1_b.saturating_sub(a_panel_bytes);
        let b_panels_per_tile = (b_tile_budget / (b_row_bytes * b_panel)).max(1);

        Self {
            a_panels_per_tile,
            b_panels_per_tile,
        }
    }
}

// ── Generic tiled reduce ─────────────────────────────────────────

/// Execute the 5-level tiling loop with a pluggable SIMD micro-kernel and
/// tile-level layout converters.
///
/// The loop nest is:
/// ```text
/// Loop 1: A tiles     (sized to L2) — convert via `ca`
///   Loop 2: B tiles   (sized to L1) — convert via `cb`
///     Loop 3: A panels (micro-panels within converted A tile)
///       Loop 4: B panels (micro-panels within converted B tile)
///         Loop 5: k (contraction dim, inside K::full_panel / K::partial_panel)
/// ```
///
/// Conversion from storage layout to kernel layout happens once per tile
/// (not per panel), amortizing cost over the entire tile.
///
/// # Safety
///
/// * `a_ptr` must be valid for `a_padded_nrows * k` elements of `AElem`.
/// * `a_padded_nrows` must be a multiple of `K::A_PANEL`.
/// * `b_ptr` must be valid for `b_nrows * k` elements of `BElem`.
/// * `scratch` must have length ≥ `a_padded_nrows` and be initialized by caller.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn tiled_reduce<A, K, LA, LB>(
    arch: A,
    ca: &LA,
    cb: &LB,
    a_ptr: *const LA::Element,
    a_padded_nrows: usize,
    b_ptr: *const LB::Element,
    b_nrows: usize,
    k: usize,
    scratch: &mut [f32],
    budget: TileBudget,
) where
    A: Architecture,
    K: Kernel<A>,
    LA: ConvertTo<A, K::Left>,
    LB: ConvertTo<A, K::Right>,
{
    let a_row_bytes = k * std::mem::size_of::<<K::Left as Layout>::Element>();
    let b_row_bytes = k * std::mem::size_of::<<K::Right as Layout>::Element>();
    let plan = FullReduce::new(a_row_bytes, b_row_bytes, K::A_PANEL, K::B_PANEL, budget);

    let b_src_panel_stride = K::B_PANEL * k;
    let b_src_tile_stride = b_src_panel_stride * plan.b_panels_per_tile;

    let a_kern_panel_stride = K::A_PANEL * k;
    let b_kern_panel_stride = K::B_PANEL * k;

    let b_remainder = b_nrows % K::B_PANEL;

    assert_eq!(
        a_padded_nrows % K::A_PANEL,
        0,
        "a_padded_nrows ({a_padded_nrows}) must be a multiple of A_PANEL ({})",
        K::A_PANEL,
    );

    // Zero-dimensional vectors have IP = 0 for every pair. Fill scratch and
    // return to avoid zero-stride infinite loops in the tiling nest.
    if k == 0 {
        if b_nrows > 0 {
            scratch[..a_padded_nrows].fill(0.0);
        }
        return;
    }

    // Allocate conversion buffers once. Identity conversions use `Buffer = ()`
    // and these calls are no-ops.
    let a_tile_rows = K::A_PANEL * plan.a_panels_per_tile;
    let b_tile_rows = K::B_PANEL * plan.b_panels_per_tile;
    let mut a_buf = ca.new_buffer(a_tile_rows, k);
    let mut b_buf = cb.new_buffer(b_tile_rows, k);

    // SAFETY: Caller guarantees b_ptr is valid for b_nrows * k elements.
    let pb_end = unsafe { b_ptr.add(b_nrows * k) };
    // SAFETY: b_remainder < B_PANEL, so pb_end - b_remainder * k is within allocation.
    let pb_full_end = unsafe { pb_end.sub(b_remainder * k) };

    // SAFETY: All pointer arithmetic stays within the respective allocations.
    unsafe {
        let mut rows_done: usize = 0;

        // Loop 1: Tiles of `A`.
        while rows_done < a_padded_nrows {
            let tile_rows = a_tile_rows.min(a_padded_nrows - rows_done);
            let pa_tile_src = a_ptr.add(rows_done * k);
            // SAFETY: rows_done < a_padded_nrows (loop condition), so the
            // pointer is in-bounds.
            let pr_tile = scratch.as_mut_ptr().add(rows_done);

            // Convert A tile from storage layout to kernel layout.
            let pa_tile = ca.convert(&mut a_buf, arch, pa_tile_src, tile_rows, k);
            let pa_tile_end = pa_tile.add(tile_rows * k);

            let mut pb_tile_src = b_ptr;

            // Loop 2: Full B-tiles (every panel in the tile is complete).
            // SAFETY: `pb_tile_src` is always in `[b_ptr, pb_full_end]` — both within
            // the same allocation — so `offset_from` is well-defined.
            while pb_full_end.offset_from(pb_tile_src) >= b_src_tile_stride as isize {
                // Convert B tile from storage layout to kernel layout.
                let pb_tile = cb.convert(&mut b_buf, arch, pb_tile_src, b_tile_rows, k);
                let pb_tile_end = pb_tile.add(b_tile_rows * k);

                let mut pa_panel = pa_tile;
                let mut pr_panel = pr_tile;

                // Loop 3: Micro-panels of `A`.
                while pa_panel < pa_tile_end {
                    let mut pb_panel = pb_tile;

                    // Loop 4: Micro-panels of `B` (all full, no remainder check).
                    while pb_panel < pb_tile_end {
                        K::full_panel(arch, pa_panel, pb_panel, k, pr_panel);
                        pb_panel = pb_panel.add(b_kern_panel_stride);
                    }

                    pa_panel = pa_panel.add(a_kern_panel_stride);
                    pr_panel = pr_panel.add(K::A_PANEL);
                }
                pb_tile_src = pb_tile_src.add(b_src_tile_stride);
            }

            // Peeled last B-tile: contains remaining full panels + remainder rows.
            if pb_tile_src < pb_end {
                let remaining_b_rows = b_nrows - ((pb_tile_src.offset_from(b_ptr) as usize) / k);
                // Convert remaining B rows.
                let pb_tile = cb.convert(&mut b_buf, arch, pb_tile_src, remaining_b_rows, k);

                let full_panels_in_remainder = remaining_b_rows / K::B_PANEL;
                let pb_full_end_local = pb_tile.add(full_panels_in_remainder * b_kern_panel_stride);

                let mut pa_panel = pa_tile;
                let mut pr_panel = pr_tile;

                // Loop 3 (peeled): Micro-panels of `A`.
                while pa_panel < pa_tile_end {
                    let mut pb_panel = pb_tile;

                    // Loop 4 (peeled): Full B-panels in the last tile.
                    while pb_panel < pb_full_end_local {
                        K::full_panel(arch, pa_panel, pb_panel, k, pr_panel);
                        pb_panel = pb_panel.add(b_kern_panel_stride);
                    }

                    // Remainder dispatch: 1..(B_PANEL-1) leftover B-rows.
                    if b_remainder > 0 {
                        K::partial_panel(arch, b_remainder, pa_panel, pb_panel, k, pr_panel);
                    }

                    pa_panel = pa_panel.add(a_kern_panel_stride);
                    pr_panel = pr_panel.add(K::A_PANEL);
                }
            }

            rows_done += tile_rows;
        }
    }
}

// ── Reduce trait for compile-time unroll reduction ───────────────

/// Compile-time unroll reduction over fixed-size arrays.
///
/// Used by the micro-kernel to reduce `UNROLL` accumulators into a single value
/// using a caller-supplied binary operator (e.g. `max_simd`).
pub(super) trait Reduce {
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

#[cfg(test)]
mod tests {
    use super::*;

    use diskann_wide::arch::Scalar;

    use super::super::f32::{F32Kernel, max_ip_kernel};
    use super::super::layouts;
    use crate::multi_vector::{BlockTransposed, MatRef, Standard};

    #[test]
    fn basic_panel_counts() {
        // 16 A-rows × 256 bytes/row = 4096 bytes per A-panel.
        // L2 budget 40960 → 40960 / 4096 = 10 A-panels.
        // One A-panel = 4096 bytes, L1 budget 36000 → 36000 - 4096 = 31904.
        // 4 B-rows × 256 bytes/row = 1024 bytes per B-panel.
        // 31904 / 1024 = 31 B-panels.
        let plan = FullReduce::new(
            256,
            256,
            16,
            4,
            TileBudget {
                l2_a: 40960,
                l1_b: 36000,
            },
        );
        assert_eq!(plan.a_panels_per_tile, 10);
        assert_eq!(plan.b_panels_per_tile, 31);
    }

    #[test]
    fn tiny_budget_clamps_to_one() {
        // Budget too small for even one panel — clamp to 1.
        let plan = FullReduce::new(1024, 1024, 16, 4, TileBudget { l2_a: 1, l1_b: 1 });
        assert_eq!(plan.a_panels_per_tile, 1);
        assert_eq!(plan.b_panels_per_tile, 1);
    }

    #[test]
    fn zero_byte_rows_clamped() {
        // Zero-byte rows (e.g. k=0) should not divide by zero.
        // FullReduce clamps row bytes to max(1), so a_row_bytes=1, b_row_bytes=1.
        let plan = FullReduce::new(
            0,
            0,
            16,
            4,
            TileBudget {
                l2_a: 100_000,
                l1_b: 50_000,
            },
        );
        // a_panels = 100_000 / (1 * 16) = 6250
        assert_eq!(plan.a_panels_per_tile, 6250);
        // a_panel_bytes = 16 * 1 = 16. b_tile_budget = 50_000 - 16 = 49_984.
        // b_panels = 49_984 / (1 * 4) = 12_496
        assert_eq!(plan.b_panels_per_tile, 12_496);
    }

    #[test]
    fn exact_fit_one_panel() {
        // Budget exactly fits one A-panel (16 × 64 = 1024 bytes).
        // No room for a second → a_panels = 1.
        let plan = FullReduce::new(
            64,
            64,
            16,
            4,
            TileBudget {
                l2_a: 1024,
                l1_b: 2048,
            },
        );
        assert_eq!(plan.a_panels_per_tile, 1);
        // L1: 2048 - 16*64(=1024) = 1024 for B. 4*64=256 per B-panel → 4 panels.
        assert_eq!(plan.b_panels_per_tile, 4);
    }

    #[test]
    fn l1_saturated_by_a_panel() {
        // A-panel alone exceeds L1 budget → b_tile_budget saturates to 0,
        // b_panels_per_tile clamps to 1.
        let plan = FullReduce::new(
            1024,
            64,
            16,
            4,
            TileBudget {
                l2_a: 100_000,
                l1_b: 100,
            },
        );
        assert_eq!(plan.b_panels_per_tile, 1);
    }

    #[test]
    #[should_panic(expected = "must be a multiple of A_PANEL")]
    fn panics_on_unaligned_a_rows() {
        let k = 4;
        // 9 is not a multiple of A_PANEL (8).
        let a = vec![0.0f32; 9 * k];
        let b = vec![0.0f32; 2 * k];
        let mut scratch = vec![f32::MIN; 16];

        let ca = layouts::BlockTransposed::<f32, 8>::new();
        let cb = layouts::RowMajor::<f32>::new();

        // SAFETY: pointers and scratch are correctly sized; we expect a panic.
        unsafe {
            super::tiled_reduce::<Scalar, F32Kernel<8>, _, _>(
                Scalar::new(),
                &ca,
                &cb,
                a.as_ptr(),
                9,
                b.as_ptr(),
                2,
                k,
                &mut scratch,
                TileBudget::default(),
            );
        }
    }

    #[test]
    fn zero_dim_fills_scratch_and_returns() {
        let a_rows = 8;
        let b_rows = 3;
        let k = 0;

        let a = Vec::<f32>::new();
        let b = Vec::<f32>::new();
        let mut scratch = vec![f32::MIN; a_rows];

        let ca = layouts::BlockTransposed::<f32, 8>::new();
        let cb = layouts::RowMajor::<f32>::new();

        // SAFETY: k == 0 so no elements are read; pointers are never dereferenced.
        unsafe {
            super::tiled_reduce::<Scalar, F32Kernel<8>, _, _>(
                Scalar::new(),
                &ca,
                &cb,
                a.as_ptr(),
                a_rows,
                b.as_ptr(),
                b_rows,
                k,
                &mut scratch,
                TileBudget::default(),
            );
        }

        for &v in &scratch {
            assert_eq!(v, 0.0, "zero-dim IP should be 0.0");
        }
    }

    #[test]
    fn zero_dim_zero_docs_leaves_scratch_untouched() {
        let a_rows = 8;
        let mut scratch = vec![f32::MIN; a_rows];

        let ca = layouts::BlockTransposed::<f32, 8>::new();
        let cb = layouts::RowMajor::<f32>::new();

        // SAFETY: k == 0, b_nrows == 0; no elements read.
        unsafe {
            super::tiled_reduce::<Scalar, F32Kernel<8>, _, _>(
                Scalar::new(),
                &ca,
                &cb,
                [].as_ptr(),
                a_rows,
                [].as_ptr(),
                0,
                0,
                &mut scratch,
                TileBudget::default(),
            );
        }

        for &v in &scratch {
            assert_eq!(v, f32::MIN, "zero docs should leave scratch untouched");
        }
    }

    #[test]
    fn reduce_folds_correctly() {
        let max = |a: f32, b: f32| a.max(b);
        assert_eq!([5.0f32].reduce(&max), 5.0);
        assert_eq!([1.0f32, 3.0].reduce(&max), 3.0);
        assert_eq!([2.0f32, 1.0, 4.0].reduce(&max), 4.0);
        assert_eq!([3.0f32, 1.0, 4.0, 2.0].reduce(&max), 4.0);
    }

    /// Verify the exact fold order of each `Reduce` impl using a
    /// non-commutative operator (subtraction).
    ///
    /// - `[a; 1]`       → `a`
    /// - `[a, b; 2]`    → `a - b`
    /// - `[a, b, c; 3]` → `(a - b) - c`         (left fold)
    /// - `[a, b, c, d; 4]` → `(a - b) - (c - d)` (balanced tree)
    #[test]
    fn reduce_fold_order() {
        let sub = |a: f32, b: f32| a - b;
        // [10]                          → 10
        assert_eq!([10.0f32].reduce(&sub), 10.0);
        // [10, 3]                       → 10 - 3 = 7
        assert_eq!([10.0f32, 3.0].reduce(&sub), 7.0);
        // [10, 3, 1]                    → (10 - 3) - 1 = 6
        assert_eq!([10.0f32, 3.0, 1.0].reduce(&sub), 6.0);
        // [10, 3, 1, 2]                 → (10 - 3) - (1 - 2) = 7 - (-1) = 8
        assert_eq!([10.0f32, 3.0, 1.0, 2.0].reduce(&sub), 8.0);
    }

    #[test]
    fn tiled_reduce_matches_naive() {
        fn naive_max_ip(
            a: &[f32],
            a_nrows: usize,
            b: &[f32],
            b_nrows: usize,
            k: usize,
        ) -> Vec<f32> {
            (0..a_nrows)
                .map(|i| {
                    (0..b_nrows)
                        .map(|j| (0..k).map(|d| a[i * k + d] * b[j * k + d]).sum::<f32>())
                        .fold(f32::MIN, f32::max)
                })
                .collect()
        }

        // Comprehensive cases matching the TEST_CASES arrays in f32/f16 modules,
        // plus additional shapes to exercise multi-tile and no-remainder paths.
        // (a_nrows, b_nrows, dim)
        let cases: &[(usize, usize, usize)] = &[
            (1, 1, 1),   // Degenerate single-element
            (1, 1, 2),   // Minimal non-trivial
            (1, 1, 4),   // Single query, single doc
            (1, 5, 8),   // Single query, multiple docs
            (5, 1, 8),   // Multiple queries, single doc
            (3, 2, 0),   // Zero dimensions, both have rows
            (3, 0, 4),   // Zero docs
            (3, 2, 3),   // Prime k
            (3, 4, 16),  // General case
            (5, 3, 5),   // Prime k, A-remainder on aarch64
            (7, 7, 32),  // Square case
            (2, 3, 7),   // k not divisible by SIMD lanes
            (2, 3, 128), // Larger dimension
            (8, 3, 4),   // Single A-panel, B remainder (3 % 2 = 1)
            (16, 5, 8),  // Two A-panels, B remainder (5 % 2 = 1)
            (16, 4, 64), // Two A-panels, no B remainder
            (17, 4, 64), // A-panel remainder
            (32, 5, 16), // Multiple full A-panels, B remainder
            (48, 3, 16), // 6 A-panels (Scalar GROUP=8)
            (16, 6, 32), // B remainder = 0 (6 % 2 = 0)
            (16, 8, 32), // No B remainder (8 % 2 = 0)
        ];

        // Two budgets: default exercises the peeled section only; tiny forces
        // a_panels_per_tile=1 and b_panels_per_tile=1 so the main loop body
        // (Loop 2) and multiple A-tile iterations (Loop 1) are entered.
        let budgets = [
            ("default", TileBudget::default()),
            ("tiny", TileBudget { l2_a: 1, l1_b: 1 }),
        ];

        for &(a_nrows, b_nrows, dim) in cases {
            let a_data: Vec<f32> = (0..a_nrows * dim).map(|i| (i + 1) as f32).collect();
            let b_data: Vec<f32> = (0..b_nrows * dim).map(|i| ((i + 1) * 2) as f32).collect();
            let expected = naive_max_ip(&a_data, a_nrows, &b_data, b_nrows, dim);

            for &(label, budget) in &budgets {
                let a_mat = MatRef::new(Standard::new(a_nrows, dim).unwrap(), &a_data).unwrap();
                let a_bt = BlockTransposed::<f32, 8>::from_matrix_view(a_mat.as_matrix_view());
                let b_mat = MatRef::new(Standard::new(b_nrows, dim).unwrap(), &b_data).unwrap();

                let mut scratch = vec![f32::MIN; a_bt.padded_nrows()];
                max_ip_kernel::<Scalar, _, 8>(
                    Scalar::new(),
                    a_bt.as_view(),
                    b_mat,
                    &mut scratch,
                    budget,
                );

                for i in 0..a_nrows {
                    assert!(
                        (scratch[i] - expected[i]).abs() < 1e-6,
                        "row {i} mismatch for ({a_nrows},{b_nrows},{dim}) budget={label}: actual={}, expected={}",
                        scratch[i],
                        expected[i],
                    );
                }
            }
        }
    }
}

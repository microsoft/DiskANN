// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Generic tiling loop and shared reduction utilities.
//!
//! This module contains the type-agnostic parts of the kernel implementation:
//!
//! - `FullReduce` — tile planner that computes A/B panel counts from cache budgets.
//! - `tiled_reduce` — the 5-level loop nest that drives any [`Kernel<A>`](super::Kernel).
//! - `Reduce` — compile-time unroll reduction trait for fixed-size accumulator arrays.

use diskann_wide::Architecture;

use super::{Kernel, l1_b_tile_budget, l2_a_tile_budget};

// ── Tile planner ─────────────────────────────────────────────────

/// A plan for performing the reducing GEMM when the contraction dimension `k`
/// is small enough that a micro-panel of `A` fits comfortably in L1 cache with room for
/// multiple micro-panels of `B`.
#[derive(Debug, Clone, Copy)]
struct FullReduce {
    /// The number of micro panels of `A` that make up a tile.
    a_panels_per_tile: usize,

    /// The number of micro panels of `B` that make up a tile.
    b_panels_per_tile: usize,
}

impl FullReduce {
    /// Compute A-tile and B-tile panel counts from cache budgets.
    ///
    /// * `a_row_bytes` — bytes per prepared A row (`k * size_of::<APrepared>()`).
    /// * `b_row_bytes` — bytes per prepared B row (`k * size_of::<BPrepared>()`).
    /// * `a_panel` — micro-kernel A panel height (`K::A_PANEL`).
    /// * `b_panel` — micro-kernel B panel width (`K::B_PANEL`).
    /// * `l2_budget` — L2 cache budget in bytes for the A tile.
    /// * `l1_budget` — L1 cache budget in bytes for the B tile.
    fn new(
        a_row_bytes: usize,
        b_row_bytes: usize,
        a_panel: usize,
        b_panel: usize,
        l2_budget: usize,
        l1_budget: usize,
    ) -> Self {
        let a_row_bytes = a_row_bytes.max(1);
        let b_row_bytes = b_row_bytes.max(1);

        let a_panels_per_tile = (l2_budget / (a_row_bytes * a_panel)).max(1);

        let a_panel_bytes = a_panel * a_row_bytes;
        let b_tile_budget = l1_budget.saturating_sub(a_panel_bytes);
        let b_panels_per_tile = (b_tile_budget / (b_row_bytes * b_panel)).max(1);

        Self {
            a_panels_per_tile,
            b_panels_per_tile,
        }
    }
}

// ── Generic tiled reduce ─────────────────────────────────────────

/// Execute the 5-level tiling loop with a pluggable SIMD micro-kernel.
///
/// This is the core scheduling primitive. The loop nest is:
/// ```text
/// Loop 1: A tiles     (sized to L2)
///   Loop 2: B tiles   (sized to L1)
///     Loop 3: A panels (micro-panels within A tile)
///       Loop 4: B panels (micro-panels within B tile)
///         K::full_panel / K::remainder_dispatch
/// ```
///
/// # Safety
///
/// * `a_ptr` must be valid for `a_available_rows * k` elements of `K::AElem`.
/// * `a_available_rows` must be a multiple of `K::A_PANEL`.
/// * `b_ptr` must be valid for `b_nrows * k` elements of `K::BElem`.
/// * `scratch` must have length ≥ `a_available_rows` and be initialized by caller.
pub(crate) unsafe fn tiled_reduce<A: Architecture, K: Kernel<A>>(
    arch: A,
    a_ptr: *const K::AElem,
    a_available_rows: usize,
    b_ptr: *const K::BElem,
    b_nrows: usize,
    k: usize,
    scratch: &mut [f32],
) {
    let a_row_bytes = k * std::mem::size_of::<K::APrepared>();
    let b_row_bytes = k * std::mem::size_of::<K::BPrepared>();
    let plan = FullReduce::new(
        a_row_bytes,
        b_row_bytes,
        K::A_PANEL,
        K::B_PANEL,
        l2_a_tile_budget(),
        l1_b_tile_budget(),
    );

    let a_panel_stride = K::A_PANEL * k;
    let b_panel_stride = K::B_PANEL * k;
    let b_tile_stride = b_panel_stride * plan.b_panels_per_tile;

    let b_remainder = b_nrows % K::B_PANEL;

    assert_eq!(
        a_available_rows % K::A_PANEL,
        0,
        "a_available_rows ({a_available_rows}) must be a multiple of A_PANEL ({})",
        K::A_PANEL,
    );

    // Zero-dimensional vectors have IP = 0 for every pair. Fill scratch and
    // return to avoid zero-stride infinite loops in the tiling nest.
    if k == 0 {
        if b_nrows > 0 {
            scratch[..a_available_rows].fill(0.0);
        }
        return;
    }

    // Staging buffers are split into independent A and B halves so the tiling
    // loop can borrow them independently — `prepare_a(&mut a_buf)` and
    // `prepare_b(&mut b_buf)` never alias. Identity kernels return empty Vecs.
    let (mut a_buf, mut b_buf) = K::new_buffers(k);

    // SAFETY: Caller guarantees b_ptr is valid for b_nrows * k elements.
    let pb_end = unsafe { b_ptr.add(b_nrows * k) };
    // SAFETY: b_remainder < B_PANEL, so pb_end - b_remainder * k is within allocation.
    let pb_full_end = unsafe { pb_end.sub(b_remainder * k) };

    // SAFETY: All pointer arithmetic stays within the respective allocations.
    unsafe {
        let a_tile_rows = K::A_PANEL * plan.a_panels_per_tile;
        let mut rows_done: usize = 0;

        // Loop 1: Tiles of `A`.
        while rows_done < a_available_rows {
            let tile_rows = a_tile_rows.min(a_available_rows - rows_done);
            let pa_tile = a_ptr.add(rows_done * k);
            let pa_tile_end = pa_tile.add(tile_rows * k);
            // SAFETY: rows_done < a_available_rows (loop condition), so the
            // pointer is in-bounds.
            let pr_tile = scratch.as_mut_ptr().add(rows_done);

            let mut pb_tile = b_ptr;

            // Loop 2: Full B-tiles (every panel in the tile is complete).
            // SAFETY: `pb_tile` is always in `[b_ptr, pb_full_end]` — both within
            // the same allocation — so `offset_from` is well-defined.
            while pb_full_end.offset_from(pb_tile) >= b_tile_stride as isize {
                let pb_tile_end = pb_tile.add(b_tile_stride);

                let mut pa_panel = pa_tile;
                let mut pr_panel = pr_tile;

                // Loop 3: Micro-panels of `A`.
                while pa_panel < pa_tile_end {
                    let prepared_a = K::prepare_a(&mut a_buf, arch, pa_panel, k);
                    let mut pb_panel = pb_tile;

                    // Loop 4: Micro-panels of `B` (all full, no remainder check).
                    while pb_panel < pb_tile_end {
                        let prepared_b = K::prepare_b(&mut b_buf, arch, pb_panel, K::B_PANEL, k);
                        K::full_panel(arch, prepared_a, prepared_b, k, pr_panel);
                        pb_panel = pb_panel.add(b_panel_stride);
                    }

                    pa_panel = pa_panel.add(a_panel_stride);
                    pr_panel = pr_panel.add(K::A_PANEL);
                }
                pb_tile = pb_tile.add(b_tile_stride);
            }

            // Peeled last B-tile: contains remaining full panels + remainder rows.
            if pb_tile < pb_end {
                let mut pa_panel = pa_tile;
                let mut pr_panel = pr_tile;

                // Loop 3 (peeled): Micro-panels of `A`.
                while pa_panel < pa_tile_end {
                    let prepared_a = K::prepare_a(&mut a_buf, arch, pa_panel, k);
                    let mut pb_panel = pb_tile;

                    // Loop 4 (peeled): Full B-panels in the last tile.
                    while pb_panel < pb_full_end {
                        let prepared_b = K::prepare_b(&mut b_buf, arch, pb_panel, K::B_PANEL, k);
                        K::full_panel(arch, prepared_a, prepared_b, k, pr_panel);
                        pb_panel = pb_panel.add(b_panel_stride);
                    }

                    // Remainder dispatch: 1..(B_PANEL-1) leftover B-rows.
                    if b_remainder > 0 {
                        let prepared_b = K::prepare_b(&mut b_buf, arch, pb_panel, b_remainder, k);
                        K::remainder_dispatch(
                            arch,
                            b_remainder,
                            prepared_a,
                            prepared_b,
                            k,
                            pr_panel,
                        );
                    }

                    pa_panel = pa_panel.add(a_panel_stride);
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
    use super::FullReduce;

    #[test]
    fn basic_panel_counts() {
        // 16 A-rows × 256 bytes/row = 4096 bytes per A-panel.
        // L2 budget 40960 → 40960 / 4096 = 10 A-panels.
        // One A-panel = 4096 bytes, L1 budget 36000 → 36000 - 4096 = 31904.
        // 4 B-rows × 256 bytes/row = 1024 bytes per B-panel.
        // 31904 / 1024 = 31 B-panels.
        let plan = FullReduce::new(256, 256, 16, 4, 40960, 36000);
        assert_eq!(plan.a_panels_per_tile, 10);
        assert_eq!(plan.b_panels_per_tile, 31);
    }

    #[test]
    fn tiny_budget_clamps_to_one() {
        // Budget too small for even one panel — clamp to 1.
        let plan = FullReduce::new(1024, 1024, 16, 4, 1, 1);
        assert_eq!(plan.a_panels_per_tile, 1);
        assert_eq!(plan.b_panels_per_tile, 1);
    }

    #[test]
    fn zero_byte_rows_clamped() {
        // Zero-byte rows (e.g. k=0) should not divide by zero.
        // FullReduce clamps row bytes to max(1), so a_row_bytes=1, b_row_bytes=1.
        let plan = FullReduce::new(0, 0, 16, 4, 100_000, 50_000);
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
        let plan = FullReduce::new(64, 64, 16, 4, 1024, 2048);
        assert_eq!(plan.a_panels_per_tile, 1);
        // L1: 2048 - 16*64(=1024) = 1024 for B. 4*64=256 per B-panel → 4 panels.
        assert_eq!(plan.b_panels_per_tile, 4);
    }

    #[test]
    fn l1_saturated_by_a_panel() {
        // A-panel alone exceeds L1 budget → b_tile_budget saturates to 0,
        // b_panels_per_tile clamps to 1.
        let plan = FullReduce::new(1024, 64, 16, 4, 100_000, 100);
        assert_eq!(plan.b_panels_per_tile, 1);
    }

    #[test]
    #[should_panic(expected = "must be a multiple of A_PANEL")]
    fn panics_on_unaligned_a_rows() {
        use super::super::f32::F32Kernel;
        use diskann_wide::arch::Scalar;

        let k = 4;
        // 9 is not a multiple of A_PANEL (8).
        let a = vec![0.0f32; 9 * k];
        let b = vec![0.0f32; 2 * k];
        let mut scratch = vec![f32::MIN; 16];

        // SAFETY: pointers and scratch are correctly sized; we expect a panic.
        unsafe {
            super::tiled_reduce::<Scalar, F32Kernel<8>>(
                Scalar::new(),
                a.as_ptr(),
                9,
                b.as_ptr(),
                2,
                k,
                &mut scratch,
            );
        }
    }

    #[test]
    fn zero_dim_fills_scratch_and_returns() {
        use super::super::f32::F32Kernel;
        use diskann_wide::arch::Scalar;

        let a_rows = 8;
        let b_rows = 3;
        let k = 0;

        let a = Vec::<f32>::new();
        let b = Vec::<f32>::new();
        let mut scratch = vec![f32::MIN; a_rows];

        // SAFETY: k == 0 so no elements are read; pointers are never dereferenced.
        unsafe {
            super::tiled_reduce::<Scalar, F32Kernel<8>>(
                Scalar::new(),
                a.as_ptr(),
                a_rows,
                b.as_ptr(),
                b_rows,
                k,
                &mut scratch,
            );
        }

        for &v in &scratch {
            assert_eq!(v, 0.0, "zero-dim IP should be 0.0");
        }
    }

    #[test]
    fn zero_dim_zero_docs_leaves_scratch_untouched() {
        use super::super::f32::F32Kernel;
        use diskann_wide::arch::Scalar;

        let a_rows = 8;
        let mut scratch = vec![f32::MIN; a_rows];

        // SAFETY: k == 0, b_nrows == 0; no elements read.
        unsafe {
            super::tiled_reduce::<Scalar, F32Kernel<8>>(
                Scalar::new(),
                [].as_ptr(),
                a_rows,
                [].as_ptr(),
                0,
                0,
                &mut scratch,
            );
        }

        for &v in &scratch {
            assert_eq!(v, f32::MIN, "zero docs should leave scratch untouched");
        }
    }

    #[test]
    fn reduce_folds_correctly() {
        use super::Reduce;

        let max = |a: f32, b: f32| a.max(b);
        assert_eq!([5.0f32].reduce(&max), 5.0);
        assert_eq!([1.0f32, 3.0].reduce(&max), 3.0);
        assert_eq!([2.0f32, 1.0, 4.0].reduce(&max), 4.0);
        assert_eq!([3.0f32, 1.0, 4.0, 2.0].reduce(&max), 4.0);
    }

    #[test]
    fn tiled_reduce_matches_naive() {
        use super::super::f32::max_ip_kernel;
        use crate::multi_vector::block_transposed::BlockTransposed;
        use crate::multi_vector::matrix::{MatRef, Standard};
        use diskann_wide::arch::Scalar;

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

        // (a_nrows, b_nrows, dim)
        let cases: &[(usize, usize, usize)] = &[
            (8, 3, 4),  // Single A-panel, B remainder (3 % 2 = 1)
            (16, 5, 8), // Two A-panels, B remainder (5 % 2 = 1)
        ];

        for &(a_nrows, b_nrows, dim) in cases {
            let a_data: Vec<f32> = (0..a_nrows * dim).map(|i| (i + 1) as f32).collect();
            let b_data: Vec<f32> = (0..b_nrows * dim).map(|i| ((i + 1) * 2) as f32).collect();

            let a_mat = MatRef::new(Standard::new(a_nrows, dim).unwrap(), &a_data).unwrap();
            let a_bt = BlockTransposed::<f32, 8>::from_matrix_view(a_mat.as_matrix_view());
            let b_mat = MatRef::new(Standard::new(b_nrows, dim).unwrap(), &b_data).unwrap();

            let mut scratch = vec![f32::MIN; a_bt.available_rows()];
            max_ip_kernel::<Scalar, 8>(Scalar::new(), a_bt.as_view(), b_mat, &mut scratch);

            let expected = naive_max_ip(&a_data, a_nrows, &b_data, b_nrows, dim);
            for i in 0..a_nrows {
                assert!(
                    (scratch[i] - expected[i]).abs() < 1e-6,
                    "row {i} mismatch for ({a_nrows},{b_nrows},{dim}): actual={}, expected={}",
                    scratch[i],
                    expected[i]
                );
            }
        }
    }
}

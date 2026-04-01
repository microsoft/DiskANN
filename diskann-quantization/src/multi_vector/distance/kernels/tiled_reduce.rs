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
    a_panels: usize,

    /// The number of micro panels of `B` that make up a tile.
    b_panels: usize,
}

impl FullReduce {
    /// Compute A-tile and B-tile panel counts from cache budgets.
    ///
    /// * `a_row_bytes` — bytes per A row (`k * size_of::<AElem>()`).
    /// * `b_row_bytes` — bytes per B row (`k * size_of::<BElem>()`).
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

        let a_panels = (l2_budget / (a_row_bytes * a_panel)).max(1);

        let a_panel_bytes = a_panel * a_row_bytes;
        let b_tile_budget = l1_budget.saturating_sub(a_panel_bytes);
        let b_panels = (b_tile_budget / (b_row_bytes * b_panel)).max(1);

        Self { a_panels, b_panels }
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
/// The last A panel may be partial (`rows < A_PANEL`) — the kernel's
/// `prepare_a` zero-pads it to a full panel so the micro-kernel body is
/// unchanged. The extra scratch entries are written but ignored by the caller.
///
/// # Safety
///
/// * `a_ptr` must be valid for `a_available_rows * k` elements of `K::AElem`.
/// * `b_ptr` must be valid for `b_nrows * k` elements of `K::BElem`.
/// * `scratch` must have length ≥ `a_available_rows` and be initialized by caller.
///   If `a_available_rows` is not a multiple of `A_PANEL`, up to `A_PANEL - 1`
///   extra entries past `scratch[a_available_rows - 1]` may be written.
pub(crate) unsafe fn tiled_reduce<A: Architecture, K: Kernel<A>>(
    arch: A,
    a_ptr: *const K::AElem,
    a_available_rows: usize,
    b_ptr: *const K::BElem,
    b_nrows: usize,
    k: usize,
    scratch: &mut [f32],
) {
    let a_row_bytes = k * std::mem::size_of::<K::AElem>();
    let b_row_bytes = k * std::mem::size_of::<K::BElem>();
    let plan = FullReduce::new(
        a_row_bytes,
        b_row_bytes,
        K::A_PANEL,
        K::B_PANEL,
        l2_a_tile_budget(),
        l1_b_tile_budget(),
    );

    let a_panel_stride = K::A_PANEL * k;
    let a_tile_stride = a_panel_stride * plan.a_panels;
    let b_panel_stride = K::B_PANEL * k;
    let b_tile_stride = b_panel_stride * plan.b_panels;

    let b_remainder = b_nrows % K::B_PANEL;
    let a_remainder = a_available_rows % K::A_PANEL;

    // Kernel owns its staging buffers (identity kernels stay zero-sized).
    let mut kernel = K::new(k);

    // SAFETY: Caller guarantees b_ptr is valid for b_nrows * k elements.
    let pb_end = unsafe { b_ptr.add(b_nrows * k) };
    // SAFETY: b_remainder < B_PANEL, so pb_end - b_remainder * k is within allocation.
    let pb_full_end = unsafe { pb_end.sub(b_remainder * k) };

    // End-of-A pointer for detecting the last (potentially partial) A panel.
    // SAFETY: Caller guarantees a_ptr is valid for a_available_rows * k elements.
    let pa_end = unsafe { a_ptr.add(a_available_rows * k) };

    // SAFETY: All pointer arithmetic stays within the respective allocations.
    unsafe {
        let a_total = a_available_rows * k;
        let mut a_offset: usize = 0;
        let mut pr_tile = scratch.as_mut_ptr();

        // Loop 1: Tiles of `A`.
        while a_offset < a_total {
            let remaining_a = a_total - a_offset;
            let pa_tile = a_ptr.add(a_offset);
            let pa_tile_end = pa_tile.add(a_tile_stride.min(remaining_a));

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
                    // Determine A panel row count: partial for the last panel.
                    let a_rows = if pa_panel.add(a_panel_stride) > pa_end && a_remainder > 0 {
                        a_remainder
                    } else {
                        K::A_PANEL
                    };
                    let prepared_a = kernel.prepare_a(arch, pa_panel, a_rows, k);
                    let mut pb_panel = pb_tile;

                    // Loop 4: Micro-panels of `B` (all full, no remainder check).
                    while pb_panel < pb_tile_end {
                        let prepared_b = kernel.prepare_b(arch, pb_panel, K::B_PANEL, k);
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
                    let a_rows = if pa_panel.add(a_panel_stride) > pa_end && a_remainder > 0 {
                        a_remainder
                    } else {
                        K::A_PANEL
                    };
                    let prepared_a = kernel.prepare_a(arch, pa_panel, a_rows, k);
                    let mut pb_panel = pb_tile;

                    // Loop 4 (peeled): Full B-panels in the last tile.
                    while pb_panel < pb_full_end {
                        let prepared_b = kernel.prepare_b(arch, pb_panel, K::B_PANEL, k);
                        K::full_panel(arch, prepared_a, prepared_b, k, pr_panel);
                        pb_panel = pb_panel.add(b_panel_stride);
                    }

                    // Remainder dispatch: 1..(B_PANEL-1) leftover B-rows.
                    if b_remainder > 0 {
                        let prepared_b = kernel.prepare_b(arch, pb_panel, b_remainder, k);
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

            a_offset += a_tile_stride;
            pr_tile = pr_tile.add(K::A_PANEL * plan.a_panels);
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
        assert_eq!(plan.a_panels, 10);
        assert_eq!(plan.b_panels, 31);
    }

    #[test]
    fn tiny_budget_clamps_to_one() {
        // Budget too small for even one panel — clamp to 1.
        let plan = FullReduce::new(1024, 1024, 16, 4, 1, 1);
        assert_eq!(plan.a_panels, 1);
        assert_eq!(plan.b_panels, 1);
    }

    #[test]
    fn zero_byte_rows_clamped() {
        // Zero-byte rows (e.g. k=0) should not divide by zero.
        // FullReduce clamps row bytes to max(1), so a_row_bytes=1, b_row_bytes=1.
        let plan = FullReduce::new(0, 0, 16, 4, 100_000, 50_000);
        // a_panels = 100_000 / (1 * 16) = 6250
        assert_eq!(plan.a_panels, 6250);
        // a_panel_bytes = 16 * 1 = 16. b_tile_budget = 50_000 - 16 = 49_984.
        // b_panels = 49_984 / (1 * 4) = 12_496
        assert_eq!(plan.b_panels, 12_496);
    }

    #[test]
    fn exact_fit_one_panel() {
        // Budget exactly fits one A-panel (16 × 64 = 1024 bytes).
        // No room for a second → a_panels = 1.
        let plan = FullReduce::new(64, 64, 16, 4, 1024, 2048);
        assert_eq!(plan.a_panels, 1);
        // L1: 2048 - 16*64(=1024) = 1024 for B. 4*64=256 per B-panel → 4 panels.
        assert_eq!(plan.b_panels, 4);
    }

    #[test]
    fn l1_saturated_by_a_panel() {
        // A-panel alone exceeds L1 budget → b_tile_budget saturates to 0,
        // b_panels clamps to 1.
        let plan = FullReduce::new(1024, 64, 16, 4, 100_000, 100);
        assert_eq!(plan.b_panels, 1);
    }
}

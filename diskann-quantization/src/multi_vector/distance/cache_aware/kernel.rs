// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Generic cache-aware tiling loop and shared reduction utilities.
//!
//! This module contains the type-agnostic parts of the cache-aware implementation:
//!
//! - `FullReduce` — tile planner that computes A/B panel counts from cache budgets.
//! - `tiled_reduce` — the 5-level loop nest that drives any `CacheAwareKernel`.
//! - `Reduce` — compile-time unroll reduction trait for fixed-size accumulator arrays.

use super::{CacheAwareKernel, L1_B_TILE_BUDGET, L2_A_TILE_BUDGET};

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
    /// * `a_row_bytes` — bytes per query row (`k * size_of::<QueryElem>()`).
    /// * `b_row_bytes` — bytes per document row (`k * size_of::<DocElem>()`).
    /// * `a_panel` — micro-kernel query panel height (`K::A_PANEL`).
    /// * `b_panel` — micro-kernel document panel width (`K::B_PANEL`).
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

/// Execute the 5-level cache-aware tiling loop with a pluggable SIMD micro-kernel.
///
/// This is the core scheduling primitive. The loop nest is:
/// ```text
/// Loop 1: A tiles     (query tiles, sized to L2)
///   Loop 2: B tiles   (doc tiles, sized to L1)
///     Loop 3: A panels (micro-panels within A tile)
///       Loop 4: B panels (micro-panels within B tile)
///         K::full_panel / K::remainder_dispatch
/// ```
///
/// # Safety
///
/// * `a_ptr` must be valid for `a_available_rows * k` elements of `K::QueryElem`.
/// * `b_ptr` must be valid for `b_nrows * k` elements of `K::DocElem`.
/// * `scratch` must have length `a_available_rows` and be initialized by caller.
/// * `a_available_rows` must be a multiple of `K::A_PANEL`.
pub(crate) unsafe fn tiled_reduce<K: CacheAwareKernel>(
    arch: diskann_wide::arch::Current,
    a_ptr: *const K::QueryElem,
    a_available_rows: usize,
    b_ptr: *const K::DocElem,
    b_nrows: usize,
    k: usize,
    scratch: &mut [f32],
) {
    debug_assert_eq!(
        a_available_rows % K::A_PANEL,
        0,
        "a_available_rows ({a_available_rows}) must be a multiple of A_PANEL ({})",
        K::A_PANEL,
    );

    let a_row_bytes = k * std::mem::size_of::<K::QueryElem>();
    let b_row_bytes = k * std::mem::size_of::<K::DocElem>();
    let plan = FullReduce::new(
        a_row_bytes,
        b_row_bytes,
        K::A_PANEL,
        K::B_PANEL,
        L2_A_TILE_BUDGET,
        L1_B_TILE_BUDGET,
    );

    let a_panel_stride = K::A_PANEL * k;
    let a_tile_stride = a_panel_stride * plan.a_panels;
    let b_panel_stride = K::B_PANEL * k;
    let b_tile_stride = b_panel_stride * plan.b_panels;

    let remainder = b_nrows % K::B_PANEL;

    // SAFETY: Caller guarantees a_ptr is valid for a_available_rows * k elements.
    let pa_end = unsafe { a_ptr.add(a_available_rows * k) };
    // SAFETY: Caller guarantees b_ptr is valid for b_nrows * k elements.
    let pb_end = unsafe { b_ptr.add(b_nrows * k) };
    // SAFETY: remainder < B_PANEL, so pb_end - remainder * k is within allocation.
    let pb_full_end = unsafe { pb_end.sub(remainder * k) };

    // SAFETY: All pointer arithmetic stays within the respective allocations.
    unsafe {
        let mut pa_tile = a_ptr;
        let mut pr_tile = scratch.as_mut_ptr();

        // Loop 1: Tiles of `A`.
        while pa_tile < pa_end {
            let remaining_a =
                (pa_end as usize - pa_tile as usize) / std::mem::size_of::<K::QueryElem>();
            let pa_tile_end = pa_tile.add(a_tile_stride.min(remaining_a));

            let mut pb_tile = b_ptr;

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
                        K::full_panel(arch, pa_panel, pb_panel, k, pr_panel);
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
                    let mut pb_panel = pb_tile;

                    // Loop 4 (peeled): Full B-panels in the last tile.
                    while pb_panel < pb_full_end {
                        K::full_panel(arch, pa_panel, pb_panel, k, pr_panel);
                        pb_panel = pb_panel.add(b_panel_stride);
                    }

                    // Remainder dispatch: 1..(B_PANEL-1) leftover B-rows.
                    if remainder > 0 {
                        K::remainder_dispatch(arch, remainder, pa_panel, pb_panel, k, pr_panel);
                    }

                    pa_panel = pa_panel.add(a_panel_stride);
                    pr_panel = pr_panel.add(K::A_PANEL);
                }
            }

            // NOTE: Use `wrapping_add` so we can still do this on the last iteration.
            pa_tile = pa_tile.wrapping_add(a_tile_stride);
            pr_tile = pr_tile.wrapping_add(K::A_PANEL * plan.a_panels);
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

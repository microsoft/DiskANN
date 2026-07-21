// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Staged tiling driver.
//!
//! Structurally identical to [`super::super::tiled_reduce`] (A-tile → B-tile), but
//! the inner work per A-panel is split into three stages: Stage A fills
//! `partial_buf` for the whole B-tile, Stage B ([`Postprocess::apply`]) turns the
//! raw `Acc` block into `Score`s, and Stage C ([`Reducer::fold_block`]) folds them
//! into the running state. The driver runs all three uniformly; for the identity
//! postprocess (`Acc == Score`) Stage B is `#[inline(always)]` and returns the
//! `partial_buf` pointer unchanged, so it compiles away (no `scored_buf`, no pass).
//!
//! Every B-tile is `≤ b_tile_rows` rows: a full tile has `b_panels_per_tile`
//! complete panels, and a final short tile may end in a `< B_PANEL` remainder
//! panel — both handled by one loop (`full_panels` + an optional `tail`), so there
//! is no separate "peeled tail" code path.

use diskann_wide::Architecture;

use super::super::TileBudget;
use super::super::layouts::Layout;
use super::{FoldCtx, Postprocess, Reducer, StagedConvert, StagedKernel, StagedPlan};
use crate::alloc::{Poly, ScopedAllocator};

/// Run the staged loop. `state` (len `a_padded_nrows`) is the persistent running
/// reduction across all B-tiles (the caller's output buffer). All transient
/// scratch — `partial` (Stage A output), the Stage-B `scored` region, and the
/// conversion buffers — is allocated **internally** from `alloc`; the caller
/// sizes nothing and hands in only the allocator.
///
/// # Safety
///
/// * `a_ptr` valid for `a_padded_nrows * k` `LA::Element`; `a_padded_nrows`
///   a multiple of `SK::A_PANEL`.
/// * `b_ptr` valid for `b_nrows * k` `LB::Element`.
#[allow(clippy::too_many_arguments, clippy::expect_used)]
pub(super) unsafe fn tiled_reduce_staged<A, SK, P, R, LA, LB>(
    arch: A,
    ca: &LA,
    cb: &LB,
    post: &P,
    a_ptr: *const LA::Element,
    a_padded_nrows: usize,
    b_ptr: *const LB::Element,
    b_nrows: usize,
    k: usize,
    state: &mut [R::State],
    alloc: ScopedAllocator<'_>,
    budget: TileBudget,
) where
    A: Architecture,
    SK: StagedKernel<A>,
    P: Postprocess<A, Acc = SK::Acc>,
    R: Reducer<A, Score = P::Score>,
    LA: StagedConvert<A, SK::Left>,
    LB: StagedConvert<A, SK::Right>,
{
    let a_panel = SK::A_PANEL;
    let b_panel = SK::B_PANEL;

    // Initialize the running reduction state.
    for s in state[..a_padded_nrows].iter_mut() {
        *s = R::init();
    }

    // Zero-dimensional vectors: every IP is 0. The caller fills the score for
    // this degenerate case; here we just avoid the zero-stride tiling nest.
    if k == 0 {
        return;
    }

    debug_assert_eq!(
        a_padded_nrows % a_panel,
        0,
        "a_padded_nrows must be a multiple of A_PANEL"
    );

    let acc_bytes = core::mem::size_of::<SK::Acc>();
    let a_row_bytes = k * core::mem::size_of::<<SK::Left as Layout>::Element>();
    let b_row_bytes = k * core::mem::size_of::<<SK::Right as Layout>::Element>();
    let plan = StagedPlan::new(
        a_row_bytes,
        b_row_bytes,
        a_panel,
        b_panel,
        acc_bytes,
        budget,
    );

    let a_tile_rows = a_panel * plan.a_panels_per_tile;
    let b_tile_rows = b_panel * plan.b_panels_per_tile;

    let a_kern_panel_stride = a_panel * k;
    let b_kern_panel_stride = b_panel * k;

    // Conversion staging buffers, also from the caller's allocator — 0-length
    // (a no-op dangling allocation) for the identity conversions every current
    // staged kernel uses. Sized by the staged-local `StagedConvert` contract, so
    // the staged driver never touches the shared `ConvertTo` machinery.
    let a_conv_len = ca.scratch_len(a_tile_rows.min(a_padded_nrows), k);
    let mut a_conv =
        Poly::<[<SK::Left as Layout>::Element], _>::new_uninit_slice(a_conv_len, alloc)
            .expect("a-side conversion scratch allocation");
    let b_conv_len = cb.scratch_len(b_tile_rows.min(b_nrows), k);
    let mut b_conv =
        Poly::<[<SK::Right as Layout>::Element], _>::new_uninit_slice(b_conv_len, alloc)
            .expect("b-side conversion scratch allocation");
    let a_conv_ptr = a_conv.as_mut_ptr().cast::<<SK::Left as Layout>::Element>();
    let b_conv_ptr = b_conv.as_mut_ptr().cast::<<SK::Right as Layout>::Element>();

    // Internal scratch, allocated from the caller's allocator — the caller sizes
    // nothing. `partial` is Stage A's output (the kernel declares its size via
    // `StagedKernel::partial_len`); `scored` is Stage B's output, sized by the
    // postprocess contract (a 0-length, no-op dangling allocation for the identity
    // postprocess). Every B-tile is `≤ b_tile_rows` wide, so `b_tile_rows` is the
    // exact upper bound on `valid_b_cols`. Both `Poly`s live to the end of the
    // call, then free via `alloc` (a global free, or a no-op for a bump allocator).
    let partial_len = SK::partial_len(k, budget);
    let mut partial = Poly::<[SK::Acc], _>::new_uninit_slice(partial_len, alloc)
        .expect("partial scratch allocation");
    let scored_len = post.scratch_len(a_panel, b_tile_rows);
    let mut scored = Poly::<[P::Score], _>::new_uninit_slice(scored_len, alloc)
        .expect("scored scratch allocation");

    let partial_ptr = partial.as_mut_ptr().cast::<SK::Acc>();
    let scored_ptr = scored.as_mut_ptr().cast::<P::Score>();
    let state_ptr = state.as_mut_ptr();

    // SAFETY: all pointer arithmetic stays within the respective allocations;
    // this mirrors `super::super::tiled_reduce`'s established bounds.
    unsafe {
        let mut rows_done: usize = 0;

        // Loop 1: A tiles.
        while rows_done < a_padded_nrows {
            let tile_rows = a_tile_rows.min(a_padded_nrows - rows_done);
            let pa_tile_src = a_ptr.add(rows_done * k);
            let pr_tile = state_ptr.add(rows_done);

            let pa_tile = ca.convert(a_conv_ptr, arch, pa_tile_src, tile_rows, k);
            let pa_tile_end = pa_tile.add(tile_rows * k);

            // Loop 2: B tiles. Each is `bt_rows = min(b_tile_rows, remaining)` —
            // full tiles end on a panel boundary (`tail == 0`); the final short
            // tile may carry a `< B_PANEL` remainder panel.
            let mut pb_tile_src = b_ptr;
            let mut b_row_offset = 0usize;
            while b_row_offset < b_nrows {
                let bt_rows = b_tile_rows.min(b_nrows - b_row_offset);
                let pb_tile = cb.convert(b_conv_ptr, arch, pb_tile_src, bt_rows, k);
                let full_panels = bt_rows / b_panel;
                let tail = bt_rows % b_panel;

                // Loop 3: A micro-panels.
                //
                // Partial-buffer granularity: one A-panel (P_a = 1) against the
                // whole B-tile (P_b = b_panels_per_tile). Loop 4 below runs ONLY
                // Stage A, so the kernel stays hot in i-cache for the entire
                // B-tile; Stage C then folds the whole block in one pass. See
                // `StagedPlan` and docs/staged_multi_vector_kernel.md §5.
                let mut pa_panel = pa_tile;
                let mut pr_panel = pr_tile;
                let mut a_row_offset = rows_done;
                while pa_panel < pa_tile_end {
                    // Stage A: fill partial_buf for this A-panel across the B-tile
                    // (the full panels, then a `< B_PANEL` remainder panel if any).
                    let mut pb_panel = pb_tile;
                    let mut col = 0usize;
                    for _ in 0..full_panels {
                        SK::full_panel(
                            arch,
                            pa_panel,
                            pb_panel,
                            k,
                            partial_ptr.add(col * a_panel),
                            a_panel,
                        );
                        pb_panel = pb_panel.add(b_kern_panel_stride);
                        col += b_panel;
                    }
                    if tail > 0 {
                        SK::partial_panel(
                            arch,
                            tail,
                            pa_panel,
                            pb_panel,
                            k,
                            partial_ptr.add(col * a_panel),
                            a_panel,
                        );
                    }

                    // Stage B (Acc -> Score): identity returns `partial_ptr` for
                    // free; a quantized post fills `scored` (the driver-allocated
                    // region), using the global (a_row_offset, b_row_offset) to
                    // index its metadata. Output column stride is contractually
                    // `a_panel`.
                    let scores = post.apply(
                        scored_ptr,
                        arch,
                        partial_ptr,
                        FoldCtx {
                            a_panel,
                            valid_b_cols: bt_rows,
                            b_stride: a_panel,
                            a_row_offset,
                            b_row_offset,
                        },
                    );
                    // Stage C: fold the scores into the running state (one fold
                    // per A-panel × B-tile — the widest, cheapest fold).
                    R::fold_block(arch, pr_panel, scores, a_panel, bt_rows, a_panel);

                    pa_panel = pa_panel.add(a_kern_panel_stride);
                    pr_panel = pr_panel.add(a_panel);
                    a_row_offset += a_panel;
                }

                pb_tile_src = pb_tile_src.add(bt_rows * k);
                b_row_offset += bt_rows;
            }

            rows_done += tile_rows;
        }
    }
}

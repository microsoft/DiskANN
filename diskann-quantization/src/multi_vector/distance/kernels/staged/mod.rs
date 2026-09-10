// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Experimental *staged* multi-vector distance kernel.
//!
//! The production kernel (`super::tiled_reduce` + `super::f32`) fuses three
//! concerns into one micro-kernel epilogue: inner-product accumulation,
//! cross-row reduction, and merge into the per-A-row score scratch. This module
//! is a parallel, *separately selectable* kernel that splits those concerns into
//! three independently-pluggable stages so future work (quantized distances,
//! other reductions) can reuse the tiling loop without forking the micro-kernel:
//!
//! * **Stage A — [`StagedKernel`]**: pure SIMD math. Writes a raw `Acc` block
//!   (`f32` here) into a per-A-panel `partial_buf`; no reduction, no merge.
//! * **Stage B — [`Postprocess`]**: `Acc` → `Score`, modeled on
//!   [`ConvertTo`](super::layouts::ConvertTo). `apply` returns a read pointer and
//!   the identity impl returns its input unchanged (reporting `scratch_len` 0), so
//!   the driver runs it uniformly and identity stays zero-cost — no extra memory
//!   pass, no boolean flag.
//! * **Stage C — [`Reducer`]**: owns the per-A-row `State` and folds `Score`
//!   blocks into it.
//!
//! The traits keep raw-pointer methods (no generic methods) so a future
//! `&dyn Postprocess` / `&dyn Reducer` switched at the per-(A-panel, B-tile)
//! boundary stays possible without a monomorphization blow-up; we keep them
//! static-generic for now.
//!
//! Scope: single K-segment (no fractured-K), f32 + V3 (AVX2/FMA) only — an
//! apples-to-apples A/B against the fused V3 kernel.

use diskann_wide::Architecture;

use super::TileBudget;
use super::layouts::Layout;

pub(super) mod arena;
pub(super) mod driver;
pub(super) mod i8;
pub(super) mod maxsim;
pub(super) mod v3;

pub(crate) use maxsim::{F32StagedScratch, StagedRun};
pub(crate) use v3::StagedF32Kernel;
// Public POC entry for the quantized (4-bit MinMax) staged kernel.
pub use i8::{QuantStagedDocs, QuantStagedQuery};

// ── Stage A: kernel ──────────────────────────────────────────────

/// Stage A micro-kernel. Computes an `A_PANEL × B_PANEL` block of raw `Acc`
/// accumulators and **writes them out** to a partial buffer — unlike
/// [`super::Kernel`], it performs no cross-row reduction or scratch merge.
///
/// # Safety
///
/// Implementors must respect the per-method pointer contracts.
pub(super) unsafe trait StagedKernel<A: Architecture> {
    /// Layout consumed by the A (left / query) side.
    type Left: Layout;
    /// Layout consumed by the B (right / document) side.
    type Right: Layout;
    /// Raw accumulator element written into `partial_buf` (`f32` for MaxSim).
    type Acc: Copy;

    /// A rows processed per invocation (= the query `BlockTransposed` GROUP).
    const A_PANEL: usize;
    /// B rows processed per full invocation.
    const B_PANEL: usize;

    /// Number of `Acc` elements the per-(A-panel, B-tile) `partial_buf` needs at
    /// contraction dim `k` under `budget`. The single source of truth for the
    /// partial size formula: the driver sizes its internal allocation from this,
    /// derived from the panel geometry + the element sizes the kernel already
    /// knows.
    fn partial_len(k: usize, budget: TileBudget) -> usize {
        let a_elem = core::mem::size_of::<<Self::Left as Layout>::Element>();
        let b_elem = core::mem::size_of::<<Self::Right as Layout>::Element>();
        let acc = core::mem::size_of::<Self::Acc>();
        StagedPlan::new(
            k * a_elem,
            k * b_elem,
            Self::A_PANEL,
            Self::B_PANEL,
            acc,
            budget,
        )
        .partial_len(Self::A_PANEL, Self::B_PANEL)
    }

    /// Write a full `A_PANEL × B_PANEL` block into `partial`.
    ///
    /// `partial` points at the first B-column of this panel; column `j`
    /// (`0..B_PANEL`) and its `A_PANEL` rows occupy `partial[j*partial_b_stride
    /// ..][..A_PANEL]` (A-major: `partial_b_stride == A_PANEL`).
    ///
    /// # Safety
    ///
    /// * `a` valid for `A_PANEL * k` `Left::Element`.
    /// * `b` valid for `B_PANEL * k` `Right::Element`.
    /// * `partial` valid for `B_PANEL` columns of `A_PANEL` `Acc` at stride
    ///   `partial_b_stride`.
    unsafe fn full_panel(
        arch: A,
        a: *const <Self::Left as Layout>::Element,
        b: *const <Self::Right as Layout>::Element,
        k: usize,
        partial: *mut Self::Acc,
        partial_b_stride: usize,
    );

    /// Like [`Self::full_panel`] but writes only `remainder` (`1..B_PANEL`)
    /// B-columns.
    ///
    /// # Safety
    ///
    /// As [`Self::full_panel`], with `b` valid for `remainder * k`
    /// `Right::Element` and only `remainder` columns written.
    unsafe fn partial_panel(
        arch: A,
        remainder: usize,
        a: *const <Self::Left as Layout>::Element,
        b: *const <Self::Right as Layout>::Element,
        k: usize,
        partial: *mut Self::Acc,
        partial_b_stride: usize,
    );
}

// ── Stage B: postprocess ─────────────────────────────────────────

/// Per-(A-panel, B-tile) context passed to [`Postprocess::apply`].
///
/// All fields are cheap `Copy` scalars the driver already tracks. The identity
/// postprocess ignores them (they vanish after inlining); a metadata-bearing
/// postprocess (e.g. the quantized path) uses the global
/// `a_row_offset`/`b_row_offset` to index its per-vector metadata.
#[derive(Debug, Clone, Copy)]
pub(super) struct FoldCtx {
    /// Rows in this A-panel (`== A_PANEL`).
    pub(super) a_panel: usize,
    /// Valid B columns in this block (`≤` the tile width).
    pub(super) valid_b_cols: usize,
    /// `Acc` column stride within the partial block (`== a_panel`).
    pub(super) b_stride: usize,
    /// Global index of this A-panel's first row.
    pub(super) a_row_offset: usize,
    /// Global index of this B-tile's first row.
    pub(super) b_row_offset: usize,
}

/// Stage B: convert one A-major block of raw `Acc` accumulators into finished
/// `Score`s, returning a read pointer to the scores.
///
/// The driver sizes the staging region from [`scratch_len`](Self::scratch_len)
/// and allocates it (from the caller's allocator), then hands it to
/// [`apply`](Self::apply) as a raw `*mut Score`. The identity impl
/// ([`Identity`](maxsim::Identity)) reports `scratch_len == 0` and returns its
/// input pointer unchanged, so the driver runs one uniform path — no boolean, no
/// branch — and the identity case is zero-cost. A non-identity impl — e.g. the
/// quantized [`MinMaxPostprocess`](i8::MinMaxPostprocess), which turns raw `i32`
/// integer dot products into f32 MinMax scores using its own captured per-vector
/// metadata — writes into `scratch` and returns a pointer into it.
///
/// # Safety
///
/// Implementors must ensure that [`apply`](Self::apply):
/// - reads at most `ctx.valid_b_cols` columns of `ctx.a_panel` `Acc` from `acc`
///   at column stride `ctx.b_stride` (never the stale padded remainder columns);
/// - writes only within the `scratch` region it was given;
/// - returns a `*const Score` valid for `ctx.valid_b_cols` columns of
///   `ctx.a_panel` `Score` at the **fixed output column stride `ctx.a_panel`**.
///   (Identity returns `acc`, already at stride `a_panel`.)
pub(super) unsafe trait Postprocess<A: Architecture> {
    /// Raw accumulator type produced by Stage A.
    type Acc: Copy;
    /// Finished score type consumed by Stage C.
    type Score: Copy;

    /// Number of `Score` elements [`apply`](Self::apply) needs for one A-panel
    /// against a B-tile of up to `max_b_cols` columns. `0` for the identity
    /// postprocess (no staging — `apply` returns `acc`); the driver allocates
    /// exactly this many `Score`s from the caller's allocator and passes the
    /// region to `apply`.
    fn scratch_len(&self, a_panel: usize, max_b_cols: usize) -> usize;

    /// Convert the `ctx.a_panel × ctx.valid_b_cols` A-major `Acc` block at `acc`
    /// (column stride `ctx.b_stride`) into `Score`s, returning a read pointer.
    /// Output is A-major at the fixed column stride `ctx.a_panel`; the identity
    /// impl returns `acc` unchanged. Metadata-bearing impls index their own
    /// per-vector metadata by `ctx.a_row_offset` / `ctx.b_row_offset`.
    ///
    /// # Safety
    ///
    /// * `acc` is valid for `ctx.valid_b_cols` columns of `ctx.a_panel` `Acc` at
    ///   stride `ctx.b_stride`.
    /// * `scratch` is valid+writable for `scratch_len(ctx.a_panel, max_b_cols)`
    ///   `Score` with `max_b_cols ≥ ctx.valid_b_cols` (dangling when that is `0`).
    unsafe fn apply(
        &self,
        scratch: *mut Self::Score,
        arch: A,
        acc: *const Self::Acc,
        ctx: FoldCtx,
    ) -> *const Self::Score;
}

// ── Stage C: reducer ─────────────────────────────────────────────

/// Stage C owns the per-A-row reduction `State` and folds `Score` blocks into
/// it. `Max` here; richer `State` shapes (argmax `(f32,u32)`, top-k) and an
/// `Output`/`finalize` step are follow-on work.
pub(super) trait Reducer<A: Architecture> {
    /// Score element folded in (matches [`Postprocess::Score`]).
    type Score: Copy;
    /// Per-A-row running state (the score scratch element).
    type State: Copy;

    /// Identity state for an A-row before any B-rows are seen.
    fn init() -> Self::State;

    /// Fold an `A_PANEL × valid_b_cols` block of `Score` (read from
    /// `partial_buf`) into `state[0..a_panel]`, in place.
    ///
    /// Column `c` (`0..valid_b_cols`), row `i` (`0..a_panel`) is at
    /// `scores[c*b_stride + i]`. Only `valid_b_cols` columns are read — the
    /// padded remainder columns hold stale data and **must not** be folded.
    ///
    /// # Safety
    ///
    /// * `state` valid+writable for `a_panel` `State`.
    /// * `scores` valid for `valid_b_cols` columns of `a_panel` `Score` at
    ///   stride `b_stride`.
    unsafe fn fold_block(
        arch: A,
        state: *mut Self::State,
        scores: *const Self::Score,
        a_panel: usize,
        valid_b_cols: usize,
        b_stride: usize,
    );
}

// ── Stage conversion: StagedConvert ──────────────────────────────

/// Staged-local tile conversion from layout `Self` to layout `To` — the staged
/// path's self-contained replacement for the shared
/// [`ConvertTo`](super::layouts::ConvertTo). Same role (convert a tile of source
/// data into the kernel's element type), but inverted ownership: instead of
/// owning a `Buffer`, the impl reports a [`scratch_len`](Self::scratch_len) and
/// the **driver** allocates that staging region from the caller's allocator and
/// hands it to [`convert`](Self::convert). The blanket identity impl reports `0`
/// and returns `src` unchanged, so identity conversions cost nothing and the
/// staged driver never touches the shared `ConvertTo` machinery.
///
/// # Safety
///
/// Implementors must ensure [`convert`](Self::convert) reads at most `rows * k`
/// source elements, writes only within the `scratch` region it was given, and
/// returns a pointer valid for `rows * k` `To::Element`.
pub(super) unsafe trait StagedConvert<A: Architecture, To: Layout>: Layout {
    /// Number of `To::Element` the driver must allocate to convert up to
    /// `max_tile_rows × k`. `0` for identity (no conversion — `convert` returns
    /// `src`, ignoring `scratch`).
    fn scratch_len(&self, max_tile_rows: usize, k: usize) -> usize;

    /// Convert `rows × k` `Self::Element` at `src` into `To::Element`, writing
    /// into `scratch` (the driver-allocated region of `scratch_len(..)`), and
    /// returning a read pointer. The identity impl returns `src` unchanged.
    ///
    /// # Safety
    ///
    /// * `src` points to `rows * k` valid `Self::Element`.
    /// * `scratch` is valid+writable for `scratch_len(max_tile_rows, k)`
    ///   `To::Element` with `max_tile_rows ≥ rows` (dangling when that is `0`).
    unsafe fn convert(
        &self,
        scratch: *mut To::Element,
        arch: A,
        src: *const Self::Element,
        rows: usize,
        k: usize,
    ) -> *const To::Element;
}

/// Identity conversion: every layout converts to itself at zero cost (no
/// scratch, returns `src`). Mirrors the shared `ConvertTo` blanket identity.
// SAFETY: identity reads nothing beyond `src`, writes nothing (`scratch_len` is
// 0), and returns exactly `src`, valid for the caller's lifetime.
unsafe impl<A: Architecture, L: Layout> StagedConvert<A, L> for L {
    fn scratch_len(&self, _max_tile_rows: usize, _k: usize) -> usize {
        0
    }

    unsafe fn convert(
        &self,
        _scratch: *mut L::Element,
        _arch: A,
        src: *const L::Element,
        _rows: usize,
        _k: usize,
    ) -> *const L::Element {
        src
    }
}

// ── Planner ──────────────────────────────────────────────────────

/// Tile-panel counts for the staged loop.
///
/// Encodes the **partial-buffer granularity** decision: the partial buffer is
/// always **one A-panel** (`P_a = 1`) wide in the A direction, against **as many
/// B-panels as co-fit L1** (`P_b = b_panels_per_tile`) in the B direction.
///
/// `P_a = 1` because extra A-panels cut no B reads (B is re-streamed per A-panel
/// regardless) — they only enlarge `partial_buf`. `P_b = co-fit` rather than
/// `1×1` because total partial traffic is invariant to the fold granularity, so
/// the widest fold minimizes fold-call / `state`-reload overhead, maximizes the
/// SIMD sweep, and keeps Stage A (kernel) and Stage C (reduce) as contiguous
/// non-interleaved phases. See `docs/staged_multi_vector_kernel.md` §5.
#[derive(Debug, Clone, Copy)]
pub(super) struct StagedPlan {
    pub(super) a_panels_per_tile: usize,
    pub(super) b_panels_per_tile: usize,
}

impl StagedPlan {
    /// Choose `a_panels_per_tile` / `b_panels_per_tile` from the cache budgets,
    /// the panel sizes, and the partial-buffer footprint.
    ///
    /// **L2** holds the A-tile (it is reused across every B-tile):
    ///
    /// ```text
    /// a_panels_per_tile · A_PANEL · a_row_bytes  ≤  l2_a
    /// ```
    ///
    /// **L1** holds *three* things at once during Stage A / Stage C, so the
    /// planner co-budgets all three against `l1_b` (a usable fraction of L1)
    /// rather than letting each independently claim the whole budget:
    ///
    /// * one A micro-panel — `A_PANEL · a_row_bytes` (re-read per B-panel);
    /// * the B-tile data — `B_TILE_ROWS · b_row_bytes` (re-read per A-panel);
    /// * `partial_buf` — `A_PANEL · B_TILE_ROWS · acc_bytes`.
    ///
    /// Each B-row added to the tile therefore costs `b_row_bytes` of document
    /// data **plus** `A_PANEL · acc_bytes` of partial scratch, so we keep the
    /// largest `B_TILE_ROWS` satisfying
    ///
    /// ```text
    /// A_PANEL·a_row_bytes + B_TILE_ROWS·(b_row_bytes + A_PANEL·acc_bytes) ≤ l1_b
    /// ```
    ///
    /// This bounds `partial_buf + B-tile` together. (For very large `k` the A
    /// micro-panel alone can approach `l1_b`; then `b_panels_per_tile` clamps to
    /// 1 and the A-panel is the limit — inherent, and identical to the fused
    /// kernel's behaviour.)
    pub(super) fn new(
        a_row_bytes: usize,
        b_row_bytes: usize,
        a_panel: usize,
        b_panel: usize,
        acc_bytes: usize,
        budget: TileBudget,
    ) -> Self {
        let a_row_bytes = a_row_bytes.max(1);
        let b_row_bytes = b_row_bytes.max(1);

        // L2: the A-tile is reused across all B-tiles, so size it to L2.
        let a_panels_per_tile = (budget.l2_a / (a_row_bytes * a_panel)).max(1);

        // L1: co-budget the A micro-panel + B-tile data + partial_buf. Each
        // B-row costs its document data plus one A_PANEL-tall partial column.
        let a_panel_bytes = a_panel * a_row_bytes;
        let bytes_per_b_row = b_row_bytes + a_panel * acc_bytes;
        let b_tile_budget = budget.l1_b.saturating_sub(a_panel_bytes);
        let b_panels_per_tile = ((b_tile_budget / bytes_per_b_row) / b_panel).max(1);

        Self {
            a_panels_per_tile,
            b_panels_per_tile,
        }
    }

    /// `partial_buf` capacity (in `Acc` elements): one A-panel (`P_a = 1`) wide,
    /// covering a full B-tile (`b_panels_per_tile · B_PANEL` rows). The driver
    /// caps every B-tile at `b_tile_rows`, so this is the exact upper bound.
    pub(super) fn partial_len(&self, a_panel: usize, b_panel: usize) -> usize {
        a_panel * self.b_panels_per_tile * b_panel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mirror of `L1_CACHE` in `super::super::TileBudget::default`.
    const L1_CACHE_BYTES: usize = 48_000;

    /// The co-budget must keep the inner-loop L1 working set — one A
    /// micro-panel + the B-tile data + `partial_buf` — within real L1 for every
    /// realistic `k`. The previous design budgeted `partial_buf` and the B-tile
    /// independently against `l1_b` and overflowed at small/moderate `k` (e.g.
    /// k=16 placed ~70 KB into a 48 KB L1); this pins the fix.
    #[test]
    fn l1_working_set_fits_for_all_k() {
        const A_PANEL: usize = 16;
        const B_PANEL: usize = 4;
        const ACC: usize = 4; // f32

        // k up to 512: beyond ~768 the A micro-panel alone exceeds L1, which is
        // inherent (the fused kernel hits the same wall) and not a planner bug.
        for &k in &[1usize, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512] {
            let row = k * 4; // f32 element
            let plan = StagedPlan::new(row, row, A_PANEL, B_PANEL, ACC, TileBudget::default());
            let b_tile_rows = plan.b_panels_per_tile * B_PANEL;

            let a_panel_bytes = A_PANEL * row;
            let b_data_bytes = b_tile_rows * row;
            let partial_bytes = A_PANEL * b_tile_rows * ACC;
            let working_set = a_panel_bytes + b_data_bytes + partial_bytes;

            assert!(
                working_set <= L1_CACHE_BYTES,
                "k={k}: L1 working set {working_set} B (a_panel={a_panel_bytes}, \
                 b_data={b_data_bytes}, partial={partial_bytes}) exceeds L1 {L1_CACHE_BYTES} B",
            );
            assert!(plan.a_panels_per_tile >= 1 && plan.b_panels_per_tile >= 1);
        }
    }
}

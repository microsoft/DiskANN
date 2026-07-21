// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Coarse tiled MaxSim. A [`TileWalk`] streams materialized [`Tile`]s — lending, so a
//! converting walk can reuse one buffer — and each tile yields panels the kernel
//! binds on by type. Three compute seams — [`Accumulate`], [`Postprocess`],
//! [`Reducer`] — fold panels into per-A-row state, which *is* the output.
//! Instantiated for 4-bit MinMax i8 ([`minmax`]) and f16 ([`f16`]); the driver and
//! seams stay generic.

use core::mem::MaybeUninit;

use crate::alloc::{AllocatorCore, Poly, ScopedAllocator};

use super::TileBudget;

mod arena;
mod f16;
mod leaves;
mod minmax;
mod tilers;

pub use f16::{QuantTiledF16Docs, QuantTiledF16Query};
pub use minmax::{QuantTiledDocs, QuantTiledQuery};

// ── Tile planning (copy of `staged::StagedPlan`) ─────────────────

/// Panel counts per tile. `a_panels` A-panels sit resident in L2; as many B-panels
/// as co-fit L1 alongside one A-panel and the partial strip.
#[derive(Clone, Copy)]
struct Plan {
    a_panels: usize,
    b_panels: usize,
}

impl Plan {
    fn new(
        a_row_bytes: usize,
        b_row_bytes: usize,
        a_panel: usize,
        b_panel: usize,
        acc_bytes: usize,
        budget: TileBudget,
    ) -> Self {
        let a_row_bytes = a_row_bytes.max(1);
        let b_row_bytes = b_row_bytes.max(1);
        let a_panels = (budget.l2_a / (a_row_bytes * a_panel)).max(1);
        let a_panel_bytes = a_panel * a_row_bytes;
        let per_b_row = b_row_bytes + a_panel * acc_bytes;
        let b_budget = budget.l1_b.saturating_sub(a_panel_bytes);
        let b_panels = ((b_budget / per_b_row) / b_panel).max(1);
        Self { a_panels, b_panels }
    }
}

// ── Intermediate strips ──────────────────────────────────────────

/// An A-major intermediate (rows = A-panel, cols = `len / rows`), read-only.
pub(crate) struct Strip<'a, T> {
    data: &'a [T],
    rows: usize,
}
/// An A-major intermediate, writable. Also the kernel's per-call output sub-view.
pub(crate) struct StripMut<'a, T> {
    data: &'a mut [T],
    rows: usize,
}

impl<T> Strip<'_, T> {
    fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    fn cols(&self) -> usize {
        if self.rows == 0 {
            0
        } else {
            self.data.len() / self.rows
        }
    }
}
impl<T> StripMut<'_, T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }
}

/// Per-block context for [`Postprocess`]: the global row offsets a metadata-bearing
/// stage indexes by.
#[derive(Clone, Copy)]
pub(crate) struct BlockCtx {
    pub a_row_offset: usize,
    pub b_row_offset: usize,
}

// ── Data side ────────────────────────────────────────────────────

/// Misuse guard for [`TileAt`]'s implicit-bounds parameter: `Bounds` and `Sealed`
/// are private, so no downstream impl can override the defaulted parameter with a
/// type that drops the `Self: 'a` implied bound.
mod sealed {
    pub trait Sealed {}
    pub struct Bounds<T>(#[allow(dead_code)] T);
    impl<T> Sealed for Bounds<T> {}
}

/// The per-lifetime half of [`TileWalk`]: at lifetime `'a`, the walk yields a tile
/// borrowing `'a`. The defaulted `B = Bounds<&'a Self>` carries the `Self: 'a`
/// implied bound through the well-formedness of `&'a Self`, which is what makes the
/// `for<'a>` bound on the driver's kernel provable (a plain GAT `where Self: 'a`
/// collapses to `'static` under that HRTB on stable).
pub(crate) trait TileAt<'a, B: sealed::Sealed = sealed::Bounds<&'a Self>> {
    type Tile: Tile;
}

/// A **lending** walk over a source: `next` reborrows `&mut self`, so a materialized
/// tile (and its panels) can borrow a buffer the walk **reuses** on the following
/// `next` — the borrow checker forbids overwriting it while a tile is still live.
/// `reset` rewinds for a re-walk (the driver re-walks B once per A-tile) without
/// reallocating the buffer.
pub(crate) trait TileWalk: for<'a> TileAt<'a> {
    fn next(&mut self) -> Option<<Self as TileAt<'_>>::Tile>;
    fn reset(&mut self);

    /// Rows in a full tile — sizes the driver's inter-stage scratch.
    fn max_tile_rows(&self) -> usize;
}

/// One materialized tile: its global offset, its row count, and its panels split
/// into full panels plus an optional short tail. The borrow into the walker's buffer
/// lives on the concrete type (e.g. `QMat<'a, T>`), surfaced via [`TileAt::Tile`], so
/// the trait itself needs no lifetime.
pub(crate) trait Tile {
    type Panel: Copy;

    /// Global A-row (queries) or B-col (docs) where this tile starts.
    fn offset(&self) -> usize;
    /// Rows in this tile.
    fn rows(&self) -> usize;
    /// Full, fixed-size panels.
    fn panels(&self) -> impl Iterator<Item = Self::Panel> + '_;
    /// The short trailing panel, if the row count isn't a whole number of panels.
    fn tail(&self) -> Option<Self::Panel>;
}

// ── Compute side ─────────────────────────────────────────────────

/// Stage A, datatype-independent facts: the accumulator type and the panel sizes.
/// Split from [`Accumulate`] so `K::Acc` stays unambiguous under the driver's
/// `for<'a, 'b>` bound over the (lifetime-carrying) panel types.
pub(crate) trait Kernel<Arch> {
    type Acc: Copy;
    const A_PANEL: usize;
    const B_PANEL: usize;
}

/// Stage A — datatype axis. One A-panel × one B-panel → an A-major `Acc` block.
/// Pinned on the `(A, B)` panel pair as type parameters, so the walks' panel types
/// select the kernel with no `Convert::To = _` join. `accumulate` is the fixed-width
/// hot path; `accumulate_tail` handles a `1..B_PANEL` remainder.
pub(crate) trait Accumulate<Arch, A, B>: Kernel<Arch> {
    fn accumulate(&self, arch: Arch, a: A, b: B, out: StripMut<'_, Self::Acc>);
    fn accumulate_tail(&self, arch: Arch, a: A, b: B, out: StripMut<'_, Self::Acc>);
}

/// Stage B — quantization axis. `Acc` strip → `Score` strip. Identity returns `acc`
/// (`Score = Acc`); a metadata stage writes `scratch` and returns it.
pub(crate) trait Postprocess<Arch, Acc> {
    type Score: Copy;

    fn scratch_len(&self, cols: usize) -> usize;
    fn apply<'s>(
        &self,
        arch: Arch,
        acc: Strip<'s, Acc>,
        scratch: StripMut<'s, Self::Score>,
        ctx: BlockCtx,
    ) -> Strip<'s, Self::Score>;
}

/// Stage C — reducer axis. Fold a `Score` strip into per-A-row `State`; `State` *is*
/// the output (the caller interprets it). `first_col` = global B offset for argmax.
pub(crate) trait Reducer<Arch, Score> {
    type State: Copy;
    const A_PANEL: usize;

    fn init() -> Self::State;
    fn fold(&self, arch: Arch, state: &mut [Self::State], scores: Strip<Score>, first_col: usize);
}

/// The zero-cost [`Postprocess`]: `Score = Acc`, returns the accumulator strip
/// untouched (no scratch, no metadata). For kernels whose `Acc` is already the score.
pub(crate) struct Identity;

impl<Arch, Acc: Copy> Postprocess<Arch, Acc> for Identity {
    type Score = Acc;

    fn scratch_len(&self, _cols: usize) -> usize {
        0
    }
    fn apply<'s>(
        &self,
        _arch: Arch,
        acc: Strip<'s, Acc>,
        _scratch: StripMut<'s, Acc>,
        _ctx: BlockCtx,
    ) -> Strip<'s, Acc> {
        acc
    }
}

// ── Scratch: uninit alloc → zeroed slice ─────────────────────────

/// Marker for element types where all-zero is a valid value, so a zeroed allocation
/// is a sound `&mut [T]`.
pub(crate) trait ZeroInit: Copy {}
impl ZeroInit for i16 {}
impl ZeroInit for i32 {}
impl ZeroInit for f32 {}
impl ZeroInit for u8 {}

fn zeroed<T: ZeroInit, A: AllocatorCore + std::fmt::Debug>(
    poly: &mut Poly<[MaybeUninit<T>], A>,
    len: usize,
) -> &mut [T] {
    let ptr = poly.as_mut_ptr().cast::<T>();
    // SAFETY: the poly owns `len` `T`-sized slots; `T: ZeroInit` ⇒ all-zero is a valid
    // `T`, so zeroing initializes every element and the slice is sound as `&mut [T]`.
    unsafe {
        core::ptr::write_bytes(ptr, 0, len);
        core::slice::from_raw_parts_mut(ptr, len)
    }
}

// ── Driver ───────────────────────────────────────────────────────

/// Per-A-row reduction into `state` (len ≥ padded A rows) via the seams. The walks
/// carry the plan (baked tile sizes); the driver reads only the B-tile width to size
/// the scratch it allocates from `alloc`, so the caller sizes nothing. B is re-walked
/// (`reset`) once per A-tile — for a convert walk that re-runs its per-tile transform,
/// which under the default budget is a single A-tile (one pass).
#[allow(clippy::too_many_arguments, clippy::expect_used)]
pub(super) fn drive<Arch, AW, BW, K, P, R>(
    arch: Arch,
    mut a_walk: AW,
    mut b_walk: BW,
    kernel: &K,
    post: &P,
    reducer: &R,
    state: &mut [R::State],
    alloc: ScopedAllocator<'_>,
) where
    Arch: Copy,
    AW: TileWalk,
    BW: TileWalk,
    K: Kernel<Arch>
        + for<'a, 'b> Accumulate<
            Arch,
            <<AW as TileAt<'a>>::Tile as Tile>::Panel,
            <<BW as TileAt<'b>>::Tile as Tile>::Panel,
        >,
    P: Postprocess<Arch, K::Acc>,
    R: Reducer<Arch, P::Score>,
    K::Acc: ZeroInit,
    P::Score: ZeroInit,
{
    const { assert!(K::A_PANEL == R::A_PANEL) }
    let a_panel = K::A_PANEL;
    let b_panel = K::B_PANEL;

    for s in state.iter_mut() {
        *s = R::init();
    }

    let b_tile_rows = b_walk.max_tile_rows();
    let strip_len = a_panel * b_tile_rows;
    let scored_len = post.scratch_len(b_tile_rows);

    let mut partial_poly =
        Poly::<[K::Acc], _>::new_uninit_slice(strip_len, alloc).expect("partial scratch");
    let mut scored_poly =
        Poly::<[P::Score], _>::new_uninit_slice(scored_len, alloc).expect("scored scratch");
    let partial = zeroed(&mut partial_poly, strip_len);
    let scored = zeroed(&mut scored_poly, scored_len);

    while let Some(a_mat) = a_walk.next() {
        // The driver has no A-tail path (it never calls `a_mat.tail()`), so a partial
        // A-panel would be silently dropped by `panels()`. Block-transposed padding
        // guarantees whole panels; assert it so an unpadded A source fails loudly.
        debug_assert_eq!(
            a_mat.rows() % a_panel,
            0,
            "A walk must yield whole A-panels; the driver has no A-tail path"
        );
        let a_tile_off = a_mat.offset();
        b_walk.reset();
        while let Some(b_mat) = b_walk.next() {
            let (cols, b_off) = (b_mat.rows(), b_mat.offset());
            let w = a_panel * cols;
            let full_w = a_panel * (cols - cols % b_panel);

            for (i, a) in a_mat.panels().enumerate() {
                let a_off = a_tile_off + i * a_panel;

                // Stage A: fill the partial strip. Full B-panels pair with equal-width
                // output blocks; the short tail dispatches its width in the kernel.
                for (b, obuf) in b_mat
                    .panels()
                    .zip(partial[..full_w].chunks_mut(a_panel * b_panel))
                {
                    kernel.accumulate(
                        arch,
                        a,
                        b,
                        StripMut {
                            data: obuf,
                            rows: a_panel,
                        },
                    );
                }
                if let Some(b) = b_mat.tail() {
                    let obuf = &mut partial[full_w..w];
                    kernel.accumulate_tail(
                        arch,
                        a,
                        b,
                        StripMut {
                            data: obuf,
                            rows: a_panel,
                        },
                    );
                }

                // Stage B: Acc → Score (identity returns the partial strip untouched).
                let scores = post.apply(
                    arch,
                    Strip {
                        data: &partial[..w],
                        rows: a_panel,
                    },
                    StripMut {
                        data: &mut scored[..post.scratch_len(cols)],
                        rows: a_panel,
                    },
                    BlockCtx {
                        a_row_offset: a_off,
                        b_row_offset: b_off,
                    },
                );

                // Stage C: fold into this A-panel's state slice.
                reducer.fold(arch, &mut state[a_off..a_off + a_panel], scores, b_off);
            }
        }
    }
}

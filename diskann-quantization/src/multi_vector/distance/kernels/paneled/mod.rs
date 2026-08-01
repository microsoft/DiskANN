// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Paneled MaxSim: a [`TileWalk`] lends cache-sized views, each [`Paneled`] into
//! register-sized panels plus a typed tail. [`Accumulate`] folds one (A-panel,
//! B-panel) pair into an accumulator slot; [`Drain`] turns the finished accumulator
//! into output it owns. [`Scratch`] is the write-side mirror of [`Paneled`], so the
//! driver assumes no layout on either side.
//!
//! Panel widths are geometry, so they live with the leaves that impose them
//! ([`leaves::A_PANEL`], [`leaves::B_PANEL`]); the panel and accumulator types carry
//! them as const parameters (`R` = A rows, `N` = B rows) so the driver stays width-
//! agnostic. Instantiated for f32 ([`float`]) and 4-bit MinMax ([`minmax`]).
//!
//! Sibling to [`tiler`](super::tiler), which keeps postprocess and reduce separate.

use super::TileBudget;

mod arena;
mod float;
mod leaves;
mod minmax;
mod strip;
mod views;

pub(crate) use strip::{Block, Strip, StripRef};

pub use float::{PaneledF32Docs, PaneledF32Query};
pub use minmax::{PaneledQuantDocs, PaneledQuantQuery};

// ── Tile planning ────────────────────────────────────────────────

/// Panel counts per tile. `a_panels` A-panels sit resident in L2; as many B-panels as
/// co-fit L1 alongside one A-panel and the accumulator.
#[derive(Clone, Copy)]
struct Plan<const R: usize, const N: usize> {
    a_panels: usize,
    b_panels: usize,
}

impl<const R: usize, const N: usize> Plan<R, N> {
    fn new(a_row_bytes: usize, b_row_bytes: usize, acc_bytes: usize, budget: TileBudget) -> Self {
        let a_row_bytes = a_row_bytes.max(1);
        let b_row_bytes = b_row_bytes.max(1);
        let a_panels = (budget.l2_a / (a_row_bytes * R)).max(1);
        let a_panel_bytes = R * a_row_bytes;
        let per_b_row = b_row_bytes + R * acc_bytes;
        let b_budget = budget.l1_b.saturating_sub(a_panel_bytes);
        let b_panels = ((b_budget / per_b_row) / N).max(1);
        Self { a_panels, b_panels }
    }

    /// Accumulator elements: one A-panel × a whole B-tile.
    fn strip_len(&self) -> usize {
        R * self.b_panels * N
    }
}

// ── Accumulator ──────────────────────────────────────────────────

/// Per-lifetime half of [`Scratch`] — same sealed-`Bounds` trick as [`TileAt`].
pub(crate) trait ScratchAt<'a, B: sealed::Sealed = sealed::Bounds<&'a mut Self>> {
    type Block;
    /// Short trailing slot; a distinct type so it selects its own [`Accumulate`] impl.
    type Short;
    /// Named so `Block`/`Short` reach bounds without a nested projection.
    type Blocks: TailIterator<Item = Self::Block, Tail = Self::Short>;
    /// What [`Drain`] reads.
    type Ref;
}

/// [`Paneled`]'s write side: a buffer that carves itself into per-B-panel slots, so
/// the driver can't tell a contiguous strip from a padded or structure-of-arrays one.
/// Not [`Paneled`] itself — `Panel: Copy` and `panels(&self)` can't yield disjoint
/// `&mut`, and the carve needs a runtime `cols` (a scratch is sized to capacity).
/// Allocation stays on the concrete type.
pub(crate) trait Scratch: for<'a> ScratchAt<'a> {
    /// Carve the live `cols` columns into one slot per B-panel plus the short trailer,
    /// which comes off the same cursor and so is provably disjoint.
    fn blocks(&mut self, cols: usize) -> <Self as ScratchAt<'_>>::Blocks;

    fn as_ref(&self, cols: usize) -> <Self as ScratchAt<'_>>::Ref;
}

// ── Data side ────────────────────────────────────────────────────

/// Misuse guard for the implicit-bounds parameter: private, so no downstream impl can
/// override the default with a type that drops the implied bound.
mod sealed {
    pub trait Sealed {}
    pub struct Bounds<T>(#[allow(dead_code)] T);
    impl<T> Sealed for Bounds<T> {}
}

/// Per-lifetime half of [`TileWalk`]. The defaulted `B = Bounds<&'a Self>` carries the
/// `Self: 'a` implied bound through well-formedness — a plain GAT `where Self: 'a`
/// collapses to `'static` under the driver's `for<'a>` bound on stable.
pub(crate) trait TileAt<'a, B: sealed::Sealed = sealed::Bounds<&'a Self>> {
    type View: Paneled;
}

/// A **lending** walk: `next` reborrows `&mut self`, so a view may borrow a buffer the
/// walk reuses on the following call. `reset` rewinds — B is re-walked once per A-tile.
pub(crate) trait TileWalk: for<'a> TileAt<'a> {
    fn next(&mut self) -> Option<Tile<<Self as TileAt<'_>>::View>>;
    fn reset(&mut self);
}

/// A lent view plus where it starts. Lifetime-free — the borrow lives on `V`.
pub(crate) struct Tile<V> {
    pub(crate) view: V,
    /// Position in the walk's own unit: A-panels for a query walk, B-rows for a doc
    /// walk. Only the [`Drain`] turns it into an output index.
    pub(crate) at: usize,
}

/// An iterator whose short trailing element has its own type. `tail` consumes the
/// exhausted iterator, so the trailer comes off the cursor the loop was already
/// advancing instead of being recomputed from the source.
pub(crate) trait TailIterator: ExactSizeIterator {
    type Tail;
    fn tail(self) -> Option<Self::Tail>;
}

/// A view that knows how it breaks into panels. `Tail` is distinct from `Panel` so the
/// short trailing panel selects its own [`Accumulate`] impl; a view that cannot tail
/// says [`NoTail`].
pub(crate) trait Paneled {
    type Panel: Copy;
    type Tail: Copy;
    /// Named so it carries `ExactSizeIterator` and the tail type into bounds, and so a
    /// k-fracturing driver could hold one across an outer loop.
    type Panels: TailIterator<Item = Self::Panel, Tail = Self::Tail>;

    fn rows(&self) -> usize;
    fn panels(&self) -> Self::Panels;
}

/// `Tail` for a view padded to whole panels. Uninhabited, so `tail()` provably
/// returns `None`.
#[derive(Clone, Copy)]
pub(crate) enum NoTail {}

// ── Compute side ─────────────────────────────────────────────────

/// One A-panel × one B-panel → an accumulator slot. Pinned on all three as type
/// parameters, so the walks' panel types select the impl.
pub(crate) trait Accumulate<Arch, A, B, O> {
    fn accumulate(&self, arch: Arch, a: A, b: B, out: O);
}

/// [`NoTail`] is uninhabited, so this one impl discharges the driver's A-tail bounds
/// for every kernel.
impl<Arch, B, O, K> Accumulate<Arch, NoTail, B, O> for K {
    #[inline(always)]
    fn accumulate(&self, _: Arch, a: NoTail, _: B, _: O) {
        match a {}
    }
}

/// Where a finished accumulator sits in the global problem. The driver counts panels
/// and never converts a count into a row — only the [`Drain`] owns `R`. B is a row
/// rather than a panel index because a tile is not panel-quantized.
#[derive(Clone, Copy)]
pub(crate) struct At {
    pub a_panel: usize,
    pub b_row: usize,
}

/// Consume a finished accumulator. The drain owns its output, so dequant, reduction
/// and scatter all live behind this one call and may be fused.
///
/// Implementations must initialize their output to the reduction's identity, and must
/// clamp their writes when the output is not padded to whole panels.
pub(crate) trait Drain<Arch, S: Scratch> {
    fn drain(&mut self, arch: Arch, acc: <S as ScratchAt<'_>>::Ref, at: At);
}

// ── Driver ───────────────────────────────────────────────────────

type PanelOf<'a, W> = <<W as TileAt<'a>>::View as Paneled>::Panel;
type TailOf<'a, W> = <<W as TileAt<'a>>::View as Paneled>::Tail;
type BlockOf<'x, S> = <S as ScratchAt<'x>>::Block;
type ShortOf<'x, S> = <S as ScratchAt<'x>>::Short;

/// One A-panel against a whole B-tile. Factored out so the driver's A-panel and
/// A-tail arms share the tail-dispatch.
#[inline(always)]
fn fill<Arch, A, BV, S, K>(arch: Arch, kernel: &K, a: A, b_view: &BV, scratch: &mut S, cols: usize)
where
    Arch: Copy,
    A: Copy,
    BV: Paneled,
    S: Scratch,
    K: for<'x> Accumulate<Arch, A, BV::Panel, BlockOf<'x, S>>
        + for<'x> Accumulate<Arch, A, BV::Tail, ShortOf<'x, S>>,
{
    let mut panels = b_view.panels();
    let mut blocks = scratch.blocks(cols);
    // What [`TailIterator`]'s `ExactSizeIterator` bound is for: `zip` stops on the
    // shorter side and silently drops the longer side's pending item, which would leave
    // both cursors at zero and slip past the tail check below.
    debug_assert_eq!(panels.len(), blocks.len(), "B view and accumulator desync");
    for (b, out) in panels.by_ref().zip(blocks.by_ref()) {
        kernel.accumulate(arch, a, b, out);
    }
    match (panels.tail(), blocks.tail()) {
        (Some(b), Some(out)) => kernel.accumulate(arch, a, b, out),
        (None, None) => {}
        _ => unreachable!("B view and accumulator disagree on tail"),
    }
}

/// Drive one A source against one B source. The walks carry the plan, `scratch` the
/// accumulator, `drain` the output — so this does no stride arithmetic and knows
/// nothing about where results go. B is re-walked once per A-tile.
///
/// `S` is not inferable (the `for<'x>` bounds project through it); call sites
/// turbofish it.
pub(super) fn drive<Arch, AW, BW, K, S, D>(
    arch: Arch,
    mut a_walk: AW,
    mut b_walk: BW,
    kernel: &K,
    scratch: &mut S,
    drain: &mut D,
) where
    Arch: Copy,
    AW: TileWalk,
    BW: TileWalk,
    S: Scratch,
    K: for<'a, 'b, 'x> Accumulate<Arch, PanelOf<'a, AW>, PanelOf<'b, BW>, BlockOf<'x, S>>
        + for<'a, 'b, 'x> Accumulate<Arch, PanelOf<'a, AW>, TailOf<'b, BW>, ShortOf<'x, S>>
        + for<'a, 'b, 'x> Accumulate<Arch, TailOf<'a, AW>, PanelOf<'b, BW>, BlockOf<'x, S>>
        + for<'a, 'b, 'x> Accumulate<Arch, TailOf<'a, AW>, TailOf<'b, BW>, ShortOf<'x, S>>,
    D: Drain<Arch, S>,
{
    while let Some(a_tile) = a_walk.next() {
        b_walk.reset();
        while let Some(b_tile) = b_walk.next() {
            let cols = b_tile.view.rows();
            let mut at = At {
                a_panel: a_tile.at,
                b_row: b_tile.at,
            };

            let mut a_panels = a_tile.view.panels();
            for a in a_panels.by_ref() {
                fill(arch, kernel, a, &b_tile.view, scratch, cols);
                drain.drain(arch, scratch.as_ref(cols), at);
                at.a_panel += 1;
            }
            if let Some(a) = a_panels.tail() {
                fill(arch, kernel, a, &b_tile.view, scratch, cols);
                drain.drain(arch, scratch.as_ref(cols), at);
            }
        }
    }
}

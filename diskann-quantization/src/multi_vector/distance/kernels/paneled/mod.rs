// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Paneled MaxSim: a [`TileWalk`] lends cache-sized views, each [`Paneled`] into
//! register-sized panels plus a typed tail. [`Accumulate`] folds one (A-panel,
//! B-panel) pair into an accumulator slot; [`Drain`] turns the finished accumulator
//! into output it owns. [`Scratch`] is the write-side mirror of [`Paneled`], so the
//! driver assumes no layout on either side.
//!
//! Position has one owner. The views state extents only — how much they hold, never
//! where it sits — and [`Scratch`] tracks the visit order itself, so each tile it lends
//! already knows where it belongs. The address a result is written to and the memory it
//! is written into are therefore decided by the same value, and a scratch that maps
//! tiles somewhere else (a matmul writing into `C`) is a different implementation
//! rather than a different driver.
//!
//! Panel widths live with the leaves that impose them, reaching the panel and
//! accumulator types as const parameters (`R` = A rows, `N` = B rows).
//!
//! Sibling to [`tiler`](super::tiler), which keeps postprocess and reduce separate.

use super::TileBudget;

mod arena;
mod float;
mod leaves;
mod minmax;
mod strip;
mod views;

pub(crate) use strip::{Block, Strip, StripScratch};

pub use float::{PaneledF32Docs, PaneledF32Query};
pub use minmax::{PaneledQuantDocs, PaneledQuantQuery};

// ── Tile planning ────────────────────────────────────────────────

/// Panel counts per tile. `a_panels` A-panels sit resident in L2; as many B-panels as
/// co-fit L1 alongside one A-panel and the accumulator.
#[derive(Clone, Copy)]
pub(crate) struct Plan<const R: usize, const N: usize> {
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

/// Per-lifetime half of [`Scratch`] — same implied-bound trick as [`TileAt`].
pub(crate) trait ScratchAt<'a, B = &'a mut Self> {
    type Tile: ScratchTile;
}

/// Lends output tiles in the driver's visit order, each stamped with where it belongs.
///
/// Exhaustive, not infallible: this is a cursor over the whole problem, so "ran out" is
/// the terminal state rather than an error. The driver is steered by the walks, so it
/// reaches the end first — a `None` there means the scratch was built for a different
/// problem than the walks were, and [`drive`] treats it as such.
pub(crate) trait Scratch: for<'a> ScratchAt<'a> {
    fn next(&mut self) -> Option<<Self as ScratchAt<'_>>::Tile>;
}

/// Per-lifetime half of [`ScratchTile`].
///
/// `Block` is the tile's own type, so what a slot carries is the scratch's choice: a
/// kernel that must know where it writes (a matmul landing in `C`) gets a `Block` that
/// says so, while MaxSim's stays a bare pointer. The driver is unaffected either way —
/// it only projects the type through.
pub(crate) trait SlotsAt<'s, B = &'s mut Self> {
    /// One B-panel's slot. Named here, rather than reached through [`Slots`], so the
    /// driver's bounds project it off the tile.
    type Block;
    type Slots: Slots<Block = Self::Block>;
}

/// One tile of output: the memory a fill writes into.
///
/// Where that memory belongs is deliberately *not* here. Position is the concrete
/// scratch's own vocabulary, and a scratch is always instantiated alongside the
/// [`Drain`] that reads it, so each pair agrees on its own terms.
pub(crate) trait ScratchTile: for<'s> SlotsAt<'s> {
    fn slots(&mut self) -> <Self as SlotsAt<'_>>::Slots;
}

/// Hands out one accumulator slot per B-panel, each disjoint from the last.
///
/// Exhaustive for the same reason as [`Scratch`]: the tile derives its own width, so a
/// tile narrower than the B-panels the walk yields is now reachable rather than
/// impossible, and silently aliasing a slot would corrupt the answer.
///
/// Alone among the write-side traits this needs no lifetime parameter, because an
/// implementor can move its buffer out of itself rather than reborrow — see
/// `BlockSlots`. That keeps the slot's own lifetime and spares the driver a third
/// binder; [`ScratchTile`] cannot do the same, since the tile must outlive `slots()`.
pub(crate) trait Slots {
    type Block;
    fn next(&mut self) -> Option<Self::Block>;
}

// ── Data side ────────────────────────────────────────────────────

/// Per-lifetime half of [`TileWalk`]. The defaulted `B = &'a Self` carries the
/// `Self: 'a` implied bound through well-formedness — a plain GAT `where Self: 'a`
/// collapses to `'static` under the driver's `for<'a>` bound on stable.
pub(crate) trait TileAt<'a, B = &'a Self> {
    type View: Paneled;
}

/// A **lending** walk: `next` reborrows `&mut self`, so a view may borrow a buffer the
/// walk reuses on the following call. `reset` rewinds — B is re-walked once per A-tile.
pub(crate) trait TileWalk: for<'a> TileAt<'a> {
    fn next(&mut self) -> Option<<Self as TileAt<'_>>::View>;
    fn reset(&mut self);
}

/// An iterator whose short trailing element has its own type. `tail` consumes the
/// exhausted iterator, so the trailer comes off the cursor the loop was already
/// advancing instead of being recomputed from the source.
///
/// `ExactSizeIterator` binds implementors, not the driver: a view that cannot state its
/// panel count exactly does not know its own geometry.
pub(crate) trait TailIterator: ExactSizeIterator {
    type Tail;
    fn tail(self) -> Option<Self::Tail>;
}

/// A view that knows how it breaks into panels. `Tail` is distinct from `Panel` so the
/// short trailing panel selects its own [`Accumulate`] impl; a view that cannot tail
/// says [`NoTail`].
///
/// Extent only, never position: a view says how much it holds, and [`Scratch`] says
/// where that lands. So the same view can be swept by a driver that visits it in a
/// different order.
pub(crate) trait Paneled {
    type Panel: Copy;
    type Tail: Copy;
    /// Named so it carries `ExactSizeIterator` and the tail type into bounds, and so a
    /// k-fracturing driver could hold one across an outer loop.
    type Panels: TailIterator<Item = Self::Panel, Tail = Self::Tail>;

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

/// [`NoTail`] is uninhabited, so this discharges the driver's A-tail bounds for every
/// kernel — and, by coherence, forbids any kernel from writing its own. Both follow
/// from the same fact: an A-tail arm is unreachable.
impl<Arch, B, O, K> Accumulate<Arch, NoTail, B, O> for K {
    #[inline(always)]
    fn accumulate(&self, _: Arch, a: NoTail, _: B, _: O) {
        match a {}
    }
}

/// Consume a finished tile. The drain owns its output, so dequant, reduction and
/// scatter all live behind this one call and may be fused.
///
/// Generic over the tile type, so a drain reads position in whatever terms its scratch
/// states it — there is no common accessor to agree on.
pub(crate) trait Drain<Arch, T> {
    fn drain(&mut self, arch: Arch, tile: &T);
}

// ── Driver ───────────────────────────────────────────────────────

type PanelOf<'a, W> = <<W as TileAt<'a>>::View as Paneled>::Panel;
type TailOf<'a, W> = <<W as TileAt<'a>>::View as Paneled>::Tail;
type TileOf<'a, S> = <S as ScratchAt<'a>>::Tile;
type BlockOf<'s, T> = <T as SlotsAt<'s>>::Block;

/// Cold, so the check costs no more than the branch the exhaustive contract already
/// pays for, and the panic's formatting machinery stays out of the inlined body.
#[cold]
#[inline(never)]
fn scratch_exhausted() -> ! {
    unreachable!("scratch ran out before the walks did — it was built for a different problem")
}

#[cold]
#[inline(never)]
fn slots_exhausted() -> ! {
    unreachable!("tile ran out of slots — its B-tile is narrower than the walk's")
}

/// One A-panel against a whole B-tile. Factored out so the driver's A-panel and
/// A-tail arms share the tail-dispatch.
#[inline(always)]
fn fill<Arch, A, BV, T, K>(arch: Arch, kernel: &K, a: A, b_view: &BV, tile: &mut T)
where
    Arch: Copy,
    A: Copy,
    BV: Paneled,
    T: ScratchTile,
    K: for<'s> Accumulate<Arch, A, BV::Panel, BlockOf<'s, T>>
        + for<'s> Accumulate<Arch, A, BV::Tail, BlockOf<'s, T>>,
{
    let mut panels = b_view.panels();
    let mut slots = tile.slots();
    for b in panels.by_ref() {
        let Some(out) = slots.next() else {
            slots_exhausted()
        };
        kernel.accumulate(arch, a, b, out);
    }
    // The tail draws from the same cursor as the full panels.
    if let Some(b) = panels.tail() {
        let Some(out) = slots.next() else {
            slots_exhausted()
        };
        kernel.accumulate(arch, a, b, out);
    }
}

/// Drive one A source against one B source. The walks carry the plan, `scratch` the
/// accumulator and where its results belong, `drain` the output — so this does no
/// stride arithmetic and knows nothing about position. B is re-walked once per A-tile.
///
/// `scratch` precedes `kernel` deliberately. The kernel's bounds project through all
/// three sources, and arguments type-check left to right, so a `scratch` placed after
/// it would still be an inference variable when those bounds are proved — forcing
/// every call site to turbofish `S`.
///
/// # Panics
///
/// If `scratch` runs out while the walks still have work — its geometry disagrees with
/// theirs, and finishing early would return a wrong answer rather than fail.
pub(super) fn drive<Arch, AW, BW, K, S, D>(
    arch: Arch,
    mut a_walk: AW,
    mut b_walk: BW,
    scratch: &mut S,
    kernel: &K,
    drain: &mut D,
) where
    Arch: Copy,
    AW: TileWalk,
    BW: TileWalk,
    S: Scratch,
    K: for<'a, 'b, 'x, 's> Accumulate<
            Arch,
            PanelOf<'a, AW>,
            PanelOf<'b, BW>,
            BlockOf<'s, TileOf<'x, S>>,
        > + for<'a, 'b, 'x, 's> Accumulate<
            Arch,
            PanelOf<'a, AW>,
            TailOf<'b, BW>,
            BlockOf<'s, TileOf<'x, S>>,
        > + for<'a, 'b, 'x, 's> Accumulate<
            Arch,
            TailOf<'a, AW>,
            PanelOf<'b, BW>,
            BlockOf<'s, TileOf<'x, S>>,
        > + for<'a, 'b, 'x, 's> Accumulate<
            Arch,
            TailOf<'a, AW>,
            TailOf<'b, BW>,
            BlockOf<'s, TileOf<'x, S>>,
        >,
    D: for<'x> Drain<Arch, TileOf<'x, S>>,
{
    while let Some(a_view) = a_walk.next() {
        b_walk.reset();
        while let Some(b_view) = b_walk.next() {
            let mut a_panels = a_view.panels();
            for panel in a_panels.by_ref() {
                let Some(mut tile) = scratch.next() else {
                    scratch_exhausted()
                };
                fill(arch, kernel, panel, &b_view, &mut tile);
                drain.drain(arch, &tile);
            }
            if let Some(panel) = a_panels.tail() {
                let Some(mut tile) = scratch.next() else {
                    scratch_exhausted()
                };
                fill(arch, kernel, panel, &b_view, &mut tile);
                drain.drain(arch, &tile);
            }
        }
    }
    // The other half of the bracket: over-running panics above, under-running lands
    // here. Not a wrong answer, so it need not cost anything in release.
    debug_assert!(
        scratch.next().is_none(),
        "scratch outlived the walks — it was built for a different problem"
    );
}

#[cfg(test)]
mod tests {
    use core::mem::size_of;

    use super::leaves::{A_PANEL, B_PANEL};
    use super::views::{DPanel, QPanel};
    use super::{Block, Strip};
    use crate::bits::{Dynamic, Static};

    /// Handles stay thin on both sides: the geometry a leaf needs is const parameters,
    /// so only what is genuinely runtime — the contraction length, and a tail's row
    /// count — is ever stored.
    ///
    /// A guard, not a curiosity: each side lost this once to a refactor that swapped a
    /// pointer for a slice, and neither showed up in a correctness test.
    #[test]
    fn handles_stay_thin() {
        let word = size_of::<*const u8>();

        // Both carry `k` rather than borrowing the other's — what makes the leaves safe.
        assert_eq!(size_of::<QPanel<'static, f32, A_PANEL>>(), 2 * word);
        assert_eq!(
            size_of::<DPanel<'static, f32, B_PANEL, Static<B_PANEL>>>(),
            2 * word
        );

        // Only the trailing panel pays for a runtime row count: `Static<N>` is a ZST.
        assert_eq!(
            size_of::<DPanel<'static, f32, B_PANEL, Dynamic>>(),
            3 * word
        );

        // The slot's extent is in the type; the tile is the checked anchor it is carved
        // from, so it keeps a length, plus the three scalars its cuts need.
        assert_eq!(size_of::<Block<'static, f32, A_PANEL, B_PANEL>>(), word);
        assert_eq!(size_of::<Strip<'static, f32, A_PANEL, B_PANEL>>(), 5 * word);
    }
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Paneled MaxSim: a [`TileWalk`] lends cache-sized views, each [`Paneled`] into
//! register-sized panels plus a typed tail. [`Accumulate`] folds one (A-panel,
//! B-panel) pair into an accumulator slot; [`Drain`] turns the finished accumulator
//! into output it owns. [`Scratch`] is the write-side mirror of [`Paneled`], so the
//! driver assumes no layout on either side.
//!
//! Position is ordinal: [`drive`] counts the panels it passes and hands a [`Drain`] an
//! A-panel index and a B-panel range, never a stride or an address. Each consumer cuts
//! for itself — `index * PANEL` starts it, and the end clamps against its own extent,
//! so a side whose type proves it never tails ([`NoTail`]) need not clamp.
//!
//! Panel widths live with the leaves that impose them, reaching the panel and
//! accumulator types as const parameters (`R` = A rows, `N` = B rows).
//!
//! Sibling to [`tiler`](super::tiler), which keeps postprocess and reduce separate.

use core::ops::Range;

use super::TileBudget;

mod arena;
mod float;
mod leaves;
mod minmax;
mod strip;
mod views;

pub(crate) use strip::{Block, Strip};

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
pub(crate) trait SlotsAt<'s, B = &'s mut Self> {
    /// One B-panel's slot. Named here, rather than reached through the iterator, so the
    /// driver's bounds project it off the scratch.
    type Block;
    /// Plain rather than lending: slots partition one buffer within a single call, so
    /// none of them borrows the cursor — unlike [`TileWalk`], which reuses one buffer
    /// *across* calls.
    type Slots: Iterator<Item = Self::Block>;
}

/// The write-side mirror of [`Paneled`]: memory a fill carves into one slot per
/// B-panel, each disjoint from the last.
pub(crate) trait Scratch: for<'s> SlotsAt<'s> {
    fn slots(&mut self) -> <Self as SlotsAt<'_>>::Slots;
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
/// kernel — and, by coherence, forbids any kernel from writing its own.
impl<Arch, B, O, K> Accumulate<Arch, NoTail, B, O> for K {
    #[inline(always)]
    fn accumulate(&self, _: Arch, a: NoTail, _: B, _: O) {
        match a {}
    }
}

/// Consume a finished accumulator. The drain owns its output, so dequant, reduction and
/// scatter all live behind this one call and may be fused.
///
/// `a_panel` and `b_panels` are ordinals in [`drive`]'s visit order, not addresses.
pub(crate) trait Drain<Arch, S> {
    fn drain(&mut self, arch: Arch, scratch: &S, a_panel: usize, b_panels: Range<usize>);
}

// ── Driver ───────────────────────────────────────────────────────

type PanelOf<'a, W> = <<W as TileAt<'a>>::View as Paneled>::Panel;
type TailOf<'a, W> = <<W as TileAt<'a>>::View as Paneled>::Tail;
type BlockOf<'s, S> = <S as SlotsAt<'s>>::Block;

#[cold]
#[inline(never)]
fn slots_exhausted() -> ! {
    unreachable!("scratch ran out of slots — it is narrower than the walk's B-tile")
}

/// One A-panel against a whole B-tile. Returns the slots it filled, which is the
/// B-tile's panel count — the driver's only measure of how far B advanced.
#[inline(always)]
fn fill<Arch, A, BV, S, K>(arch: Arch, kernel: &K, a: A, b_view: &BV, scratch: &mut S) -> usize
where
    Arch: Copy,
    A: Copy,
    BV: Paneled,
    S: Scratch,
    K: for<'s> Accumulate<Arch, A, BV::Panel, BlockOf<'s, S>>
        + for<'s> Accumulate<Arch, A, BV::Tail, BlockOf<'s, S>>,
{
    let mut panels = b_view.panels();
    let mut slots = scratch.slots();
    let mut filled = 0;
    for b in panels.by_ref() {
        let Some(out) = slots.next() else {
            slots_exhausted()
        };
        kernel.accumulate(arch, a, b, out);
        filled += 1;
    }
    // The tail draws from the same cursor as the full panels.
    if let Some(b) = panels.tail() {
        let Some(out) = slots.next() else {
            slots_exhausted()
        };
        kernel.accumulate(arch, a, b, out);
        filled += 1;
    }
    filled
}

/// Drive one A source against one B source. The walks carry the plan, `scratch` the
/// accumulator, `drain` the output — so this does no stride arithmetic and knows
/// nothing about position beyond counting the panels it has passed. B is re-walked once
/// per A-tile.
///
/// `scratch` precedes `kernel` deliberately. The kernel's bounds project through all
/// three sources, and arguments type-check left to right, so a `scratch` placed after
/// it would still be an inference variable when those bounds are proved — forcing
/// every call site to turbofish `S`.
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
    K: for<'a, 'b, 's> Accumulate<Arch, PanelOf<'a, AW>, PanelOf<'b, BW>, BlockOf<'s, S>>
        + for<'a, 'b, 's> Accumulate<Arch, PanelOf<'a, AW>, TailOf<'b, BW>, BlockOf<'s, S>>
        + for<'a, 'b, 's> Accumulate<Arch, TailOf<'a, AW>, PanelOf<'b, BW>, BlockOf<'s, S>>
        + for<'a, 'b, 's> Accumulate<Arch, TailOf<'a, AW>, TailOf<'b, BW>, BlockOf<'s, S>>,
    D: Drain<Arch, S>,
{
    let mut a_base = 0;
    while let Some(a_view) = a_walk.next() {
        b_walk.reset();
        let mut b_base = 0;
        // Every B-tile re-sweeps the same A-panels, so both are rewritten identically
        // each pass and read after the last. An A-tile with no B-tiles advances
        // neither, which is unobservable: no drain fires, and a B source empty for one
        // A-tile is empty for all.
        let mut a_end = a_base;
        let mut b_used = 0;
        while let Some(b_view) = b_walk.next() {
            let mut a_panel = a_base;
            let mut a_panels = a_view.panels();
            for panel in a_panels.by_ref() {
                b_used = fill(arch, kernel, panel, &b_view, scratch);
                drain.drain(arch, scratch, a_panel, b_base..b_base + b_used);
                a_panel += 1;
            }
            if let Some(panel) = a_panels.tail() {
                b_used = fill(arch, kernel, panel, &b_view, scratch);
                drain.drain(arch, scratch, a_panel, b_base..b_base + b_used);
                a_panel += 1;
            }
            a_end = a_panel;
            b_base += b_used;
        }
        a_base = a_end;
    }
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
    /// Each side lost this once already to a refactor that swapped a pointer for a
    /// slice, and neither showed up in a test. The difference is too small for an A/B
    /// of two builds to resolve, so diff the emitted asm rather than re-benchmarking.
    #[test]
    fn handles_stay_thin() {
        let word = size_of::<*const u8>();

        // Both panels carry `k`: the price of each stating its own extent rather than
        // borrowing the other's, which is what lets the leaves be safe.
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

        // The slot's extent is in the type; the strip is the checked anchor it is
        // carved from, so it keeps a length.
        assert_eq!(size_of::<Block<'static, f32, A_PANEL, B_PANEL>>(), word);
        assert_eq!(size_of::<Strip<'static, f32, A_PANEL, B_PANEL>>(), 2 * word);
    }
}

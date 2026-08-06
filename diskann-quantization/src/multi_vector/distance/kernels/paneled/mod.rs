// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Paneled MaxSim: a [`TileWalk`] lends cache-sized views, each [`Paneled`] into
//! register-sized panels plus a typed tail. [`Accumulate`] folds one (A-panel,
//! B-panel) pair into an accumulator slot; [`Drain`] turns the finished accumulator
//! into output it owns. [`Scratch`] is the write-side mirror of [`Paneled`], so the
//! driver assumes no layout on either side.
//!
//! Panel widths are geometry, so they live with the leaves that impose them; the panel
//! and accumulator types carry them as const parameters (`R` = A rows, `N` = B rows) so
//! the driver stays width-agnostic.
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

/// Per-lifetime half of [`Scratch`] — same implied-bound trick as [`TileAt`].
pub(crate) trait ScratchAt<'a, B = &'a mut Self> {
    /// One B-panel's slot.
    type Block;
    /// Named so `Block` reaches bounds without a nested projection.
    type Slots: Slots<Block = Self::Block>;
}

/// Hands out one accumulator slot per B-panel, each disjoint from the last.
///
/// Infallible: a scratch is sized for the widest tile it will ever see, so "ran out" is
/// not a state the driver can reach, and the fill loop carries no `Option`. Overdrawing
/// is a planning bug; implementations must fail loudly rather than alias.
pub(crate) trait Slots {
    type Block;
    fn next(&mut self) -> Self::Block;
}

/// [`Paneled`]'s write side: a buffer that carves itself into per-B-panel slots, so
/// the driver can't tell a contiguous strip from a padded or structure-of-arrays one.
/// Not [`Paneled`] itself — `Panel: Copy` and `panels(&self)` can't yield disjoint
/// `&mut`. Allocation stays on the concrete type.
///
/// Slots are uniform: a short B-tail writes a prefix of a full-width slot, and the
/// [`Drain`] is told how much is live, so no second width has to be threaded through.
pub(crate) trait Scratch: for<'a> ScratchAt<'a> {
    fn slots(&mut self) -> <Self as ScratchAt<'_>>::Slots;
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
/// Each element carries its own [`Geo`], because the view is what knows how it is laid
/// out; the driver only forwards what it is handed.
///
/// `ExactSizeIterator` binds implementors, not the driver: a view that cannot state its
/// panel count exactly does not know its own geometry.
pub(crate) trait TailIterator: ExactSizeIterator {
    type Tail;
    fn tail(self) -> Option<(Self::Tail, Geo)>;
}

/// A view that knows how it breaks into panels. `Tail` is distinct from `Panel` so the
/// short trailing panel selects its own [`Accumulate`] impl; a view that cannot tail
/// says [`NoTail`].
pub(crate) trait Paneled {
    type Panel: Copy;
    type Tail: Copy;
    /// Named so it carries `ExactSizeIterator` and the tail type into bounds, and so a
    /// k-fracturing driver could hold one across an outer loop.
    type Panels: TailIterator<Item = (Self::Panel, Geo), Tail = Self::Tail>;

    /// Where this view sits in the global problem. A view is a sub-view of something,
    /// so it is the only thing that can say where it came from.
    fn geo(&self) -> Geo;

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

/// A contiguous run of **real** vectors, `[start, end)`, in the global problem's
/// numbering. Purely logical: a view that pads reports what is real, and a drain that
/// pads owns that itself — so no consumer has to know anyone else's padding rule.
///
/// This is the driver's whole vocabulary for position, so a walk must yield vectors
/// that are contiguous and monotone. A gathering or permuting walk cannot be described
/// this way; it needs an id source, which is a different trait rather than a wider
/// struct.
#[derive(Clone, Copy)]
pub(crate) struct Geo {
    pub(crate) start: usize,
    pub(crate) end: usize,
}

impl Geo {
    /// Real vectors covered — the reduction's live width.
    pub(crate) fn len(self) -> usize {
        self.end - self.start
    }

    /// For indexing a buffer held in the same numbering.
    pub(crate) fn range(self) -> Range<usize> {
        self.start..self.end
    }
}

/// Where a finished accumulator sits in the global problem. Both sides mean the same
/// thing — real vectors — whatever granularity the driver happens to drain them at.
#[derive(Clone, Copy)]
pub(crate) struct Region {
    pub(crate) a: Geo,
    pub(crate) b: Geo,
}

/// Consume a finished accumulator. The drain owns its output, so dequant, reduction
/// and scatter all live behind this one call and may be fused.
///
/// `region` also carries the live extent: the accumulator is sized for the widest tile,
/// so a drain that folded its capacity instead of `region.b.len()` would fold the
/// *previous* tile's values back in.
pub(crate) trait Drain<Arch, S: Scratch> {
    fn drain(&mut self, arch: Arch, acc: &S, region: Region);
}

// ── Driver ───────────────────────────────────────────────────────

type PanelOf<'a, W> = <<W as TileAt<'a>>::View as Paneled>::Panel;
type TailOf<'a, W> = <<W as TileAt<'a>>::View as Paneled>::Tail;
type BlockOf<'x, S> = <S as ScratchAt<'x>>::Block;

/// One A-panel against a whole B-tile. Factored out so the driver's A-panel and
/// A-tail arms share the tail-dispatch.
#[inline(always)]
fn fill<Arch, A, BV, S, K>(arch: Arch, kernel: &K, a: A, b_view: &BV, scratch: &mut S)
where
    Arch: Copy,
    A: Copy,
    BV: Paneled,
    S: Scratch,
    K: for<'x> Accumulate<Arch, A, BV::Panel, BlockOf<'x, S>>
        + for<'x> Accumulate<Arch, A, BV::Tail, BlockOf<'x, S>>,
{
    let mut panels = b_view.panels();
    let mut slots = scratch.slots();
    // The geos are the drain's business; this side only feeds the kernel.
    for (b, _) in panels.by_ref() {
        kernel.accumulate(arch, a, b, slots.next());
    }
    // The tail draws from the same cursor as the full panels.
    if let Some((b, _)) = panels.tail() {
        kernel.accumulate(arch, a, b, slots.next());
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
        + for<'a, 'b, 'x> Accumulate<Arch, PanelOf<'a, AW>, TailOf<'b, BW>, BlockOf<'x, S>>
        + for<'a, 'b, 'x> Accumulate<Arch, TailOf<'a, AW>, PanelOf<'b, BW>, BlockOf<'x, S>>
        + for<'a, 'b, 'x> Accumulate<Arch, TailOf<'a, AW>, TailOf<'b, BW>, BlockOf<'x, S>>,
    D: Drain<Arch, S>,
{
    while let Some(a_view) = a_walk.next() {
        b_walk.reset();
        while let Some(b_view) = b_walk.next() {
            let b = b_view.geo();
            let mut a_panels = a_view.panels();
            for (panel, a) in a_panels.by_ref() {
                fill(arch, kernel, panel, &b_view, scratch);
                drain.drain(arch, scratch, Region { a, b });
            }
            if let Some((panel, a)) = a_panels.tail() {
                fill(arch, kernel, panel, &b_view, scratch);
                drain.drain(arch, scratch, Region { a, b });
            }
        }
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
    /// A guard, not a curiosity: each side lost this once already to a refactor that
    /// swapped a pointer for a slice to simplify a cursor, and neither showed up in a
    /// test. The choice is a type-system one — an A/B of two builds cannot resolve a
    /// difference this small, so diff the emitted asm rather than re-benchmarking.
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
        // carved from, so only the strip keeps a length.
        assert_eq!(size_of::<Block<'static, f32, A_PANEL, B_PANEL>>(), word);
        assert_eq!(size_of::<Strip<'static, f32, A_PANEL, B_PANEL>>(), 2 * word);
    }
}

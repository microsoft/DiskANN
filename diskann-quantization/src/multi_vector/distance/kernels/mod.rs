/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Cache-tiled MaxSim: a [`TileWalk`] lends cache-sized tiles, each [`Paneled`] into the
//! panels one leaf call consumes, plus a typed tail. [`Accumulate`] folds one (A-panel,
//! B-panel) pair into a slot of the [`Scratch`]; [`Drain`] consumes the finished
//! accumulator.
//!
//! Position is ordinal. [`drive`] counts the panels it passes and hands a [`Drain`] an
//! A-panel index and a B-panel range, never a stride or an address. A drain scales those
//! ordinals by its own panel width and clamps the end against its own extent. A side
//! whose tail type is uninhabited ([`NoTail`]) need not clamp at all, and that is a
//! type-level fact, not a convention.
//!
//! Naming follows the layer: `A` and `B` are positions in the contraction, type names say
//! which layout a thing is, and only the entry points and the [`Drain`] speak of queries
//! and documents.

use core::num::NonZeroUsize;
use core::ops::Range;

mod f16;
mod float;
mod leaves;
mod strip;
mod tiles;

pub(crate) use f16::MaxIpF16;
pub(crate) use float::MaxIp;

//////////////////////////////
// Tile budget and planning //
//////////////////////////////

/// Cache budgets fed to [`Plan::new`].
#[derive(Debug, Clone, Copy)]
struct TileBudget {
    /// L2 bytes reserved for A tiles.
    l2_a: usize,
    /// L1 bytes reserved for B tiles, before the resident A-panel is subtracted.
    l1_b: usize,
}

impl Default for TileBudget {
    // TODO: Replace hardcoded fallbacks with detected cache sizes
    // (e.g. via `diskann_platform`, env-var override, or runtime query).
    fn default() -> Self {
        const L2_CACHE: usize = 1_250_000; // 1.25 MB fallback
        const L1_CACHE: usize = 48_000; // 48 KB fallback

        Self {
            // 50% of L2 for A tiles; remainder for B streaming + pollution.
            l2_a: L2_CACHE / 2,
            // 75% of L1 for B tiles; A micro-panel subtracted at runtime.
            l1_b: L1_CACHE * 3 / 4,
        }
    }
}

/// Panel counts per tile: `a_panels` A-panels resident in L2, and as many B-panels as
/// co-fit L1 alongside one A-panel *and* the accumulator columns they feed, never more
/// than the B-rows on hand.
#[derive(Debug, Clone, Copy)]
struct Plan<const AR: usize, const BR: usize> {
    a_panels: NonZeroUsize,
    b_panels: NonZeroUsize,
}

/// A panel count of at least one: a budget too small for even a single panel still has to
/// make progress, and a zero-wide tile would stall the walk it strides.
fn at_least_one_panel(panels: usize) -> NonZeroUsize {
    NonZeroUsize::new(panels).unwrap_or(NonZeroUsize::MIN)
}

impl<const AR: usize, const BR: usize> Plan<AR, BR> {
    /// `b_panels` is reconciled against `b_rows`, so a plan belongs to the B side it was
    /// built for. Reused against a longer one it still computes the right answer, but tiles
    /// far more narrowly than the cache allows. `a_panels` needs no such reconciliation: it only
    /// feeds a walk stride, which the cursor already bounds by the data it holds, whereas
    /// `b_panels` also sizes the accumulator strip.
    ///
    /// Row sizes are [`NonZeroUsize`] because they reach a divisor. The entries reject an
    /// empty contraction before planning.
    fn new(
        a_row_bytes: NonZeroUsize,
        b_row_bytes: NonZeroUsize,
        b_rows: usize,
        acc_bytes: usize,
        budget: TileBudget,
    ) -> Self {
        // Dividing by the row width and then the panel width, instead of by their product,
        // leaves both divisors provably non-zero without a check, and leaves no
        // intermediate that could overflow.
        let a_panels = at_least_one_panel(budget.l2_a / a_row_bytes / AR);

        // A B-row costs its own bytes plus the accumulator column it fills, and one
        // A-panel stays resident alongside.
        let per_b_row = b_row_bytes.saturating_add(AR * acc_bytes);
        let b_budget = budget.l1_b.saturating_sub(AR * a_row_bytes.get());
        let cache_fit = (b_budget / per_b_row) / BR;

        // Never plan wider than the B-rows on hand: a cache-sized tile over a short B side
        // would size accumulator columns no fill can reach.
        let b_panels = at_least_one_panel(cache_fit.min(b_rows.div_ceil(BR)));

        Self { a_panels, b_panels }
    }

    /// Accumulator elements for one A-panel against the widest B-tile this plan allows.
    ///
    /// Sizing the scratch below this trips [`slots_exhausted`].
    fn strip_len(&self) -> usize {
        AR * self.b_panels.get() * BR
    }
}

///////////////
// Read side //
///////////////

/// Per-lifetime half of [`TileWalk`].
///
/// The defaulted `B = &'a Self` carries the `Self: 'a` implied bound through
/// well-formedness. A plain GAT `where Self: 'a` collapses to `'static` under [`drive`]'s
/// `for<'a>` bound on stable.
trait TileAt<'a, B = &'a Self> {
    type Tile: Paneled;
}

/// A **lending** walk: `next` reborrows `&mut self`, so a tile may borrow a buffer the walk
/// reuses on the following call. That is what lets a walk convert as it goes. `reset`
/// rewinds, because B is re-walked once per A-tile.
trait TileWalk: for<'a> TileAt<'a> {
    fn next(&mut self) -> Option<<Self as TileAt<'_>>::Tile>;
    fn reset(&mut self);
}

/// An iterator whose short trailing element has its own type.
///
/// `tail` consumes the exhausted iterator. The trailer comes off the cursor the loop was
/// already advancing and is never recomputed from the source.
trait TailIterator: Iterator {
    type Tail;
    fn tail(self) -> Option<Self::Tail>;
}

/// A tile that knows how it breaks into panels.
///
/// Extent only, never position: a tile says how much it holds, and the driver decides where
/// that lands.
trait Paneled {
    type Panel: Copy;
    type Tail: Copy;
    type Panels: TailIterator<Item = Self::Panel, Tail = Self::Tail>;

    fn panels(&self) -> Self::Panels;
}

/// [`Paneled::Tail`] for a side padded to whole panels. Uninhabited, so `tail()` provably
/// returns `None` and no consumer on that side needs to clamp.
#[derive(Clone, Copy)]
enum NoTail {}

////////////////
// Write side //
////////////////

/// Per-lifetime half of [`Scratch`], using the same implied-bound trick as [`TileAt`].
trait SlotsAt<'s, B = &'s mut Self> {
    /// Named here instead of being reached through the iterator, so [`drive`]'s bounds can
    /// project it off the scratch.
    type Slot;
    /// Plain, not lending. Slots partition one buffer within a single call, so none of them
    /// borrows the cursor.
    type Slots: Iterator<Item = Self::Slot>;
}

/// The write-side mirror of [`Paneled`]: memory a fill carves into one slot per B-panel,
/// each disjoint from the last.
trait Scratch: for<'s> SlotsAt<'s> {
    fn slots(&mut self) -> <Self as SlotsAt<'_>>::Slots;
}

//////////////////
// Compute side //
//////////////////

/// One A-panel × one B-panel → one accumulator slot.
///
/// Pinned on all three as type parameters, which lets the walks' panel types select the
/// impl.
trait Accumulate<Arch, A, B, O> {
    fn accumulate(&self, arch: Arch, a: A, b: B, out: O);
}

/// [`NoTail`] is uninhabited, so this discharges [`drive`]'s A-tail bounds for every
/// kernel, and by coherence forbids any kernel from writing its own.
impl<Arch, B, O, K> Accumulate<Arch, NoTail, B, O> for K {
    #[inline(always)]
    fn accumulate(&self, _: Arch, a: NoTail, _: B, _: O) {
        match a {}
    }
}

/// Consume a finished accumulator.
///
/// `a_panel` and `b_panels` are ordinals in [`drive`]'s visit order, not addresses.
trait Drain<Arch, S> {
    fn drain(&mut self, arch: Arch, scratch: &S, a_panel: usize, b_panels: Range<usize>);
}

////////////
// Driver //
////////////

type PanelOf<'a, W> = <<W as TileAt<'a>>::Tile as Paneled>::Panel;
type TailOf<'a, W> = <<W as TileAt<'a>>::Tile as Paneled>::Tail;
type SlotOf<'s, S> = <S as SlotsAt<'s>>::Slot;

#[cold]
#[inline(never)]
fn slots_exhausted() -> ! {
    unreachable!("scratch ran out of slots: narrower than the walk's B-tile")
}

/// One A-panel against a whole B-tile.
///
/// Returns the slots it filled, which is ground truth for how far B advanced: a count of
/// whole panels would exclude the tail.
#[inline(always)]
fn fill<Arch, A, BT, S, K>(arch: Arch, kernel: &K, a: A, b_tile: &BT, scratch: &mut S) -> usize
where
    Arch: Copy,
    A: Copy,
    BT: Paneled,
    S: Scratch,
    K: for<'s> Accumulate<Arch, A, BT::Panel, SlotOf<'s, S>>
        + for<'s> Accumulate<Arch, A, BT::Tail, SlotOf<'s, S>>,
{
    let mut panels = b_tile.panels();
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

/// Drive one A source against one B source, re-walking B once per A-tile.
///
/// `scratch` precedes `kernel` deliberately: `S` must be resolved before the kernel's
/// [`Accumulate`] bounds are proved, or their slot type is still `<_ as SlotsAt>::Slot`
/// and no impl matches.
fn drive<Arch, AW, BW, K, S, D>(
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
    K: for<'a, 'b, 's> Accumulate<Arch, PanelOf<'a, AW>, PanelOf<'b, BW>, SlotOf<'s, S>>
        + for<'a, 'b, 's> Accumulate<Arch, PanelOf<'a, AW>, TailOf<'b, BW>, SlotOf<'s, S>>
        + for<'a, 'b, 's> Accumulate<Arch, TailOf<'a, AW>, PanelOf<'b, BW>, SlotOf<'s, S>>
        + for<'a, 'b, 's> Accumulate<Arch, TailOf<'a, AW>, TailOf<'b, BW>, SlotOf<'s, S>>,
    D: Drain<Arch, S>,
{
    let mut a_base = 0;
    while let Some(a_tile) = a_walk.next() {
        b_walk.reset();
        let mut b_base = 0;
        // Last pass wins: every B-tile re-sweeps the same A-panels, so both counters are
        // rewritten identically each pass and read after the last. An A-tile with no
        // B-tiles advances neither, which is unobservable. No drain fires, and a B source
        // empty for one A-tile is empty for all.
        let mut a_end = a_base;
        let mut b_used = 0;

        while let Some(b_tile) = b_walk.next() {
            let mut a_panel = a_base;
            let mut a_panels = a_tile.panels();

            for panel in a_panels.by_ref() {
                b_used = fill(arch, kernel, panel, &b_tile, scratch);
                drain.drain(arch, scratch, a_panel, b_base..b_base + b_used);
                a_panel += 1;
            }
            if let Some(panel) = a_panels.tail() {
                b_used = fill(arch, kernel, panel, &b_tile, scratch);
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
    use super::*;

    fn nz(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).unwrap()
    }

    #[test]
    fn plan_reserves_l1_for_the_resident_a_panel_and_accumulator() {
        // 64-byte rows, AR = 16, BR = 4, 4-byte accumulator.
        // l2_a 40960 / (64 * 16) = 40 A-panels.
        // l1_b 36000 - 16 * 64 = 34976 for B; per B-row = 64 + 16 * 4 = 128;
        // 34976 / 128 = 273 rows -> 273 / 4 = 68 B-panels.
        let plan = Plan::<16, 4>::new(
            nz(64),
            nz(64),
            usize::MAX,
            4,
            TileBudget {
                l2_a: 40960,
                l1_b: 36000,
            },
        );
        assert_eq!((plan.a_panels.get(), plan.b_panels.get()), (40, 68));
        assert_eq!(plan.strip_len(), 16 * 68 * 4);
    }

    #[test]
    fn plan_never_tiles_past_the_b_rows_on_hand() {
        // The cache would take 68 B-panels, but 10 B-rows fill three.
        let budget = TileBudget {
            l2_a: 40960,
            l1_b: 36000,
        };
        let plan = Plan::<16, 4>::new(nz(64), nz(64), 10, 4, budget);
        assert_eq!(plan.b_panels.get(), 3);
        assert_eq!(plan.strip_len(), 16 * 3 * 4);
    }

    #[test]
    fn plan_clamps_to_one_panel_per_tile() {
        let plan = Plan::<16, 4>::new(
            nz(1024),
            nz(1024),
            usize::MAX,
            4,
            TileBudget { l2_a: 1, l1_b: 1 },
        );
        assert_eq!((plan.a_panels.get(), plan.b_panels.get()), (1, 1));
        assert_eq!(plan.strip_len(), 16 * 4);
    }
}

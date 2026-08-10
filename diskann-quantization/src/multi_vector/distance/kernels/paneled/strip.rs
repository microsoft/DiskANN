// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! The default [`Scratch`]: one buffer, lent out once per (A-panel, B-tile) as a
//! [`Strip`] of `R` rows A-major carved into `N`-column [`Block`]s. MaxSim collapses
//! each tile into the running max before the next one starts, so every tile is the
//! same memory at a different position. Borrows its buffer, so the allocator stays at
//! the call site.

use core::marker::PhantomData;
use core::mem::MaybeUninit;

use super::{Plan, Scratch, ScratchAt, ScratchTile, Slots, SlotsAt};
use crate::alloc::{AllocatorCore, Poly};

/// Marker for element types where all-zero is a valid value.
pub(crate) trait ZeroInit: Copy {}
impl ZeroInit for i32 {}
impl ZeroInit for f32 {}

/// The accumulator buffer and a cursor over the driver's A-tile → B-tile → A-panel
/// order, paired on every [`Scratch::next`].
///
/// The order is load-bearing, not incidental: a B-tile stays resident while the
/// A-panels sweep past it, so A-panels are re-enumerated once per B-tile. That rewind
/// is why the cursor keeps the tile origin rather than a single counter.
pub(crate) struct StripScratch<'a, T, const R: usize, const N: usize> {
    buf: &'a mut [T],
    a_padded: usize,
    b_rows: usize,
    a_tile_rows: usize,
    b_tile_rows: usize,
    /// First row of the A-panel the next tile will cover.
    a: usize,
    /// First row of the A-tile being swept — where `a` rewinds to per B-tile.
    a_tile: usize,
    b: usize,
}

/// One tile of accumulator: the buffer, sized to the widest B-tile the plan allows,
/// plus where this use of it belongs.
///
/// Position is kept, never handed out. Every consumer wants a cut of a buffer held in
/// the tile's own numbering, so the tile makes the cut — which states the padding rule
/// once and leaves no loose coordinate to apply to the wrong side.
pub(crate) struct Strip<'a, T, const R: usize, const N: usize> {
    buf: &'a mut [T],
    a_row: usize,
    b_row: usize,
    b_live: usize,
}

/// One B-panel's slot: `R` rows by `N` columns, A-major.
///
/// Slots are packed, so column `c` of the strip sits at `c * R` whichever slot it falls
/// in — the live columns stay one contiguous run, which is what lets a
/// [`Drain`](super::Drain) fold a whole tile in one pass.
///
/// Thin: `R * N` is already in the type, and the `split_at_mut` that carves the slot is
/// where the length still means something, so the handle keeps only the pointer.
pub(crate) struct Block<'a, T, const R: usize, const N: usize> {
    ptr: *mut T,
    _lifetime: PhantomData<&'a mut [T]>,
}

/// Splits slots off the front of a [`Strip`], so disjointness is the borrow checker's
/// job rather than an invariant to uphold by hand.
pub(crate) struct BlockSlots<'a, T, const R: usize, const N: usize>(&'a mut [T]);

impl<'a, T, const R: usize, const N: usize> Slots for BlockSlots<'a, T, R, N> {
    type Block = Block<'a, T, R, N>;

    fn next(&mut self) -> Option<Block<'a, T, R, N>> {
        // `take` re-lends the buffer for `'a` rather than the `&mut self` borrow, which
        // is what lets a slot outlive the cursor call that produced it.
        let (slot, rest) = core::mem::take(&mut self.0).split_at_mut_checked(R * N)?;
        self.0 = rest;
        // The split is what proves the slots disjoint, so the pointer inherits that
        // guarantee rather than resting on an invariant upheld by hand.
        Some(Block {
            ptr: slot.as_mut_ptr(),
            _lifetime: PhantomData,
        })
    }
}

impl<'a, T, const R: usize, const N: usize> StripScratch<'a, T, R, N> {
    /// `a_padded` is the A extent including panel padding.
    fn new(buf: &'a mut [T], plan: Plan<R, N>, a_padded: usize, b_rows: usize) -> Self {
        Self {
            buf,
            a_padded,
            b_rows,
            a_tile_rows: plan.a_panels * R,
            b_tile_rows: plan.b_panels * N,
            a: 0,
            a_tile: 0,
            b: 0,
        }
    }

    /// `(a_row, b_row, b_live)` for the next tile, or `None` once the traversal is
    /// complete.
    fn advance(&mut self) -> Option<(usize, usize, usize)> {
        if self.a_padded == 0 || self.b_rows == 0 {
            return None;
        }
        // Padding is to whole panels, so `a` lands exactly on the tile end rather than
        // stepping past it, and the finished tile's end is the next tile's start.
        if self.a >= (self.a_tile + self.a_tile_rows).min(self.a_padded) {
            self.b += self.b_tile_rows;
            if self.b >= self.b_rows {
                self.b = 0;
                self.a_tile = self.a;
                if self.a_tile >= self.a_padded {
                    return None;
                }
            }
            self.a = self.a_tile;
        }
        // B is not padded, so the last tile is short; A always covers a whole panel.
        let live = self.b_tile_rows.min(self.b_rows - self.b);
        let at = (self.a, self.b, live);
        self.a += R;
        Some(at)
    }
}

impl<'a, T: ZeroInit, const R: usize, const N: usize> StripScratch<'a, T, R, N> {
    /// Zeroing is what makes the `&mut [T]` sound; the kernels overwrite every live
    /// column before it is read.
    pub(crate) fn from_uninit<A: AllocatorCore + std::fmt::Debug>(
        poly: &'a mut Poly<[MaybeUninit<T>], A>,
        len: usize,
        plan: Plan<R, N>,
        a_padded: usize,
        b_rows: usize,
    ) -> Self {
        let ptr = poly.as_mut_ptr().cast::<T>();
        // SAFETY: the poly owns `len` `T`-sized slots; `T: ZeroInit` ⇒ all-zero is a
        // valid `T`, so zeroing initializes every element and the slice is sound.
        let buf = unsafe {
            core::ptr::write_bytes(ptr, 0, len);
            core::slice::from_raw_parts_mut(ptr, len)
        };
        Self::new(buf, plan, a_padded, b_rows)
    }
}

impl<'a, T, const R: usize, const N: usize> ScratchAt<'a> for StripScratch<'_, T, R, N> {
    type Tile = Strip<'a, T, R, N>;
}

impl<T, const R: usize, const N: usize> Scratch for StripScratch<'_, T, R, N> {
    fn next(&mut self) -> Option<Strip<'_, T, R, N>> {
        let (a_row, b_row, b_live) = self.advance()?;
        Some(Strip {
            buf: &mut *self.buf,
            a_row,
            b_row,
            b_live,
        })
    }
}

impl<'s, T, const R: usize, const N: usize> SlotsAt<'s> for Strip<'_, T, R, N> {
    type Block = Block<'s, T, R, N>;
    type Slots = BlockSlots<'s, T, R, N>;
}

impl<T, const R: usize, const N: usize> ScratchTile for Strip<'_, T, R, N> {
    fn slots(&mut self) -> BlockSlots<'_, T, R, N> {
        BlockSlots(&mut *self.buf)
    }
}

impl<T, const R: usize, const N: usize> Block<'_, T, R, N> {
    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<T, const R: usize, const N: usize> Strip<'_, T, R, N> {
    /// This tile's `R` rows of `buf`, which must be indexed in A's **padded**
    /// numbering. A tiles are always whole panels, so the cut is fixed-width and never
    /// clamps — which is why the A-side metadata is allocated padded.
    ///
    /// # Panics
    ///
    /// If `buf` is not padded to whole panels.
    pub(crate) fn a_rows<'x, U>(&self, buf: &'x [U]) -> &'x [U] {
        &buf[self.a_row..][..R]
    }

    /// See [`a_rows`](Self::a_rows).
    pub(crate) fn a_rows_mut<'x, U>(&self, buf: &'x mut [U]) -> &'x mut [U] {
        &mut buf[self.a_row..][..R]
    }

    /// This tile's **real** rows of `buf`, which must be indexed in B's numbering. B is
    /// not padded, so the last tile is short.
    ///
    /// # Panics
    ///
    /// If `buf` is shorter than the B extent the scratch was built for.
    pub(crate) fn b_rows<'x, U>(&self, buf: &'x [U]) -> &'x [U] {
        &buf[self.b_row..][..self.b_live]
    }

    /// The accumulator's live prefix: `R` rows for each **real** B row. The rest is
    /// capacity left over from a wider tile, so a consumer handed the whole buffer
    /// would fold the *previous* tile's values back in.
    ///
    /// # Panics
    ///
    /// If the live extent outruns the strip — a planning bug, since the strip is sized
    /// to the widest B-tile the plan allows.
    pub(crate) fn columns(&self) -> &[T] {
        &self.buf[..self.b_live * R]
    }
}

#[cfg(test)]
mod tests {
    use super::{Plan, Scratch, StripScratch};
    use crate::multi_vector::distance::kernels::paneled::leaves::{A_PANEL, B_PANEL};

    /// Enumerate the positions the driver's loop nest visits, mirroring `block_range` /
    /// `row_range` clamping directly. Deliberately naive: this is the oracle.
    fn reference<const R: usize, const N: usize>(
        a_rows: usize,
        b_rows: usize,
        a_panels: usize,
        b_panels: usize,
    ) -> Vec<(usize, usize, usize)> {
        let nb = a_rows.div_ceil(R);
        let mut out = Vec::new();
        let mut blk = 0;
        while blk < nb {
            let take = a_panels.min(nb - blk);
            let mut b = 0;
            while b < b_rows {
                let rows = (b_panels * N).min(b_rows - b);
                for p in 0..take {
                    out.push(((blk + p) * R, b, rows));
                }
                b += rows;
            }
            blk += take;
        }
        out
    }

    fn walk<const R: usize, const N: usize>(
        a_rows: usize,
        b_rows: usize,
        a_panels: usize,
        b_panels: usize,
    ) -> Vec<(usize, usize, usize)> {
        let plan = Plan::<R, N> { a_panels, b_panels };
        // The cursor never touches the buffer, so an empty one exercises it in full.
        let mut scratch =
            StripScratch::<f32, R, N>::new(&mut [], plan, a_rows.div_ceil(R) * R, b_rows);
        let mut out = Vec::new();
        // Bounded so a cursor that fails to terminate fails the test instead of hanging.
        for _ in 0..10_000 {
            match scratch.next() {
                Some(s) => out.push((s.a_row, s.b_row, s.b_live)),
                None => return out,
            }
        }
        panic!("the cursor did not terminate");
    }

    #[test]
    fn cursor_replays_the_driver_order() {
        for a_rows in [0, 1, 15, 16, 17, 31, 32, 33, 64, 100] {
            for b_rows in [0, 1, 3, 4, 5, 7, 8, 12, 13, 40] {
                for a_panels in [1, 2, 3, 5] {
                    for b_panels in [1, 2, 3, 7] {
                        assert_eq!(
                            walk::<A_PANEL, B_PANEL>(a_rows, b_rows, a_panels, b_panels),
                            reference::<A_PANEL, B_PANEL>(a_rows, b_rows, a_panels, b_panels),
                            "a_rows={a_rows} b_rows={b_rows} a_panels={a_panels} b_panels={b_panels}"
                        );
                    }
                }
            }
        }
    }
}

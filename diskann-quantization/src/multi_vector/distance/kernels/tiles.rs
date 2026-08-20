/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The read ladder: walks lend tiles, tiles cut into panels.
//!
//! The two sides read different layouts. A carries a [`BlockTransposedRef`], whose blocks
//! already interleave `AR` rows, so a panel is one block and a leaf loads `AR` contiguous
//! rows per contraction step. B carries a [`Standard`] matrix, plain row-major, so a panel
//! is `BR` consecutive rows.
//!
//! Neither panel stores `k`: it is `len() / AR` on the block-transposed side, and the leaf
//! checks the row-major panel against it, so a mismatched pair is caught where it is
//! used, not assumed.

use core::num::NonZeroUsize;

use super::{NoTail, Paneled, TailIterator, TileAt, TileWalk};
use crate::bits::{Dynamic, Length, Static};
use crate::multi_vector::{BlockTransposedRef, MatRef, Standard};

///////////////////////////
// Block-transposed side //
///////////////////////////

/// A single block of a [`BlockTransposedRef`]: `AR` rows × `k` columns, column-major
/// within the block.
#[derive(Clone, Copy)]
pub(super) struct BlockTransposedPanel<'a, T, const AR: usize>(&'a [T]);

impl<'a, T, const AR: usize> BlockTransposedPanel<'a, T, AR> {
    pub(super) fn as_slice(&self) -> &'a [T] {
        self.0
    }

    /// The contraction length.
    pub(super) fn k(&self) -> usize {
        self.0.len() / AR
    }
}

/// The panel iterator of a [`BlockTransposedTile`].
pub(super) struct BlockTransposedPanels<'a, T, const AR: usize> {
    rest: &'a [T],
    stride: usize,
}

impl<'a, T, const AR: usize> Iterator for BlockTransposedPanels<'a, T, AR> {
    type Item = BlockTransposedPanel<'a, T, AR>;

    fn next(&mut self) -> Option<Self::Item> {
        let (panel, rest) = self.rest.split_at_checked(self.stride)?;
        self.rest = rest;
        Some(BlockTransposedPanel(panel))
    }
}

impl<T, const AR: usize> TailIterator for BlockTransposedPanels<'_, T, AR> {
    type Tail = NoTail;

    fn tail(self) -> Option<NoTail> {
        None
    }
}

/// A view over consecutive blocks of a [`BlockTransposedRef<T, AR>`](BlockTransposedRef).
///
/// Its [`Paneled`] impl yields one [`BlockTransposedPanel`] per block.
pub(super) struct BlockTransposedTile<'a, T, const AR: usize> {
    data: &'a [T],
    k: NonZeroUsize,
}

impl<'a, T, const AR: usize> BlockTransposedTile<'a, T, AR> {
    /// # Panics
    ///
    /// Debug-only: panics unless `data` is a whole number of `AR × k` blocks, which is
    /// what makes [`NoTail`] honest.
    pub(super) fn new(data: &'a [T], k: NonZeroUsize) -> Self {
        debug_assert!(data.len().is_multiple_of(AR * k.get()));
        Self { data, k }
    }
}

impl<'a, T: Copy, const AR: usize> Paneled for BlockTransposedTile<'a, T, AR> {
    type Panel = BlockTransposedPanel<'a, T, AR>;
    /// Block-transposed storage is padded out to whole `AR`-row blocks, so a run of blocks
    /// cannot end in a partial one.
    type Tail = NoTail;
    type Panels = BlockTransposedPanels<'a, T, AR>;

    fn panels(&self) -> Self::Panels {
        BlockTransposedPanels {
            rest: self.data,
            stride: AR * self.k.get(),
        }
    }
}

////////////////////
// Row-major side //
////////////////////

/// Up to `BR` consecutive rows of a [`Standard`] matrix, `k` elements each.
///
/// `L` is [`Static<BR>`] for a whole panel and [`Dynamic`] for the trailing one, which is
/// how the tail reaches a leaf that can unroll for its width.
#[derive(Clone, Copy)]
pub(super) struct RowMajorPanel<'a, T, const BR: usize, L> {
    data: &'a [T],
    rows: L,
}

impl<'a, T, const BR: usize, L: Length> RowMajorPanel<'a, T, BR, L> {
    pub(super) fn as_slice(&self) -> &'a [T] {
        self.data
    }

    pub(super) fn rows(&self) -> usize {
        self.rows.value()
    }
}

/// The panel iterator of a [`RowMajorTile`].
pub(super) struct RowMajorPanels<'a, T, const BR: usize> {
    rest: &'a [T],
    /// The contraction length, not a panel stride. Only the tail needs a divisor, and what
    /// it divides by is this, not `BR * k`.
    k: NonZeroUsize,
}

impl<'a, T, const BR: usize> Iterator for RowMajorPanels<'a, T, BR> {
    type Item = RowMajorPanel<'a, T, BR, Static<BR>>;

    fn next(&mut self) -> Option<Self::Item> {
        let (data, rest) = self.rest.split_at_checked(BR * self.k.get())?;
        self.rest = rest;
        Some(RowMajorPanel {
            data,
            rows: Static::<BR>,
        })
    }
}

impl<'a, T, const BR: usize> TailIterator for RowMajorPanels<'a, T, BR> {
    type Tail = RowMajorPanel<'a, T, BR, Dynamic>;

    fn tail(self) -> Option<Self::Tail> {
        debug_assert!(
            self.rest.len() < BR * self.k.get(),
            "tail taken before the iterator was exhausted"
        );
        (!self.rest.is_empty()).then(|| RowMajorPanel {
            data: self.rest,
            rows: Dynamic(self.rest.len() / self.k.get()),
        })
    }
}

/// A view over consecutive rows of a [`Standard`] matrix.
///
/// Its [`Paneled`] impl yields one [`RowMajorPanel`] per `BR` rows, plus a short tail.
pub(super) struct RowMajorTile<'a, T, const BR: usize> {
    data: &'a [T],
    k: NonZeroUsize,
}

impl<'a, T, const BR: usize> RowMajorTile<'a, T, BR> {
    /// # Panics
    ///
    /// Debug-only: panics unless `data` is a whole number of `k`-element rows.
    pub(super) fn new(data: &'a [T], k: NonZeroUsize) -> Self {
        debug_assert!(data.len().is_multiple_of(k.get()));
        Self { data, k }
    }
}

impl<'a, T: Copy, const BR: usize> Paneled for RowMajorTile<'a, T, BR> {
    type Panel = RowMajorPanel<'a, T, BR, Static<BR>>;
    /// A row count need not divide `BR`, so a tile can end in a partial panel.
    type Tail = RowMajorPanel<'a, T, BR, Dynamic>;
    type Panels = RowMajorPanels<'a, T, BR>;

    fn panels(&self) -> Self::Panels {
        RowMajorPanels {
            rest: self.data,
            k: self.k,
        }
    }
}

///////////
// Walks //
///////////

/// The contraction length as a [`NonZeroUsize`], which walks stride by and panels divide
/// by. Carrying the invariant in the type keeps those sites free of a zero check.
///
/// # Panics
///
/// Panics if `k` is zero. The entry guards that case before any walk is built. This is
/// the backstop for constructors reachable without it.
#[track_caller]
#[expect(
    clippy::expect_used,
    reason = "a walk cannot represent an empty contraction"
)]
pub(super) fn contraction(k: usize) -> NonZeroUsize {
    NonZeroUsize::new(k).expect("walk requires a non-empty contraction")
}

/// Elements spanned by `panels` panels of `width` rows, each row `k` long.
///
/// Overflowing the product needs a contraction near `usize::MAX` and no entry admits one,
/// so a wrapped result is kept non-zero instead of being detected. A stride of zero would
/// leave [`Cursor`] handing out empty tiles without ever advancing.
pub(super) fn tile_stride(panels: NonZeroUsize, width: usize, k: NonZeroUsize) -> NonZeroUsize {
    NonZeroUsize::new(panels.get() * width * k.get()).unwrap_or(NonZeroUsize::MIN)
}

/// A cursor over a contiguous source, cut into tiles of `stride` elements.
///
/// Shared by both sides, and by the widening walks, because the only difference is how a
/// tile is interpreted.
pub(super) struct Cursor<'a, T> {
    data: &'a [T],
    stride: NonZeroUsize,
    cur: usize,
}

impl<'a, T> Cursor<'a, T> {
    pub(super) fn new(data: &'a [T], stride: NonZeroUsize) -> Self {
        Self {
            data,
            stride,
            cur: 0,
        }
    }

    /// The next tile, clamped to what remains. The last one may be short.
    pub(super) fn next(&mut self) -> Option<&'a [T]> {
        let rest = self.data.get(self.cur..)?;
        if rest.is_empty() {
            return None;
        }
        let take = self.stride.get().min(rest.len());
        self.cur += take;
        Some(&rest[..take])
    }

    pub(super) fn reset(&mut self) {
        self.cur = 0;
    }

    /// The longest tile this cursor can yield, which is the size a converting walk must stage.
    pub(super) fn widest(&self) -> usize {
        self.stride.get().min(self.data.len())
    }
}

/// Walks the padded storage of a [`BlockTransposedRef`], `a_panels` blocks at a time.
pub(super) struct BlockTransposedWalk<'a, T, const AR: usize> {
    cursor: Cursor<'a, T>,
    k: NonZeroUsize,
}

impl<'a, T: Copy, const AR: usize> BlockTransposedWalk<'a, T, AR> {
    /// # Panics
    ///
    /// Panics if `view` has no columns.
    pub(super) fn new(view: BlockTransposedRef<'a, T, AR>, a_panels: NonZeroUsize) -> Self {
        let k = contraction(view.padded_ncols());
        Self {
            cursor: Cursor::new(view.as_slice(), tile_stride(a_panels, AR, k)),
            k,
        }
    }
}

impl<'t, T: Copy, const AR: usize> TileAt<'t> for BlockTransposedWalk<'_, T, AR> {
    type Tile = BlockTransposedTile<'t, T, AR>;
}

impl<T: Copy, const AR: usize> TileWalk for BlockTransposedWalk<'_, T, AR> {
    fn next(&mut self) -> Option<BlockTransposedTile<'_, T, AR>> {
        let k = self.k;
        self.cursor
            .next()
            .map(|data| BlockTransposedTile::new(data, k))
    }

    fn reset(&mut self) {
        self.cursor.reset();
    }
}

/// Walks a [`Standard`] matrix, `b_panels * BR` rows at a time.
pub(super) struct RowMajorWalk<'a, T, const BR: usize> {
    cursor: Cursor<'a, T>,
    k: NonZeroUsize,
}

impl<'a, T, const BR: usize> RowMajorWalk<'a, T, BR> {
    /// # Panics
    ///
    /// Panics if `mat` has zero-length rows.
    pub(super) fn new(mat: MatRef<'a, Standard<T>>, b_panels: NonZeroUsize) -> Self {
        let k = contraction(mat.vector_dim());
        Self {
            cursor: Cursor::new(mat.as_slice(), tile_stride(b_panels, BR, k)),
            k,
        }
    }
}

impl<'t, T: Copy, const BR: usize> TileAt<'t> for RowMajorWalk<'_, T, BR> {
    type Tile = RowMajorTile<'t, T, BR>;
}

impl<T: Copy, const BR: usize> TileWalk for RowMajorWalk<'_, T, BR> {
    fn next(&mut self) -> Option<RowMajorTile<'_, T, BR>> {
        let k = self.k;
        self.cursor.next().map(|data| RowMajorTile::new(data, k))
    }

    fn reset(&mut self) {
        self.cursor.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Handles are passed by value into every leaf call. Keeping them register-sized is why
    /// `k` is derived instead of stored.
    #[test]
    fn handles_stay_thin() {
        use core::mem::size_of;
        assert_eq!(
            size_of::<BlockTransposedPanel<'_, f32, 16>>(),
            2 * size_of::<usize>()
        );
        assert_eq!(
            size_of::<RowMajorPanel<'_, f32, 4, Static<4>>>(),
            2 * size_of::<usize>()
        );
        assert_eq!(
            size_of::<RowMajorPanel<'_, f32, 4, Dynamic>>(),
            3 * size_of::<usize>()
        );
    }

    #[test]
    fn row_major_tile_splits_into_whole_panels_plus_a_dynamic_tail() {
        let data: Vec<u8> = (0..7 * 3).map(|i| i as u8).collect();
        let tile = RowMajorTile::<u8, 2>::new(&data, NonZeroUsize::new(3).unwrap());

        let mut panels = tile.panels();
        assert_eq!(panels.next().unwrap().as_slice(), &[0, 1, 2, 3, 4, 5]);
        let rest: Vec<_> = panels.by_ref().map(|p| p.rows()).collect();
        assert_eq!(rest, [2, 2]);

        let tail = panels.tail().unwrap();
        assert_eq!(tail.rows(), 1);
        assert_eq!(tail.as_slice(), &[18, 19, 20]);
    }

    #[test]
    fn a_tile_that_divides_evenly_has_no_tail() {
        let data: Vec<u8> = (0..6 * 3).map(|i| i as u8).collect();
        let mut panels = RowMajorTile::<u8, 2>::new(&data, NonZeroUsize::new(3).unwrap()).panels();
        assert_eq!(panels.by_ref().count(), 3);
        assert!(panels.tail().is_none());
    }

    #[test]
    fn a_walk_yields_short_final_tile_then_stops_and_rewinds() {
        let data: Vec<u8> = (0..5 * 3).map(|i| i as u8).collect();
        let mat = MatRef::new(Standard::new(5, 3).unwrap(), &data).unwrap();
        let mut walk = RowMajorWalk::<u8, 2>::new(mat, NonZeroUsize::new(1).unwrap());

        assert_eq!(walk.next().unwrap().data.len(), 6);
        assert_eq!(walk.next().unwrap().data.len(), 6);
        assert_eq!(walk.next().unwrap().data.len(), 3);
        assert!(walk.next().is_none());

        walk.reset();
        assert_eq!(walk.next().unwrap().data.len(), 6);
    }
}

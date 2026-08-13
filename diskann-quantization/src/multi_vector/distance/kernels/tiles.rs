// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! The read ladder: walks lend tiles, tiles cut into panels.
//!
//! A query panel is one block-transposed block — `AR` rows interleaved column-major, so
//! the leaf loads `AR` contiguous rows per contraction step. A doc panel is `BR` row-major
//! rows. Neither carries `k`: it is `len() / AR` on the query side, and the leaf checks the
//! doc panel against it, so a mismatched pair is caught where it is used rather than
//! assumed.

use core::slice::ChunksExact;

use super::{NoTail, Paneled, TailIterator, TileAt, TileWalk};
use crate::bits::{Dynamic, Length, Static};
use crate::multi_vector::{BlockTransposedRef, MatRef, Standard};

// ── Query side ───────────────────────────────────────────────────

/// One block-transposed block: `AR` rows × `k` columns, column-major within the block.
#[derive(Clone, Copy)]
pub(super) struct QueryPanel<'a, T, const AR: usize>(&'a [T]);

impl<'a, T, const AR: usize> QueryPanel<'a, T, AR> {
    pub(super) fn as_slice(&self) -> &'a [T] {
        self.0
    }

    /// The contraction length.
    pub(super) fn k(&self) -> usize {
        self.0.len() / AR
    }
}

pub(super) struct QueryPanels<'a, T, const AR: usize>(ChunksExact<'a, T>);

impl<'a, T, const AR: usize> Iterator for QueryPanels<'a, T, AR> {
    type Item = QueryPanel<'a, T, AR>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(QueryPanel)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<T, const AR: usize> ExactSizeIterator for QueryPanels<'_, T, AR> {}

impl<T, const AR: usize> TailIterator for QueryPanels<'_, T, AR> {
    type Tail = NoTail;

    fn tail(self) -> Option<NoTail> {
        None
    }
}

/// A run of whole blocks — block-transposed storage is padded to `AR`, hence [`NoTail`].
pub(super) struct QueryTile<'a, T, const AR: usize> {
    data: &'a [T],
    k: usize,
}

impl<'a, T, const AR: usize> QueryTile<'a, T, AR> {
    /// # Panics
    ///
    /// Debug-only: panics unless `data` is a whole number of `AR × k` blocks, which is
    /// what makes [`NoTail`] honest.
    pub(super) fn new(data: &'a [T], k: usize) -> Self {
        debug_assert!(k > 0 && data.len().is_multiple_of(AR * k));
        Self { data, k }
    }
}

impl<'a, T: Copy, const AR: usize> Paneled for QueryTile<'a, T, AR> {
    type Panel = QueryPanel<'a, T, AR>;
    type Tail = NoTail;
    type Panels = QueryPanels<'a, T, AR>;

    fn panels(&self) -> Self::Panels {
        QueryPanels(self.data.chunks_exact(AR * self.k))
    }
}

// ── Doc side ─────────────────────────────────────────────────────

/// Up to `BR` row-major rows of `k` elements.
///
/// `L` is [`Static<BR>`] for a whole panel and [`Dynamic`] for the trailing one, which is
/// how the tail reaches a leaf that can unroll for its width.
#[derive(Clone, Copy)]
pub(super) struct DocPanel<'a, T, const BR: usize, L> {
    data: &'a [T],
    rows: L,
}

impl<'a, T, const BR: usize, L: Length> DocPanel<'a, T, BR, L> {
    pub(super) fn as_slice(&self) -> &'a [T] {
        self.data
    }

    pub(super) fn rows(&self) -> usize {
        self.rows.value()
    }
}

pub(super) struct DocPanels<'a, T, const BR: usize> {
    chunks: ChunksExact<'a, T>,
    k: usize,
}

impl<'a, T, const BR: usize> Iterator for DocPanels<'a, T, BR> {
    type Item = DocPanel<'a, T, BR, Static<BR>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.chunks.next().map(|data| DocPanel {
            data,
            rows: Static::<BR>,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.chunks.size_hint()
    }
}

impl<T, const BR: usize> ExactSizeIterator for DocPanels<'_, T, BR> {}

impl<'a, T, const BR: usize> TailIterator for DocPanels<'a, T, BR> {
    type Tail = DocPanel<'a, T, BR, Dynamic>;

    fn tail(self) -> Option<Self::Tail> {
        let data = self.chunks.remainder();
        (!data.is_empty()).then(|| DocPanel {
            data,
            rows: Dynamic(data.len() / self.k),
        })
    }
}

/// Row-major rows; only the final tile of a walk may be short.
pub(super) struct DocTile<'a, T, const BR: usize> {
    data: &'a [T],
    k: usize,
}

impl<'a, T, const BR: usize> DocTile<'a, T, BR> {
    /// # Panics
    ///
    /// Debug-only: panics unless `data` is a whole number of `k`-element rows.
    pub(super) fn new(data: &'a [T], k: usize) -> Self {
        debug_assert!(k > 0 && data.len().is_multiple_of(k));
        Self { data, k }
    }
}

impl<'a, T: Copy, const BR: usize> Paneled for DocTile<'a, T, BR> {
    type Panel = DocPanel<'a, T, BR, Static<BR>>;
    type Tail = DocPanel<'a, T, BR, Dynamic>;
    type Panels = DocPanels<'a, T, BR>;

    fn panels(&self) -> Self::Panels {
        DocPanels {
            chunks: self.data.chunks_exact(BR * self.k),
            k: self.k,
        }
    }
}

// ── Walks ────────────────────────────────────────────────────────

/// A cursor over a contiguous source, cut into tiles of `stride` elements.
///
/// Shared by both sides — and by the widening walks — because the only difference is how
/// a tile is interpreted.
pub(super) struct Cursor<'a, T> {
    data: &'a [T],
    stride: usize,
    cur: usize,
}

impl<'a, T> Cursor<'a, T> {
    pub(super) fn new(data: &'a [T], stride: usize) -> Self {
        debug_assert!(stride > 0);
        Self {
            data,
            stride,
            cur: 0,
        }
    }

    pub(super) fn next(&mut self) -> Option<&'a [T]> {
        let rest = self.data.get(self.cur..)?;
        if rest.is_empty() {
            return None;
        }
        let take = self.stride.min(rest.len());
        self.cur += take;
        Some(&rest[..take])
    }

    pub(super) fn reset(&mut self) {
        self.cur = 0;
    }

    /// The longest tile this cursor can yield — the size a converting walk must stage.
    pub(super) fn widest(&self) -> usize {
        self.stride.min(self.data.len())
    }
}

/// Walks the padded storage of a block-transposed query, `a_panels` blocks at a time.
pub(super) struct QueryWalk<'a, T, const AR: usize> {
    cursor: Cursor<'a, T>,
    k: usize,
}

impl<'a, T: Copy, const AR: usize> QueryWalk<'a, T, AR> {
    /// # Panics
    ///
    /// Panics if `view` has no columns.
    pub(super) fn new(view: BlockTransposedRef<'a, T, AR>, a_panels: usize) -> Self {
        let k = view.padded_ncols();
        assert!(k > 0, "QueryWalk requires a non-empty contraction");
        Self {
            cursor: Cursor::new(view.as_slice(), a_panels * AR * k),
            k,
        }
    }
}

impl<'t, T: Copy, const AR: usize> TileAt<'t> for QueryWalk<'_, T, AR> {
    type Tile = QueryTile<'t, T, AR>;
}

impl<T: Copy, const AR: usize> TileWalk for QueryWalk<'_, T, AR> {
    fn next(&mut self) -> Option<QueryTile<'_, T, AR>> {
        let k = self.k;
        self.cursor.next().map(|data| QueryTile::new(data, k))
    }

    fn reset(&mut self) {
        self.cursor.reset();
    }
}

/// Walks a row-major doc matrix, `b_panels * BR` rows at a time.
pub(super) struct DocWalk<'a, T, const BR: usize> {
    cursor: Cursor<'a, T>,
    k: usize,
}

impl<'a, T, const BR: usize> DocWalk<'a, T, BR> {
    /// # Panics
    ///
    /// Panics if `docs` has zero-length rows.
    pub(super) fn new(docs: MatRef<'a, Standard<T>>, b_panels: usize) -> Self {
        let k = docs.vector_dim();
        assert!(k > 0, "DocWalk requires a non-empty contraction");
        Self {
            cursor: Cursor::new(docs.as_slice(), b_panels * BR * k),
            k,
        }
    }
}

impl<'t, T: Copy, const BR: usize> TileAt<'t> for DocWalk<'_, T, BR> {
    type Tile = DocTile<'t, T, BR>;
}

impl<T: Copy, const BR: usize> TileWalk for DocWalk<'_, T, BR> {
    fn next(&mut self) -> Option<DocTile<'_, T, BR>> {
        let k = self.k;
        self.cursor.next().map(|data| DocTile::new(data, k))
    }

    fn reset(&mut self) {
        self.cursor.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Handles are passed by value into every leaf call; keeping them register-sized is
    /// the reason `k` is derived rather than stored.
    #[test]
    fn handles_stay_thin() {
        use core::mem::size_of;
        assert_eq!(size_of::<QueryPanel<'_, f32, 16>>(), 2 * size_of::<usize>());
        assert_eq!(
            size_of::<DocPanel<'_, f32, 4, Static<4>>>(),
            2 * size_of::<usize>()
        );
        assert_eq!(
            size_of::<DocPanel<'_, f32, 4, Dynamic>>(),
            3 * size_of::<usize>()
        );
    }

    #[test]
    fn doc_tile_splits_into_whole_panels_plus_a_dynamic_tail() {
        let data: Vec<u8> = (0..7 * 3).map(|i| i as u8).collect();
        let tile = DocTile::<u8, 2>::new(&data, 3);

        let mut panels = tile.panels();
        assert_eq!(panels.len(), 3);
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
        let mut panels = DocTile::<u8, 2>::new(&data, 3).panels();
        assert_eq!(panels.by_ref().count(), 3);
        assert!(panels.tail().is_none());
    }

    #[test]
    fn a_walk_yields_short_final_tile_then_stops_and_rewinds() {
        let data: Vec<u8> = (0..5 * 3).map(|i| i as u8).collect();
        let docs = MatRef::new(Standard::new(5, 3).unwrap(), &data).unwrap();
        let mut walk = DocWalk::<u8, 2>::new(docs, 1);

        assert_eq!(walk.next().unwrap().data.len(), 6);
        assert_eq!(walk.next().unwrap().data.len(), 6);
        assert_eq!(walk.next().unwrap().data.len(), 3);
        assert!(walk.next().is_none());

        walk.reset();
        assert_eq!(walk.next().unwrap().data.len(), 6);
    }
}

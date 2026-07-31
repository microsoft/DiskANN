// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! The two views and the walks that lend them. Both views are the real matrix types —
//! [`BlockTransposedRef`] for A, [`MatRef`] for B behind a [`Rows`] adapter (a
//! row-major matrix doesn't imply a panel height). Each sub-views itself, so a walk is
//! a cursor and nothing else.

use super::{NoTail, Paneled, Tile, TileAt, TileWalk};
use crate::multi_vector::{BlockTransposedRef, MatRef, Standard};

/// Rows per A-panel (the block-transposed group and the drain's state block).
pub(crate) const A_PANEL: usize = 16;
/// Rows per full B-panel (the kernel's micro-panel / max unroll).
pub(crate) const B_PANEL: usize = 4;

// ── Panels ───────────────────────────────────────────────────────

/// One block-transposed A block: `R` rows × `k` `T`.
pub(crate) struct QPanel<'a, T, const R: usize> {
    data: &'a [T],
    k: usize,
}
/// One row-major B panel: exactly `N` rows × `k` `T`. `k` travels with the A panel,
/// and the row count is `N` by construction, so neither is stored.
pub(crate) struct DPanel<'a, T, const N: usize> {
    data: &'a [T],
}
/// The short trailing B panel: `rows` in `1..N`. Distinct from [`DPanel`] so it
/// selects its own [`Accumulate`](super::Accumulate) impl, and so a full panel carries
/// no runtime row count.
pub(crate) struct DTail<'a, T> {
    data: &'a [T],
    rows: usize,
}

impl<T, const R: usize> Clone for QPanel<'_, T, R> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T, const R: usize> Copy for QPanel<'_, T, R> {}
impl<T, const N: usize> Clone for DPanel<'_, T, N> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T, const N: usize> Copy for DPanel<'_, T, N> {}
impl<T> Clone for DTail<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for DTail<'_, T> {}

impl<T, const R: usize> QPanel<'_, T, R> {
    pub(crate) fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    pub(crate) fn k(&self) -> usize {
        self.k
    }
}
impl<T, const N: usize> DPanel<'_, T, N> {
    pub(crate) fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
}
impl<T> DTail<'_, T> {
    pub(crate) fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }
}

// ── Views ────────────────────────────────────────────────────────

/// A block-transposed matrix' blocks *are* the A-panels, so the real type is the view.
/// The remainder block is zero-padded to a full `R` rows, hence [`NoTail`]; rows past
/// `nrows()` score against padding and are dropped by the caller. `P` only widens
/// `padded_ncols`, the contraction length the kernel sees.
impl<'a, T: Copy, const R: usize, const P: usize> Paneled for BlockTransposedRef<'a, T, R, P> {
    type Panel = QPanel<'a, T, R>;
    type Tail = NoTail;

    fn rows(&self) -> usize {
        self.nrows()
    }
    fn panels(&self) -> impl Iterator<Item = QPanel<'a, T, R>> + '_ {
        let (v, k) = (*self, self.padded_ncols());
        (0..self.num_blocks()).filter_map(move |b| {
            Some(QPanel {
                data: v.block_slice(b)?,
                k,
            })
        })
    }
    fn tail(&self) -> Option<NoTail> {
        None
    }
}

/// Cut a row-major matrix into `N`-row panels. All the geometry stays on the matrix.
pub(crate) struct Rows<const N: usize, V>(pub(crate) V);

impl<'a, const N: usize, T: Copy> Paneled for Rows<N, MatRef<'a, Standard<T>>> {
    type Panel = DPanel<'a, T, N>;
    type Tail = DTail<'a, T>;

    fn rows(&self) -> usize {
        self.0.num_vectors()
    }
    fn panels(&self) -> impl Iterator<Item = DPanel<'a, T, N>> + '_ {
        let (data, k) = (self.0.as_slice(), self.0.vector_dim());
        (0..self.rows() / N).map(move |p| DPanel {
            data: &data[p * N * k..(p + 1) * N * k],
        })
    }
    fn tail(&self) -> Option<DTail<'a, T>> {
        let (data, k, n) = (self.0.as_slice(), self.0.vector_dim(), self.rows());
        let rem = n % N;
        (rem > 0).then(|| DTail {
            data: &data[(n - rem) * k..],
            rows: rem,
        })
    }
}

// ── Walks ────────────────────────────────────────────────────────

/// A cursor over A-blocks, lending `tile_panels` of them at a time.
pub(crate) struct QueryWalk<'s, T: Copy, const R: usize, const P: usize = 1> {
    src: BlockTransposedRef<'s, T, R, P>,
    tile_panels: usize,
    cur: usize,
}

/// A cursor over B-rows, lending `tile_panels` `N`-row panels' worth at a time.
pub(crate) struct DocWalk<'s, T: Copy, const N: usize> {
    src: MatRef<'s, Standard<T>>,
    tile_panels: usize,
    cur: usize,
}

impl<'s, T: Copy, const R: usize, const P: usize> QueryWalk<'s, T, R, P> {
    pub(crate) fn new(src: BlockTransposedRef<'s, T, R, P>, tile_panels: usize) -> Self {
        Self {
            src,
            tile_panels,
            cur: 0,
        }
    }
}
impl<'s, T: Copy, const N: usize> DocWalk<'s, T, N> {
    pub(crate) fn new(src: MatRef<'s, Standard<T>>, tile_panels: usize) -> Self {
        Self {
            src,
            tile_panels,
            cur: 0,
        }
    }
}

impl<'a, T: Copy, const R: usize, const P: usize> TileAt<'a> for QueryWalk<'_, T, R, P> {
    type View = BlockTransposedRef<'a, T, R, P>;
}
impl<T: Copy, const R: usize, const P: usize> TileWalk for QueryWalk<'_, T, R, P> {
    fn next(&mut self) -> Option<Tile<BlockTransposedRef<'_, T, R, P>>> {
        let view = self.src.block_range(self.cur, self.tile_panels)?;
        let at = self.cur;
        self.cur += view.num_blocks();
        Some(Tile { view, at })
    }
    fn reset(&mut self) {
        self.cur = 0;
    }
}

impl<'a, T: Copy, const N: usize> TileAt<'a> for DocWalk<'_, T, N> {
    type View = Rows<N, MatRef<'a, Standard<T>>>;
}
impl<T: Copy, const N: usize> TileWalk for DocWalk<'_, T, N> {
    fn next(&mut self) -> Option<Tile<Rows<N, MatRef<'_, Standard<T>>>>> {
        let view = self.src.row_range(self.cur, self.tile_panels * N)?;
        let at = self.cur;
        self.cur += view.num_vectors();
        Some(Tile {
            view: Rows(view),
            at,
        })
    }
    fn reset(&mut self) {
        self.cur = 0;
    }
}

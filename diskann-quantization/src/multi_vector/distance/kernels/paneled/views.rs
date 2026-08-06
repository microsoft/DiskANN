// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! The two views and the walks that lend them. The views are the real matrix types
//! rather than wrappers around them — the B side needs one adapter, [`RowPanels`], only
//! because a row-major matrix doesn't imply a panel height. Each sub-views itself, so a
//! walk is a cursor and nothing else.

use core::marker::PhantomData;

use super::{Geo, NoTail, Paneled, TailIterator, TileAt, TileWalk};
use crate::bits::{Dynamic, Length, Static};
use crate::multi_vector::{BlockTransposedRef, MatRef, Standard};

// ── Panels ───────────────────────────────────────────────────────

/// One block-transposed A block: `R` rows × `k` `T`.
pub(crate) struct QPanel<'a, T, const R: usize> {
    ptr: *const T,
    k: usize,
    _lifetime: PhantomData<&'a [T]>,
}

/// One row-major B panel: `L` rows × `k` `T`, where `L.value() <= N`. A full panel is
/// `Static<N>` — a ZST; only the trailing panel is `Dynamic`, in `1..N`.
///
/// `k` is carried rather than borrowed from the A panel it pairs with: a handle whose
/// validity rests on a field in some *other* value cannot be checked where it is used,
/// which is what would force the leaves to be `unsafe`.
pub(crate) struct DPanel<'a, T, const N: usize, L: Length> {
    ptr: *const T,
    k: usize,
    rows: L,
    _lifetime: PhantomData<&'a [T]>,
}

impl<T, const R: usize> Clone for QPanel<'_, T, R> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T, const R: usize> Copy for QPanel<'_, T, R> {}
impl<T, const N: usize, L: Length> Clone for DPanel<'_, T, N, L> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T, const N: usize, L: Length> Copy for DPanel<'_, T, N, L> {}

impl<'a, T, const R: usize> QPanel<'a, T, R> {
    /// # Safety
    ///
    /// `ptr` must be valid for reads of `R * k` `T` for `'a`.
    unsafe fn new(ptr: *const T, k: usize) -> Self {
        Self {
            ptr,
            k,
            _lifetime: PhantomData,
        }
    }
}

impl<'a, T, const N: usize, L: Length> DPanel<'a, T, N, L> {
    /// # Safety
    ///
    /// `ptr` must be valid for reads of `rows.value() * k` `T` for `'a`, and
    /// `rows.value()` must be at most `N`.
    unsafe fn new(ptr: *const T, k: usize, rows: L) -> Self {
        debug_assert!(rows.value() <= N, "panel taller than its type claims");
        Self {
            ptr,
            k,
            rows,
            _lifetime: PhantomData,
        }
    }
}

impl<T, const R: usize> QPanel<'_, T, R> {
    pub(crate) fn as_ptr(&self) -> *const T {
        self.ptr
    }
    pub(crate) fn k(&self) -> usize {
        self.k
    }
}
impl<T, const N: usize, L: Length> DPanel<'_, T, N, L> {
    pub(crate) fn as_ptr(&self) -> *const T {
        self.ptr
    }
    pub(crate) fn k(&self) -> usize {
        self.k
    }
    pub(crate) fn rows(&self) -> usize {
        self.rows.value()
    }
}

// ── Panel iterators ──────────────────────────────────────────────

/// Walks a block-transposed matrix' blocks. The remainder block is zero-padded to a
/// full `R` rows, so there is no tail.
pub(crate) struct QPanelIter<'a, T: Copy, const R: usize, const P: usize> {
    view: BlockTransposedRef<'a, T, R, P>,
    cur: usize,
    base: Geo,
}

impl<'a, T: Copy, const R: usize, const P: usize> Iterator for QPanelIter<'a, T, R, P> {
    type Item = (QPanel<'a, T, R>, Geo);

    fn next(&mut self) -> Option<Self::Item> {
        let data = self.view.block_slice(self.cur)?;
        // The trailing block is padded to a full `R`, so the real extent clamps to what
        // the view says is real — the kernel still writes the whole panel.
        let start = self.base.start + self.cur * R;
        let geo = Geo {
            start,
            end: (start + R).min(self.base.end),
        };
        self.cur += 1;
        let k = self.view.padded_ncols();
        // SAFETY: `block_slice` returns a checked `R * k` slice borrowed from the view.
        Some((unsafe { QPanel::new(data.as_ptr(), k) }, geo))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // `next` bails without bumping `cur` once `block_slice` says no, so `cur` stops
        // at `num_blocks()` and this can't wrap.
        let n = self.view.num_blocks() - self.cur;
        (n, Some(n))
    }
}

impl<T: Copy, const R: usize, const P: usize> ExactSizeIterator for QPanelIter<'_, T, R, P> {}

impl<T: Copy, const R: usize, const P: usize> TailIterator for QPanelIter<'_, T, R, P> {
    type Tail = NoTail;
    fn tail(self) -> Option<(NoTail, Geo)> {
        None
    }
}

/// Walks a row-major matrix' full `N`-row panels; [`TailIterator::tail`] hands back the
/// short trailer from the same cursor.
///
/// `rest` is what has not been lent yet — the panel count and the trailer's height are
/// read off it.
///
/// # Safety invariants
///
/// `ptr` is valid for reads of `rest.len() * k` `T` for `'a`. `ptr` and `rest.start`
/// advance together — `N` rows *is* `N * k` elements — so the two cannot drift apart.
pub(crate) struct DPanelIter<'a, T, const N: usize> {
    ptr: *const T,
    k: usize,
    rest: Geo,
    _lifetime: PhantomData<&'a [T]>,
}

impl<'a, T, const N: usize> Iterator for DPanelIter<'a, T, N> {
    type Item = (DPanel<'a, T, N, Static<N>>, Geo);

    fn next(&mut self) -> Option<Self::Item> {
        if self.rest.len() < N {
            return None;
        }
        let ptr = self.ptr;
        // SAFETY: `rest.len() >= N`, so the invariant covers another `N * k` and the
        // bump stays inside the allocation.
        self.ptr = unsafe { self.ptr.add(N * self.k) };
        // A full panel is exactly `N` real rows — the short trailer belongs to the tail,
        // so nothing here clamps.
        let geo = Geo {
            start: self.rest.start,
            end: self.rest.start + N,
        };
        self.rest.start = geo.end;
        // SAFETY: as above — `ptr` covers exactly `N * k` readable `T`.
        Some((unsafe { DPanel::new(ptr, self.k, Static) }, geo))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.rest.len() / N;
        (n, Some(n))
    }
}

impl<T, const N: usize> ExactSizeIterator for DPanelIter<'_, T, N> {}

impl<'a, T, const N: usize> TailIterator for DPanelIter<'a, T, N> {
    type Tail = DPanel<'a, T, N, Dynamic>;

    fn tail(self) -> Option<(Self::Tail, Geo)> {
        let rows = self.rest.len();
        debug_assert!(rows < N, "tail taken before the panels are exhausted");
        if rows == 0 {
            return None;
        }
        // SAFETY: the panels are exhausted, so the cursor sits on the trailer, which
        // the invariant covers for `rows * k` readable `T`.
        let panel = unsafe { DPanel::new(self.ptr, self.k, Dynamic(rows)) };
        // What is left over *is* the trailer, so the cursor's own extent is its geo.
        Some((panel, self.rest))
    }
}

// ── Views ────────────────────────────────────────────────────────

/// A block-transposed sub-view plus where it came from. `block_range` hands back a view
/// that has lost its origin, so the walk pairs it back up here — the same job
/// [`RowPanels`] does on the B side.
///
/// The remainder block is zero-padded to a full `R` rows, hence [`NoTail`]; the padding
/// rows are excluded from the [`Geo`]. `P` only widens `padded_ncols`, the contraction
/// length the kernel sees.
pub(crate) struct BlockPanels<'a, T: Copy, const R: usize, const P: usize> {
    view: BlockTransposedRef<'a, T, R, P>,
    start: usize,
}

impl<'a, T: Copy, const R: usize, const P: usize> Paneled for BlockPanels<'a, T, R, P> {
    type Panel = QPanel<'a, T, R>;
    type Tail = NoTail;
    type Panels = QPanelIter<'a, T, R, P>;

    fn geo(&self) -> Geo {
        Geo {
            start: self.start,
            end: self.start + self.view.nrows(),
        }
    }

    fn panels(&self) -> QPanelIter<'a, T, R, P> {
        QPanelIter {
            view: self.view,
            cur: 0,
            base: self.geo(),
        }
    }
}

/// Cut a row-major matrix into `N`-row panels. All the geometry stays on the matrix;
/// only the origin, which sub-viewing drops, is carried alongside.
pub(crate) struct RowPanels<const N: usize, V> {
    view: V,
    start: usize,
}

impl<'a, const N: usize, T: Copy> Paneled for RowPanels<N, MatRef<'a, Standard<T>>> {
    type Panel = DPanel<'a, T, N, Static<N>>;
    type Tail = DPanel<'a, T, N, Dynamic>;
    type Panels = DPanelIter<'a, T, N>;

    fn geo(&self) -> Geo {
        Geo {
            start: self.start,
            end: self.start + self.view.num_vectors(),
        }
    }

    fn panels(&self) -> DPanelIter<'a, T, N> {
        // The checked slice is what establishes `DPanelIter`'s invariant.
        DPanelIter {
            ptr: self.view.as_slice().as_ptr(),
            k: self.view.vector_dim(),
            rest: self.geo(),
            _lifetime: PhantomData,
        }
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
    type View = BlockPanels<'a, T, R, P>;
}
impl<T: Copy, const R: usize, const P: usize> TileWalk for QueryWalk<'_, T, R, P> {
    fn next(&mut self) -> Option<BlockPanels<'_, T, R, P>> {
        let view = self.src.block_range(self.cur, self.tile_panels)?;
        let start = self.cur * R;
        self.cur += view.num_blocks();
        Some(BlockPanels { view, start })
    }
    fn reset(&mut self) {
        self.cur = 0;
    }
}

impl<'a, T: Copy, const N: usize> TileAt<'a> for DocWalk<'_, T, N> {
    type View = RowPanels<N, MatRef<'a, Standard<T>>>;
}
impl<T: Copy, const N: usize> TileWalk for DocWalk<'_, T, N> {
    fn next(&mut self) -> Option<RowPanels<N, MatRef<'_, Standard<T>>>> {
        let view = self.src.row_range(self.cur, self.tile_panels * N)?;
        let start = self.cur;
        self.cur += view.num_vectors();
        Some(RowPanels { view, start })
    }
    fn reset(&mut self) {
        self.cur = 0;
    }
}

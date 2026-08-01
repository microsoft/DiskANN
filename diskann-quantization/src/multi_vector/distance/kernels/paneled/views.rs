// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! The two views and the walks that lend them. Both views are the real matrix types —
//! [`BlockTransposedRef`] for A, [`MatRef`] for B behind a [`Rows`] adapter (a
//! row-major matrix doesn't imply a panel height). Each sub-views itself, so a walk is
//! a cursor and nothing else.

use core::marker::PhantomData;

use super::{NoTail, Paneled, TailIterator, Tile, TileAt, TileWalk};
use crate::bits::{Dynamic, Length, Static};
use crate::multi_vector::{BlockTransposedRef, MatRef, Standard};

// ── Panels ───────────────────────────────────────────────────────

/// One block-transposed A block: `R` rows × `k` `T`.
pub(crate) struct QPanel<'a, T, const R: usize> {
    ptr: *const T,
    k: usize,
    _lifetime: PhantomData<&'a [T]>,
}

/// One row-major B panel: `L` rows × `k` `T`, where `L.value() <= N`. `k` travels with
/// the A panel, so it is not stored. A full panel is `Static<N>` — a ZST, so the whole
/// handle is one pointer; the trailing panel is `Dynamic` in `1..N`.
pub(crate) struct DPanel<'a, T, const N: usize, L: Length> {
    ptr: *const T,
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
    /// `ptr` must be valid for reads of `rows.value() * k` `T` for `'a`, where `k` is
    /// the contraction length carried by the A panel it is paired with, and
    /// `rows.value()` must be at most `N`.
    unsafe fn new(ptr: *const T, rows: L) -> Self {
        debug_assert!(rows.value() <= N, "panel taller than its type claims");
        Self {
            ptr,
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
    pub(crate) fn rows(&self) -> usize {
        self.rows.value()
    }
}

// ── Panel iterators ──────────────────────────────────────────────

/// Walks a block-transposed matrix' blocks. The remainder block is zero-padded to a
/// full `R` rows, so there is no tail.
pub(crate) struct QPanelIter<'a, T: Copy, const R: usize, const P: usize> {
    view: BlockTransposedRef<'a, T, R, P>,
    k: usize,
    cur: usize,
    end: usize,
}

impl<'a, T: Copy, const R: usize, const P: usize> Iterator for QPanelIter<'a, T, R, P> {
    type Item = QPanel<'a, T, R>;

    fn next(&mut self) -> Option<Self::Item> {
        let data = self.view.block_slice(self.cur)?;
        self.cur += 1;
        // SAFETY: `block_slice` returns a checked `R * k` slice borrowed from the view.
        Some(unsafe { QPanel::new(data.as_ptr(), self.k) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.end - self.cur;
        (n, Some(n))
    }
}

impl<T: Copy, const R: usize, const P: usize> ExactSizeIterator for QPanelIter<'_, T, R, P> {}

impl<T: Copy, const R: usize, const P: usize> TailIterator for QPanelIter<'_, T, R, P> {
    type Tail = NoTail;
    fn tail(self) -> Option<NoTail> {
        None
    }
}

/// Walks a row-major matrix' full `N`-row panels; [`TailIterator::tail`] hands back the
/// short trailer from the same cursor.
///
/// # Safety invariants
///
/// `ptr` is valid for reads of `(full * N + tail_rows) * k` `T` for `'a`, and only ever
/// advances.
pub(crate) struct DPanelIter<'a, T, const N: usize> {
    ptr: *const T,
    k: usize,
    full: usize,
    tail_rows: usize,
    _lifetime: PhantomData<&'a [T]>,
}

impl<'a, T, const N: usize> Iterator for DPanelIter<'a, T, N> {
    type Item = DPanel<'a, T, N, Static<N>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.full == 0 {
            return None;
        }
        let ptr = self.ptr;
        // SAFETY: the invariant covers `full` more panels of `N * k`, so the bump stays
        // inside the allocation.
        self.ptr = unsafe { self.ptr.add(N * self.k) };
        self.full -= 1;
        // SAFETY: as above — `ptr` covers exactly `N * k` readable `T`.
        Some(unsafe { DPanel::new(ptr, Static) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.full, Some(self.full))
    }
}

impl<T, const N: usize> ExactSizeIterator for DPanelIter<'_, T, N> {}

impl<'a, T, const N: usize> TailIterator for DPanelIter<'a, T, N> {
    type Tail = DPanel<'a, T, N, Dynamic>;

    fn tail(self) -> Option<Self::Tail> {
        debug_assert_eq!(self.full, 0, "tail taken before the panels are exhausted");
        // SAFETY: the panels are exhausted, so the cursor sits on the trailer, which
        // the invariant covers for `tail_rows * k` readable `T`.
        (self.tail_rows > 0).then(|| unsafe { DPanel::new(self.ptr, Dynamic(self.tail_rows)) })
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
    type Panels = QPanelIter<'a, T, R, P>;

    fn rows(&self) -> usize {
        self.nrows()
    }
    fn panels(&self) -> QPanelIter<'a, T, R, P> {
        QPanelIter {
            view: *self,
            k: self.padded_ncols(),
            cur: 0,
            end: self.num_blocks(),
        }
    }
}

/// Cut a row-major matrix into `N`-row panels. All the geometry stays on the matrix.
pub(crate) struct Rows<const N: usize, V>(pub(crate) V);

impl<'a, const N: usize, T: Copy> Paneled for Rows<N, MatRef<'a, Standard<T>>> {
    type Panel = DPanel<'a, T, N, Static<N>>;
    type Tail = DPanel<'a, T, N, Dynamic>;
    type Panels = DPanelIter<'a, T, N>;

    fn rows(&self) -> usize {
        self.0.num_vectors()
    }
    fn panels(&self) -> DPanelIter<'a, T, N> {
        let (n, k) = (self.rows(), self.0.vector_dim());
        // The checked slice is what establishes `DPanelIter`'s invariant.
        DPanelIter {
            ptr: self.0.as_slice().as_ptr(),
            k,
            full: n / N,
            tail_rows: n % N,
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

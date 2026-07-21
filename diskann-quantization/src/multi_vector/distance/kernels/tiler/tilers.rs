// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Block-transposed A / row-major B walks, generic over the element type. Two
//! flavours share one tile/panel shape:
//!
//! - **Identity** ([`QueryWalk`] / [`DocWalk`]): tiles reborrow the source (`i8`
//!   path — `i16` / `u8`, and any pre-materialised `f32`).
//! - **Convert** ([`QueryConvertWalk`] / [`DocConvertWalk`]): each tile widens its
//!   `f16` source into a **reused** `f32` buffer (allocated once from the arena) and
//!   hands back the *same* `f32` tile the identity-`f32` walk would — so one `f32`
//!   kernel serves both. The lending [`TileWalk`] contract makes the reuse sound: a
//!   live tile borrows the buffer, so `next` (which overwrites it) can't be called
//!   until the tile is dropped.

use super::{Tile, TileAt, TileWalk};

/// Rows per A-panel (the block-transposed group and the reducer's state block).
pub(crate) const A_PANEL: usize = 16;
/// Rows per full B-panel (the kernel's micro-panel / max unroll).
pub(crate) const B_PANEL: usize = 4;

// ── Panels ───────────────────────────────────────────────────────

/// One block-transposed query block: `A_PANEL` rows × `k` `T`.
pub(crate) struct QPanel<'a, T> {
    data: &'a [T],
    k: usize,
}
/// One row-major doc panel: `rows` (`1..=B_PANEL`) × `k` `T`. The contraction `k`
/// travels with the query panel, so it isn't repeated here.
pub(crate) struct DPanel<'a, T> {
    data: &'a [T],
    rows: usize,
}

impl<T> Clone for QPanel<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for QPanel<'_, T> {}
impl<T> Clone for DPanel<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for DPanel<'_, T> {}

impl<T> QPanel<'_, T> {
    pub(crate) fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    pub(crate) fn k(&self) -> usize {
        self.k
    }
}
impl<T> DPanel<'_, T> {
    pub(crate) fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }
}

// ── Materialized tiles ───────────────────────────────────────────

/// One A-tile: a run of whole `A_PANEL`-row blocks.
pub(crate) struct QMat<'a, T> {
    data: &'a [T],
    offset: usize,
    k: usize,
}
/// One B-tile: a run of docs (last panel possibly short).
pub(crate) struct DMat<'a, T> {
    data: &'a [T],
    offset: usize,
    k: usize,
}

impl<'a, T> Tile for QMat<'a, T> {
    type Panel = QPanel<'a, T>;

    fn offset(&self) -> usize {
        self.offset
    }
    fn rows(&self) -> usize {
        if self.k == 0 {
            0
        } else {
            self.data.len() / self.k
        }
    }
    fn panels(&self) -> impl Iterator<Item = QPanel<'a, T>> + '_ {
        let (data, k) = (self.data, self.k);
        let block = A_PANEL * k;
        let n = if block == 0 { 0 } else { data.len() / block };
        (0..n).map(move |p| QPanel {
            data: &data[p * block..(p + 1) * block],
            k,
        })
    }
    fn tail(&self) -> Option<QPanel<'a, T>> {
        None // the query is padded to a whole number of A_PANEL blocks
    }
}

impl<'a, T> Tile for DMat<'a, T> {
    type Panel = DPanel<'a, T>;

    fn offset(&self) -> usize {
        self.offset
    }
    fn rows(&self) -> usize {
        if self.k == 0 {
            0
        } else {
            self.data.len() / self.k
        }
    }
    fn panels(&self) -> impl Iterator<Item = DPanel<'a, T>> + '_ {
        let (data, k) = (self.data, self.k);
        let full = self.rows() / B_PANEL;
        (0..full).map(move |p| DPanel {
            data: &data[p * B_PANEL * k..(p + 1) * B_PANEL * k],
            rows: B_PANEL,
        })
    }
    fn tail(&self) -> Option<DPanel<'a, T>> {
        let (data, k) = (self.data, self.k);
        let rem = self.rows() % B_PANEL;
        (rem > 0).then(|| DPanel {
            data: &data[(self.rows() - rem) * k..],
            rows: rem,
        })
    }
}

// ── Identity walks (reborrow the source) ─────────────────────────

/// Block-transposed query source, walked `tile_panels` `A_PANEL`-row blocks at a time.
pub(crate) struct QueryWalk<'s, T> {
    src: &'s [T],
    k: usize,
    tile_panels: usize,
    cur: usize,
    ti: usize,
}
/// Row-major doc source, walked `tile_panels` `B_PANEL`-row panels at a time.
pub(crate) struct DocWalk<'s, T> {
    src: &'s [T],
    k: usize,
    tile_panels: usize,
    cur: usize,
    ti: usize,
}

impl<'s, T> QueryWalk<'s, T> {
    pub(crate) fn new(src: &'s [T], k: usize, tile_panels: usize) -> Self {
        Self {
            src,
            k,
            tile_panels,
            cur: 0,
            ti: 0,
        }
    }
}
impl<'s, T> DocWalk<'s, T> {
    pub(crate) fn new(src: &'s [T], k: usize, tile_panels: usize) -> Self {
        Self {
            src,
            k,
            tile_panels,
            cur: 0,
            ti: 0,
        }
    }
}

impl<'a, 's, T> TileAt<'a> for QueryWalk<'s, T> {
    type Tile = QMat<'a, T>;
}
impl<'s, T> TileWalk for QueryWalk<'s, T> {
    fn next(&mut self) -> Option<QMat<'_, T>> {
        let src = self.src;
        if self.cur >= src.len() {
            return None;
        }
        let span = (self.tile_panels * A_PANEL * self.k).max(1);
        let (start, ti) = (self.cur, self.ti);
        let end = (start + span).min(src.len());
        self.cur = end;
        self.ti += 1;
        Some(QMat {
            data: &src[start..end],
            offset: ti * self.tile_panels * A_PANEL,
            k: self.k,
        })
    }
    fn reset(&mut self) {
        self.cur = 0;
        self.ti = 0;
    }
    fn max_tile_rows(&self) -> usize {
        self.tile_panels * A_PANEL
    }
}

impl<'a, 's, T> TileAt<'a> for DocWalk<'s, T> {
    type Tile = DMat<'a, T>;
}
impl<'s, T> TileWalk for DocWalk<'s, T> {
    fn next(&mut self) -> Option<DMat<'_, T>> {
        let src = self.src;
        if self.cur >= src.len() {
            return None;
        }
        let span = (self.tile_panels * B_PANEL * self.k).max(1);
        let (start, ti) = (self.cur, self.ti);
        let end = (start + span).min(src.len());
        self.cur = end;
        self.ti += 1;
        Some(DMat {
            data: &src[start..end],
            offset: ti * self.tile_panels * B_PANEL,
            k: self.k,
        })
    }
    fn reset(&mut self) {
        self.cur = 0;
        self.ti = 0;
    }
    fn max_tile_rows(&self) -> usize {
        self.tile_panels * B_PANEL
    }
}

// ── Convert walks (widen f16 → f32 into a reused buffer) ──────────

/// Block-transposed `f16` query, widened per tile into `buf` (`≥ max_tile_rows·k`).
pub(crate) struct QueryConvertWalk<'s, 'buf> {
    src: &'s [half::f16],
    k: usize,
    tile_panels: usize,
    cur: usize,
    ti: usize,
    buf: &'buf mut [f32],
}
/// Row-major `f16` docs, widened per tile into `buf` (`≥ max_tile_rows·k`).
pub(crate) struct DocConvertWalk<'s, 'buf> {
    src: &'s [half::f16],
    k: usize,
    tile_panels: usize,
    cur: usize,
    ti: usize,
    buf: &'buf mut [f32],
}

impl<'s, 'buf> QueryConvertWalk<'s, 'buf> {
    pub(crate) fn new(
        src: &'s [half::f16],
        k: usize,
        tile_panels: usize,
        buf: &'buf mut [f32],
    ) -> Self {
        Self {
            src,
            k,
            tile_panels,
            cur: 0,
            ti: 0,
            buf,
        }
    }
}
impl<'s, 'buf> DocConvertWalk<'s, 'buf> {
    pub(crate) fn new(
        src: &'s [half::f16],
        k: usize,
        tile_panels: usize,
        buf: &'buf mut [f32],
    ) -> Self {
        Self {
            src,
            k,
            tile_panels,
            cur: 0,
            ti: 0,
            buf,
        }
    }
}

impl<'a, 's, 'buf> TileAt<'a> for QueryConvertWalk<'s, 'buf> {
    type Tile = QMat<'a, f32>;
}
impl<'s, 'buf> TileWalk for QueryConvertWalk<'s, 'buf> {
    fn next(&mut self) -> Option<QMat<'_, f32>> {
        let (src, k) = (self.src, self.k);
        if self.cur >= src.len() {
            return None;
        }
        let span = (self.tile_panels * A_PANEL * k).max(1);
        let (start, ti) = (self.cur, self.ti);
        let end = (start + span).min(src.len());
        let len = end - start;
        for i in 0..len {
            self.buf[i] = diskann_wide::cast_f16_to_f32(src[start + i]);
        }
        self.cur = end;
        self.ti += 1;
        Some(QMat {
            data: &self.buf[..len],
            offset: ti * self.tile_panels * A_PANEL,
            k,
        })
    }
    fn reset(&mut self) {
        self.cur = 0;
        self.ti = 0;
    }
    fn max_tile_rows(&self) -> usize {
        self.tile_panels * A_PANEL
    }
}

impl<'a, 's, 'buf> TileAt<'a> for DocConvertWalk<'s, 'buf> {
    type Tile = DMat<'a, f32>;
}
impl<'s, 'buf> TileWalk for DocConvertWalk<'s, 'buf> {
    fn next(&mut self) -> Option<DMat<'_, f32>> {
        let (src, k) = (self.src, self.k);
        if self.cur >= src.len() {
            return None;
        }
        let span = (self.tile_panels * B_PANEL * k).max(1);
        let (start, ti) = (self.cur, self.ti);
        let end = (start + span).min(src.len());
        let len = end - start;
        for i in 0..len {
            self.buf[i] = diskann_wide::cast_f16_to_f32(src[start + i]);
        }
        self.cur = end;
        self.ti += 1;
        Some(DMat {
            data: &self.buf[..len],
            offset: ti * self.tile_panels * B_PANEL,
            k,
        })
    }
    fn reset(&mut self) {
        self.cur = 0;
        self.ti = 0;
    }
    fn max_tile_rows(&self) -> usize {
        self.tile_panels * B_PANEL
    }
}

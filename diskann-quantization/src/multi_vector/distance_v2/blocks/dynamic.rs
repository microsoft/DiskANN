/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use crate::multi_vector::distance_v2::{
    bounds::{self, Bound},
    num::{AllColumns, Elements},
    ptr::Slice,
};

use super::fixed;

//----------//
// RowMajor //
//----------//

#[derive(Debug, Clone, Copy)]
pub(crate) struct RowMajor<'a, T> {
    ptr: Slice<'a, T>,
    rows: usize,
    cols: Bound,
}

impl<'a, T> RowMajor<'a, T> {
    pub(crate) unsafe fn new(ptr: Slice<'a, T>, rows: usize, cols: Bound) -> Self {
        bounds::check_eq!(ptr.len(), Bound::new(rows) * cols);

        Self { ptr, rows, cols }
    }

    pub(crate) const fn as_ptr(&self) -> Slice<'_, T> {
        self.ptr
    }

    pub(crate) const fn nrows(&self) -> usize {
        self.rows
    }

    pub(crate) const fn ncols(&self) -> Bound {
        self.cols
    }

    pub(crate) unsafe fn materialize<const NR: usize>(&self) -> fixed::FullRowMajor<'_, T, NR> {
        debug_assert_eq!(NR, self.rows);
        fixed::FullRowMajor::new(self.ptr, self.ncols())
    }

    pub(crate) unsafe fn subslice(
        &self,
        cols: AllColumns,
        row: usize,
        rows: usize,
    ) -> RowMajor<'a, T> {
        debug_assert!(row <= self.nrows());

        let stride = self.stride(cols);

        Self::new(
            self.ptr.add(stride * row).truncate(stride * rows),
            rows,
            self.cols,
        )
    }

    pub(crate) fn stride(&self, k: AllColumns) -> Elements<T> {
        bounds::check_eq!(self.cols, k.value());
        Elements::new(k.value().get())
    }

    pub(crate) fn block<const ROWS: usize>(
        &self,
        k: AllColumns,
        row: usize,
    ) -> fixed::FullRowMajor<'a, T, ROWS> {
        let stride = self.stride(k);

        fixed::FullRowMajor::new(
            unsafe {
                self.ptr
                    .add(stride * row)
                    .truncate(Elements::new(ROWS) * k.value().get())
            },
            self.cols,
        )
    }

    /// Partition the matrix into bands consisting of `nr` rows (with the last group being
    /// potentially smaller). Provide all sub-matrices to `f`.
    ///
    /// # Safety
    ///
    /// Self must have `k` columns.
    pub(crate) unsafe fn visit_all_rows<F>(&self, nr: NonZeroUsize, k: AllColumns, mut f: F)
    where
        F: FnMut(RowMajor<'_, T>),
    {
        let stride = self.stride(k);

        let mut r = 0;
        while r != self.nrows() {
            let rows = (self.nrows() - r).min(nr.get());

            let sub = Self::new(
                self.ptr.add(stride * r).truncate(stride * rows),
                rows,
                self.ncols(),
            );

            f(sub);

            r += rows;
        }
    }

    /// TODO: A `RowMajor` with a fixed upper capacity.
    #[must_use = "the remainder needs to be handled separately"]
    pub(crate) unsafe fn visit_all_rows_fixed<const NR: usize>(
        &self,
        k: AllColumns,
        mut f: impl FnMut(fixed::FullRowMajor<'_, T, NR>),
    ) -> Option<RowMajor<'_, T>> {
        const { assert!(NR > 0) };

        let full_rows = self.nrows() - self.nrows() % NR;
        let stride = self.stride(k);

        for r in (0..full_rows).step_by(NR) {
            let sub = fixed::FullRowMajor::new(
                self.ptr
                    .add(stride * r)
                    .truncate(Elements::new(NR) * k.value().get()),
                self.ncols(),
            );

            f(sub);
        }

        let remaining = self.nrows() - full_rows;
        if remaining != 0 {
            Some(RowMajor::new(
                self.ptr
                    .add(stride * full_rows)
                    .truncate(Elements::new(remaining) * k.value().get()),
                remaining,
                self.ncols(),
            ))
        } else {
            None
        }
    }
}

//----------------//
// BlockTranspose //
//----------------//

#[derive(Debug, Clone, Copy)]
pub(crate) struct BlockTransposed<'a, T, const GROUP: usize> {
    ptr: Slice<'a, T>,
    blocks: usize,
    cols: Bound,
}

impl<'a, T, const GROUP: usize> BlockTransposed<'a, T, GROUP> {
    pub(crate) fn new(ptr: Slice<'a, T>, blocks: usize, cols: Bound) -> Self {
        bounds::check_eq!(
            ptr.len(),
            Bound::new(blocks) * Bound::new(GROUP) * cols,
            "invalid block-transposed access",
        );

        Self { ptr, blocks, cols }
    }

    pub(crate) const fn as_ptr(&self) -> Slice<'_, T> {
        self.ptr
    }

    pub(crate) const fn blocks(&self) -> usize {
        self.blocks
    }

    pub(crate) const fn ncols(&self) -> Bound {
        self.cols
    }

    fn block_stride(&self, k: AllColumns) -> Elements<T> {
        bounds::check_eq!(self.cols, k.value());
        Elements::new(GROUP * k.value().get())
    }

    pub(crate) fn block(
        &self,
        k: AllColumns,
        block: usize,
    ) -> fixed::FullBlockTranspose<'a, T, GROUP> {
        debug_assert!(block < self.blocks);
        let block_stride = self.block_stride(k);

        fixed::FullBlockTranspose::new(
            unsafe { self.ptr.add(block_stride * block).truncate(block_stride) },
            self.cols,
        )
    }
}

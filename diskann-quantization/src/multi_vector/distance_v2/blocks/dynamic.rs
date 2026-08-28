/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::multi_vector::distance_v2::{
    Check, Length,
    num::{AllColumns, Elements},
    ptr::Slice,
    bounds,
};

use super::fixed;

//----------//
// RowMajor //
//----------//

#[derive(Debug, Clone, Copy)]
pub(crate) struct RowMajor<'a, T> {
    ptr: Slice<'a, T>,
    rows: usize,
    cols: Length,
}

impl<'a, T> RowMajor<'a, T> {
    pub(crate) fn new(ptr: Slice<'a, T>, rows: usize, cols: Length) -> Self {
        bounds::check_eq!(ptr.length(), Length::new(rows) * cols);

        Self { ptr, rows, cols }
    }

    pub(crate) const fn as_ptr(&self) -> Slice<'_, T> {
        self.ptr
    }

    pub(crate) const fn nrows(&self) -> usize {
        self.rows
    }

    pub(crate) const fn ncols(&self) -> Length {
        self.cols
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
        Elements::new(k.value())
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
                    .truncate(Elements::new(ROWS) * k.value())
            },
            self.cols,
        )
    }
}

//----------------//
// BlockTranspose //
//----------------//

#[derive(Debug, Clone, Copy)]
pub(crate) struct BlockTransposed<'a, T, const GROUP: usize> {
    ptr: Slice<'a, T>,
    blocks: usize,
    cols: Length,
}

impl<'a, T, const GROUP: usize> BlockTransposed<'a, T, GROUP> {
    pub(crate) fn new(ptr: Slice<'a, T>, blocks: usize, cols: Length) -> Self {
        bounds::check_eq!(
            ptr.length(),
            Length::new(blocks) * Length::new(GROUP) * cols,
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

    pub(crate) const fn ncols(&self) -> Length {
        self.cols
    }

    fn block_stride(&self, k: AllColumns) -> Elements<T> {
        bounds::check_eq!(self.cols, k.value());
        Elements::new(GROUP * k.value())
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

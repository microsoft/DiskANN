/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::multi_vector::distance_v2::{
    Check, Length,
    num::{AllColumns, Elements},
    ptr::Slice,
};

use super::fixed;

//----------//
// RowMajor //
//----------//

#[derive(Debug, Clone, Copy)]
pub(super) struct RowMajor<'a, T> {
    ptr: Slice<'a, T>,
    rows: Length,
    cols: Length,
}

impl<'a, T> RowMajor<'a, T> {
    pub(super) fn new(ptr: Slice<'a, T>, rows: Length, cols: Length) -> Self {
        ptr.length().check_with(Check::eq(), || rows * cols);

        Self { ptr, rows, cols }
    }

    pub(super) const fn as_ptr(&self) -> Slice<'_, T> {
        self.ptr
    }

    pub(super) const fn nrows(&self) -> Length {
        self.rows
    }

    pub(super) const fn ncols(&self) -> Length {
        self.cols
    }
}

#[derive(Debug)]
pub(super) struct FullRowMajorIter<'a, T, const ROWS: usize> {
    ptr: Slice<'a, T>,
    rows_remaining: usize,
    cols: Length,
}

impl<'a, T, const ROWS: usize> FullRowMajorIter<'a, T, ROWS> {
    fn stride(&self, k: AllColumns) -> Elements<T> {
        self.cols.check(Check::eq(), k.value());
        Elements::new(k.value())
    }

    unsafe fn next(&mut self, k: AllColumns) -> Option<fixed::FullRowMajor<'a, T, ROWS>> {
        self.cols.check(Check::eq(), k.value());

        if self.rows_remaining < ROWS {
            None
        } else {
            let stride = self.stride(k);

            let ptr = self.ptr;
            self.ptr = unsafe { self.ptr.add(stride) };
            self.rows_remaining -= ROWS;
            Some(fixed::FullRowMajor::new(
                ptr.truncate(stride * ROWS),
                self.cols,
            ))
        }
    }

    fn try_cast<const FEWER: usize>(&self) -> Option<fixed::FullRowMajor<'a, T, FEWER>> {
        if self.rows_remaining == FEWER {
            Some(fixed::FullRowMajor::new(self.ptr, self.cols))
        } else {
            None
        }
    }
}

//----------------//
// BlockTranspose //
//----------------//

#[derive(Debug, Clone, Copy)]
pub(super) struct BlockTransposed<'a, T, const GROUP: usize> {
    ptr: Slice<'a, T>,
    blocks: Length,
    cols: Length,
}

impl<'a, T, const GROUP: usize> BlockTransposed<'a, T, GROUP> {
    pub(super) fn new(ptr: Slice<'a, T>, blocks: Length, cols: Length) -> Self {
        ptr.length()
            .check_with(Check::eq(), || blocks * Length::new(GROUP) * cols);

        Self { ptr, blocks, cols }
    }

    pub(super) const fn as_ptr(&self) -> Slice<'_, T> {
        self.ptr
    }

    pub(super) const fn blocks(&self) -> Length {
        self.blocks
    }

    pub(super) const fn ncols(&self) -> Length {
        self.cols
    }

    fn block_stride(&self, k: AllColumns) -> Elements<T> {
        self.cols.check(Check::eq(), k.value());
        Elements::new(GROUP * k.value())
    }

    pub(super) fn block(
        &self,
        k: AllColumns,
        block: usize,
    ) -> fixed::FullBlockTranspose<'a, T, GROUP> {
        self.blocks.check(Check::gt(), block);
        let block_stride = self.block_stride(k);

        fixed::FullBlockTranspose::new(
            unsafe { self.ptr.add(block_stride * block).truncate(block_stride) },
            self.cols,
        )
    }
}

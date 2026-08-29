/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::multi_vector::distance_v2::{bounds, num::Elements, ptr::Slice, Check, Length};

//----------//
// RowMajor //
//----------//

/// A block of `ROWS` rows of a matrix with element type `T`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FullRowMajor<'a, T, const ROWS: usize> {
    ptr: Slice<'a, T>,
    cols: Length,
}

impl<'a, T, const ROWS: usize> FullRowMajor<'a, T, ROWS> {
    pub(crate) fn new(ptr: Slice<'a, T>, cols: Length) -> Self {
        bounds::check_eq!(ptr.length(), cols * Length::new(ROWS));

        Self { ptr, cols }
    }

    pub(crate) const fn as_ptr(&self) -> Slice<'_, T> {
        self.ptr
    }

    pub(crate) const fn nrows(&self) -> usize {
        ROWS
    }

    pub(crate) const fn ncols(&self) -> Length {
        self.cols
    }

    pub(crate) fn stride(&self, cols: usize) -> Elements<T> {
        bounds::check_eq!(self.cols, cols);
        Elements::new(cols)
    }
}

//----------------//
// BlockTranspose //
//----------------//

#[derive(Debug, Clone, Copy)]
pub(crate) struct FullBlockTranspose<'a, T, const GROUP: usize, const PACK: usize = 1> {
    ptr: Slice<'a, T>,
    cols: Length,
}

impl<'a, T, const GROUP: usize, const PACK: usize> FullBlockTranspose<'a, T, GROUP, PACK> {
    pub(crate) fn new(ptr: Slice<'a, T>, cols: Length) -> Self {
        cols.with(|cols| {
            let expected = GROUP * cols.next_multiple_of(PACK);
            bounds::check_eq!(ptr.length(), expected);
        });

        Self { ptr, cols }
    }

    pub(crate) const fn as_ptr(&self) -> Slice<'_, T> {
        self.ptr
    }

    pub(crate) const fn group(&self) -> usize {
        GROUP
    }

    pub(crate) const fn pack(&self) -> usize {
        PACK
    }

    pub(crate) const fn ncols(&self) -> Length {
        self.cols
    }

    pub(crate) fn stride(&self, cols: usize) -> Elements<T> {
        bounds::check_eq!(self.cols, cols);
        Elements::new(GROUP * PACK)
    }
}

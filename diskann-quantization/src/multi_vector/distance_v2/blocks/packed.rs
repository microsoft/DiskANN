/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use crate::multi_vector::distance_v2::{
    bounds::{self, Bound},
    num::{DimK, Elements},
    ptr::Slice,
};

#[derive(Debug, Clone, Copy)]
pub(crate) struct View<'a, T, const EXTENT: usize> {
    ptr: Slice<'a, T>,
    blocks: usize,
    k: Bound,
}

impl<'a, T, const EXTENT: usize> View<'a, T, EXTENT> {
    pub(crate) fn new(ptr: Slice<'a, T>, blocks: usize, k: Bound) -> Self {
        bounds::check_eq!(
            ptr.len(),
            Bound::new(blocks) * Bound::new(EXTENT) * k,
            "invalid block-transposed access",
        );

        Self { ptr, blocks, k }
    }

    pub(crate) const fn as_ptr(&self) -> Slice<'_, T> {
        self.ptr
    }

    pub(crate) const fn blocks(&self) -> usize {
        self.blocks
    }

    pub(crate) const fn k(&self) -> Bound {
        self.k
    }

    fn block_stride(&self, k: DimK) -> Elements<T> {
        bounds::check_eq!(self.k, k.value());
        Elements::new(EXTENT * k.value().get())
    }

    pub(crate) fn block(
        &self,
        k: DimK,
        block: usize,
    ) -> Panel<'a, T, EXTENT> {
        debug_assert!(block < self.blocks);
        let block_stride = self.block_stride(k);

        Panel::new(
            unsafe { self.ptr.add(block_stride * block).truncate(block_stride) },
            self.k,
        )
    }
}

//-------//
// Panel //
//-------//


#[derive(Debug, Clone, Copy)]
pub(crate) struct Panel<'a, T, const EXTENT: usize, const SUB_EXTENT: usize = 1> {
    ptr: Slice<'a, T>,
    k: Bound,
}

impl<'a, T, const EXTENT: usize, const SUB_EXTENT: usize> Panel<'a, T, EXTENT, SUB_EXTENT> {
    pub(crate) fn new(ptr: Slice<'a, T>, k: Bound) -> Self {
        k.with(|k| {
            let expected = EXTENT * k.next_multiple_of(SUB_EXTENT);
            bounds::check_eq!(ptr.len(), expected);
        });

        Self { ptr, k }
    }

    pub(crate) const fn as_ptr(&self) -> Slice<'_, T> {
        self.ptr
    }

    pub(crate) const fn group(&self) -> usize {
        EXTENT
    }

    pub(crate) const fn pack(&self) -> usize {
        SUB_EXTENT
    }

    pub(crate) const fn k(&self) -> Bound {
        self.k
    }

    pub(crate) fn stride(&self, k: DimK) -> Elements<T> {
        bounds::check_eq!(self.k, k);
        Elements::new(EXTENT * SUB_EXTENT)
    }
}

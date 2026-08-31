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
pub(crate) struct View<'a, T, const SZ: usize> {
    ptr: Slice<'a, T>,
    blocks: NonZeroUsize,
    k: Bound,
}

impl<'a, T, const SZ: usize> View<'a, T, SZ> {
    pub(crate) fn new(ptr: Slice<'a, T>, blocks: NonZeroUsize, k: Bound) -> Self {
        bounds::check_eq!(
            ptr.len(),
            Bound::new(blocks.get()) * Bound::new(SZ) * k,
            "invalid block-transposed access",
        );

        Self { ptr, blocks, k }
    }

    pub(crate) const fn blocks(&self) -> NonZeroUsize {
        self.blocks
    }

    pub(crate) const fn k(&self) -> Bound {
        self.k
    }

    pub(crate) fn block_stride(&self, k: DimK) -> Elements<T> {
        bounds::check_eq!(self.k, k.value());
        Elements::new(SZ * k.value().get())
    }

    pub(crate) fn extent(&self) -> NonZeroUsize {
        const { assert!(SZ != 0) };
        self.blocks.saturating_mul(NonZeroUsize::new(SZ).unwrap())
    }

    pub(crate) unsafe fn visit_sub_views<F>(&self, sub_blocks: NonZeroUsize, k: DimK, mut f: F)
    where
        F: FnMut(View<'_, T, SZ>, usize),
    {
        let stride = self.block_stride(k);

        let mut i = 0;

        // The loop bound is a bit funky because it is setup to give us a `NonZeroUsize` for
        // free. Once it returns `None`, we know `i == self.extent()` and we're done.
        while let Some(remaining) = NonZeroUsize::new(self.blocks().get() - i) {
            let this_blocks = remaining.min(sub_blocks);

            let sub = Self::new(
                unsafe {
                    self.ptr
                        .add(stride * i)
                        .truncate(stride * this_blocks.get())
                },
                this_blocks,
                self.k(),
            );

            f(sub, i);

            i += this_blocks.get();
        }
    }

    pub(crate) unsafe fn visit_panels<F>(&self, k: DimK, mut f: F)
    where
        F: FnMut(Panel<'_, T, SZ>, usize),
    {
        let stride = self.block_stride(k);
        for b in 0..self.blocks().get() {
            let panel = Panel::new(unsafe { self.ptr.add(stride * b).truncate(stride) }, self.k);
            f(panel, b);
        }
    }
}

//-------//
// Panel //
//-------//

#[derive(Debug, Clone, Copy)]
pub(crate) struct Panel<'a, T, const SZ: usize, const PACK: usize = 1> {
    ptr: Slice<'a, T>,
    k: Bound,
}

impl<'a, T, const SZ: usize, const PACK: usize> Panel<'a, T, SZ, PACK> {
    pub(crate) fn new(ptr: Slice<'a, T>, k: Bound) -> Self {
        k.with(|k| {
            let expected = SZ * k.next_multiple_of(PACK);
            bounds::check_eq!(ptr.len(), expected);
        });

        Self { ptr, k }
    }

    pub(crate) const fn as_ptr(&self) -> Slice<'_, T> {
        self.ptr
    }

    pub(crate) const fn k(&self) -> Bound {
        self.k
    }

    pub(crate) fn stride(&self, k: DimK) -> Elements<T> {
        bounds::check_eq!(self.k, k);
        Elements::new(SZ * PACK)
    }
}

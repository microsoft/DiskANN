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
pub(crate) struct View<'a, T> {
    ptr: Slice<'a, T>,
    extent: NonZeroUsize,
    k: Bound,
}

impl<'a, T> View<'a, T> {
    pub(crate) unsafe fn new(ptr: Slice<'a, T>, extent: NonZeroUsize, k: Bound) -> Self {
        bounds::check_eq!(ptr.len(), Bound::new(extent.get()) * k);

        Self { ptr, extent, k }
    }

    pub(crate) const fn as_ptr(&self) -> Slice<'_, T> {
        self.ptr
    }

    pub(crate) unsafe fn as_std_slice(&self, k: DimK) -> &[T] {
        let len = self.stride(k) * self.extent().get();
        unsafe { self.ptr.as_std_slice(len.value()) }
    }

    pub(crate) const fn extent(&self) -> NonZeroUsize {
        self.extent
    }

    pub(crate) const fn k(&self) -> Bound {
        self.k
    }

    pub(crate) unsafe fn materialize<const EXTENT: usize>(&self) -> Panel<'_, T, EXTENT> {
        debug_assert_eq!(EXTENT, self.extent().get());
        Panel::new(self.ptr, self.k())
    }

    pub(crate) unsafe fn subslice(
        &self,
        k: DimK,
        start: usize,
        length: NonZeroUsize,
    ) -> View<'a, T> {
        debug_assert!(start <= self.extent().get());

        let stride = self.stride(k);

        Self::new(
            self.ptr.add(stride * start).truncate(stride * length.get()),
            length,
            self.k,
        )
    }

    pub(crate) fn stride(&self, k: DimK) -> Elements<T> {
        bounds::check_eq!(self.k, k.value());
        Elements::new(k.value().get())
    }

    pub(crate) fn block<const EXTENT: usize>(&self, k: DimK, row: usize) -> Panel<'a, T, EXTENT> {
        let stride = self.stride(k);

        Panel::new(
            unsafe {
                self.ptr
                    .add(stride * row)
                    .truncate(Elements::new(EXTENT) * k.value().get())
            },
            self.k,
        )
    }

    /// Partition the matrix into bands consisting of `nr` rows (with the last group being
    /// potentially smaller). Provide all sub-matrices to `f`.
    ///
    /// # Safety
    ///
    /// Self must have `k` columns.
    pub(crate) unsafe fn visit_sub_views<F>(&self, sub_extent: NonZeroUsize, k: DimK, mut f: F)
    where
        F: FnMut(View<'_, T>),
    {
        let stride = self.stride(k);

        let mut i = 0;

        // The loop bound is a bit funky because it is setup to give us a `NonZeroUsize` for
        // free. Once it returns `None`, we know `i == self.extent()` and we're done.
        while let Some(remaining) = NonZeroUsize::new(self.extent().get() - i) {
            let this_extent = remaining.min(sub_extent);

            let sub = Self::new(
                self.ptr
                    .add(stride * i)
                    .truncate(stride * this_extent.get()),
                this_extent,
                self.k(),
            );

            f(sub);

            i += this_extent.get();
        }
    }

    /// TODO: A `View` with a fixed upper capacity.
    #[must_use = "the remainder needs to be handled separately"]
    pub(crate) unsafe fn visit_panels<const EXTENT: usize>(
        &self,
        k: DimK,
        mut f: impl FnMut(Panel<'_, T, EXTENT>),
    ) -> Option<View<'_, T>> {
        const { assert!(EXTENT > 0) };

        let full_groups = self.extent().get() - self.extent().get() % EXTENT;
        let stride = self.stride(k);

        for r in (0..full_groups).step_by(EXTENT) {
            let sub = Panel::new(
                self.ptr
                    .add(stride * r)
                    .truncate(Elements::new(EXTENT) * k.value().get()),
                self.k(),
            );

            f(sub);
        }

        if let Some(remaining) = NonZeroUsize::new(self.extent().get() - full_groups) {
            Some(View::new(
                self.ptr
                    .add(stride * full_groups)
                    .truncate(Elements::new(remaining.get()) * k.value().get()),
                remaining,
                self.k(),
            ))
        } else {
            None
        }
    }
}

/// A block of `EXTENT` rows of a matrix with element type `T`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Panel<'a, T, const EXTENT: usize> {
    ptr: Slice<'a, T>,
    k: Bound,
}

impl<'a, T, const EXTENT: usize> Panel<'a, T, EXTENT> {
    pub(crate) fn new(ptr: Slice<'a, T>, k: Bound) -> Self {
        bounds::check_eq!(ptr.len(), k * Bound::new(EXTENT));

        Self { ptr, k }
    }

    pub(crate) const fn as_ptr(&self) -> Slice<'_, T> {
        self.ptr
    }

    pub(crate) const fn extent(&self) -> usize {
        EXTENT
    }

    pub(crate) const fn k(&self) -> Bound {
        self.k
    }

    pub(crate) fn stride(&self, k: DimK) -> Elements<T> {
        bounds::check_eq!(self.k, k);
        Elements::new(k.value().get())
    }
}

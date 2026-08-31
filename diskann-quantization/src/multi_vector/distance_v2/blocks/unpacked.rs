/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

#[cfg(test)]
use diskann_utils::views::MatrixView;

use crate::multi_vector::distance_v2::{
    bounds::{self, Bound},
    num::{DimK, Elements},
    ptr::Slice,
};

/// A view over an unpacked 2-dimensional matrix.
///
/// This struct avoids mentioning "rows" and "columns" as its use is contextual. Instead,
/// consider memory as divided contiguous "bands", each containing "k" elements of type `T`.
/// Bands are laid out sequentially and contiguously.
///
/// [`Self::extent`] describes the number of bands.
///
/// For row-major matrices, "bands" is interpreted as "rows". For column-major, "bands" is
/// interpreted as "columns".
#[derive(Debug, Clone, Copy)]
pub(crate) struct View<'a, T> {
    ptr: Slice<'a, T>,
    extent: NonZeroUsize,
    k: Bound,
}

impl<'a, T> View<'a, T> {
    /// Construct a new [`View`] over `ptr`.
    ///
    /// # Safety
    ///
    /// The true length of `ptr` must be exactly `extent * k`.
    pub(crate) unsafe fn new(ptr: Slice<'a, T>, extent: NonZeroUsize, k: DimK) -> Self {
        let k: usize = k.value().get();
        bounds::check_eq!(ptr.len(), extent.get() * k);
        unsafe { Self::new_inner(ptr, extent, Bound::new(k)) }
    }

    /// Construct a [`View`] from a [`MatrixView`].
    #[cfg(test)]
    pub(crate) fn from_view(v: MatrixView<'a, T>) -> Option<Self> {
        let extent = NonZeroUsize::new(v.nrows())?;
        let k = DimK::new(NonZeroUsize::new(v.ncols())?);
        Some(unsafe { Self::new(Slice::new(v.into_inner()), extent, k) })
    }

    unsafe fn new_inner(ptr: Slice<'a, T>, extent: NonZeroUsize, k: Bound) -> Self {
        bounds::check_eq!(ptr.len(), Bound::new(extent.get()) * k);
        Self { ptr, extent, k }
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

    pub(crate) fn stride(&self, k: DimK) -> Elements<T> {
        bounds::check_eq!(self.k, k.value());
        Elements::new(k.value().get())
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

            let sub = unsafe {
                Self::new_inner(
                    self.ptr
                        .add(stride * i)
                        .truncate(stride * this_extent.get()),
                    this_extent,
                    self.k(),
                )
            };

            f(sub);

            i += this_extent.get();
        }
    }

    /// TODO: A `View` with a fixed upper capacity.
    #[must_use = "the remainder needs to be handled separately"]
    pub(crate) unsafe fn visit_panels<const EXTENT: usize>(
        &self,
        k: DimK,
        mut f: impl FnMut(Panel<'_, T, EXTENT>, usize),
    ) -> Option<Remainder<'_, T, EXTENT>> {
        const { assert!(EXTENT > 0) };

        let full_groups = self.extent().get() - self.extent().get() % EXTENT;
        let stride = self.stride(k);

        for r in (0..full_groups).step_by(EXTENT) {
            let sub = unsafe {
                Panel::new_inner(
                    self.ptr
                        .add(stride * r)
                        .truncate(Elements::new(EXTENT) * k.value().get()),
                    self.k(),
                )
            };

            f(sub, r);
        }

        if let Some(remaining) = NonZeroUsize::new(self.extent().get() - full_groups) {
            Some(unsafe {
                Remainder::new_inner(
                    self.ptr
                        .add(stride * full_groups)
                        .truncate(Elements::new(remaining.get()) * k.value().get()),
                    full_groups,
                    remaining,
                    self.k(),
                )
            })
        } else {
            None
        }
    }

    // NOTE: This function is not safe outside of a `cfg(test)` context.
    #[cfg(test)]
    fn get(&self, band: usize, offset: usize) -> &T {
        assert!(self.extent().get() > band);
        bounds::check_gt!(self.k(), offset);

        let stride = self.stride(DimK::new(NonZeroUsize::new(self.k().value()).unwrap()));

        // SAFETY: All operations are safe under `cfg(test)`.
        unsafe {
            self.ptr
                .add(stride * band + Elements::new(offset))
                .truncate(Elements::new(1))
                .as_ref()
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
    unsafe fn new_inner(ptr: Slice<'a, T>, k: Bound) -> Self {
        bounds::check_eq!(ptr.len(), k * Bound::new(EXTENT));
        Self { ptr, k }
    }

    pub(crate) const fn as_ptr(&self) -> Slice<'_, T> {
        self.ptr
    }

    #[cfg(test)]
    unsafe fn as_std_slice(self, k: DimK) -> &'a [T] {
        bounds::check_eq!(self.k(), k);
        unsafe { self.ptr.as_std_slice(EXTENT * k.value().get()) }
    }

    pub(crate) const fn k(&self) -> Bound {
        self.k
    }

    pub(crate) fn stride(&self, k: DimK) -> Elements<T> {
        bounds::check_eq!(self.k(), k);
        Elements::new(k.value().get())
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Remainder<'a, T, const CAPACITY: usize> {
    ptr: Slice<'a, T>,
    start: usize,
    extent: NonZeroUsize,
    k: Bound,
}

impl<'a, T, const CAPACITY: usize> Remainder<'a, T, CAPACITY> {
    unsafe fn new_inner(ptr: Slice<'a, T>, start: usize, extent: NonZeroUsize, k: Bound) -> Self {
        bounds::check_eq!(ptr.len(), Bound::new(extent.get()) * k);
        Self {
            ptr,
            start,
            extent,
            k,
        }
    }

    pub(crate) fn extent(&self) -> NonZeroUsize {
        self.extent
    }

    pub(crate) fn start(&self) -> usize {
        self.start
    }

    fn k(&self) -> Bound {
        self.k
    }

    pub(crate) fn try_as_panel<const EXTENT: usize>(self) -> Option<Panel<'a, T, EXTENT>> {
        if self.extent().get() == EXTENT {
            Some(unsafe { Panel::new_inner(self.ptr, self.k()) })
        } else {
            None
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test {
    use super::*;

    use diskann_utils::views::Matrix;
    use rand::{SeedableRng, rngs::StdRng};

    use crate::multi_vector::distance_v2::test_util;

    #[test]
    fn test_view() {
        let mut rng = StdRng::seed_from_u64(0x32a99c1210);
        for nrows in (1..100).step_by(7) {
            for ncols in (1..20).step_by(3) {
                test_view_inner(
                    NonZeroUsize::new(nrows).unwrap(),
                    NonZeroUsize::new(ncols).unwrap(),
                    &mut rng,
                    format_args!("ncols = {ncols}, nrows = {nrows}"),
                )
            }
        }
    }

    fn test_view_inner(
        nrows: NonZeroUsize,
        ncols: NonZeroUsize,
        rng: &mut impl rand::Rng,
        ctx: std::fmt::Arguments<'_>,
    ) {
        let k = DimK::new(ncols);

        let mut mat = Matrix::new(0.0f32, nrows.get(), ncols.get());
        test_util::TestDistr::fill(mat.as_mut_slice(), rng);

        let view = View::from_view(mat.as_view()).unwrap();
        assert_eq!(view.extent().get(), mat.nrows());
        assert_eq!(view.k().value(), mat.ncols());

        assert_eq!(
            unsafe { view.as_std_slice(k) },
            mat.as_slice(),
            "underlying slices must be unchanged -- {ctx}",
        );

        visit_panels::<1>(view, mat.as_view(), ctx);
        visit_panels::<2>(view, mat.as_view(), ctx);
        visit_panels::<3>(view, mat.as_view(), ctx);
        visit_panels::<4>(view, mat.as_view(), ctx);
        visit_panels::<6>(view, mat.as_view(), ctx);
    }

    fn visit_panels<const N: usize>(
        dut: View<'_, f32>,
        reference: MatrixView<'_, f32>,
        ctx: std::fmt::Arguments<'_>,
    ) {
        let dimk = DimK::new(NonZeroUsize::new(dut.k().value()).unwrap());
        let mut count = 0;
        let ncols = reference.ncols();

        let visitor = |panel: Panel<'_, f32, N>, start| {
            assert_eq!(start, count, "{ctx}");
            let s = unsafe { panel.as_std_slice(dimk) };
            assert_eq!(s, &reference.as_slice()[ncols * start..ncols * (start + N)],);

            count += N;
        };

        let remainder = unsafe { dut.visit_panels::<N>(dimk, visitor) };

        match remainder {
            None => assert!(dut.extent().get().is_multiple_of(N), "{ctx}"),
            Some(remainder) => {
                assert_eq!(
                    remainder.extent().get(),
                    dut.extent().get() % N,
                    "{ctx}"
                );

                let start = remainder.start();
                assert_eq!(start, count, "{ctx}");
                assert_eq!(start + remainder.extent().get(), reference.nrows(), "{ctx}");

                let expected = &reference.as_slice()[ncols * start..];

                let mut passed = false;

                macro_rules! check {
                    ($N:literal) => {
                        if let Some(panel) = remainder.try_as_panel::<$N>() {
                            assert_eq!(
                                unsafe { panel.as_std_slice(dimk) },
                                expected,
                                "-- N = {}, {ctx}",
                                $N,
                            );
                            assert!(!passed, "-- N = {}, {ctx}", $N);
                            passed = true;
                        }
                    }
                }

                check!(1);
                check!(2);
                check!(3);
                check!(4);
                check!(5);
                check!(6);

                assert!(passed, "{ctx}");
            }
        }
    }
}

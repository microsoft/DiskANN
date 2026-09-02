/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann_utils::views::MatrixView;

use crate::matrix_kernels::{
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
///
/// # Class Invariants
///
/// The bound `ptr.len()` must be equal to `self.extent * self.k`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct View<'a, T> {
    ptr: Slice<'a, T>,
    extent: NonZeroUsize,
    k: Bound,
}

impl<'a, T> View<'a, T> {
    /// Construct a [`View`] from a [`MatrixView`].
    ///
    /// Since [`MatrixView`]s are interpreted as "row-major", the value `k` will be derived
    /// from `v.ncols()` and the extent will be taken from `v.nrows()`.
    ///
    /// Returns `None` if either dimension is zero.
    pub(crate) fn from_matrix_view(v: MatrixView<'a, T>) -> Option<Self> {
        let extent = NonZeroUsize::new(v.nrows())?;
        let k = DimK::new(NonZeroUsize::new(v.ncols())?);

        // SAFETY: The `MatrixView` ensures that the inner slice has size `extent * k`.
        Some(unsafe { Self::new(Slice::new(v.into_inner()), extent, k) })
    }

    /// Construct a new [`View`] over `ptr`.
    ///
    /// # Safety
    ///
    /// The true length of `ptr` must be exactly `extent * k`.
    pub(crate) unsafe fn new(ptr: Slice<'a, T>, extent: NonZeroUsize, k: DimK) -> Self {
        let k: usize = k.value().get();
        bounds::check_eq!(ptr.len(), extent.get() * k);

        // SAFETY: Inherited from caller
        unsafe { Self::new_inner(ptr, extent, Bound::new(k)) }
    }

    /// # Safety
    ///
    /// The true length of `ptr` must be exactly `extent * k`.
    unsafe fn new_inner(ptr: Slice<'a, T>, extent: NonZeroUsize, k: Bound) -> Self {
        bounds::check_eq!(ptr.len(), Bound::new(extent.get()) * k);
        Self { ptr, extent, k }
    }

    /// View the underlying memory as a `&[T]`.
    ///
    /// # Safety
    ///
    /// `k` must equal the contraction dimension tracked by [`Self::k`].
    pub(in crate::matrix_kernels) unsafe fn as_std_slice(&self, k: DimK) -> &[T] {
        let len = self.stride(k) * self.extent().get();

        // SAFETY: By class invariant, the true ize f the underlying slice will be `len`.
        unsafe { self.ptr.as_std_slice(len.value()) }
    }

    /// Return the number of contiguous bands in `self`.
    ///
    /// Each band contains [`Self::k`] elements.
    pub(in crate::matrix_kernels) const fn extent(&self) -> NonZeroUsize {
        self.extent
    }

    /// Return the number of elements in each "band" of `self`.
    ///
    /// This is tracked as a [`Bound`]. Its provenance comes from the constructors of `self`.
    pub(in crate::matrix_kernels) const fn k(&self) -> Bound {
        self.k
    }

    /// Return the number of elements in each "band" of self.
    ///
    /// The value `k` must be equal to [`Self::k`].
    pub(in crate::matrix_kernels) fn stride(&self, k: DimK) -> Elements<T> {
        bounds::check_eq!(self.k, k.value());
        Elements::new(k.value().get())
    }

    /// Partition the matrix into sub-views each containing at most `sub_extent` bands, with
    /// the last one potentially containing fewer.
    ///
    /// Provide all sub-views to `f` in memory order. The second argument to `f` is the index
    /// of the first band in the sub-view within `self`.
    ///
    /// # Safety
    ///
    /// The bound [`Self::k`] must be equal to `k`.
    #[inline(always)]
    pub(in crate::matrix_kernels) unsafe fn visit_sub_views<F>(
        &self,
        sub_extent: NonZeroUsize,
        k: DimK,
        mut f: F,
    ) where
        F: FnMut(View<'_, T>, usize),
    {
        let stride = self.stride(k);

        let mut i = 0;

        // The loop bound is a bit funky because it is setup to give us a `NonZeroUsize` for
        // free. Once it returns `None`, we know `i == self.extent()` and we're done.
        while let Some(remaining) = NonZeroUsize::new(self.extent().get() - i) {
            let this_extent = remaining.min(sub_extent);

            // SAFETY: If `k` is correct:
            //
            // * The pointer offset and truncation are valid.
            // * The `Slice` provided to `new_inner` has a length of `this_extent * k`.
            let sub = unsafe {
                Self::new_inner(
                    self.ptr
                        .add(stride * i)
                        .truncate(stride * this_extent.get()),
                    this_extent,
                    self.k(),
                )
            };

            f(sub, i);

            i += this_extent.get();
        }
    }

    /// Partition the matrix into panels each containing exactly `EXTENT` bands.
    ///
    /// Provide all panels to `visitor` in memory order. The visitor receives the index of
    /// each panel's first band within `self`.
    ///
    /// If [`Self::extent`] is not a multiple of `EXTENT`, then a [`Remainder`] will be
    /// returned for the remaining bands. The remainder starts immediately after the visited
    /// panels.
    ///
    /// # Safety
    ///
    /// The bound [`Self::k`] must be equal to `k`.
    #[inline(always)]
    #[must_use = "the remainder needs to be handled separately"]
    pub(in crate::matrix_kernels) unsafe fn visit_panels<const EXTENT: usize>(
        &self,
        k: DimK,
        mut visitor: impl PanelVisitor<T, EXTENT>,
    ) -> Option<Remainder<'_, T, EXTENT>> {
        const { assert!(EXTENT > 0) };

        let full_groups = self.extent().get() - self.extent().get() % EXTENT;
        let stride = self.stride(k);

        for r in (0..full_groups).step_by(EXTENT) {
            // SAFETY: By class invariant, `self.ptr.len() == self.extent * self.k`.
            //
            // The caller asserts that `k == self.k`.
            //
            // Since `r < self.extent()`:
            //
            // * The pointer offset if valid.
            // * The truncation is valid.
            // * The size of the resulting slice is equal to `EXTENT * k`.
            let sub = unsafe {
                Panel::new_inner(
                    self.ptr
                        .add(stride * r)
                        .truncate(Elements::new(EXTENT) * k.value().get()),
                    self.k(),
                )
            };

            visitor.visit(sub, r);
        }

        NonZeroUsize::new(self.extent().get() - full_groups).map(|remaining| {
            // SAFETY: Following the logic above, the pointer offset, truncation, and
            // length are all correct.
            //
            // Further, `remaining` is guaranteed to be strictly less than `EXTENT`.
            unsafe {
                Remainder::new_inner(
                    self.ptr
                        .add(stride * full_groups)
                        .truncate(Elements::new(remaining.get()) * k.value().get()),
                    full_groups,
                    remaining,
                    self.k(),
                )
            }
        })
    }
}

/// A visitor for [`View::visit_panels`].
///
/// This is expressed as a custom trait rather than a closure because rustc/LLVM do not
/// seem to reliably inline closures, which is necessary when embedding visitation in a
/// context with `target_feature` enabled.
pub(in crate::matrix_kernels) trait PanelVisitor<T, const N: usize> {
    /// Visit a [`Panel`] from [`View::visit_panels`]. Argument `start` gives the index
    /// of the first band in `panel` in the parent [`View`].
    fn visit(&mut self, panel: Panel<'_, T, N>, start: usize);
}

impl<T, const N: usize, F> PanelVisitor<T, N> for F
where
    F: FnMut(Panel<'_, T, N>, usize),
{
    #[inline(always)]
    fn visit(&mut self, panel: Panel<'_, T, N>, start: usize) {
        (self)(panel, start)
    }
}

#[cfg(test)]
impl<'a, T> View<'a, T> {
    fn checked_as_std_slice(&self) -> &[T] {
        let k = DimK::from_bound(self.k());
        // SAFETY: Checked in test builds.
        unsafe { self.as_std_slice(k) }
    }

    fn checked_visit_sub_views<F>(&self, sub_extent: NonZeroUsize, f: F)
    where
        F: FnMut(View<'_, T>, usize),
    {
        let k = DimK::from_bound(self.k());
        // SAFETY: Checked in test builds.
        unsafe { self.visit_sub_views(sub_extent, k, f) }
    }

    fn checked_visit_panels<const EXTENT: usize>(
        &self,
        f: impl FnMut(Panel<'_, T, EXTENT>, usize),
    ) -> Option<Remainder<'_, T, EXTENT>> {
        let k = DimK::from_bound(self.k());
        // SAFETY: Checked in test builds.
        unsafe { self.visit_panels(k, f) }
    }
}

/// A block of `EXTENT` bands of a matrix of type `T`.
///
/// # Class Invariants
///
/// The bound `ptr.len()` must be equal to `EXTENT * self.k`.
#[derive(Debug, Clone, Copy)]
pub(in crate::matrix_kernels) struct Panel<'a, T, const EXTENT: usize> {
    ptr: Slice<'a, T>,
    k: Bound,
}

impl<'a, T, const EXTENT: usize> Panel<'a, T, EXTENT> {
    /// Construct a new [`Panel`].
    ///
    /// # Safety
    ///
    /// The bound `ptr.len()` must be equal to exactly `k * EXTENT`.
    #[cfg(test)]
    pub(in crate::matrix_kernels) unsafe fn new(ptr: Slice<'a, T>, k: DimK) -> Self {
        let k: usize = k.value().get();
        bounds::check_eq!(ptr.len(), k * EXTENT);
        // SAFETY: Inherited from caller.
        unsafe { Self::new_inner(ptr, Bound::new(k)) }
    }

    /// # Safety
    ///
    /// The bound `ptr.len()` must be equal to exactly `k * EXTENT`.
    unsafe fn new_inner(ptr: Slice<'a, T>, k: Bound) -> Self {
        bounds::check_eq!(ptr.len(), k * Bound::new(EXTENT));
        Self { ptr, k }
    }

    /// Return the base span of this panel as a [`Slice`].
    pub(in crate::matrix_kernels) const fn as_ptr(&self) -> Slice<'_, T> {
        self.ptr
    }

    /// Return the contraction dimension of `self`.
    ///
    /// This is inherited from all constructors.
    pub(in crate::matrix_kernels) const fn k(&self) -> Bound {
        self.k
    }

    /// Return the number of elements in each band of `self`.
    ///
    /// `k` must equal the contraction dimension tracked by [`Self::k`].
    pub(in crate::matrix_kernels) fn stride(&self, k: DimK) -> Elements<T> {
        bounds::check_eq!(self.k(), k);
        Elements::new(k.value().get())
    }

    /// Return the contents of `self` as a `&[T]`.
    ///
    /// # Safety
    ///
    /// `k` must equal the contraction dimension tracked by [`Self::k`].
    #[cfg(test)]
    unsafe fn as_std_slice(&self, k: DimK) -> &[T] {
        bounds::check_eq!(self.k(), k);

        // SAFETY: Since `k == self.k`, the underlying slice has a length of exactly
        // `EXTENT * k`.
        unsafe { self.ptr.as_std_slice(EXTENT * k.value().get()) }
    }
}

#[cfg(test)]
impl<T, const EXTENT: usize> Panel<'_, T, EXTENT> {
    fn checked_as_std_slice(&self) -> &[T] {
        let k = DimK::from_bound(self.k());
        // SAFETY: Checked in test builds.
        unsafe { self.as_std_slice(k) }
    }
}

/// A nonempty trailing view containing fewer than `CAPACITY` bands.
///
/// # Class Invariants
///
/// * The bound `ptr.len()` must be equal to `self.extent * self.k`.
/// * `self.extent` must be **strictly less** than `CAPACITY`.
#[derive(Debug, Clone, Copy)]
pub(in crate::matrix_kernels) struct Remainder<'a, T, const CAPACITY: usize> {
    ptr: Slice<'a, T>,
    _start: usize,
    extent: NonZeroUsize,
    k: Bound,
}

impl<'a, T, const CAPACITY: usize> Remainder<'a, T, CAPACITY> {
    /// # Safety
    ///
    /// * `ptr.len()` must be equal to `extent * k`.
    /// * `extent` must be strictly less than `CAPACITY`.
    unsafe fn new_inner(ptr: Slice<'a, T>, start: usize, extent: NonZeroUsize, k: Bound) -> Self {
        bounds::check_eq!(ptr.len(), Bound::new(extent.get()) * k);
        bounds::check_lt!(
            Bound::new(extent.get()),
            CAPACITY,
            "remainder must be strictly less than capacity"
        );

        Self {
            ptr,
            _start: start,
            extent,
            k,
        }
    }

    /// Return the number of bands in `self`.
    ///
    /// This is guaranteed to be less than `CAPACITY`.
    pub(in crate::matrix_kernels) fn extent(&self) -> NonZeroUsize {
        self.extent
    }

    /// Return the index of the first band in `self`'s immediate parent [`View`].
    #[cfg_attr(
        not(test),
        expect(unused, reason = "this completes an API but is not used yet")
    )]
    pub(in crate::matrix_kernels) fn start(&self) -> usize {
        self._start
    }

    /// Return the number of elements in each "band" of `self`.
    fn k(&self) -> Bound {
        self.k
    }

    /// Return a [`Panel`] if [`Self::extent`] is equal to `EXTENT`.
    ///
    /// Otherwise, returns `None`.
    pub(in crate::matrix_kernels) fn try_as_panel<const EXTENT: usize>(
        self,
    ) -> Option<Panel<'a, T, EXTENT>> {
        if self.extent().get() == EXTENT {
            // SAFETY: By class invariant, `self.ptr.len() == EXTENT * self.k()`.
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
    use diskann_utils::views::{Init, Matrix};

    use crate::matrix_kernels::test_util::{assert_contains, panic_message_for};

    #[test]
    fn test_visit_panels() {
        for nrows in (1..100).step_by(7) {
            for ncols in (1..20).step_by(3) {
                test_visit_panels_inner(
                    NonZeroUsize::new(nrows).unwrap(),
                    NonZeroUsize::new(ncols).unwrap(),
                    format_args!("ncols = {ncols}, nrows = {nrows}"),
                )
            }
        }
    }

    fn test_visit_panels_inner(
        nrows: NonZeroUsize,
        ncols: NonZeroUsize,
        ctx: std::fmt::Arguments<'_>,
    ) {
        let mat = {
            let mut i = 0.0;
            let init = Init(|| {
                let v = i;
                i += 1.0;
                v
            });
            Matrix::new(init, nrows.get(), ncols.get())
        };

        let view = View::from_matrix_view(mat.as_view()).unwrap();
        assert_eq!(view.extent().get(), mat.nrows());
        assert_eq!(view.k().value(), mat.ncols());

        assert_eq!(
            view.checked_as_std_slice(),
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
        let mut count = 0;
        let ncols = reference.ncols();

        let visitor = |panel: Panel<'_, f32, N>, start| {
            assert_eq!(start, count, "{ctx}");
            let s = panel.checked_as_std_slice();
            assert_eq!(s, &reference.as_slice()[ncols * start..ncols * (start + N)],);

            count += N;
        };

        let remainder = dut.checked_visit_panels::<N>(visitor);

        match remainder {
            None => assert!(dut.extent().get().is_multiple_of(N), "{ctx}"),
            Some(remainder) => {
                assert_eq!(remainder.extent().get(), dut.extent().get() % N, "{ctx}");

                let start = remainder.start();
                assert_eq!(start, count, "{ctx}");
                assert_eq!(start + remainder.extent().get(), reference.nrows(), "{ctx}");

                let expected = &reference.as_slice()[ncols * start..];

                let mut passed = false;

                macro_rules! check {
                    ($N:literal) => {
                        if let Some(panel) = remainder.try_as_panel::<$N>() {
                            assert_eq!(
                                panel.checked_as_std_slice(),
                                expected,
                                "-- N = {}, {ctx}",
                                $N,
                            );
                            assert!(!passed, "-- N = {}, {ctx}", $N);
                            passed = true;
                        }
                    };
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

    #[test]
    fn test_visit_sub_views() {
        for nrows in (1..100).step_by(7) {
            for ncols in (1..20).step_by(3) {
                test_sub_views_inner(
                    NonZeroUsize::new(nrows).unwrap(),
                    NonZeroUsize::new(ncols).unwrap(),
                    format_args!("ncols = {ncols}, nrows = {nrows}"),
                )
            }
        }
    }

    fn test_sub_views_inner(
        nrows: NonZeroUsize,
        ncols: NonZeroUsize,
        ctx: std::fmt::Arguments<'_>,
    ) {
        let mat = {
            let mut i = 0.0;
            let init = Init(|| {
                let v = i;
                i += 1.0;
                v
            });
            Matrix::new(init, nrows.get(), ncols.get())
        };

        let view = View::from_matrix_view(mat.as_view()).unwrap();
        assert_eq!(view.extent().get(), mat.nrows());
        assert_eq!(view.k().value(), mat.ncols());

        assert_eq!(
            view.checked_as_std_slice(),
            mat.as_slice(),
            "underlying slices must be unchanged -- {ctx}",
        );

        let sub_extents: Vec<NonZeroUsize> = [
            1,
            nrows.get().div_ceil(10) + 1,
            nrows.get() / 2,
            nrows.get() - 1,
            nrows.get(),
            nrows.get() + 1,
        ]
        .map(NonZeroUsize::new)
        .into_iter()
        .flatten()
        .collect();

        for sub_extent in sub_extents.into_iter() {
            let mut count = 0;
            let nc = ncols.get();

            view.checked_visit_sub_views(sub_extent, |v, start| {
                assert_eq!(start, count, "{ctx}");
                assert!(v.extent() <= sub_extent, "{ctx}");
                assert_eq!(
                    v.checked_as_std_slice(),
                    &mat.as_slice()[start * nc..(start + v.extent().get()) * nc],
                );

                count += v.extent().get();
            });

            assert_eq!(count, nrows.get());
        }
    }

    #[test]
    fn test_rejects_inconsistent_lengths() {
        let data = [0u8; 7];
        let extent = NonZeroUsize::new(2).unwrap();
        let k = DimK::new(NonZeroUsize::new(3).unwrap());

        for len in [5, 7] {
            let message = panic_message_for(|| {
                // SAFETY: The deliberate length mismatch is caught while bounds are retained.
                let _ = unsafe { View::new(Slice::new(&data[..len]), extent, k) };
            });
            assert_contains!(message, "equal to 6");

            let message = panic_message_for(|| {
                // SAFETY: The deliberate length mismatch is caught while bounds are retained.
                let _ =
                    unsafe { Panel::<_, 2>::new_inner(Slice::new(&data[..len]), Bound::new(3)) };
            });
            assert_contains!(message, "equal to 6");

            let message = panic_message_for(|| {
                // SAFETY: The deliberate length mismatch is caught while bounds are retained.
                let _ = unsafe {
                    Remainder::<_, 3>::new_inner(Slice::new(&data[..len]), 0, extent, Bound::new(3))
                };
            });
            assert_contains!(message, "equal to 6");
        }

        let message = panic_message_for(|| {
            // SAFETY: The deliberate capacity mismatch is caught while bounds are retained.
            let _ = unsafe {
                Remainder::<_, 2>::new_inner(Slice::new(&data[..6]), 0, extent, Bound::new(3))
            };
        });
        assert_contains!(message, "remainder must be strictly less than capacity");
    }

    #[test]
    fn test_rejects_inconsistent_k() {
        let data = [0u8; 6];
        let actual_k = DimK::new(NonZeroUsize::new(3).unwrap());
        let wrong_k = DimK::new(NonZeroUsize::new(2).unwrap());

        // SAFETY: `data` contains two bands of three elements.
        let view = unsafe { View::new(Slice::new(&data), NonZeroUsize::new(2).unwrap(), actual_k) };

        assert_k_mismatch(|| {
            let _ = view.stride(wrong_k);
        });
        assert_k_mismatch(|| {
            // SAFETY: The deliberate K mismatch is caught before materialization.
            let _ = unsafe { view.as_std_slice(wrong_k) };
        });
        assert_k_mismatch(|| {
            // SAFETY: The deliberate K mismatch is caught before pointer arithmetic.
            unsafe {
                view.visit_sub_views(NonZeroUsize::new(1).unwrap(), wrong_k, |_, _| {});
            }
        });
        assert_k_mismatch(|| {
            // SAFETY: The deliberate K mismatch is caught before pointer arithmetic.
            let _ = unsafe { view.visit_panels::<2>(wrong_k, |_: Panel<'_, _, _>, _| {}) };
        });

        // SAFETY: `data` contains exactly two bands of three elements.
        let panel = unsafe { Panel::<_, 2>::new_inner(Slice::new(&data), Bound::new(3)) };

        assert_k_mismatch(|| {
            let _ = panel.stride(wrong_k);
        });
        assert_k_mismatch(|| {
            // SAFETY: The deliberate K mismatch is caught before materialization.
            let _ = unsafe { panel.as_std_slice(wrong_k) };
        });
    }

    fn assert_k_mismatch(f: impl FnOnce() + std::panic::UnwindSafe) {
        let message = panic_message_for(f);
        assert_contains!(message, "equal to 2");
    }
}

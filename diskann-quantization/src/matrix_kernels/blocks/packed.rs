/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use crate::matrix_kernels::{
    bounds::{self, Bound},
    num::{DimK, Elements},
    ptr::Slice,
};

#[cfg(test)]
use crate::multi_vector::BlockTransposedRef;

#[derive(Debug, Clone, Copy)]
pub(crate) struct View<'a, T, const SZ: usize> {
    ptr: Slice<'a, T>,
    blocks: NonZeroUsize,
    k: Bound,
}

impl<'a, T, const SZ: usize> View<'a, T, SZ> {
    pub(crate) unsafe fn new(ptr: Slice<'a, T>, blocks: NonZeroUsize, k: DimK) -> Self {
        bounds::check_eq!(
            ptr.len(),
            blocks.get() * SZ * k.value().get(),
            "invalid block-transposed access",
        );

        unsafe { Self::new_inner(ptr, blocks, Bound::new(k.value().get())) }
    }

    unsafe fn new_inner(ptr: Slice<'a, T>, blocks: NonZeroUsize, k: Bound) -> Self {
        bounds::check_eq!(
            ptr.len(),
            Bound::new(blocks.get()) * Bound::new(SZ) * k,
            "invalid block-transposed access",
        );

        Self { ptr, blocks, k }
    }

    /// Construct a [`View`] from a [`BlockTransposedRef`].
    #[cfg(test)]
    pub(crate) fn from_block_transposed(v: BlockTransposedRef<'a, T, SZ>) -> Self
    where
        T: Copy,
    {
        let blocks = NonZeroUsize::new(v.num_blocks()).unwrap();
        let k = DimK::new(NonZeroUsize::new(v.ncols()).unwrap());
        unsafe { Self::new(Slice::new(v.as_slice()), blocks, k) }
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

            let sub = unsafe {
                Self::new_inner(
                    self.ptr
                        .add(stride * i)
                        .truncate(stride * this_blocks.get()),
                    this_blocks,
                    self.k(),
                )
            };

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
            let panel =
                unsafe { Panel::new_inner(self.ptr.add(stride * b).truncate(stride), self.k) };
            f(panel, b);
        }
    }
}

#[cfg(test)]
impl<'a, T, const SZ: usize> View<'a, T, SZ> {
    fn checked_visit_sub_views<F>(&self, sub_blocks: NonZeroUsize, f: F)
    where
        F: FnMut(View<'_, T, SZ>, usize),
    {
        let k = DimK::from_bound(self.k());
        unsafe { self.visit_sub_views(sub_blocks, k, f) }
    }

    fn checked_visit_panels<F>(&self, f: F)
    where
        F: FnMut(Panel<'_, T, SZ>, usize),
    {
        let k = DimK::from_bound(self.k());
        unsafe { self.visit_panels(k, f) }
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
    pub(crate) unsafe fn new(ptr: Slice<'a, T>, k: DimK) -> Self {
        bounds::check_eq!(ptr.len(), SZ * k.value().get().next_multiple_of(PACK));
        unsafe { Self::new_inner(ptr, Bound::new(k.value().get())) }
    }

    unsafe fn new_inner(ptr: Slice<'a, T>, k: Bound) -> Self {
        k.with(|k| bounds::check_eq!(ptr.len(), SZ * k.next_multiple_of(PACK)));
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

#[cfg(test)]
impl<'a, T, const SZ: usize, const PACK: usize> Panel<'a, T, SZ, PACK> {
    fn checked_as_std_slice(self) -> &'a [T] {
        let len = SZ * self.k().value().next_multiple_of(PACK);
        // SAFETY: Bounds are retained under `cfg(test)`.
        unsafe { self.ptr.as_std_slice(len) }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use diskann_utils::views::{Init, Matrix, MatrixView};

    use crate::matrix_kernels::test_util::{assert_contains, panic_message_for};

    #[test]
    fn test_visit_panels() {
        for blocks in (1..50).step_by(7) {
            for k in (1..20).step_by(3) {
                let blocks = NonZeroUsize::new(blocks).unwrap();
                let k = NonZeroUsize::new(k).unwrap();
                let ctx = format_args!("blocks = {blocks}, k = {k}");

                test_visit_panels_inner::<1>(blocks, k, ctx);
                test_visit_panels_inner::<3>(blocks, k, ctx);
                test_visit_panels_inner::<4>(blocks, k, ctx);
            }
        }
    }

    fn test_visit_panels_inner<const SZ: usize>(
        blocks: NonZeroUsize,
        k: NonZeroUsize,
        ctx: std::fmt::Arguments<'_>,
    ) {
        let matrix = test_matrix(blocks.get() * SZ, k.get());
        let packed = pack::<SZ>(matrix.as_view());
        let dim_k = DimK::new(k);

        // SAFETY: `packed` contains `blocks` complete blocks of `SZ` rows and `k` columns.
        let view = unsafe { View::<_, SZ>::new(Slice::new(&packed), blocks, dim_k) };

        assert_eq!(view.blocks(), blocks, "{ctx}");
        assert_eq!(view.extent().get(), matrix.nrows(), "{ctx}");
        assert_eq!(view.k().value(), matrix.ncols(), "{ctx}");
        assert_eq!(
            view.block_stride(dim_k).value(),
            SZ * matrix.ncols(),
            "{ctx}",
        );

        let mut count = 0;
        view.checked_visit_panels(|panel, start| {
            assert_eq!(start, count, "{ctx}");
            assert_panel(panel, matrix.as_view(), start, ctx);
            count += 1;
        });

        assert_eq!(count, blocks.get(), "{ctx}");
    }

    #[test]
    fn test_visit_sub_views() {
        for blocks in (1..50).step_by(7) {
            for k in (1..20).step_by(3) {
                let blocks = NonZeroUsize::new(blocks).unwrap();
                let k = NonZeroUsize::new(k).unwrap();
                let ctx = format_args!("blocks = {blocks}, k = {k}");

                test_visit_sub_views_inner::<1>(blocks, k, ctx);
                test_visit_sub_views_inner::<3>(blocks, k, ctx);
                test_visit_sub_views_inner::<4>(blocks, k, ctx);
            }
        }
    }

    fn test_visit_sub_views_inner<const SZ: usize>(
        blocks: NonZeroUsize,
        k: NonZeroUsize,
        ctx: std::fmt::Arguments<'_>,
    ) {
        let matrix = test_matrix(blocks.get() * SZ, k.get());
        let packed = pack::<SZ>(matrix.as_view());

        // SAFETY: `packed` contains `blocks` complete blocks of `SZ` rows and `k` columns.
        let view = unsafe { View::<_, SZ>::new(Slice::new(&packed), blocks, DimK::new(k)) };

        let sub_blocks = [
            1,
            blocks.get().div_ceil(10) + 1,
            blocks.get() / 2,
            blocks.get() - 1,
            blocks.get(),
            blocks.get() + 1,
        ]
        .map(NonZeroUsize::new)
        .into_iter()
        .flatten();

        for sub_blocks in sub_blocks {
            let mut count = 0;

            view.checked_visit_sub_views(sub_blocks, |sub_view, start| {
                assert_eq!(start, count, "{ctx}");
                assert!(sub_view.blocks() <= sub_blocks, "{ctx}");
                assert_eq!(
                    sub_view.extent().get(),
                    sub_view.blocks().get() * SZ,
                    "{ctx}",
                );
                assert_eq!(sub_view.k(), view.k(), "{ctx}");

                let mut panel_count = 0;
                sub_view.checked_visit_panels(|panel, panel_start| {
                    assert_eq!(panel_start, panel_count, "{ctx}");
                    assert_panel(panel, matrix.as_view(), start + panel_start, ctx);
                    panel_count += 1;
                });

                assert_eq!(panel_count, sub_view.blocks().get(), "{ctx}");
                count += sub_view.blocks().get();
            });

            assert_eq!(count, blocks.get(), "{ctx}");
        }
    }

    fn assert_panel<const SZ: usize>(
        panel: Panel<'_, f32, SZ>,
        reference: MatrixView<'_, f32>,
        block: usize,
        ctx: std::fmt::Arguments<'_>,
    ) {
        let k = reference.ncols();
        let dim_k = DimK::new(NonZeroUsize::new(k).unwrap());
        let packed = panel.checked_as_std_slice();

        assert_eq!(panel.k().value(), k, "{ctx}");
        assert_eq!(panel.stride(dim_k).value(), SZ, "{ctx}");

        for col in 0..k {
            for row in 0..SZ {
                assert_eq!(
                    packed[col * SZ + row],
                    reference[(block * SZ + row, col)],
                    "{ctx}, block = {block}, row = {row}, col = {col}",
                );
            }
        }
    }

    #[test]
    fn test_rejects_inconsistent_lengths() {
        let data = [0u8; 25];
        let blocks = NonZeroUsize::new(2).unwrap();

        let k = DimK::new(NonZeroUsize::new(3).unwrap());

        for len in [23, 25] {
            let message = panic_message_for(|| {
                // SAFETY: The deliberate length mismatch is caught while bounds are retained.
                let _ = unsafe { View::<_, 4>::new(Slice::new(&data[..len]), blocks, k) };
            });
            assert_contains!(message, "invalid block-transposed access");
        }

        let data = [0u8; 13];
        for len in [11, 13] {
            let message = panic_message_for(|| {
                // SAFETY: The deliberate length mismatch is caught while bounds are retained.
                let _ = unsafe { Panel::<_, 4>::new(Slice::new(&data[..len]), k) };
            });
            assert_contains!(message, "equal to 12");
        }

        let data = [0u8; 17];
        for len in [15, 17] {
            let message = panic_message_for(|| {
                // SAFETY: The deliberate length mismatch is caught while bounds are retained.
                let _ = unsafe { Panel::<_, 4, 2>::new(Slice::new(&data[..len]), k) };
            });
            assert_contains!(message, "equal to 16");
        }
    }

    #[test]
    fn test_rejects_inconsistent_k() {
        let data = [0u8; 24];
        let actual_k = DimK::new(NonZeroUsize::new(3).unwrap());
        let wrong_k = DimK::new(NonZeroUsize::new(2).unwrap());

        // SAFETY: `data` contains two complete blocks of four rows and three columns.
        let view = unsafe {
            View::<_, 4>::new(Slice::new(&data), NonZeroUsize::new(2).unwrap(), actual_k)
        };

        assert_k_mismatch(|| {
            let _ = view.block_stride(wrong_k);
        });
        assert_k_mismatch(|| {
            // SAFETY: The deliberate K mismatch is caught before pointer arithmetic.
            unsafe {
                view.visit_sub_views(NonZeroUsize::new(1).unwrap(), wrong_k, |_, _| {});
            }
        });
        assert_k_mismatch(|| {
            // SAFETY: The deliberate K mismatch is caught before pointer arithmetic.
            unsafe {
                view.visit_panels(wrong_k, |_, _| {});
            }
        });

        let panel = unsafe { Panel::<_, 4>::new(Slice::new(&data[..12]), actual_k) };
        assert_k_mismatch(|| {
            let _ = panel.stride(wrong_k);
        });
    }

    fn assert_k_mismatch(f: impl FnOnce() + std::panic::UnwindSafe) {
        let message = panic_message_for(f);
        assert_contains!(message, "equal to 2");
    }

    fn test_matrix(nrows: usize, ncols: usize) -> Matrix<f32> {
        let mut value = 0.0;
        Matrix::new(
            Init(|| {
                let current = value;
                value += 1.0;
                current
            }),
            nrows,
            ncols,
        )
    }

    fn pack<const SZ: usize>(matrix: MatrixView<'_, f32>) -> Vec<f32> {
        assert!(matrix.nrows().is_multiple_of(SZ));

        let mut packed = Vec::with_capacity(matrix.as_slice().len());
        for block in 0..matrix.nrows() / SZ {
            for col in 0..matrix.ncols() {
                for row in 0..SZ {
                    packed.push(matrix[(block * SZ + row, col)]);
                }
            }
        }

        packed
    }
}

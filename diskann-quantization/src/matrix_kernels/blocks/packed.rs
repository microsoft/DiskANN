/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use crate::{
    matrix_kernels::{
        bounds::{self, Bound},
        num::{DimK, Elements},
        ptr::Slice,
    },
    multi_vector::BlockTransposedRef,
};

//--------//
// Layout //
//--------//

/// Index conversions for one block of `SZ` bands and `k` logical columns.
///
/// Columns are gathered into groups of `PACK`. Group `g` stores columns
/// `[g * PACK, (g + 1) * PACK)` of all `SZ` bands contiguously, one band after another.
/// When `PACK` does not divide `k`, the final group is padded to `PACK` columns.
///
/// Intra-block offsets do not depend on `k`; only the block stride does.
pub(in crate::matrix_kernels) struct Layout<const SZ: usize, const PACK: usize>;

impl<const SZ: usize, const PACK: usize> Layout<SZ, PACK> {
    const ASSERTIONS: () = {
        assert!(SZ > 0, "group size may not be zero");
        assert!(PACK > 0, "packing factor may not be zero");
        assert!(
            SZ.is_multiple_of(PACK),
            "the group size must be a multiple of PACK"
        );
    };

    /// The number of groups needed to hold `k` logical columns.
    pub(in crate::matrix_kernels) const fn groups(k: usize) -> usize {
        let () = Self::ASSERTIONS;
        k.div_ceil(PACK)
    }

    /// The number of physical columns in a block holding `k` logical columns.
    pub(in crate::matrix_kernels) const fn padded_k(k: usize) -> usize {
        Self::groups(k) * PACK
    }

    /// The number of physical elements in a block holding `k` logical columns.
    pub(in crate::matrix_kernels) const fn block_len(k: usize) -> usize {
        SZ * Self::padded_k(k)
    }

    /// Convert a logical `(band, col)` coordinate to its offset within a block.
    #[cfg(test)]
    pub(in crate::matrix_kernels) const fn linear(band: usize, col: usize) -> usize {
        Self::group_offset(col / PACK) + band * PACK + col % PACK
    }

    /// The offset of group `group` within a block.
    pub(in crate::matrix_kernels) const fn group_offset(group: usize) -> usize {
        let () = Self::ASSERTIONS;
        group * SZ * PACK
    }

    /// Convert an offset within a block to its `(band, col)` coordinate.
    ///
    /// The returned column may lie in the padding past the logical `k`.
    #[cfg(test)]
    pub(in crate::matrix_kernels) const fn logical(linear: usize) -> (usize, usize) {
        let () = Self::ASSERTIONS;
        let group = linear / (SZ * PACK);
        let within = linear % (SZ * PACK);
        (within / PACK, group * PACK + within % PACK)
    }
}

//------//
// View //
//------//

/// A view over packed memory.
///
/// Each block contains `SZ` bands of `k` logical columns laid out according to
/// [`Layout`]. A block occupies [`Layout::block_len`] elements. This matches the
/// physical layout of [`BlockTransposedRef`]. No block may be partially filled.
///
/// # Class Invariants
///
/// * The tracked length `ptr.len()` must be equal to `blocks * Layout::block_len(k)`.
/// * `SZ` and `PACK` may not be zero and `PACK` must divide `SZ`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct View<'a, T, const SZ: usize, const PACK: usize = 1> {
    ptr: Slice<'a, T>,
    blocks: NonZeroUsize,
    k: Bound,
}

impl<'a, T, const SZ: usize, const PACK: usize> View<'a, T, SZ, PACK> {
    /// Construct a [`View`] from a [`BlockTransposedRef`].
    ///
    /// The mapping of parameters is as follows:
    ///
    /// * The group size `SZ` is taken from the `GROUP` const-generic on [`BlockTransposedRef`].
    /// * The packing factor `PACK` is taken from the `PACK` const-generic.
    /// * `k` is [`BlockTransposedRef::ncols`].
    /// * The number of blocks is [`BlockTransposedRef::num_blocks`].
    ///
    /// Returns `None` if any of the runtime values is zero.
    pub(crate) fn from_block_transposed(v: BlockTransposedRef<'a, T, SZ, PACK>) -> Option<Self>
    where
        T: Copy,
    {
        let blocks = NonZeroUsize::new(v.num_blocks())?;
        let k = DimK::new(NonZeroUsize::new(v.ncols())?);

        // SAFETY: `BlockTransposedRef` stores exactly `num_blocks * GROUP * padded_ncols`
        // elements, where `padded_ncols == Layout::padded_k(ncols)`.
        Some(unsafe { Self::new(Slice::new(v.as_slice()), blocks, k) })
    }

    /// # Safety
    ///
    /// `ptr.len()` must be exactly equal to `blocks * Layout::<SZ, PACK>::block_len(k)`.
    pub(in crate::matrix_kernels) unsafe fn new(
        ptr: Slice<'a, T>,
        blocks: NonZeroUsize,
        k: DimK,
    ) -> Self {
        bounds::check_eq!(
            ptr.len(),
            blocks.get() * Layout::<SZ, PACK>::block_len(k.value().get()),
            "invalid block-transposed access",
        );

        // SAFETY: Inherited from caller.
        unsafe { Self::new_inner(ptr, blocks, Bound::new(k.value().get())) }
    }

    /// # Safety
    ///
    /// `ptr.len()` must be exactly equal to `blocks * Layout::<SZ, PACK>::block_len(k)`.
    unsafe fn new_inner(ptr: Slice<'a, T>, blocks: NonZeroUsize, k: Bound) -> Self {
        k.with(|k| {
            bounds::check_eq!(
                ptr.len(),
                blocks.get() * Layout::<SZ, PACK>::block_len(k),
                "invalid block-transposed access",
            );
        });

        Self { ptr, blocks, k }
    }

    /// Return the number of blocks in the [`View`].
    pub(in crate::matrix_kernels) const fn blocks(&self) -> NonZeroUsize {
        self.blocks
    }

    /// Return the logical contraction dimension of `self`.
    ///
    /// This is inherited from all constructors.
    pub(in crate::matrix_kernels) const fn k(&self) -> Bound {
        self.k
    }

    /// Return the number of physical elements in each block.
    ///
    /// `k` must be equal to the contraction dimension tracked by [`Self::k`].
    pub(in crate::matrix_kernels) fn block_stride(&self, k: DimK) -> Elements<T> {
        bounds::check_eq!(self.k, k.value());
        Elements::new(Layout::<SZ, PACK>::block_len(k.value().get()))
    }

    /// Return the number of bands stored in `self`.
    ///
    /// This is equal to `Self::blocks() * SZ`.
    #[cfg(test)]
    fn extent(&self) -> NonZeroUsize {
        const { assert!(SZ != 0) };
        self.blocks.saturating_mul(NonZeroUsize::new(SZ).unwrap())
    }

    /// Partition the view into sub-views each containing at most `sub_blocks` blocks, with
    /// the last one potentially containing fewer.
    ///
    /// Provide all sub-views to `f` in memory order. The second argument to `f` is the index
    /// of the first **block** in the sub-view within `self`.
    ///
    /// # Safety
    ///
    /// The bound [`Self::k`] must be equal to `k`.
    pub(in crate::matrix_kernels) unsafe fn visit_sub_views<F>(
        &self,
        sub_blocks: NonZeroUsize,
        k: DimK,
        mut f: F,
    ) where
        F: FnMut(View<'_, T, SZ, PACK>, usize),
    {
        let stride = self.block_stride(k);

        let mut i = 0;

        // The loop bound is a bit funky because it is setup to give us a `NonZeroUsize` for
        // free. Once it returns `None`, we know `i == self.blocks()` and we're done.
        while let Some(remaining) = NonZeroUsize::new(self.blocks().get() - i) {
            let this_blocks = remaining.min(sub_blocks);

            // SAFETY: By class invariant, `self.ptr.len() == self.blocks * stride`.
            //
            // The caller asserts that `k == self.k`.
            //
            // Since `i < self.blocks()`:
            //
            // * The pointer offset is valid.
            // * The truncation is valid.
            // * The size of the resulting slice is equal to `this_blocks * stride`.
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

    /// Partition the view into panels each containing exactly `SZ` bands.
    ///
    /// Provide all panels to `f` in memory order. The callback receives the index of
    /// each panel's first block within `self`.
    ///
    /// # Safety
    ///
    /// The bound [`Self::k`] must be equal to `k`.
    pub(in crate::matrix_kernels) unsafe fn visit_panels<F>(&self, k: DimK, mut f: F)
    where
        F: FnMut(Panel<'_, T, SZ, PACK>, usize),
    {
        let stride = self.block_stride(k);
        for b in 0..self.blocks().get() {
            // SAFETY: By class invariant, `self.ptr.len() == self.blocks * stride`.
            //
            // The caller asserts that `k == self.k`.
            //
            // Since `b < self.blocks()`:
            //
            // * The pointer offset is valid.
            // * The truncation is valid.
            // * The size of the resulting slice is equal to `stride`.
            let panel =
                unsafe { Panel::new_inner(self.ptr.add(stride * b).truncate(stride), self.k) };
            f(panel, b);
        }
    }
}

#[cfg(test)]
impl<T, const SZ: usize, const PACK: usize> View<'_, T, SZ, PACK> {
    fn checked_visit_sub_views<F>(&self, sub_blocks: NonZeroUsize, f: F)
    where
        F: FnMut(View<'_, T, SZ, PACK>, usize),
    {
        let k = DimK::from_bound(self.k());
        // SAFETY: Checked in test builds.
        unsafe { self.visit_sub_views(sub_blocks, k, f) }
    }

    fn checked_visit_panels<F>(&self, f: F)
    where
        F: FnMut(Panel<'_, T, SZ, PACK>, usize),
    {
        let k = DimK::from_bound(self.k());
        // SAFETY: Checked in test builds.
        unsafe { self.visit_panels(k, f) }
    }
}

//-------//
// Panel //
//-------//

/// A single block of `SZ` bands and `k` logical columns laid out according to [`Layout`].
///
/// # Class Invariants
///
/// The bound `ptr.len()` must be equal to `Layout::<SZ, PACK>::block_len(k)`.
#[derive(Debug, Clone, Copy)]
pub(in crate::matrix_kernels) struct Panel<'a, T, const SZ: usize, const PACK: usize = 1> {
    ptr: Slice<'a, T>,
    k: Bound,
}

impl<'a, T, const SZ: usize, const PACK: usize> Panel<'a, T, SZ, PACK> {
    /// # Safety
    ///
    /// `ptr.len()` must be equal to `Layout::<SZ, PACK>::block_len(k)`.
    #[cfg(test)]
    pub(in crate::matrix_kernels) unsafe fn new(ptr: Slice<'a, T>, k: DimK) -> Self {
        // SAFETY: Inherited from caller.
        unsafe { Self::new_inner(ptr, Bound::new(k.value().get())) }
    }

    /// # Safety
    ///
    /// `ptr.len()` must be equal to `Layout::<SZ, PACK>::block_len(k)`.
    unsafe fn new_inner(ptr: Slice<'a, T>, k: Bound) -> Self {
        k.with(|k| bounds::check_eq!(ptr.len(), Layout::<SZ, PACK>::block_len(k)));
        Self { ptr, k }
    }

    /// Return the base span of this panel as a [`Slice`].
    pub(in crate::matrix_kernels) const fn as_ptr(&self) -> Slice<'_, T> {
        self.ptr
    }

    /// Return the logical contraction dimension of `self`.
    ///
    /// This is inherited from all constructors.
    pub(in crate::matrix_kernels) const fn k(&self) -> Bound {
        self.k
    }

    /// Return group `group` as an `SZ x PACK` row-major [`Patch`].
    ///
    /// Row `r` of the patch holds columns `[group * PACK, (group + 1) * PACK)` of band `r`.
    /// For the final group, columns at or beyond [`Self::k`] are padding.
    ///
    /// # Safety
    ///
    /// `group` must be strictly less than `Layout::<SZ, PACK>::groups(k)`, where `k` is
    /// the contraction dimension tracked by [`Self::k`].
    pub(in crate::matrix_kernels) unsafe fn group(&self, group: usize) -> Patch<'a, T, SZ, PACK> {
        self.k.with(|k| {
            bounds::check_lt!(
                Bound::new(group),
                Layout::<SZ, PACK>::groups(k),
                "packed group out of bounds",
            );
        });
        // SAFETY: `group < groups(k)` by the caller's contract, so the span
        // `[group_offset(group), group_offset(group) + SZ * PACK)` lies within
        // `block_len(k)`.
        unsafe {
            Patch::new(
                self.ptr
                    .add(Elements::new(Layout::<SZ, PACK>::group_offset(group)))
                    .truncate(Elements::new(SZ * PACK)),
            )
        }
    }
}

#[cfg(test)]
impl<'a, T, const SZ: usize, const PACK: usize> Panel<'a, T, SZ, PACK> {
    fn checked_as_std_slice(self) -> &'a [T] {
        let len = Layout::<SZ, PACK>::block_len(self.k().value());
        // SAFETY: Bounds are retained under `cfg(test)`.
        unsafe { self.ptr.as_std_slice(len) }
    }

    fn checked_group(self, group: usize) -> Patch<'a, T, SZ, PACK> {
        assert!(group < Layout::<SZ, PACK>::groups(self.k().value()));
        // SAFETY: Checked immediately above.
        unsafe { self.group(group) }
    }
}

//-------//
// Patch //
//-------//

/// A fixed-size `ROWS x COLS` row-major view.
///
/// # Class Invariants
///
/// The underlying span contains exactly `ROWS * COLS` elements.
#[derive(Debug, Clone, Copy)]
pub(in crate::matrix_kernels) struct Patch<'a, T, const ROWS: usize, const COLS: usize> {
    ptr: Slice<'a, T>,
}

impl<'a, T, const ROWS: usize, const COLS: usize> Patch<'a, T, ROWS, COLS> {
    /// # Safety
    ///
    /// `ptr` must span exactly `ROWS * COLS` elements.
    unsafe fn new(ptr: Slice<'a, T>) -> Self {
        bounds::check_eq!(ptr.len(), ROWS * COLS, "invalid patch length");
        Self { ptr }
    }

    /// Construct a [`Patch`] over a row-major array.
    #[cfg(test)]
    pub(in crate::matrix_kernels) fn from_array(values: &'a [[T; COLS]; ROWS]) -> Self {
        // SAFETY: The flattened array has exactly `ROWS * COLS` elements.
        unsafe { Self::new(Slice::new(values.as_flattened())) }
    }

    /// Return the base span of this patch as a [`Slice`] of `ROWS * COLS` elements.
    pub(in crate::matrix_kernels) const fn as_ptr(&self) -> Slice<'a, T> {
        self.ptr
    }

    /// Return the patch as a row-major array.
    pub(in crate::matrix_kernels) fn as_array(&self) -> &'a [[T; COLS]; ROWS] {
        // SAFETY: By class invariant, `ptr` spans exactly `ROWS * COLS` contiguous
        // elements, which has the same layout as `[[T; COLS]; ROWS]`.
        unsafe { &*self.ptr.as_ptr().cast::<[[T; COLS]; ROWS]>() }
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

                test_visit_panels_inner::<1, 1>(blocks, k, ctx);
                test_visit_panels_inner::<3, 1>(blocks, k, ctx);
                test_visit_panels_inner::<4, 1>(blocks, k, ctx);
            }
        }

        // `PACK > 1` with every residue of `k` modulo `PACK`.
        let ks: &[usize] = if cfg!(miri) {
            &[1, 3, 5, 9]
        } else {
            &[
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 23, 24, 25,
            ]
        };
        for blocks in (1..20).step_by(if cfg!(miri) { 9 } else { 3 }) {
            for &k in ks {
                let blocks = NonZeroUsize::new(blocks).unwrap();
                let k = NonZeroUsize::new(k).unwrap();
                let ctx = format_args!("blocks = {blocks}, k = {k}");

                test_visit_panels_inner::<4, 2>(blocks, k, ctx);
                test_visit_panels_inner::<4, 4>(blocks, k, ctx);
                test_visit_panels_inner::<8, 4>(blocks, k, ctx);
                test_visit_panels_inner::<16, 4>(blocks, k, ctx);
                test_visit_panels_inner::<8, 8>(blocks, k, ctx);
                test_visit_panels_inner::<16, 8>(blocks, k, ctx);
            }
        }
    }

    fn test_visit_panels_inner<const SZ: usize, const PACK: usize>(
        blocks: NonZeroUsize,
        k: NonZeroUsize,
        ctx: std::fmt::Arguments<'_>,
    ) {
        let matrix = test_matrix(blocks.get() * SZ, k.get());
        let packed = pack::<SZ, PACK>(matrix.as_view());
        let dim_k = DimK::new(k);

        // SAFETY: `packed` contains `blocks` complete blocks of `SZ` rows and `k` columns.
        let view = unsafe { View::<_, SZ, PACK>::new(Slice::new(&packed), blocks, dim_k) };

        assert_eq!(view.blocks(), blocks, "{ctx}");
        assert_eq!(view.extent().get(), matrix.nrows(), "{ctx}");
        assert_eq!(view.k().value(), matrix.ncols(), "{ctx}");
        assert_eq!(
            view.block_stride(dim_k).value(),
            SZ * k.get().next_multiple_of(PACK),
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

                test_visit_sub_views_inner::<1, 1>(blocks, k, ctx);
                test_visit_sub_views_inner::<3, 1>(blocks, k, ctx);
                test_visit_sub_views_inner::<4, 1>(blocks, k, ctx);
            }
        }

        let ks: &[usize] = if cfg!(miri) {
            &[1, 3, 5, 9]
        } else {
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17]
        };
        for blocks in (1..20).step_by(if cfg!(miri) { 9 } else { 3 }) {
            for &k in ks {
                let blocks = NonZeroUsize::new(blocks).unwrap();
                let k = NonZeroUsize::new(k).unwrap();
                let ctx = format_args!("blocks = {blocks}, k = {k}");

                test_visit_sub_views_inner::<4, 4>(blocks, k, ctx);
                test_visit_sub_views_inner::<16, 4>(blocks, k, ctx);
                test_visit_sub_views_inner::<16, 8>(blocks, k, ctx);
            }
        }
    }

    fn test_visit_sub_views_inner<const SZ: usize, const PACK: usize>(
        blocks: NonZeroUsize,
        k: NonZeroUsize,
        ctx: std::fmt::Arguments<'_>,
    ) {
        let matrix = test_matrix(blocks.get() * SZ, k.get());
        let packed = pack::<SZ, PACK>(matrix.as_view());

        // SAFETY: `packed` contains `blocks` complete blocks of `SZ` rows and `k` columns.
        let view = unsafe { View::<_, SZ, PACK>::new(Slice::new(&packed), blocks, DimK::new(k)) };

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
                assert_eq!(sub_view.k().value(), view.k().value(), "{ctx}");

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

    fn assert_panel<const SZ: usize, const PACK: usize>(
        panel: Panel<'_, f32, SZ, PACK>,
        reference: MatrixView<'_, f32>,
        block: usize,
        ctx: std::fmt::Arguments<'_>,
    ) {
        let k = reference.ncols();
        let packed = panel.checked_as_std_slice();

        assert_eq!(panel.k().value(), k, "{ctx}");

        assert_eq!(packed.len(), SZ * k.next_multiple_of(PACK), "{ctx}");
        for (index, &value) in packed.iter().enumerate() {
            let (row, col) = Layout::<SZ, PACK>::logical(index);
            let expected = if col < k {
                reference[(block * SZ + row, col)]
            } else {
                PADDING
            };
            assert_eq!(
                value, expected,
                "{ctx}, block = {block}, row = {row}, col = {col}",
            );
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

    /// Sentinel for padded columns in manually packed test data.
    const PADDING: f32 = -1.0;

    /// Pack `matrix` by enumerating the documented layout order directly:
    /// block, then group, then band, then lane.
    fn pack<const SZ: usize, const PACK: usize>(matrix: MatrixView<'_, f32>) -> Vec<f32> {
        assert!(matrix.nrows().is_multiple_of(SZ));
        let k = matrix.ncols();

        let mut packed = Vec::new();
        for block in 0..matrix.nrows() / SZ {
            for group in 0..k.div_ceil(PACK) {
                for row in 0..SZ {
                    for lane in 0..PACK {
                        let col = group * PACK + lane;
                        packed.push(if col < k {
                            matrix[(block * SZ + row, col)]
                        } else {
                            PADDING
                        });
                    }
                }
            }
        }

        packed
    }

    #[test]
    fn test_layout_conversions() {
        check_layout::<1, 1>();
        check_layout::<4, 1>();
        check_layout::<4, 2>();
        check_layout::<8, 4>();
        check_layout::<16, 4>();
        check_layout::<16, 8>();
    }

    fn check_layout<const SZ: usize, const PACK: usize>() {
        for k in 1..=(3 * PACK + 1) {
            let groups = k.div_ceil(PACK);
            assert_eq!(Layout::<SZ, PACK>::groups(k), groups);
            assert_eq!(Layout::<SZ, PACK>::padded_k(k), groups * PACK);
            assert_eq!(Layout::<SZ, PACK>::block_len(k), SZ * groups * PACK);

            // Every physical offset, in the documented enumeration order, maps to a unique
            // coordinate and back.
            let mut linear = 0;
            for group in 0..groups {
                for band in 0..SZ {
                    for lane in 0..PACK {
                        let col = group * PACK + lane;
                        assert_eq!(Layout::<SZ, PACK>::linear(band, col), linear);
                        assert_eq!(Layout::<SZ, PACK>::logical(linear), (band, col));
                        linear += 1;
                    }
                }
            }
            assert_eq!(linear, Layout::<SZ, PACK>::block_len(k));
        }
    }

    #[test]
    fn test_group_patches() {
        check_group_patches::<4, 1>();
        check_group_patches::<4, 2>();
        check_group_patches::<8, 4>();
        check_group_patches::<16, 4>();
        check_group_patches::<16, 8>();
    }

    fn check_group_patches<const SZ: usize, const PACK: usize>() {
        for k in (1..=(4 * PACK + 1)).step_by(if cfg!(miri) { PACK + 1 } else { 1 }) {
            let k = NonZeroUsize::new(k).unwrap();
            let blocks = NonZeroUsize::new(3).unwrap();
            let matrix = test_matrix(blocks.get() * SZ, k.get());
            let packed = pack::<SZ, PACK>(matrix.as_view());

            // SAFETY: `packed` holds `blocks` complete blocks of `SZ` rows and `k` columns.
            let view =
                unsafe { View::<_, SZ, PACK>::new(Slice::new(&packed), blocks, DimK::new(k)) };

            view.checked_visit_panels(|panel, block| {
                for group in 0..Layout::<SZ, PACK>::groups(k.get()) {
                    let patch = panel.checked_group(group);
                    assert_eq!(patch.as_ptr().len().value(), SZ * PACK);
                    for (row, values) in patch.as_array().iter().enumerate() {
                        for (lane, &value) in values.iter().enumerate() {
                            let col = group * PACK + lane;
                            let expected = if col < k.get() {
                                matrix[(block * SZ + row, col)]
                            } else {
                                PADDING
                            };
                            assert_eq!(
                                value, expected,
                                "SZ = {SZ}, PACK = {PACK}, k = {k}, group = {group}",
                            );
                        }
                    }
                }
            });
        }
    }

    #[test]
    fn test_group_out_of_bounds() {
        let data = [0u8; 16];
        // Three logical columns occupy two groups of two.
        let k = DimK::new(NonZeroUsize::new(3).unwrap());
        // SAFETY: `data` spans `4 * padded_k(3) = 16` elements.
        let panel = unsafe { Panel::<_, 4, 2>::new(Slice::new(&data), k) };
        let message = panic_message_for(|| {
            // SAFETY: The deliberate out-of-bounds group is caught under `cfg(test)`.
            let _ = unsafe { panel.group(2) };
        });
        assert_contains!(message, "packed group out of bounds");
    }

    #[test]
    fn test_from_block_transposed_uses_logical_columns() {
        check_from_block_transposed::<4, 1>();
        check_from_block_transposed::<4, 2>();
        check_from_block_transposed::<8, 4>();
        check_from_block_transposed::<16, 8>();
    }

    fn check_from_block_transposed<const SZ: usize, const PACK: usize>() {
        use crate::multi_vector::BlockTransposed;

        for nrows in [1, SZ - 1, SZ, SZ + 1, 2 * SZ + 1]
            .into_iter()
            .filter(|&n| n > 0)
        {
            for ncols in 1..(3 * PACK + 2) {
                let mut matrix = BlockTransposed::<f32, SZ, PACK>::new(nrows, ncols);
                for row in 0..nrows {
                    let mut row_mut = matrix.get_row_mut(row).unwrap();
                    for col in 0..ncols {
                        row_mut.set(col, (row * 1000 + col) as f32);
                    }
                }

                let view = View::<_, SZ, PACK>::from_block_transposed(matrix.as_view()).unwrap();

                assert_eq!(view.blocks().get(), nrows.div_ceil(SZ));
                assert_eq!(view.k().value(), ncols);

                view.checked_visit_panels(|panel, block| {
                    let packed = panel.checked_as_std_slice();
                    for (index, &value) in packed.iter().enumerate() {
                        let (row, col) = Layout::<SZ, PACK>::logical(index);
                        let logical = block * SZ + row;
                        let expected = if logical < nrows && col < ncols {
                            (logical * 1000 + col) as f32
                        } else {
                            0.0
                        };
                        assert_eq!(value, expected, "nrows = {nrows}, ncols = {ncols}");
                    }
                });
            }
        }
    }
}

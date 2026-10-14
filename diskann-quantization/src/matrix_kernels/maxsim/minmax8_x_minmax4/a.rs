/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Borrowed packed A panels with synchronized compensation and valid-row extents.

use super::layout::{BLOCK, Layout};
use crate::{
    matrix_kernels::{
        blocks::packed,
        bounds::{self, Bound},
        num::Bytes,
        ptr::Slice,
    },
    minmax::MinMaxCompensation,
    multi_vector::BlockTransposedRef,
};

#[derive(Clone, Copy)]
pub(crate) struct AView<'a, const MR: usize, const PACK: usize> {
    values: packed::View<'a, u8, MR, PACK>,
    compensation: &'a [[MinMaxCompensation; MR]],
    rows: usize,
    layout: &'a Layout,
}

impl<'a, const MR: usize, const PACK: usize> AView<'a, MR, PACK> {
    /// Bind already-prepared values to row-matched compensation.
    ///
    /// The owner must establish the even/odd ordering and zero row/column padding.
    /// This checks extents, not the contents of the prepared buffer.
    pub(crate) fn new(
        values: BlockTransposedRef<'a, u8, MR, PACK>,
        compensation: &'a [[MinMaxCompensation; MR]],
        layout: &'a Layout,
    ) -> Option<Self> {
        const { assert!(BLOCK.is_multiple_of(PACK)) };
        assert_eq!(values.ncols(), layout.padded_dim(), "query layout mismatch");
        assert_eq!(
            values.num_blocks(),
            compensation.len(),
            "query metadata mismatch"
        );
        Some(Self {
            values: packed::View::from_block_transposed(values)?,
            compensation,
            rows: values.nrows(),
            layout,
        })
    }

    pub(super) fn rows(self) -> usize {
        self.rows
    }
    pub(super) fn layout(self) -> &'a Layout {
        self.layout
    }

    /// Working-set bytes per panel, including row/column padding and compensation.
    pub(super) fn panel_bytes(self) -> Bytes {
        let values = self.values.block_stride(self.layout.k()).bytes();
        Bytes::new(values.value() + std::mem::size_of::<[MinMaxCompensation; MR]>())
    }

    #[inline(always)]
    pub(super) fn visit_panels(self, mut visit: impl FnMut(APanel<'_, MR, PACK>, usize)) {
        // SAFETY: Each packed panel has matching metadata and zero row/column padding.
        unsafe {
            self.values.visit_panels(self.layout.k(), |values, block| {
                visit(
                    APanel {
                        values,
                        compensation: &self.compensation[block],
                        valid_rows: (self.rows - block * MR).min(MR),
                        layout: self.layout,
                    },
                    block * MR,
                );
            });
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct APanel<'a, const MR: usize, const PACK: usize> {
    values: packed::Panel<'a, u8, MR, PACK>,
    compensation: &'a [MinMaxCompensation; MR],
    valid_rows: usize,
    layout: &'a Layout,
}

impl<const MR: usize, const PACK: usize> APanel<'_, MR, PACK> {
    pub(super) fn layout(&self) -> &Layout {
        self.layout
    }

    pub(super) fn valid_rows(&self) -> usize {
        self.valid_rows
    }
    pub(super) fn compensation(&self) -> &[MinMaxCompensation; MR] {
        self.compensation
    }

    /// # Safety
    ///
    /// group < K / PACK. Query padding makes every group a complete MR x PACK patch.
    pub(super) unsafe fn group(&self, group: usize) -> Slice<'_, u8> {
        bounds::check_lt!(
            Bound::new(group),
            self.values.packs(self.layout.k()),
            "query group out of bounds"
        );
        // SAFETY: The caller provides an in-bounds group; each spans MR * PACK elements.
        unsafe {
            self.values
                .as_ptr()
                .add(self.values.pack_stride() * group)
                .truncate(self.values.pack_stride())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_vector::BlockTransposed;

    fn check_panel_bytes<const MR: usize, const PACK: usize>() {
        for dim in [1, 63, 64, 65, 129] {
            let layout = Layout::new(dim).unwrap();
            for rows in [1, MR, MR + 1] {
                let values = BlockTransposed::<u8, MR, PACK>::new(rows, layout.padded_dim());
                let metadata = vec![[MinMaxCompensation::default(); MR]; rows.div_ceil(MR)];
                let view = AView::new(values.as_view(), &metadata, &layout).unwrap();
                let bytes = std::mem::size_of_val(values.as_slice())
                    + std::mem::size_of_val(metadata.as_slice());
                assert_eq!(view.panel_bytes().value() * metadata.len(), bytes);
            }
        }
    }

    #[test]
    fn panel_bytes_include_padding_and_compensation() {
        check_panel_bytes::<8, 4>();
        check_panel_bytes::<16, 4>();
        check_panel_bytes::<16, 8>();
    }

    #[test]
    fn query_groups_keep_geometry_and_reject_invalid_indices() {
        use crate::matrix_kernels::test_util::panic_message_for;

        let layout = Layout::new(128).unwrap();
        let mut values = BlockTransposed::<u8, 8, 4>::new(8, 128);
        for row in 0..8 {
            for col in 0..128 {
                values
                    .get_row_mut(row)
                    .unwrap()
                    .set(col, (row * 17 + col) as u8);
            }
        }
        let metadata = [[MinMaxCompensation::default(); 8]];
        let view = AView::new(values.as_view(), &metadata, &layout).unwrap();
        view.visit_panels(|panel, _| {
            for group in 0..32 {
                // SAFETY: All 32 complete groups contain MR * PACK = 32 bytes.
                let values = unsafe { panel.group(group).as_std_slice(32) };
                let expected: [[u8; 4]; 8] = core::array::from_fn(|row| {
                    core::array::from_fn(|i| (row * 17 + group * 4 + i) as u8)
                });
                assert_eq!(values, expected.as_flattened());
            }
            let message = panic_message_for(|| {
                // SAFETY: Under cfg(test), the logical group check precedes pointer arithmetic.
                unsafe { panel.group(32) };
            });
            assert!(message.contains("query group out of bounds"), "{message}");
        });
    }
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Shared query/decoder protocol: within each 64-dimensional block, even dimensions
//! precede odd dimensions. Padding is zero; only the original dimension is compensated.

use std::num::NonZeroUsize;

use crate::{
    matrix_kernels::{blocks::packed, num::DimK},
    multi_vector::BlockTransposed,
};

#[derive(Debug, Clone, Copy)]
pub(super) struct EvenOdd64Layout {
    dim: usize,
    padded: usize,
}

impl EvenOdd64Layout {
    pub(super) const BLOCK: usize = 64;
    pub(super) const PACKED_BYTES: usize = Self::BLOCK / 2;

    #[expect(
        clippy::expect_used,
        reason = "unrepresentable dimensions fail before allocation"
    )]
    pub(super) fn new(dim: usize) -> Self {
        let padded = dim
            .checked_next_multiple_of(Self::BLOCK)
            .expect("EvenOdd64Layout padded dimension overflow");
        Self { dim, padded }
    }

    pub(super) fn dim(self) -> usize {
        self.dim
    }

    pub(super) fn padded(self) -> usize {
        self.padded
    }

    pub(super) fn source_dimension(position: usize) -> usize {
        let within = position % Self::BLOCK;
        position / Self::BLOCK * Self::BLOCK
            + 2 * (within % Self::PACKED_BYTES)
            + within / Self::PACKED_BYTES
    }
}

/// Only row-wise writes in original dimension order can populate this storage.
#[derive(Debug)]
pub(crate) struct PackedQuery<const MR: usize, const PACK: usize> {
    values: BlockTransposed<u8, MR, PACK>,
    layout: EvenOdd64Layout,
}

impl<const MR: usize, const PACK: usize> PackedQuery<MR, PACK> {
    pub(crate) fn new(rows: usize, dim: usize) -> Self {
        const {
            assert!(PACK > 0 && EvenOdd64Layout::BLOCK.is_multiple_of(PACK));
        }
        let layout = EvenOdd64Layout::new(dim);
        Self {
            values: BlockTransposed::new(rows, layout.padded()),
            layout,
        }
    }

    #[expect(
        clippy::expect_used,
        reason = "invalid internal row indices must fail explicitly"
    )]
    pub(crate) fn set_row(&mut self, row: usize, values: &[u8]) {
        assert_eq!(
            values.len(),
            self.layout.dim(),
            "query row dimension mismatch"
        );
        let mut output = self
            .values
            .get_row_mut(row)
            .expect("query row out of bounds");
        for position in 0..self.layout.padded() {
            let source = EvenOdd64Layout::source_dimension(position);
            output.set(position, values.get(source).copied().unwrap_or(0));
        }
    }

    pub(crate) fn nrows(&self) -> usize {
        self.values.nrows()
    }

    /// Empty queries and zero-dimensional queries do not need a driver.
    pub(crate) fn as_view(&self) -> Option<PackedQueryView<'_, MR, PACK>> {
        Some(PackedQueryView {
            values: packed::View::from_block_transposed(self.values.as_view())?,
            layout: self.layout,
            k: DimK::new(NonZeroUsize::new(self.layout.padded())?),
            nrows: self.nrows(),
        })
    }
}

/// A nonempty query with proven [`EvenOdd64Layout`] ordering and zero row/column padding.
#[derive(Clone, Copy)]
pub(crate) struct PackedQueryView<'a, const MR: usize, const PACK: usize> {
    values: packed::View<'a, u8, MR, PACK>,
    layout: EvenOdd64Layout,
    k: DimK,
    nrows: usize,
}

impl<'a, const MR: usize, const PACK: usize> PackedQueryView<'a, MR, PACK> {
    pub(super) fn values(self) -> packed::View<'a, u8, MR, PACK> {
        self.values
    }

    pub(super) fn layout(self) -> EvenOdd64Layout {
        self.layout
    }

    pub(super) fn k(self) -> DimK {
        self.k
    }

    pub(super) fn nrows(self) -> usize {
        self.nrows
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    pub(in crate::matrix_kernels::maxsim::minmax8_x_minmax4) const DIMS: &[usize] = &[
        0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 127, 128, 129, 249, 250, 255, 256, 257, 1024, 1025,
    ];

    fn check<const MR: usize, const PACK: usize>() {
        for &dim in DIMS {
            for rows in [0, 1, MR - 1, MR, MR + 1, 3 * MR + 1] {
                let mut storage = PackedQuery::<MR, PACK>::new(rows, dim);
                for row in 0..rows {
                    let values: Vec<_> =
                        (0..dim).map(|d| ((row * 17 + d) % 255 + 1) as u8).collect();
                    storage.set_row(row, &values);
                }
                let k = dim.div_ceil(64) * 64;
                assert_eq!(storage.values.ncols(), k);
                let mut expected = vec![0; rows.div_ceil(MR) * MR * k];
                // Derive destinations from original dimensions, independently of the
                // production mapping from packed position back to original dimension.
                for row in 0..rows {
                    for d in 0..dim {
                        let p = (d / 64) * 64 + (d % 2) * 32 + (d % 64) / 2;
                        let offset = (row / MR) * MR * k
                            + (p / PACK) * MR * PACK
                            + (row % MR) * PACK
                            + p % PACK;
                        expected[offset] = ((row * 17 + d) % 255 + 1) as u8;
                    }
                }
                assert_eq!(
                    storage.values.as_slice(),
                    expected,
                    "rows={rows}, dim={dim}"
                );
                assert_eq!(storage.as_view().is_none(), rows == 0 || dim == 0);
            }
        }
    }

    #[test]
    fn query_layout_and_padding() {
        check::<8, 4>();
        check::<16, 4>();
        check::<16, 8>();
    }

    #[test]
    #[should_panic(expected = "EvenOdd64Layout padded dimension overflow")]
    fn padding_overflow() {
        EvenOdd64Layout::new(usize::MAX);
    }

    #[test]
    #[should_panic(expected = "query row dimension mismatch")]
    fn invalid_row_dimension() {
        PackedQuery::<8, 4>::new(1, 9).set_row(0, &[0; 8]);
    }
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Query storage and the dimension order shared with decoded document panels.

use std::num::NonZeroUsize;

use crate::{
    matrix_kernels::{
        blocks::packed,
        num::{Bytes, DimK},
    },
    minmax::{MinMaxCompensation, MinMaxMeta},
    multi_vector::{BlockTransposed, MatRef, block_transposed::BlockLayout},
};

/// Stateless dimension order: each block's even dimensions precede its odd dimensions.
pub(super) struct EvenOdd;

impl EvenOdd {
    pub(super) const BLOCK: usize = 64;
    pub(super) const PACKED_BYTES: usize = Self::BLOCK / 2;

    #[expect(
        clippy::expect_used,
        reason = "unrepresentable dimensions fail before allocation"
    )]
    pub(super) fn padded(dim: usize) -> usize {
        dim.checked_next_multiple_of(Self::BLOCK)
            .expect("query padded dimension overflow")
    }

    pub(super) const fn position(d: usize) -> usize {
        d / Self::BLOCK * Self::BLOCK + d % 2 * Self::PACKED_BYTES + d % Self::BLOCK / 2
    }

    fn query_index<const MR: usize, const PACK: usize>(row: usize, d: usize, k: usize) -> usize {
        BlockLayout::<MR, PACK>::linear_index(row, Self::position(d), k)
    }
}

/// Structure-of-arrays coefficients for one complete query panel.
#[derive(Debug, Clone, Copy)]
pub(super) struct QueryCompensation<const MR: usize> {
    pub(super) scale: [f32; MR],
    pub(super) bias: [f32; MR],
    pub(super) scaled_sum: [f32; MR],
}

impl<const MR: usize> Default for QueryCompensation<MR> {
    fn default() -> Self {
        Self {
            scale: [0.0; MR],
            bias: [0.0; MR],
            scaled_sum: [0.0; MR],
        }
    }
}

/// Values and compensation are populated together and exposed only through paired panels.
#[derive(Debug)]
pub(crate) struct PackedQuery<const MR: usize, const PACK: usize> {
    values: BlockTransposed<u8, MR, PACK>,
    compensation: Vec<QueryCompensation<MR>>,
    dim: usize,
}

impl<const MR: usize, const PACK: usize> PackedQuery<MR, PACK> {
    pub(crate) fn new(query: MatRef<'_, MinMaxMeta<8>>) -> Self {
        let mut result = Self::empty(query.num_vectors(), query.repr().intrinsic_dim());
        for (i, row) in query.rows().enumerate() {
            result.set_row(i, row.vector().as_slice(), row.meta());
        }
        result
    }

    fn empty(rows: usize, dim: usize) -> Self {
        const {
            assert!(PACK > 0 && EvenOdd::BLOCK.is_multiple_of(PACK));
        }
        let k = EvenOdd::padded(dim);
        let panels = rows.div_ceil(MR);
        Self {
            values: BlockTransposed::new(rows, k),
            compensation: vec![QueryCompensation::default(); panels],
            dim,
        }
    }

    fn set_row(&mut self, row: usize, values: &[u8], meta: MinMaxCompensation) {
        assert!(row < self.nrows(), "query row out of bounds");
        assert_eq!(values.len(), self.dim, "query row dimension mismatch");
        let k = self.values.ncols();
        let output = self.values.as_mut_slice();
        for (d, &value) in values.iter().enumerate() {
            output[EvenOdd::query_index::<MR, PACK>(row, d, k)] = value;
        }
        let block = &mut self.compensation[row / MR];
        block.scale[row % MR] = meta.a;
        block.bias[row % MR] = meta.b;
        block.scaled_sum[row % MR] = meta.n;
    }

    pub(crate) fn nrows(&self) -> usize {
        self.values.nrows()
    }

    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    /// Empty or zero-dimensional queries are handled before entering the driver.
    pub(crate) fn as_view(&self) -> Option<PackedQueryView<'_, MR, PACK>> {
        Some(PackedQueryView {
            values: packed::View::from_block_transposed(self.values.as_view())?,
            compensation: &self.compensation,
            k: DimK::new(NonZeroUsize::new(self.values.ncols())?),
            dim: self.dim,
            nrows: self.nrows(),
        })
    }
}

/// A nonempty query with zero padding and a contraction dimension divisible by PACK.
#[derive(Clone, Copy)]
pub(crate) struct PackedQueryView<'a, const MR: usize, const PACK: usize> {
    values: packed::View<'a, u8, MR, PACK>,
    compensation: &'a [QueryCompensation<MR>],
    k: DimK,
    dim: usize,
    nrows: usize,
}

impl<const MR: usize, const PACK: usize> PackedQueryView<'_, MR, PACK> {
    pub(super) fn k(self) -> DimK {
        self.k
    }

    pub(super) fn dim(self) -> usize {
        self.dim
    }

    pub(super) fn nrows(self) -> usize {
        self.nrows
    }

    #[expect(
        clippy::expect_used,
        reason = "working-set overflow fails before allocating scratch"
    )]
    pub(super) fn panel_bytes(self) -> Bytes {
        Bytes::new(
            self.values
                .block_stride(self.k())
                .bytes()
                .value()
                .checked_add(std::mem::size_of::<QueryCompensation<MR>>())
                .expect("query panel size overflow"),
        )
    }

    pub(super) fn visit_panels(self, mut visit: impl FnMut(APanel<'_, MR, PACK>, usize)) {
        // SAFETY: The packed values and K come from the same validated query.
        unsafe {
            self.values.visit_panels(self.k(), |values, block| {
                let start = block * MR;
                visit(
                    APanel {
                        values,
                        compensation: &self.compensation[block],
                        k: self.k,
                        dim: self.dim,
                        valid_rows: (self.nrows - start).min(MR),
                    },
                    start,
                );
            });
        }
    }
}

/// One query panel in EvenOdd order, including its matching compensation and row tail.
#[derive(Clone, Copy)]
pub(super) struct APanel<'a, const MR: usize, const PACK: usize> {
    values: packed::Panel<'a, u8, MR, PACK>,
    compensation: &'a QueryCompensation<MR>,
    k: DimK,
    dim: usize,
    valid_rows: usize,
}

impl<'a, const MR: usize, const PACK: usize> APanel<'a, MR, PACK> {
    pub(super) fn k(self) -> DimK {
        self.k
    }
    pub(super) fn dim(self) -> usize {
        self.dim
    }
    pub(super) fn valid_rows(self) -> usize {
        self.valid_rows
    }
    pub(super) fn compensation(self) -> &'a QueryCompensation<MR> {
        self.compensation
    }

    /// # Safety
    ///
    /// `group < self.k().value().get() / PACK`.
    pub(super) unsafe fn group(self, group: usize) -> packed::Patch<'a, u8, MR, PACK> {
        // SAFETY: Inherited from caller; K is a multiple of PACK.
        unsafe { self.values.group(group) }
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    pub(in crate::matrix_kernels::maxsim::minmax8_x_minmax4) const DIMS: &[usize] = &[
        0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 127, 128, 129, 249, 250, 255, 256, 257, 1024, 1025,
    ];

    pub(in crate::matrix_kernels::maxsim::minmax8_x_minmax4) fn query<
        const MR: usize,
        const PACK: usize,
    >(
        rows: usize,
        dim: usize,
        mut values: impl FnMut(usize, usize) -> u8,
        mut meta: impl FnMut(usize) -> MinMaxCompensation,
    ) -> PackedQuery<MR, PACK> {
        let mut query = PackedQuery::empty(rows, dim);
        for row in 0..rows {
            let values: Vec<_> = (0..dim).map(|d| values(row, d)).collect();
            query.set_row(row, &values, meta(row));
        }
        query
    }

    pub(in crate::matrix_kernels::maxsim::minmax8_x_minmax4) fn panel<
        'a,
        const MR: usize,
        const PACK: usize,
    >(
        values: &'a [u8],
        compensation: &'a QueryCompensation<MR>,
        dim: usize,
    ) -> APanel<'a, MR, PACK> {
        let k = dim.next_multiple_of(EvenOdd::BLOCK);
        assert!(k.is_multiple_of(PACK));
        assert_eq!(values.len(), MR * k);
        let k = DimK::new(NonZeroUsize::new(k).unwrap());
        APanel {
            // SAFETY: The test supplies an explicitly sized packed panel.
            values: unsafe {
                packed::Panel::new(crate::matrix_kernels::ptr::Slice::new(values), k)
            },
            compensation,
            k,
            dim,
            valid_rows: MR,
        }
    }

    fn check<const MR: usize, const PACK: usize>() {
        for &dim in DIMS {
            for rows in [0, 1, MR - 1, MR, MR + 1, 3 * MR + 1] {
                let mut storage = PackedQuery::<MR, PACK>::empty(rows, dim);
                let k = dim.div_ceil(64) * 64;
                for generation in [0, 127] {
                    for row in 0..rows {
                        let values: Vec<_> = (0..dim)
                            .map(|d| ((row * 17 + d + generation) % 255 + 1) as u8)
                            .collect();
                        storage.set_row(
                            row,
                            &values,
                            MinMaxCompensation {
                                a: row as f32 + 1.0,
                                b: -2.0,
                                n: generation as f32,
                                ..Default::default()
                            },
                        );
                    }
                    let mut expected = vec![0; rows.div_ceil(MR) * MR * k];
                    for row in 0..rows {
                        for d in 0..dim {
                            let offset =
                                BlockLayout::<MR, PACK>::linear_index(row, EvenOdd::position(d), k);
                            expected[offset] = ((row * 17 + d + generation) % 255 + 1) as u8;
                        }
                    }
                    assert_eq!(storage.values.as_slice(), expected);
                    for (block, meta) in storage.compensation.iter().enumerate() {
                        for lane in 0..MR {
                            let row = block * MR + lane;
                            assert_eq!(
                                meta.scale[lane],
                                if row < rows { row as f32 + 1.0 } else { 0.0 }
                            );
                            assert_eq!(meta.bias[lane], if row < rows { -2.0 } else { 0.0 });
                            assert_eq!(
                                meta.scaled_sum[lane],
                                if row < rows { generation as f32 } else { 0.0 }
                            );
                        }
                    }
                    if let Some(view) = storage.as_view() {
                        let mut visited = 0;
                        view.visit_panels(|panel, start| {
                            assert_eq!(start, visited);
                            assert_eq!(panel.k().value().get(), k);
                            assert_eq!(panel.dim(), dim);
                            assert!(std::ptr::eq(
                                panel.compensation(),
                                &storage.compensation[start / MR]
                            ));
                            visited += panel.valid_rows();
                        });
                        assert_eq!(visited, rows);
                    } else {
                        assert!(rows == 0 || dim == 0);
                    }
                }
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
    fn even_odd_positions() {
        for block in 0..3 {
            let base = block * EvenOdd::BLOCK;
            let order: Vec<_> = (base..base + EvenOdd::BLOCK)
                .step_by(2)
                .chain((base + 1..base + EvenOdd::BLOCK).step_by(2))
                .collect();
            for (position, d) in order.into_iter().enumerate() {
                assert_eq!(EvenOdd::position(d), base + position);
            }
        }
    }

    #[test]
    #[should_panic(expected = "query padded dimension overflow")]
    fn padding_overflow() {
        EvenOdd::padded(usize::MAX);
    }

    #[test]
    #[should_panic(expected = "query row dimension mismatch")]
    fn invalid_row_dimension() {
        PackedQuery::<8, 4>::empty(1, 9).set_row(0, &[0; 8], MinMaxCompensation::default());
    }
}

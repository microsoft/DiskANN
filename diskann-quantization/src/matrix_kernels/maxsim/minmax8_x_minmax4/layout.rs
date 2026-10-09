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
    multi_vector::{
        BlockTransposed, MatRef, Overflow,
        block_transposed::{ColumnLayout, ColumnOrder, EvenOdd},
    },
};

type Order = EvenOdd<64>;
pub(super) const BLOCK: usize = <Order as ColumnOrder>::BLOCK;

/// Geometry shared by query packing and decoded document views.
///
/// The 64-dimensional policy preserves the existing MinMax8 format: one pair of
/// 32-byte nibble channels per block. It is not an ISA-wide optimality claim.
/// Decoders only expand bytes into channels; this layout determines their destinations.
#[derive(Debug)]
pub(super) struct Layout {
    columns: ColumnLayout<Order>,
    panel_bytes: Bytes,
    decoded_row_bytes: Bytes,
    max_b_rows: usize,
}

impl Layout {
    pub(super) fn new<const MR: usize, const PACK: usize, const NR: usize>(
        dim: usize,
    ) -> Result<Self, Overflow> {
        const {
            assert!(MR > 0 && PACK > 0 && NR > 0);
            assert!(MR.is_multiple_of(PACK) && BLOCK.is_multiple_of(PACK));
            assert!(BLOCK == 2 * super::decode::BLOCK_BYTES);
        }
        let columns = ColumnLayout::new(dim)?;
        let k = columns.padded_ncols();
        let query_overflow = || Overflow::for_type::<u8>(MR, dim);
        let panel_bytes = MR
            .checked_mul(k)
            .and_then(|n| n.checked_add(std::mem::size_of::<QueryCompensation<MR>>()))
            .ok_or_else(query_overflow)?;
        Overflow::check_byte_budget::<u8>(panel_bytes, MR, dim)?;
        let doc_overflow = || Overflow::for_type::<u8>(NR, dim);
        let decoded_row_bytes = k
            .checked_add(std::mem::size_of::<MinMaxCompensation>())
            .ok_or_else(doc_overflow)?;
        let decoded_panel_bytes = NR.checked_mul(decoded_row_bytes).ok_or_else(doc_overflow)?;
        Overflow::check_byte_budget::<u8>(decoded_panel_bytes, NR, dim)?;
        Ok(Self {
            columns,
            panel_bytes: Bytes::new(panel_bytes),
            decoded_row_bytes: Bytes::new(decoded_row_bytes),
            max_b_rows: isize::MAX as usize / decoded_row_bytes,
        })
    }

    pub(super) fn dim(&self) -> usize {
        self.columns.ncols()
    }

    pub(super) fn nonempty(&self) -> Option<RowLayout<'_>> {
        Some(RowLayout {
            layout: self,
            k: DimK::new(NonZeroUsize::new(self.columns.padded_ncols())?),
        })
    }

    #[expect(
        clippy::expect_used,
        reason = "validated blocks contain two complete nibble channels"
    )]
    fn channels<'a>(
        &self,
        block: &'a mut [u8; BLOCK],
        base: usize,
    ) -> (
        &'a mut [u8; super::decode::BLOCK_BYTES],
        &'a mut [u8; super::decode::BLOCK_BYTES],
    ) {
        let high = self.columns.physical_col(base + 1) - base;
        let (low, high) = block.split_at_mut(high);
        (
            low.try_into().expect("complete low-nibble channel"),
            high.try_into().expect("complete high-nibble channel"),
        )
    }

    pub(super) fn decode_row(
        &self,
        arch: impl super::decode::Decoder,
        packed: &[u8],
        output: &mut [u8],
    ) {
        assert_eq!(
            packed.len(),
            self.dim().div_ceil(2),
            "packed code length mismatch"
        );
        assert_eq!(
            output.len(),
            self.columns.padded_ncols(),
            "decoded row length mismatch"
        );
        let full = self.dim() / BLOCK;
        for (index, (source, block)) in packed[..full * super::decode::BLOCK_BYTES]
            .as_chunks::<{ super::decode::BLOCK_BYTES }>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<BLOCK>().0)
            .enumerate()
        {
            let (low, high) = self.channels(block, index * BLOCK);
            arch.unpack_block(source, low, high);
        }
        if !self.dim().is_multiple_of(BLOCK) {
            let base = full * BLOCK;
            let block = &mut output.as_chunks_mut::<BLOCK>().0[full];
            let (low, high) = self.channels(block, base);
            arch.unpack_tail(&packed[full * super::decode::BLOCK_BYTES..], low, high);
            if !self.dim().is_multiple_of(2) {
                let physical = self.columns.physical_col(self.dim());
                debug_assert_eq!(self.columns.logical_col(physical), self.dim());
                block[physical - base] = 0;
            }
        }
    }
}

/// Nonempty views borrow their dimensions and byte extents from the prepared layout.
#[derive(Clone, Copy)]
pub(super) struct RowLayout<'a> {
    layout: &'a Layout,
    k: DimK,
}

impl RowLayout<'_> {
    pub(super) fn dim(self) -> usize {
        self.layout.dim()
    }
    pub(super) fn k(self) -> DimK {
        self.k
    }
    pub(super) fn decoded_row_bytes(self) -> Bytes {
        self.layout.decoded_row_bytes
    }
    pub(super) fn max_b_rows(self) -> usize {
        self.layout.max_b_rows
    }
    pub(super) fn decode_row(
        self,
        arch: impl super::decode::Decoder,
        packed: &[u8],
        output: &mut [u8],
    ) {
        self.layout.decode_row(arch, packed, output);
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
pub(crate) struct PackedQuery<const MR: usize, const PACK: usize, const NR: usize> {
    values: BlockTransposed<u8, MR, PACK>,
    compensation: Vec<QueryCompensation<MR>>,
    layout: Layout,
}

impl<const MR: usize, const PACK: usize, const NR: usize> PackedQuery<MR, PACK, NR> {
    pub(crate) fn new(query: MatRef<'_, MinMaxMeta<8>>) -> Result<Self, Overflow> {
        let mut result = Self::empty(query.num_vectors(), query.repr().intrinsic_dim())?;
        for (i, row) in query.rows().enumerate() {
            result.set_row(i, row.vector().as_slice(), row.meta());
        }
        Ok(result)
    }

    fn empty(rows: usize, dim: usize) -> Result<Self, Overflow> {
        let layout = Layout::new::<MR, PACK, NR>(dim)?;
        let panels = rows.div_ceil(MR);
        Overflow::check_byte_budget::<QueryCompensation<MR>>(panels, rows, dim)?;
        Ok(Self {
            values: BlockTransposed::try_new(rows, layout.columns.padded_ncols())?,
            compensation: vec![QueryCompensation::default(); panels],
            layout,
        })
    }

    fn set_row(&mut self, row: usize, values: &[u8], meta: MinMaxCompensation) {
        assert!(row < self.nrows(), "query row out of bounds");
        assert_eq!(values.len(), self.dim(), "query row dimension mismatch");
        let output = self.values.as_mut_slice();
        for (d, &value) in values.iter().enumerate() {
            output[self.layout.columns.packed_index::<MR, PACK>(row, d)] = value;
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
        self.layout.dim()
    }

    /// Empty or zero-dimensional queries are handled before entering the driver.
    pub(crate) fn as_view(&self) -> Option<PackedQueryView<'_, MR, PACK, NR>> {
        Some(PackedQueryView {
            values: packed::View::from_block_transposed(self.values.as_view())?,
            compensation: &self.compensation,
            layout: self.layout.nonempty()?,
            nrows: self.nrows(),
        })
    }
}

/// A nonempty query with zero padding and a contraction dimension divisible by PACK.
#[derive(Clone, Copy)]
pub(crate) struct PackedQueryView<'a, const MR: usize, const PACK: usize, const NR: usize> {
    values: packed::View<'a, u8, MR, PACK>,
    compensation: &'a [QueryCompensation<MR>],
    layout: RowLayout<'a>,
    nrows: usize,
}

impl<'a, const MR: usize, const PACK: usize, const NR: usize> PackedQueryView<'a, MR, PACK, NR> {
    pub(super) fn k(self) -> DimK {
        self.layout.k()
    }

    pub(super) fn dim(self) -> usize {
        self.layout.dim()
    }

    pub(super) fn nrows(self) -> usize {
        self.nrows
    }

    pub(super) fn panel_bytes(self) -> Bytes {
        self.layout.layout.panel_bytes
    }

    pub(super) fn layout(self) -> RowLayout<'a> {
        self.layout
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
                        layout: self.layout,
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
    layout: RowLayout<'a>,
    valid_rows: usize,
}

impl<'a, const MR: usize, const PACK: usize> APanel<'a, MR, PACK> {
    pub(super) fn k(self) -> DimK {
        self.layout.k()
    }
    pub(super) fn dim(self) -> usize {
        self.layout.dim()
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
    use crate::multi_vector::block_transposed::BlockLayout;

    pub(in crate::matrix_kernels::maxsim::minmax8_x_minmax4) const DIMS: &[usize] = &[
        0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 127, 128, 129, 249, 250, 255, 256, 257, 1024, 1025,
    ];

    pub(in crate::matrix_kernels::maxsim::minmax8_x_minmax4) fn query<
        const MR: usize,
        const PACK: usize,
        const NR: usize,
    >(
        rows: usize,
        dim: usize,
        mut values: impl FnMut(usize, usize) -> u8,
        mut meta: impl FnMut(usize) -> MinMaxCompensation,
    ) -> PackedQuery<MR, PACK, NR> {
        let mut query = PackedQuery::empty(rows, dim).unwrap();
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
        layout: RowLayout<'a>,
    ) -> APanel<'a, MR, PACK> {
        let k = layout.k();
        assert!(k.value().get().is_multiple_of(PACK));
        assert_eq!(values.len(), MR * k.value().get());
        APanel {
            // SAFETY: The test supplies an explicitly sized packed panel.
            values: unsafe {
                packed::Panel::new(crate::matrix_kernels::ptr::Slice::new(values), k)
            },
            compensation,
            layout,
            valid_rows: MR,
        }
    }

    pub(in crate::matrix_kernels::maxsim::minmax8_x_minmax4) fn position(d: usize) -> usize {
        let base = d / 64 * 64;
        base + (d % 64) / 2 + if d.is_multiple_of(2) { 0 } else { 32 }
    }

    fn check<const MR: usize, const PACK: usize, const NR: usize>() {
        for &dim in DIMS {
            for rows in [0, 1, MR - 1, MR, MR + 1, 3 * MR + 1] {
                let mut storage = PackedQuery::<MR, PACK, NR>::empty(rows, dim).unwrap();
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
                            let offset = BlockLayout::<MR, PACK>::linear_index(row, position(d), k);
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
                        assert_eq!(
                            view.panel_bytes().value(),
                            view.values.block_stride(view.k()).bytes().value()
                                + std::mem::size_of::<QueryCompensation<MR>>(),
                        );
                        let mut visited = 0;
                        view.visit_panels(|panel, start| {
                            assert!(std::ptr::eq(panel.layout.layout, &storage.layout));
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
        check::<8, 4, 6>();
        check::<8, 4, 8>();
        check::<16, 4, 6>();
        check::<16, 8, 8>();
    }

    #[test]
    fn even_odd_positions() {
        let columns = ColumnLayout::<Order>::new(3 * BLOCK).unwrap();
        for block in 0..3 {
            let base = block * BLOCK;
            let order: Vec<_> = (base..base + BLOCK)
                .step_by(2)
                .chain((base + 1..base + BLOCK).step_by(2))
                .collect();
            for (position, d) in order.into_iter().enumerate() {
                assert_eq!(columns.physical_col(d), base + position);
            }
        }
    }

    #[test]
    fn padding_overflow() {
        assert!(Layout::new::<8, 4, 6>(usize::MAX).is_err());
    }

    #[test]
    fn checked_working_set_extents() {
        check_extents::<8, 4, 6>();
        check_extents::<8, 4, 8>();
        check_extents::<16, 4, 6>();
        check_extents::<16, 8, 8>();
        assert!(PackedQuery::<8, 4, 6>::empty(usize::MAX / 64, 1).is_err());
        assert!(PackedQuery::<8, 4, 6>::empty(usize::MAX, 0).is_err());
    }

    fn check_extents<const MR: usize, const PACK: usize, const NR: usize>() {
        for &dim in DIMS {
            let layout = Layout::new::<MR, PACK, NR>(dim).unwrap();
            let k = dim.div_ceil(64) * 64;
            let row_bytes = k + std::mem::size_of::<MinMaxCompensation>();
            assert_eq!(layout.decoded_row_bytes.value(), row_bytes);
            assert_eq!(
                layout.panel_bytes.value(),
                MR * k + std::mem::size_of::<QueryCompensation<MR>>()
            );
            assert!(layout.max_b_rows * row_bytes <= isize::MAX as usize);
            assert!((layout.max_b_rows + 1) * row_bytes > isize::MAX as usize);
        }
        let limit = ((isize::MAX as usize - std::mem::size_of::<QueryCompensation<MR>>()) / MR)
            .min(isize::MAX as usize / NR - std::mem::size_of::<MinMaxCompensation>());
        let k = limit / BLOCK * BLOCK;
        assert!(Layout::new::<MR, PACK, NR>(k).is_ok());
        assert!(Layout::new::<MR, PACK, NR>(k + 1).is_err());
    }

    #[test]
    #[should_panic(expected = "query row dimension mismatch")]
    fn invalid_row_dimension() {
        PackedQuery::<8, 4, 6>::empty(1, 9).unwrap().set_row(
            0,
            &[0; 8],
            MinMaxCompensation::default(),
        );
    }
}

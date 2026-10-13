/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{
    matrix_kernels::maxsim::minmax8_x_minmax4::layout::{AView, Layout},
    minmax::{MinMaxCompensation, MinMaxMeta},
    multi_vector::{BlockTransposed, MatRef, Overflow},
};

#[derive(Debug)]
pub(super) struct PackedQuery<const MR: usize, const PACK: usize> {
    values: BlockTransposed<u8, MR, PACK>,
    compensation: Vec<[MinMaxCompensation; MR]>,
    layout: Layout,
}

impl<const MR: usize, const PACK: usize> PackedQuery<MR, PACK> {
    #[expect(
        clippy::expect_used,
        reason = "enumerated query rows match the packed allocation"
    )]
    pub(super) fn new(query: MatRef<'_, MinMaxMeta<8>>) -> Result<Self, Overflow> {
        const { assert!(MR > 0 && PACK > 0 && MR.is_multiple_of(PACK)) };
        let rows = query.num_vectors();
        let layout = Layout::new(query.repr().intrinsic_dim())?;
        Overflow::check_byte_budget::<[MinMaxCompensation; MR]>(
            rows.div_ceil(MR),
            rows,
            layout.dim(),
        )?;
        let mut values = BlockTransposed::try_new(rows, layout.padded_dim())?;
        let mut compensation = vec![[MinMaxCompensation::default(); MR]; rows.div_ceil(MR)];
        for (row, source) in query.rows().enumerate() {
            let mut output = values.get_row_mut(row).expect("query row in bounds");
            for (d, &value) in source.vector().as_slice().iter().enumerate() {
                output.set(Layout::position(d), value);
            }
            compensation[row / MR][row % MR] = source.meta();
        }
        Ok(Self {
            values,
            compensation,
            layout,
        })
    }

    pub(super) fn rows(&self) -> usize {
        self.values.nrows()
    }

    pub(super) fn dim(&self) -> usize {
        self.layout.dim()
    }

    pub(super) fn view(&self) -> Option<AView<'_, MR, PACK>> {
        AView::new(self.values.as_view(), &self.compensation, &self.layout)
    }
}

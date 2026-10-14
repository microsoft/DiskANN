/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{
    matrix_kernels::maxsim::minmax8_x_minmax4::{a::AView, layout::Layout},
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_vector::{Defaulted, Mat};
    use diskann_utils::ReborrowMut;

    fn check_packing<const MR: usize, const PACK: usize>() {
        for rows in [0, 1, MR - 1, MR, MR + 1] {
            for dim in [0usize, 1, 31, 32, 33, 63, 64, 65, 127, 128, 129] {
                let mut query = Mat::new(MinMaxMeta::<8>::new(rows, dim), Defaulted).unwrap();
                for (row, mut source) in query.reborrow_mut().rows_mut().enumerate() {
                    for d in 0..dim {
                        source
                            .vector_mut()
                            .set(d, ((row * 37 + d * 19) % 255 + 1) as i64)
                            .unwrap();
                    }
                    source.set_meta(MinMaxCompensation {
                        dim: dim as u32,
                        a: row as f32 + 1.0,
                        b: -(row as f32) - 2.0,
                        n: row as f32 + 3.0,
                        ..Default::default()
                    });
                }
                let packed = PackedQuery::<MR, PACK>::new(query.as_view()).unwrap();
                assert_eq!(packed.view().is_none(), rows == 0 || dim == 0);
                let padded = dim.next_multiple_of(64);
                // Construct the order independently of Layout::position.
                let order: Vec<_> = (0..padded)
                    .step_by(64)
                    .flat_map(|base| {
                        (0..2).flat_map(move |parity| (base + parity..base + 64).step_by(2))
                    })
                    .collect();
                for (index, group) in packed.values.as_slice().chunks_exact(MR * PACK).enumerate() {
                    let panel = index / (padded / PACK);
                    let k = index % (padded / PACK) * PACK;
                    for row in 0..MR {
                        for lane in 0..PACK {
                            let original_row = panel * MR + row;
                            let d = order[k + lane];
                            let expected = if original_row < rows && d < dim {
                                ((original_row * 37 + d * 19) % 255 + 1) as u8
                            } else {
                                0
                            };
                            assert_eq!(
                                group[row * PACK + lane],
                                expected,
                                "rows={rows}, dim={dim}, MR={MR}, PACK={PACK}, panel={panel}, row={row}, k={k}, lane={lane}"
                            );
                        }
                    }
                }
                for (panel, metadata) in packed.compensation.iter().enumerate() {
                    for (row, actual) in metadata.iter().enumerate() {
                        let expected = query
                            .get_row(panel * MR + row)
                            .map(|row| row.meta())
                            .unwrap_or_default();
                        assert_eq!(*actual, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn prepared_queries_preserve_order_padding_and_metadata() {
        check_packing::<8, 4>();
        check_packing::<16, 4>();
        check_packing::<16, 8>();
    }
}

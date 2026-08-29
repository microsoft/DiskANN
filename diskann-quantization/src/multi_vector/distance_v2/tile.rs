/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use crate::multi_vector::distance_v2::{
    bounds::{Bound},
    blocks, kernel,
    num::AllColumns,
    ptr::{MutSlice, Slice},
};

#[derive(Debug, Clone, Copy)]
struct Budget {
    ablocks: usize,
    brows: usize,
}

pub fn example_maxsim_f32(
    a: crate::multi_vector::BlockTransposedRef<'_, f32, 16>,
    b: diskann_utils::views::MatrixView<'_, f32>,
    c: &mut [f32],
    budget: Budget,
) {
    assert_eq!(a.ncols(), b.ncols());
    assert_eq!(c.len(), a.padded_nrows());

    let cols = NonZeroUsize::new(a.ncols()).unwrap();

    let a = blocks::dynamic::BlockTransposed::<f32, 16>::new(
        Slice::new(a.as_slice()),
        a.num_blocks(),
        Bound::new(cols.get()),
    );

    let b = blocks::dynamic::RowMajor::<f32>::new(
        Slice::new(b.as_slice()),
        b.nrows(),
        Bound::new(b.ncols()),
    );

    c.fill(f32::NEG_INFINITY);

    let c = MutSlice::new(c);

    example_maxsim_f32_inner(diskann_wide::ARCH, a, b, c, AllColumns::new(cols), budget);
}

#[inline(never)]
fn example_maxsim_f32_inner<A>(
    arch: A,
    a: blocks::dynamic::BlockTransposed<'_, f32, 16>,
    b: blocks::dynamic::RowMajor<'_, f32>,
    mut c: MutSlice<'_, f32>,
    cols: AllColumns,
    budget: Budget,
) where
    A: diskann_wide::Architecture,
    for<'a> kernel::maxsim::f32::BlockWithRowMajor<'a, A, 16, 6>: kernel::Kernel,
{
    let mut ablock = 0;
    while ablock != a.blocks() {
        let these_a_blocks = (a.blocks() - ablock).min(budget.ablocks);

        let mut brow = 0;
        while brow != b.nrows() {
            let these_b_rows = (b.nrows() - brow).min(budget.brows);
            let subb = unsafe { b.subslice(cols, brow, these_b_rows) };

            for ablock_offset in 0..these_a_blocks {
                let suba = a.block(cols, ablock + ablock_offset);

                let mut subc = c.subslice(16 * (ablock + ablock_offset), Bound::new(16));

                let mut kernel = kernel::maxsim::f32::BlockWithRowMajor {
                    kernel: kernel::maxsim::MaxSim::new(arch),
                    a: suba,
                    b: subb,
                    c: unsafe { subc.materialize::<16>() },
                    cols,
                };

                <_ as kernel::Kernel>::run(&mut kernel);
            }

            brow += these_b_rows;
        }

        ablock += these_a_blocks;
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use diskann_utils::views::MatrixView;

    use super::*;

    fn naive_maxsim(a: MatrixView<'_, f32>, b: MatrixView<'_, f32>) -> Vec<f32> {
        a.row_iter()
            .map(|arow| {
                b.row_iter()
                    .map(|brow| arow.iter().zip(brow).map(|(a, b)| a * b).sum::<f32>())
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect()
    }

    fn run_case(narows: usize, nbrows: usize, cols: usize, budget: Budget) {
        let adata = (0..narows * cols)
            .map(|i| ((17 * i + 5) % 23) as f32 / 7.0 - 1.5)
            .collect::<Vec<_>>();
        let bdata = (0..nbrows * cols)
            .map(|i| ((11 * i + 3) % 19) as f32 / 5.0 - 1.75)
            .collect::<Vec<_>>();

        let aview = MatrixView::try_from(adata.as_slice(), narows, cols).unwrap();
        let bview = MatrixView::try_from(bdata.as_slice(), nbrows, cols).unwrap();
        let expected = naive_maxsim(aview, bview);
        let a = crate::multi_vector::BlockTransposed::<f32, 16>::from_matrix_view(aview);
        let mut actual = vec![f32::NAN; a.padded_nrows()];

        example_maxsim_f32(a.as_view(), bview, &mut actual, budget);

        for (row, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = 1e-5 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "row {row}: expected {expected}, got {actual}"
            );
        }

        assert!(
            actual[narows..].iter().all(|&value| value == 0.0),
            "padded A rows should have zero inner products"
        );
    }

    #[test]
    fn preserves_negative_maxima() {
        let adata = vec![1.0; 16 * 3];
        let bdata = [-1.0, -2.0, -3.0];
        let aview = MatrixView::try_from(adata.as_slice(), 16, 3).unwrap();
        let bview = MatrixView::try_from(bdata.as_slice(), 1, 3).unwrap();
        let a = crate::multi_vector::BlockTransposed::<f32, 16>::from_matrix_view(aview);
        let mut actual = vec![f32::NAN; 16];

        example_maxsim_f32(
            a.as_view(),
            bview,
            &mut actual,
            Budget {
                ablocks: 1,
                brows: 1,
            },
        );

        assert_eq!(actual, vec![-6.0; 16]);
    }

    #[test]
    fn handles_full_and_remainder_b_panels() {
        for nbrows in 1..=8 {
            run_case(
                16,
                nbrows,
                7,
                Budget {
                    ablocks: 1,
                    brows: nbrows,
                },
            );
        }
    }

    #[test]
    fn tiles_across_a_and_b() {
        run_case(
            21,
            11,
            9,
            Budget {
                ablocks: 1,
                brows: 5,
            },
        );
    }
}

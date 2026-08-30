/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use crate::multi_vector::distance_v2::{
    blocks,
    bounds::Bound,
    kernel,
    num::DimK,
    ptr::{MutSlice, Slice},
};

#[derive(Debug, Clone, Copy)]
struct Budget {
    ablocks: NonZeroUsize,
    brows: NonZeroUsize,
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

    let ablocks = NonZeroUsize::new(a.num_blocks()).unwrap();

    let a = blocks::packed::View::<f32, 16>::new(
        Slice::new(a.as_slice()),
        ablocks,
        Bound::new(cols.get()),
    );

    let brows = NonZeroUsize::new(b.nrows()).unwrap();

    let b = unsafe {
        blocks::unpacked::View::<f32>::new(Slice::new(b.as_slice()), brows, Bound::new(b.ncols()))
    };

    c.fill(f32::NEG_INFINITY);

    let c = MutSlice::new(c);

    example_maxsim_f32_inner(diskann_wide::ARCH, a, b, c, DimK::new(cols), budget);
}

#[inline(never)]
fn example_maxsim_f32_inner<A>(
    arch: A,
    a: blocks::packed::View<'_, f32, 16>,
    b: blocks::unpacked::View<'_, f32>,
    mut c: MutSlice<'_, f32>,
    k: DimK,
    budget: Budget,
) where
    A: diskann_wide::Architecture,
    for<'a> kernel::maxsim::f32::BlockWithRowMajor<'a, A, 16, 4>: kernel::PanelKernel,
{
    unsafe {
        a.visit_sub_views(budget.ablocks, k, |suba, a_block_base| {
            b.visit_sub_views(budget.brows, k, |subb| {
                suba.visit_panels(k, |apanel, a_block_offset| {
                    let mut subc =
                        unsafe { c.subslice(16 * (a_block_base + a_block_offset), Bound::new(16)) };

                    let mut kernel = unsafe {
                        kernel::maxsim::f32::BlockWithRowMajor::new(
                            kernel::maxsim::MaxSim::new(arch),
                            apanel,
                            subb,
                            unsafe { subc.materialize::<16>() },
                            k,
                        )
                    };

                    <_ as kernel::PanelKernel>::panel_kernel(&mut kernel);
                })
            })
        });
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
                ablocks: NonZeroUsize::new(1).unwrap(),
                brows: NonZeroUsize::new(1).unwrap(),
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
                    ablocks: NonZeroUsize::new(1).unwrap(),
                    brows: NonZeroUsize::new(nbrows).unwrap(),
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
                ablocks: NonZeroUsize::new(1).unwrap(),
                brows: NonZeroUsize::new(5).unwrap(),
            },
        );
    }
}

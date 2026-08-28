/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::multi_vector::distance_v2::{
    ptr::{Slice, MutSlice},
    blocks,
    Length,
}

#[derive(Debug, Clone, Copy)]
struct Budget {
    ablocks: usize,
    brows: usize,
}

pub fn example_maxsim_f32(
    a: crate::multi_vector::BlockTransposeRef<'_, f32, 16>,
    b: diskann_utils::views::MatrixView<'_, f32>,
    c: &mut [f32],
    budget: Budget,
) {
    assert_eq!(a.ncols(), b.ncols());
    assert_eq!(c.len(), a.padded_rows());

    let cols = a.ncols();

    let a = blocks::dynamic::BlockTransposed::<f32, 16>::new(
        Slice::new(a.as_slice()),
        Length::from_fn(|| a.blocks()),
        Length::new(cols),
    );

    let b = blocks::dynamic::RowMajor::<f32>::new(
        Slice::new(b.as_slice()),
        Length::new(b.nrows()),
        Length::new(b.ncols()),
    );

    let c = MutSlice::new(c);

    example_maxsim_f32_inner(diskann_wide::arch::Current, a, b, c, cols, budget);
}

fn example_maxsim_f32_inner<A>(
    arch: A,
    a: blocks::dynamic::BlockTransposed<'_, f32, 16>,
    b: blocks::dynamic::RowMajor<'_, f32>,
    c: MutSlice<'_, f32>,
    cols: AllColumns,
    budget: Budget,
) {
    const BROW: usize = 4;

    let mut ablock = 0;
    while ablock != a.blocks() {
        let these_a_blocks = (a.blocks() - ablock).min(budget.ablocks);

        let mut brow = 0;
        while brow != b.rows() {
            let these_b_rows = (b.rows() - brow).min(budget.brows);
            let subb = b.subslice(cols, brow, these_b_rows);

            for ablock_offset in 0..these_a_blocks {
                let suba = a.block(cols, ablock + ablock_offset);
            }
        }
    }
}


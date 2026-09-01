/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! V3 (AVX2+FMA) f32 leaf, panels of 16 × 4.

use diskann_wide::arch::x86_64::V3;
use diskann_wide::{SIMDMinMax, SIMDMulAdd, SIMDVector};

use crate::bits::Length;
use crate::multi_vector::distance::kernels::tiles::{BlockTransposedPanel, RowMajorPanel};

diskann_wide::alias!(f32s = <V3>::f32x8);

/// Two `f32x8` register tiles per B-row: enough independent accumulators to cover FMA
/// latency at `B_PANEL = 4` without spilling.
pub(crate) const A_PANEL: usize = 2 * f32s::LANES;
pub(crate) const B_PANEL: usize = 4;

/// Register tiles spanned by one A-panel.
const REGS: usize = A_PANEL / f32s::LANES;

/// Fold the maximum of `A_PANEL × UNROLL` inner products into `state`.
///
/// # Panics
///
/// Panics if `b` does not hold `UNROLL` rows of `a`'s contraction length. That extent is
/// what every unchecked access below relies on.
#[inline(always)]
pub(crate) fn f32_max_microkernel<const UNROLL: usize, L: Length>(
    arch: V3,
    a: BlockTransposedPanel<'_, f32, A_PANEL>,
    b: RowMajorPanel<'_, f32, B_PANEL, L>,
    state: &mut [f32; A_PANEL],
) {
    const { assert!(UNROLL >= 1 && UNROLL <= B_PANEL) }

    let k = a.k();
    let a_data = a.as_slice();
    let b_data = b.as_slice();
    assert_eq!(b_data.len(), UNROLL * k, "B panel extent");

    let ap = a_data.as_ptr();
    let bp = b_data.as_ptr();

    let mut acc = [[f32s::default(arch); REGS]; UNROLL];
    let b_row: [usize; UNROLL] = core::array::from_fn(|j| k * j);

    for i in 0..k {
        let a_col: [f32s; REGS] = core::array::from_fn(|r| {
            // SAFETY: `r < REGS` and `A_PANEL == REGS * LANES`, so the load ends at or
            // before `A_PANEL * (i + 1)`, hence at or before `A_PANEL * k`, which is at
            // most `a_data.len()` because `k` is that length divided by `A_PANEL`.
            unsafe { f32s::load_simd(arch, ap.add(A_PANEL * i + r * f32s::LANES)) }
        });

        for (acc_j, &row) in acc.iter_mut().zip(&b_row) {
            // SAFETY: `row == k * j` for some `j < UNROLL`, so `i + row < UNROLL * k`,
            // the asserted length of `b_data`.
            let b = unsafe { bp.add(i + row).read() };

            let bj = f32s::splat(arch, b);
            for (acc_jr, a_r) in acc_j.iter_mut().zip(&a_col) {
                *acc_jr = a_r.mul_add_simd(bj, *acc_jr);
            }
        }
    }

    // `run` seeds state with a finite identity, and every update preserves that invariant.
    // Keeping it as the right operand also makes x86 max ignore a NaN from a document.
    let (rows, _) = state.as_chunks_mut::<{ f32s::LANES }>();
    let mut merged: [f32s; REGS] = core::array::from_fn(|r| f32s::from_array(arch, rows[r]));
    for column in &acc {
        for (dst, src) in merged.iter_mut().zip(column) {
            *dst = src.max_simd(*dst);
        }
    }

    for (m, row) in merged.iter().zip(rows) {
        *row = m.to_array();
    }
}

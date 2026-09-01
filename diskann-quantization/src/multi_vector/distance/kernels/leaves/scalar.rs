/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Emulated f32 leaf, panels of 8 × 2.
//!
//! Geometry is narrower than the V3 leaf because [`Emulated`](diskann_wide::Emulated)
//! lanes are an unrolled loop over scalars, not registers: 8 × 2 matches the `Strategy2x1`
//! shape used by the scalar distance functions elsewhere in the crate.
//!
//! The inner loop multiplies and adds separately instead of calling `mul_add_simd`, which
//! on x86-64 without hardware FMA would drop into libm's software `fma()`.

use diskann_wide::arch::Scalar;
use diskann_wide::{SIMDMinMax, SIMDVector};

use crate::bits::Length;
use crate::multi_vector::distance::kernels::tiles::{BlockTransposedPanel, RowMajorPanel};

diskann_wide::alias!(f32s = <Scalar>::f32x8);

pub(crate) const A_PANEL: usize = f32s::LANES;
pub(crate) const B_PANEL: usize = 2;

/// Vector chunks spanned by one A-panel.
const REGS: usize = A_PANEL / f32s::LANES;

/// Fold the maximum of `A_PANEL × UNROLL` inner products into `state`.
///
/// # Panics
///
/// Panics if `b` does not hold `UNROLL` rows of `a`'s contraction length. That extent is
/// what every unchecked access below relies on.
#[inline(always)]
pub(crate) fn f32_max_microkernel<const UNROLL: usize, L: Length>(
    arch: Scalar,
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
                *acc_jr = *a_r * bj + *acc_jr;
            }
        }
    }

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

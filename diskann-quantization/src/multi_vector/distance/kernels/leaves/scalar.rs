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

use super::WAYS;
use crate::bits::Length;
use crate::multi_vector::distance::kernels::strip::Slot;
use crate::multi_vector::distance::kernels::tiles::{BlockTransposedPanel, RowMajorPanel};

diskann_wide::alias!(f32s = <Scalar>::f32x8);

pub(crate) const A_PANEL: usize = f32s::LANES;
pub(crate) const B_PANEL: usize = 2;

/// Vector chunks spanned by one A-panel.
const REGS: usize = A_PANEL / f32s::LANES;

/// `A_PANEL × UNROLL` inner products, stored column-major into `out`.
///
/// # Panics
///
/// Panics if `b` does not hold `UNROLL` rows of `a`'s contraction length. That extent is
/// what every unchecked access below relies on.
#[inline(always)]
pub(crate) fn f32_store_microkernel<const UNROLL: usize, L: Length>(
    arch: Scalar,
    a: BlockTransposedPanel<'_, f32, A_PANEL>,
    b: RowMajorPanel<'_, f32, B_PANEL, L>,
    mut out: Slot<'_, f32, A_PANEL, B_PANEL>,
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

    for (acc_j, col) in acc.iter().zip(out.columns()) {
        let (tiles, _) = col.as_chunks_mut::<{ f32s::LANES }>();
        for (acc_jr, dst) in acc_j.iter().zip(tiles) {
            // SAFETY: `as_chunks_mut` yields exactly `LANES` elements.
            unsafe { acc_jr.store_simd(dst.as_mut_ptr()) };
        }
    }
}

/// Merge each column of `acc` into the running per-A-row maxima in `state`.
///
/// The maxima are held in registers for the whole sweep, so `state` is read once on entry
/// and written once on exit, not once per column.
#[inline(always)]
pub(crate) fn max_into_rows(arch: Scalar, acc: &[[f32; A_PANEL]], state: &mut [f32; A_PANEL]) {
    let mut chains = [[f32s::splat(arch, f32::MIN); REGS]; WAYS];

    let mut groups = acc.chunks_exact(WAYS);
    for group in groups.by_ref() {
        max_into_chains(arch, &mut chains, group);
    }
    // Fewer than WAYS columns remain, so each still lands on its own chain.
    max_into_chains(arch, &mut chains, groups.remainder());

    let mut merged = chains[0];
    for chain in &chains[1..] {
        for (m, c) in merged.iter_mut().zip(chain) {
            *m = m.max_simd(*c);
        }
    }

    let (rows, _) = state.as_chunks_mut::<{ f32s::LANES }>();
    for (m, row) in merged.iter().zip(rows) {
        *row = m.max_simd(f32s::from_array(arch, *row)).to_array();
    }
}

/// Merge up to [`WAYS`] consecutive columns, one per chain.
#[inline(always)]
fn max_into_chains(arch: Scalar, chains: &mut [[f32s; REGS]; WAYS], src: &[[f32; A_PANEL]]) {
    for (chain, column) in chains.iter_mut().zip(src) {
        let (tiles, _) = column.as_chunks::<{ f32s::LANES }>();
        for (c, tile) in chain.iter_mut().zip(tiles) {
            *c = c.max_simd(f32s::from_array(arch, *tile));
        }
    }
}

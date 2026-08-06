// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! V3 SIMD leaves, and the panel geometry they impose. The store-out micro-kernels do
//! the same math as `tiler`'s, so an A/B between the two experiments measures the
//! abstraction rather than a different kernel; [`score_fold_strip`] is the one with no
//! counterpart there — it fuses `tiler`'s separate dequant and max passes.
//!
//! Every leaf is safe to call: they take the panel and slot handles whole, so each one's
//! requirements are discharged here rather than promised at every call site. The
//! remaining `unsafe` is the SIMD loads and stores, each in bounds by the lines above it.

use diskann_wide::arch::x86_64::V3;
use diskann_wide::{SIMDCast, SIMDDotProduct, SIMDMinMax, SIMDMulAdd, SIMDReinterpret, SIMDVector};

use super::strip::{Block, Strip};
use super::views::{DPanel, QPanel};
use crate::bits::Length;
use crate::minmax::MinMaxCompensation;

diskann_wide::alias!(i16s = <V3>::i16x16);
diskann_wide::alias!(i32s = <V3>::i32x8);
diskann_wide::alias!(u32s = <V3>::u32x8);
diskann_wide::alias!(f32s = <V3>::f32x8);

/// Rows per A-panel: every leaf below emits exactly two 32-bit SIMD registers of rows,
/// so a wider ISA's leaves get their own width.
pub(super) const A_PANEL: usize = 2 * f32s::LANES;

/// Max B-rows per kernel call — a register-budget choice: `B_PANEL` × two accumulator
/// registers, plus two A registers, must fit the architectural file.
pub(super) const B_PANEL: usize = 4;

/// Integer store-out micro-kernel: [`A_PANEL`] A-rows × `UNROLL` B-rows. `k` must be
/// even — the loads take column pairs.
///
/// # Panics
///
/// If the panels disagree on `k`, or `b` is not `UNROLL` rows — both driver bugs.
#[inline(always)]
pub(super) fn int_store_microkernel<const UNROLL: usize, L: Length>(
    arch: V3,
    a: QPanel<'_, i16, A_PANEL>,
    b: DPanel<'_, u8, B_PANEL, L>,
    mut out: Block<'_, i32, A_PANEL, B_PANEL>,
) {
    // The i16 half-loads and the i32 stores must span the same rows; that relation
    // holds across two different register types, so it is not self-evident.
    const {
        assert!(
            i16s::LANES == A_PANEL,
            "leaf loads A_PANEL i16 per A column-pair half"
        )
    }
    // Bounds the store below inside the slot's `A_PANEL * B_PANEL`.
    const { assert!(UNROLL <= B_PANEL, "unroll wider than the slot") }

    let k = a.k();
    assert_eq!(
        k,
        b.k(),
        "panels paired across different contraction lengths"
    );
    assert_eq!(b.rows(), UNROLL, "panel height must match the unroll");
    debug_assert_eq!(k % 2, 0, "the integer leaf loads column pairs");
    let (a_packed, b_ptr, partial) = (a.as_ptr(), b.as_ptr(), out.as_mut_ptr());

    let mut p0 = [i32s::default(arch); UNROLL];
    let mut p1 = [i32s::default(arch); UNROLL];
    let offsets: [usize; UNROLL] = core::array::from_fn(|j| k * j);

    let a_pair_stride = 2 * A_PANEL;
    let a_half = A_PANEL;
    let pairs = k / 2;

    for p in 0..pairs {
        // SAFETY: `a` is `A_PANEL * k` i16, so it holds `pairs` column-pairs of
        // `2 * A_PANEL`, and `p < pairs`.
        let (a0, a1) = unsafe {
            (
                i16s::load_simd(arch, a_packed.add(a_pair_stride * p)),
                i16s::load_simd(arch, a_packed.add(a_pair_stride * p + a_half)),
            )
        };

        for j in 0..UNROLL {
            // SAFETY: `b` is `UNROLL * k` u8 (both asserted), row j starts at
            // `offsets[j] = k * j`, and `2 * p + 1 < k`.
            let (d0, d1) = unsafe {
                let base = 2 * p + offsets[j];
                (
                    u32::from(b_ptr.add(base).read()),
                    u32::from(b_ptr.add(base + 1).read()),
                )
            };
            let packed = d0 | (d1 << 16);
            let bcast: i16s = u32s::splat(arch, packed).reinterpret_simd();
            p0[j] = p0[j].dot_simd(a0, bcast);
            p1[j] = p1[j].dot_simd(a1, bcast);
        }
    }

    for j in 0..UNROLL {
        // SAFETY: the slot is `A_PANEL * B_PANEL` writable i32 and `UNROLL <= B_PANEL`,
        // so column j occupies `[j * A_PANEL, (j + 1) * A_PANEL)` inside it.
        unsafe {
            p0[j].store_simd(partial.add(j * A_PANEL));
            p1[j].store_simd(partial.add(j * A_PANEL + i32s::LANES));
        }
    }
}

/// f32 store-out micro-kernel: [`A_PANEL`] A-rows × `UNROLL` B-rows of inner product.
///
/// # Panics
///
/// If the panels disagree on `k`, or `b` is not `UNROLL` rows — both driver bugs.
#[inline(always)]
pub(super) fn f32_store_microkernel<const UNROLL: usize, L: Length>(
    arch: V3,
    a: QPanel<'_, f32, A_PANEL>,
    b: DPanel<'_, f32, B_PANEL, L>,
    mut out: Block<'_, f32, A_PANEL, B_PANEL>,
) {
    // Bounds the store below inside the slot's `A_PANEL * B_PANEL`.
    const { assert!(UNROLL <= B_PANEL, "unroll wider than the slot") }

    let k = a.k();
    assert_eq!(
        k,
        b.k(),
        "panels paired across different contraction lengths"
    );
    assert_eq!(b.rows(), UNROLL, "panel height must match the unroll");
    let (a_packed, b_ptr, partial) = (a.as_ptr(), b.as_ptr(), out.as_mut_ptr());

    let mut p0 = [f32s::default(arch); UNROLL];
    let mut p1 = [f32s::default(arch); UNROLL];
    let offsets: [usize; UNROLL] = core::array::from_fn(|j| k * j);

    let a_stride = A_PANEL;
    let a_half = f32s::LANES;

    for i in 0..k {
        // SAFETY: `a` is `A_PANEL * k` f32 and `i < k`, so both halves of column `i`
        // are in bounds.
        let (a0, a1) = unsafe {
            (
                f32s::load_simd(arch, a_packed.add(a_stride * i)),
                f32s::load_simd(arch, a_packed.add(a_stride * i + a_half)),
            )
        };
        for j in 0..UNROLL {
            // SAFETY: `b` is `UNROLL * k` f32 (both asserted), row j starts at
            // `offsets[j] = k * j`, and `i < k`.
            let bj = unsafe { f32s::splat(arch, b_ptr.add(i + offsets[j]).read_unaligned()) };
            p0[j] = a0.mul_add_simd(bj, p0[j]);
            p1[j] = a1.mul_add_simd(bj, p1[j]);
        }
    }

    for j in 0..UNROLL {
        // SAFETY: the slot is `A_PANEL * B_PANEL` writable f32 and `UNROLL <= B_PANEL`,
        // so column j occupies `[j * A_PANEL, (j + 1) * A_PANEL)` inside it.
        unsafe {
            p0[j].store_simd(partial.add(j * A_PANEL));
            p1[j].store_simd(partial.add(j * A_PANEL + f32s::LANES));
        }
    }
}

/// Fold an [`A_PANEL`]×`cols` A-major f32 strip into the running max.
///
/// # Panics
///
/// If `state` is not exactly one A-panel, or the strip holds fewer than `cols` columns.
#[inline(always)]
pub(super) fn fold_strip(
    arch: V3,
    state: &mut [f32],
    strip: &Strip<'_, f32, A_PANEL, B_PANEL>,
    cols: usize,
) {
    let lanes = f32s::LANES;
    // The slicing is the bounds check: both loads below are inside these lengths.
    let acc = strip.columns(cols);
    assert_eq!(state.len(), A_PANEL, "the fold writes a whole A-panel");
    let (state, acc) = (state.as_mut_ptr(), acc.as_ptr());

    // SAFETY: `state` is A_PANEL = 2·LANES writable f32, and `acc` is `cols * A_PANEL`
    // readable f32, so column `c < cols` and both its halves are in bounds.
    unsafe {
        let mut m0 = f32s::load_simd(arch, state);
        let mut m1 = f32s::load_simd(arch, state.add(lanes));
        for c in 0..cols {
            let col = acc.add(c * A_PANEL);
            m0 = m0.max_simd(f32s::load_simd(arch, col));
            m1 = m1.max_simd(f32s::load_simd(arch, col.add(lanes)));
        }
        m0.store_simd(state);
        m1.store_simd(state.add(lanes));
    }
}

/// 4-bit MinMax dequant of an [`A_PANEL`]×`cols` A-major `i32` strip, folded straight
/// into the running max — the score never reaches memory.
///
/// # Panics
///
/// If `state` or `q_meta` is not one A-panel, or `d_meta` or the strip holds fewer than
/// `cols` columns.
#[inline(always)]
pub(super) fn score_fold_strip(
    arch: V3,
    strip: &Strip<'_, i32, A_PANEL, B_PANEL>,
    state: &mut [f32],
    cols: usize,
    q_meta: &[MinMaxCompensation],
    d_meta: &[MinMaxCompensation],
    dim: f32,
) {
    let lanes = f32s::LANES;
    // The slicing is the bounds check: every load below is inside these lengths.
    let acc = strip.columns(cols);
    let d_meta = &d_meta[..cols];
    assert_eq!(state.len(), A_PANEL, "the fold writes a whole A-panel");
    assert_eq!(q_meta.len(), A_PANEL, "one compensation per A-panel row");
    let (state, acc) = (state.as_mut_ptr(), acc.as_ptr());

    let mut qa = [0.0f32; A_PANEL];
    let mut qb = [0.0f32; A_PANEL];
    let mut qn = [0.0f32; A_PANEL];
    for (i, qm) in q_meta.iter().enumerate() {
        qa[i] = qm.a;
        qb[i] = qm.b;
        qn[i] = qm.n;
    }
    // SAFETY: each array holds exactly A_PANEL = 2·LANES f32, and `state` is A_PANEL
    // writable f32, so both halves of each are in bounds.
    let (qa0, qa1, qb0, qb1, qn0, qn1, mut m0, mut m1) = unsafe {
        (
            f32s::load_simd(arch, qa.as_ptr()),
            f32s::load_simd(arch, qa.as_ptr().add(lanes)),
            f32s::load_simd(arch, qb.as_ptr()),
            f32s::load_simd(arch, qb.as_ptr().add(lanes)),
            f32s::load_simd(arch, qn.as_ptr()),
            f32s::load_simd(arch, qn.as_ptr().add(lanes)),
            f32s::load_simd(arch, state),
            f32s::load_simd(arch, state.add(lanes)),
        )
    };

    for (c, dm) in d_meta.iter().enumerate() {
        let a_c = f32s::splat(arch, dm.a);
        let b_c = f32s::splat(arch, dm.b);
        let c_c = f32s::splat(arch, dm.n + dm.b * dim);
        let col = c * A_PANEL;
        // SAFETY: `acc` is `cols * A_PANEL` readable i32 and `c < cols`, so the column
        // and both its halves are in bounds.
        unsafe {
            let raw0 = i32s::load_simd(arch, acc.add(col)).simd_cast();
            let raw1 = i32s::load_simd(arch, acc.add(col + lanes)).simd_cast();
            let s0 = a_c.mul_add_simd(qa0 * raw0, b_c.mul_add_simd(qn0, c_c * qb0));
            let s1 = a_c.mul_add_simd(qa1 * raw1, b_c.mul_add_simd(qn1, c_c * qb1));
            m0 = m0.max_simd(s0);
            m1 = m1.max_simd(s1);
        }
    }

    // SAFETY: `state` is A_PANEL writable f32, as loaded above.
    unsafe {
        m0.store_simd(state);
        m1.store_simd(state.add(lanes));
    }
}

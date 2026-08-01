// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! V3 SIMD leaves, and the panel geometry they impose. The store-out micro-kernels are
//! byte-identical to `tiler`'s, so an A/B between the two experiments measures only the
//! abstraction; [`score_fold_strip`] is the one that differs — it fuses `tiler`'s
//! separate dequant and max passes.

use diskann_wide::arch::x86_64::V3;
use diskann_wide::{SIMDCast, SIMDDotProduct, SIMDMinMax, SIMDMulAdd, SIMDReinterpret, SIMDVector};

use crate::minmax::MinMaxCompensation;

diskann_wide::alias!(i16s = <V3>::i16x16);
diskann_wide::alias!(i32s = <V3>::i32x8);
diskann_wide::alias!(u32s = <V3>::u32x8);
diskann_wide::alias!(f32s = <V3>::f32x8);

/// Rows per A-panel: every leaf below emits exactly two 32-bit SIMD registers of rows.
/// Derived rather than written, so a wider ISA's leaves get their own width.
pub(super) const A_PANEL: usize = 2 * f32s::LANES;

/// Max B-rows per kernel call. Not derived — a register-budget choice: `B_PANEL` × two
/// accumulator registers, plus two A registers, must fit the architectural file.
pub(super) const B_PANEL: usize = 4;

/// Integer store-out micro-kernel: [`A_PANEL`] A-rows × `UNROLL` B-rows.
///
/// # Safety
///
/// 1. `a_packed` points to an `A_PANEL × k` block-transposed `i16` block (`k` even).
/// 2. `b` points to `UNROLL` rows of `k` contiguous `u8` (`k` even).
/// 3. `partial` is valid for `UNROLL` columns of `A_PANEL` `i32` at stride `A_PANEL`.
#[inline(always)]
pub(super) unsafe fn int_store_microkernel<const UNROLL: usize>(
    arch: V3,
    a_packed: *const i16,
    b: *const u8,
    k: usize,
    partial: *mut i32,
) {
    // The i16 half-loads and the i32 stores must span the same rows; that relation
    // holds across two different register types, so it is not self-evident.
    const {
        assert!(
            i16s::LANES == A_PANEL,
            "leaf loads A_PANEL i16 per A column-pair half"
        )
    }
    let mut p0 = [i32s::default(arch); UNROLL];
    let mut p1 = [i32s::default(arch); UNROLL];
    let offsets: [usize; UNROLL] = core::array::from_fn(|j| k * j);

    let a_pair_stride = 2 * A_PANEL;
    let a_half = A_PANEL;
    let pairs = k / 2;

    for p in 0..pairs {
        // SAFETY: precondition 1 — the A block has `pairs` col-pairs of 2·A_PANEL i16.
        let (a0, a1) = unsafe {
            (
                i16s::load_simd(arch, a_packed.add(a_pair_stride * p)),
                i16s::load_simd(arch, a_packed.add(a_pair_stride * p + a_half)),
            )
        };

        for j in 0..UNROLL {
            // SAFETY: precondition 2 — B row j is `offsets[j]` in, `2*p+1 < k`.
            let (d0, d1) = unsafe {
                let base = 2 * p + offsets[j];
                (
                    u32::from(b.add(base).read()),
                    u32::from(b.add(base + 1).read()),
                )
            };
            let packed = d0 | (d1 << 16);
            let bcast: i16s = u32s::splat(arch, packed).reinterpret_simd();
            p0[j] = p0[j].dot_simd(a0, bcast);
            p1[j] = p1[j].dot_simd(a1, bcast);
        }
    }

    for j in 0..UNROLL {
        // SAFETY: precondition 3 — column j occupies [j*A_PANEL, j*A_PANEL+A_PANEL) i32.
        unsafe {
            p0[j].store_simd(partial.add(j * A_PANEL));
            p1[j].store_simd(partial.add(j * A_PANEL + i32s::LANES));
        }
    }
}

/// f32 store-out micro-kernel: [`A_PANEL`] A-rows × `UNROLL` B-rows of inner product.
///
/// # Safety
///
/// 1. `a_packed` points to an `A_PANEL × k` block-transposed `f32` block (`PACK = 1`).
/// 2. `b` points to `UNROLL` rows of `k` contiguous `f32`.
/// 3. `partial` is valid for `UNROLL` columns of `A_PANEL` `f32` at stride `A_PANEL`.
#[inline(always)]
pub(super) unsafe fn f32_store_microkernel<const UNROLL: usize>(
    arch: V3,
    a_packed: *const f32,
    b: *const f32,
    k: usize,
    partial: *mut f32,
) {
    let mut p0 = [f32s::default(arch); UNROLL];
    let mut p1 = [f32s::default(arch); UNROLL];
    let offsets: [usize; UNROLL] = core::array::from_fn(|j| k * j);

    let a_stride = A_PANEL;
    let a_half = f32s::LANES;

    for i in 0..k {
        // SAFETY: precondition 1 — the A block has `k` columns of A_PANEL f32.
        let (a0, a1) = unsafe {
            (
                f32s::load_simd(arch, a_packed.add(a_stride * i)),
                f32s::load_simd(arch, a_packed.add(a_stride * i + a_half)),
            )
        };
        for j in 0..UNROLL {
            // SAFETY: precondition 2 — B row j is `offsets[j]` in, `i < k`.
            let bj = unsafe { f32s::splat(arch, b.add(i + offsets[j]).read_unaligned()) };
            p0[j] = a0.mul_add_simd(bj, p0[j]);
            p1[j] = a1.mul_add_simd(bj, p1[j]);
        }
    }

    for j in 0..UNROLL {
        // SAFETY: precondition 3 — column j occupies [j*A_PANEL, j*A_PANEL+A_PANEL) f32.
        unsafe {
            p0[j].store_simd(partial.add(j * A_PANEL));
            p1[j].store_simd(partial.add(j * A_PANEL + f32s::LANES));
        }
    }
}

/// Fold an [`A_PANEL`]×`cols` A-major f32 strip into the running max.
///
/// # Safety
///
/// `state` writable for `A_PANEL` `f32`; `acc` valid for `cols` columns of `A_PANEL` `f32`.
#[inline(always)]
pub(super) unsafe fn fold_strip(arch: V3, state: *mut f32, acc: *const f32, cols: usize) {
    let lanes = f32s::LANES;
    // SAFETY: `state` writable for A_PANEL; `acc` valid for `cols` columns of A_PANEL.
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
/// # Safety
///
/// `acc` valid for `cols` columns of `A_PANEL` `i32` (stride `A_PANEL`); `state`
/// writable for `A_PANEL` `f32`; `q_meta.len() >= A_PANEL`; `d_meta.len() >= cols`.
#[inline(always)]
pub(super) unsafe fn score_fold_strip(
    arch: V3,
    acc: *const i32,
    state: *mut f32,
    cols: usize,
    q_meta: &[MinMaxCompensation],
    d_meta: &[MinMaxCompensation],
    dim: f32,
) {
    let lanes = f32s::LANES;

    let mut qa = [0.0f32; A_PANEL];
    let mut qb = [0.0f32; A_PANEL];
    let mut qn = [0.0f32; A_PANEL];
    for i in 0..A_PANEL {
        let qm = q_meta[i];
        qa[i] = qm.a;
        qb[i] = qm.b;
        qn[i] = qm.n;
    }
    // SAFETY: each array holds exactly A_PANEL = 2·LANES f32; `state` writable for A_PANEL.
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

    for (c, dm) in d_meta.iter().enumerate().take(cols) {
        let a_c = f32s::splat(arch, dm.a);
        let b_c = f32s::splat(arch, dm.b);
        let c_c = f32s::splat(arch, dm.n + dm.b * dim);
        let col = c * A_PANEL;
        // SAFETY: `col + 2·LANES <= cols*A_PANEL`; `acc` valid for that many i32.
        unsafe {
            let raw0 = i32s::load_simd(arch, acc.add(col)).simd_cast();
            let raw1 = i32s::load_simd(arch, acc.add(col + lanes)).simd_cast();
            let s0 = a_c.mul_add_simd(qa0 * raw0, b_c.mul_add_simd(qn0, c_c * qb0));
            let s1 = a_c.mul_add_simd(qa1 * raw1, b_c.mul_add_simd(qn1, c_c * qb1));
            m0 = m0.max_simd(s0);
            m1 = m1.max_simd(s1);
        }
    }

    // SAFETY: `state` writable for A_PANEL f32.
    unsafe {
        m0.store_simd(state);
        m1.store_simd(state.add(lanes));
    }
}

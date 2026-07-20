// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! The three SIMD leaves, copied verbatim from `staged`. Keeping the inner math
//! byte-identical is the point: any A/B against `staged` measures only the
//! abstraction, not a different kernel. These are the only `unsafe` here outside
//! the scratch bridge.
//!
//! All three assume the intermediate strip is A-major with column stride
//! `A_PANEL = 16 = 2·LANES`.

use diskann_wide::arch::x86_64::V3;
use diskann_wide::{SIMDCast, SIMDDotProduct, SIMDMinMax, SIMDMulAdd, SIMDReinterpret, SIMDVector};

use crate::minmax::MinMaxCompensation;

diskann_wide::alias!(i16s = <V3>::i16x16);
diskann_wide::alias!(i32s = <V3>::i32x8);
diskann_wide::alias!(u32s = <V3>::u32x8);
diskann_wide::alias!(f32s = <V3>::f32x8);

/// Stage A — integer store-out micro-kernel: 16 A-rows × `UNROLL` B-cols.
///
/// # Safety
///
/// 1. `a_packed` points to a `16 × k` block-transposed `i16` block (`k` even).
/// 2. `b` points to `UNROLL` rows of `k` contiguous `u8` (`k` even).
/// 3. `partial` is valid for `UNROLL` columns of 16 `i32` at stride `b_stride`.
#[inline(always)]
pub(super) unsafe fn int_store_microkernel<const UNROLL: usize>(
    arch: V3,
    a_packed: *const i16,
    b: *const u8,
    k: usize,
    partial: *mut i32,
    b_stride: usize,
) {
    let mut p0 = [i32s::default(arch); UNROLL];
    let mut p1 = [i32s::default(arch); UNROLL];
    let offsets: [usize; UNROLL] = core::array::from_fn(|j| k * j);

    let a_pair_stride = 2 * i16s::LANES;
    let a_half = i16s::LANES;
    let pairs = k / 2;

    for p in 0..pairs {
        // SAFETY: precondition 1 — the query block has `pairs` col-pairs of 32 i16.
        let (a0, a1) = unsafe {
            (
                i16s::load_simd(arch, a_packed.add(a_pair_stride * p)),
                i16s::load_simd(arch, a_packed.add(a_pair_stride * p + a_half)),
            )
        };

        for j in 0..UNROLL {
            // SAFETY: precondition 2 — doc col j is `offsets[j]` in, `2*p+1 < k`.
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
        // SAFETY: precondition 3 — column j occupies [j*b_stride, j*b_stride+16) i32.
        unsafe {
            p0[j].store_simd(partial.add(j * b_stride));
            p1[j].store_simd(partial.add(j * b_stride + i32s::LANES));
        }
    }
}

/// Stage A (f32) — store-out micro-kernel: 16 A-rows × `UNROLL` B-cols of f32 IP.
///
/// Mirrors [`int_store_microkernel`] but for f32 in/out with no packing; the max
/// reduction is deferred to the `Max` reducer (via [`fold_strip`]).
///
/// # Safety
///
/// 1. `a_packed` points to a `16 × k` block-transposed `f32` block (`PACK = 1`).
/// 2. `b` points to `UNROLL` rows of `k` contiguous `f32`.
/// 3. `partial` is valid for `UNROLL` columns of 16 `f32` at stride `b_stride`.
#[inline(always)]
pub(super) unsafe fn f32_store_microkernel<const UNROLL: usize>(
    arch: V3,
    a_packed: *const f32,
    b: *const f32,
    k: usize,
    partial: *mut f32,
    b_stride: usize,
) {
    let mut p0 = [f32s::default(arch); UNROLL];
    let mut p1 = [f32s::default(arch); UNROLL];
    let offsets: [usize; UNROLL] = core::array::from_fn(|j| k * j);

    let a_stride = 2 * f32s::LANES;
    let a_half = f32s::LANES;

    for i in 0..k {
        // SAFETY: precondition 1 — the query block has `k` columns of 16 f32.
        let (a0, a1) = unsafe {
            (
                f32s::load_simd(arch, a_packed.add(a_stride * i)),
                f32s::load_simd(arch, a_packed.add(a_stride * i + a_half)),
            )
        };
        for j in 0..UNROLL {
            // SAFETY: precondition 2 — doc col j is `offsets[j]` in, `i < k`.
            let bj = unsafe { f32s::splat(arch, b.add(i + offsets[j]).read_unaligned()) };
            p0[j] = a0.mul_add_simd(bj, p0[j]);
            p1[j] = a1.mul_add_simd(bj, p1[j]);
        }
    }

    for j in 0..UNROLL {
        // SAFETY: precondition 3 — column j occupies [j*b_stride, j*b_stride+16) f32.
        unsafe {
            p0[j].store_simd(partial.add(j * b_stride));
            p1[j].store_simd(partial.add(j * b_stride + f32s::LANES));
        }
    }
}

/// Stage B — 4-bit MinMax dequant of one 16×`cols` A-major `i32` strip into f32.
///
/// # Safety
///
/// `acc` valid for `cols` columns of 16 `i32` (stride 16); `out` writable for the
/// same shape of `f32`; `q_meta.len() >= 16`; `d_meta.len() >= cols`.
#[inline(always)]
pub(super) unsafe fn score_strip(
    arch: V3,
    acc: *const i32,
    out: *mut f32,
    cols: usize,
    q_meta: &[MinMaxCompensation],
    d_meta: &[MinMaxCompensation],
    dim: f32,
) {
    let lanes = f32s::LANES;

    let mut qa = [0.0f32; 16];
    let mut qb = [0.0f32; 16];
    let mut qn = [0.0f32; 16];
    for i in 0..16 {
        let qm = q_meta[i];
        qa[i] = qm.a;
        qb[i] = qm.b;
        qn[i] = qm.n;
    }
    // SAFETY: each array holds exactly 16 = 2·LANES f32.
    let (qa0, qa1, qb0, qb1, qn0, qn1) = unsafe {
        (
            f32s::load_simd(arch, qa.as_ptr()),
            f32s::load_simd(arch, qa.as_ptr().add(lanes)),
            f32s::load_simd(arch, qb.as_ptr()),
            f32s::load_simd(arch, qb.as_ptr().add(lanes)),
            f32s::load_simd(arch, qn.as_ptr()),
            f32s::load_simd(arch, qn.as_ptr().add(lanes)),
        )
    };

    for (c, dm) in d_meta.iter().enumerate().take(cols) {
        let a_c = f32s::splat(arch, dm.a);
        let b_c = f32s::splat(arch, dm.b);
        let c_c = f32s::splat(arch, dm.n + dm.b * dim);
        let col = c * 16;
        // SAFETY: `col + 2·LANES <= cols*16`; `acc`/`out` valid for that many.
        unsafe {
            let raw0 = i32s::load_simd(arch, acc.add(col)).simd_cast();
            let raw1 = i32s::load_simd(arch, acc.add(col + lanes)).simd_cast();
            let s0 = a_c.mul_add_simd(qa0 * raw0, b_c.mul_add_simd(qn0, c_c * qb0));
            let s1 = a_c.mul_add_simd(qa1 * raw1, b_c.mul_add_simd(qn1, c_c * qb1));
            s0.store_simd(out.add(col));
            s1.store_simd(out.add(col + lanes));
        }
    }
}

/// Stage C — fold a 16×`cols` A-major f32 score strip into the 16-wide running max.
///
/// # Safety
///
/// `state` writable for 16 `f32`; `scores` valid for `cols` columns of 16 `f32`.
#[inline(always)]
pub(super) unsafe fn fold_strip(arch: V3, state: *mut f32, scores: *const f32, cols: usize) {
    let lanes = f32s::LANES;
    // SAFETY: `state` writable for 16; `scores` valid for `cols` columns of 16.
    unsafe {
        let mut a0 = f32s::load_simd(arch, state);
        let mut a1 = f32s::load_simd(arch, state.add(lanes));
        for c in 0..cols {
            let col = scores.add(c * 16);
            a0 = a0.max_simd(f32s::load_simd(arch, col));
            a1 = a1.max_simd(f32s::load_simd(arch, col.add(lanes)));
        }
        a0.store_simd(state);
        a1.store_simd(state.add(lanes));
    }
}

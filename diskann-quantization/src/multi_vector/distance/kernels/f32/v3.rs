// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! V3 (AVX2+FMA) f32 micro-kernel (16×4).

use diskann_wide::arch::x86_64::V3;
use diskann_wide::{SIMDMinMax, SIMDMulAdd, SIMDVector};

use super::super::Kernel;
use super::super::layouts;
use super::super::tiled_reduce::Reduce;
use super::F32Kernel;

diskann_wide::alias!(f32s = <V3>::f32x8);

// SAFETY: F32Kernel's `full_panel` and `partial_panel` only access
// A_PANEL(16) * k A elements, UNROLL * k B elements, and A_PANEL(16)
// scratch elements — all within the bounds guaranteed by `tiled_reduce`.
unsafe impl Kernel<V3> for F32Kernel<16> {
    type Left = layouts::BlockTransposed<f32, 16>;
    type Right = layouts::RowMajor<f32>;
    const A_PANEL: usize = 16;
    const B_PANEL: usize = 4;

    #[inline(always)]
    unsafe fn full_panel(arch: V3, a: *const f32, b: *const f32, k: usize, r: *mut f32) {
        // SAFETY: pointer validity per Kernel<V3> contract.
        unsafe { f32_microkernel::<{ Self::B_PANEL }>(arch, a, b, k, r) }
    }

    #[inline(always)]
    unsafe fn partial_panel(
        arch: V3,
        remainder: usize,
        a: *const f32,
        b: *const f32,
        k: usize,
        r: *mut f32,
    ) {
        // SAFETY: pointer validity per Kernel<V3> contract.
        unsafe {
            match remainder {
                1 => f32_microkernel::<1>(arch, a, b, k, r),
                2 => f32_microkernel::<2>(arch, a, b, k, r),
                3 => f32_microkernel::<3>(arch, a, b, k, r),
                _ => unreachable!(
                    "unexpected remainder {remainder} for B_PANEL={}",
                    Self::B_PANEL
                ),
            }
        }
    }
}

// ── V3 f32 micro-kernel ─────────────────────────────────────────

/// SIMD micro-kernel: processes 16 A rows × `UNROLL` B rows.
///
/// Accumulates via FMA into two `f32x8` register tiles, reduces across the
/// `UNROLL` B lanes with `max_simd`, then merges into the scratch buffer `r`.
///
/// # Safety
///
/// 1. `a_packed` must point to `A_PANEL(16) × k` contiguous `f32` values.
/// 2. `b` must point to `UNROLL` rows of `k` contiguous `f32` values.
/// 3. `r` must point to at least `A_PANEL(16)` writable `f32` values.
#[inline(always)]
unsafe fn f32_microkernel<const UNROLL: usize>(
    arch: V3,
    a_packed: *const f32,
    b: *const f32,
    k: usize,
    r: *mut f32,
) where
    [f32s; UNROLL]: Reduce<Element = f32s>,
{
    let op = |x: f32s, y: f32s| x.max_simd(y);

    let mut p0 = [f32s::default(arch); UNROLL];
    let mut p1 = [f32s::default(arch); UNROLL];
    let offsets: [usize; UNROLL] = core::array::from_fn(|i| k * i);

    let a_stride = 2 * f32s::LANES;
    let a_stride_half = f32s::LANES;

    for i in 0..k {
        // SAFETY: By preconditions 1 and 2; i < k and j < UNROLL.
        unsafe {
            let a0 = f32s::load_simd(arch, a_packed.add(a_stride * i));
            let a1 = f32s::load_simd(arch, a_packed.add(a_stride * i + a_stride_half));

            for j in 0..UNROLL {
                let bj = f32s::splat(arch, b.add(i + offsets[j]).read_unaligned());
                p0[j] = a0.mul_add_simd(bj, p0[j]);
                p1[j] = a1.mul_add_simd(bj, p1[j]);
            }
        }
    }

    // SAFETY: By precondition 3; LANES < A_PANEL so both halves are in-bounds.
    let mut r0 = unsafe { f32s::load_simd(arch, r) };
    // SAFETY: By precondition 3; r.add(LANES) is still within the A_PANEL-sized scratch.
    let mut r1 = unsafe { f32s::load_simd(arch, r.add(f32s::LANES)) };

    r0 = op(r0, p0.reduce(&op));
    r1 = op(r1, p1.reduce(&op));

    // SAFETY: By precondition 3.
    unsafe { r0.store_simd(r) };
    // SAFETY: By precondition 3; r.add(LANES) is still within the A_PANEL-sized scratch.
    unsafe { r1.store_simd(r.add(f32s::LANES)) };
}

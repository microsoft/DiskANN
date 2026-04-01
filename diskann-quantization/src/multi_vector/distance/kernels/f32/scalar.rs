// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Scalar (emulated) f32 micro-kernel (8×2) and Neon delegation.
//!
//! Uses `Emulated<f32, 8>` for arithmetic — 8 multiply-accumulate operations
//! per inner iteration, 8 scalar comparisons per `max_simd`. Geometry is
//! A_PANEL=8 (1 × f32x8), B_PANEL=2 (matching the `Strategy2x1` pattern used
//! by scalar distance functions elsewhere in the codebase).
//!
//! The inner loop uses separate multiply and add (`a * b + acc`) instead of
//! `mul_add_simd` to avoid calling into libm's software `fma()` routine on
//! x86-64 targets without hardware FMA support.

use std::marker::PhantomData;

use diskann_wide::arch::Scalar;
use diskann_wide::{SIMDMinMax, SIMDVector};

use super::super::Kernel;
use super::super::tiled_reduce::Reduce;
use super::F32Kernel;

diskann_wide::alias!(f32s = <Scalar>::f32x8);

// SAFETY: F32Kernel's `full_panel` and `remainder_dispatch` only access
// A_PANEL(8) * k A elements, UNROLL * k B elements, and A_PANEL(8)
// scratch elements — all within the bounds guaranteed by `tiled_reduce`.
unsafe impl Kernel<Scalar> for F32Kernel<Scalar, 8> {
    type AElem = f32;
    type BElem = f32;
    type APrepared = f32;
    type BPrepared = f32;
    const A_PANEL: usize = 8;
    const B_PANEL: usize = 2;

    fn new(_k: usize) -> Self {
        F32Kernel(PhantomData)
    }

    #[inline(always)]
    unsafe fn prepare_a(
        &mut self,
        _arch: Scalar,
        src: *const f32,
        _rows: usize,
        _k: usize,
    ) -> *const f32 {
        src
    }

    #[inline(always)]
    unsafe fn prepare_b(
        &mut self,
        _arch: Scalar,
        src: *const f32,
        _rows: usize,
        _k: usize,
    ) -> *const f32 {
        src
    }

    #[inline(always)]
    unsafe fn full_panel(arch: Scalar, a: *const f32, b: *const f32, k: usize, r: *mut f32) {
        // SAFETY: Caller guarantees pointer validity per Kernel<Scalar> contract.
        unsafe { scalar_f32_microkernel::<{ Self::B_PANEL }>(arch, a, b, k, r) }
    }

    #[inline(always)]
    unsafe fn remainder_dispatch(
        arch: Scalar,
        remainder: usize,
        a: *const f32,
        b: *const f32,
        k: usize,
        r: *mut f32,
    ) {
        // SAFETY: Caller guarantees pointer validity per Kernel<Scalar> contract.
        unsafe {
            match remainder {
                1 => scalar_f32_microkernel::<1>(arch, a, b, k, r),
                _ => {
                    debug_assert!(
                        false,
                        "unexpected remainder {remainder} for B_PANEL={}",
                        Self::B_PANEL
                    )
                }
            }
        }
    }
}

// ── Neon delegation ──────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
use diskann_wide::arch::aarch64::Neon;

// SAFETY: Delegates to the scalar microkernel via the zero-cost From<Neon> for Scalar
// conversion. The scalar emulated path is always correct.
#[cfg(target_arch = "aarch64")]
unsafe impl Kernel<Neon> for F32Kernel<Neon, 8> {
    type AElem = f32;
    type BElem = f32;
    type APrepared = f32;
    type BPrepared = f32;
    const A_PANEL: usize = 8;
    const B_PANEL: usize = 2;

    fn new(_k: usize) -> Self {
        F32Kernel(PhantomData)
    }

    #[inline(always)]
    unsafe fn prepare_a(
        &mut self,
        _arch: Neon,
        src: *const f32,
        _rows: usize,
        _k: usize,
    ) -> *const f32 {
        src
    }

    #[inline(always)]
    unsafe fn prepare_b(
        &mut self,
        _arch: Neon,
        src: *const f32,
        _rows: usize,
        _k: usize,
    ) -> *const f32 {
        src
    }

    #[inline(always)]
    unsafe fn full_panel(arch: Neon, a: *const f32, b: *const f32, k: usize, r: *mut f32) {
        // SAFETY: Caller guarantees pointer validity; Scalar::from(Neon) is safe.
        unsafe { scalar_f32_microkernel::<{ Self::B_PANEL }>(Scalar::from(arch), a, b, k, r) }
    }

    #[inline(always)]
    unsafe fn remainder_dispatch(
        arch: Neon,
        remainder: usize,
        a: *const f32,
        b: *const f32,
        k: usize,
        r: *mut f32,
    ) {
        // SAFETY: Caller guarantees pointer validity; Scalar::from(Neon) is safe.
        unsafe {
            let arch = Scalar::from(arch);
            match remainder {
                1 => scalar_f32_microkernel::<1>(arch, a, b, k, r),
                _ => {
                    debug_assert!(
                        false,
                        "unexpected remainder {remainder} for B_PANEL={}",
                        Self::B_PANEL
                    )
                }
            }
        }
    }
}

// ── Scalar f32 micro-kernel ──────────────────────────────────────

/// Emulated micro-kernel: processes 8 A rows × `UNROLL` B rows.
///
/// Uses separate multiply and add (`a * b + acc`) rather than `mul_add_simd`
/// to avoid calling libm's software `fma()` on x86-64 without hardware FMA.
/// A single register tile covers A_PANEL = 8 = f32s::LANES. B_PANEL=2
/// follows the `Strategy2x1` pattern from scalar distance functions.
///
/// # Safety
///
/// * `a_packed` must point to `A_PANEL(8) × k` contiguous `f32` values.
/// * `b` must point to `UNROLL` rows of `k` contiguous `f32` values.
/// * `r` must point to at least `A_PANEL(8)` writable `f32` values.
#[inline(always)]
pub(in crate::multi_vector::distance::kernels) unsafe fn scalar_f32_microkernel<
    const UNROLL: usize,
>(
    arch: Scalar,
    a_packed: *const f32,
    b: *const f32,
    k: usize,
    r: *mut f32,
) where
    [f32s; UNROLL]: Reduce<Element = f32s>,
{
    let op = |x: f32s, y: f32s| x.max_simd(y);

    let mut p0 = [f32s::default(arch); UNROLL];
    let offsets: [usize; UNROLL] = core::array::from_fn(|i| k * i);

    let a_stride = f32s::LANES;

    for i in 0..k {
        // SAFETY: a_packed points to A_PANEL * k contiguous f32s (one micro-panel).
        // b points to UNROLL rows of k contiguous f32s each. All reads are in-bounds.
        unsafe {
            let a0 = f32s::load_simd(arch, a_packed.add(a_stride * i));

            for j in 0..UNROLL {
                let bj = f32s::splat(arch, b.add(i + offsets[j]).read_unaligned());
                p0[j] = a0 * bj + p0[j];
            }
        }
    }

    // SAFETY: r points to at least A_PANEL = 8 writable f32s (1 × f32x8).
    let mut r0 = unsafe { f32s::load_simd(arch, r) };

    r0 = op(r0, p0.reduce(&op));

    // SAFETY: r points to at least A_PANEL = 8 writable f32s (1 × f32x8).
    unsafe { r0.store_simd(r) };
}

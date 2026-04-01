// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Scalar (emulated) and Neon f16 kernel implementations (8×2).
//!
//! Both A and B panels are converted from `half::f16` to `f32` in the
//! `prepare_a` / `prepare_b` hooks using scalar `half::f16::to_f32()`.
//! All micro-kernel work — both full-panel and remainder dispatch —
//! delegates to
//! [`scalar_f32_microkernel`](super::super::f32::scalar::scalar_f32_microkernel).

use std::marker::PhantomData;

use diskann_wide::arch::Scalar;

use super::super::Kernel;
use super::super::f32::scalar::scalar_f32_microkernel;
use super::F16Kernel;

// ── Scalar f16→f32 panel conversion ─────────────────────────────

/// Convert `count` f16 values to f32 using scalar operations.
///
/// # Safety
///
/// * `src` must point to `count` contiguous `half::f16` values.
/// * `dst` must point to writable space for `count` `f32` values.
#[inline(always)]
unsafe fn convert_f16_to_f32_scalar(src: *const half::f16, dst: *mut f32, count: usize) {
    for i in 0..count {
        // SAFETY: i < count, and caller guarantees src/dst validity for count elements.
        unsafe {
            let val = src.add(i).read_unaligned().to_f32();
            dst.add(i).write(val);
        }
    }
}

// ── Kernel<Scalar> ──────────────────────────────────────────────

// SAFETY: prepare_a / prepare_b convert f16→f32 into self-owned buffers sized
// to A_PANEL*k / B_PANEL*k (allocated in `new`). prepare_a zero-pads partial
// panels. full_panel and remainder_dispatch delegate to scalar_f32_microkernel
// which reads within A_PANEL*k + B_PANEL*k prepared elements and writes
// A_PANEL scratch f32s — all within caller-guaranteed bounds.
unsafe impl Kernel<Scalar> for F16Kernel<Scalar, 8> {
    type AElem = half::f16;
    type BElem = half::f16;
    type APrepared = f32;
    type BPrepared = f32;
    const A_PANEL: usize = 8;
    const B_PANEL: usize = 2;

    fn new(k: usize) -> Self {
        F16Kernel {
            a_buf: vec![0.0f32; Self::A_PANEL * k],
            b_buf: vec![0.0f32; Self::B_PANEL * k],
            _arch: PhantomData,
        }
    }

    #[inline(always)]
    unsafe fn prepare_a(
        &mut self,
        _arch: Scalar,
        src: *const half::f16,
        rows: usize,
        k: usize,
    ) -> *const f32 {
        let dst = self.a_buf.as_mut_ptr();
        // SAFETY: Caller guarantees src has rows * k f16 values. self.a_buf
        // has A_PANEL * k f32 slots (rows <= A_PANEL).
        unsafe { convert_f16_to_f32_scalar(src, dst, rows * k) };
        // Zero-pad the remainder so the micro-kernel sees a full A_PANEL panel.
        self.a_buf[rows * k..].fill(0.0);
        dst
    }

    #[inline(always)]
    unsafe fn prepare_b(
        &mut self,
        _arch: Scalar,
        src: *const half::f16,
        rows: usize,
        k: usize,
    ) -> *const f32 {
        let dst = self.b_buf.as_mut_ptr();
        // SAFETY: Caller guarantees src has rows * k f16 values. self.b_buf
        // has B_PANEL * k f32 slots (rows <= B_PANEL).
        unsafe { convert_f16_to_f32_scalar(src, dst, rows * k) };
        dst
    }

    #[inline(always)]
    unsafe fn full_panel(arch: Scalar, a: *const f32, b: *const f32, k: usize, r: *mut f32) {
        // A and B are already f32 after preparation — reuse the f32 micro-kernel.
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
        // Both A and B are prepared f32 — reuse the f32 micro-kernel.
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

// SAFETY: Delegates to the scalar implementation via zero-cost From<Neon> for Scalar.
#[cfg(target_arch = "aarch64")]
unsafe impl Kernel<Neon> for F16Kernel<Neon, 8> {
    type AElem = half::f16;
    type BElem = half::f16;
    type APrepared = f32;
    type BPrepared = f32;
    const A_PANEL: usize = 8;
    const B_PANEL: usize = 2;

    fn new(k: usize) -> Self {
        F16Kernel {
            a_buf: vec![0.0f32; Self::A_PANEL * k],
            b_buf: vec![0.0f32; Self::B_PANEL * k],
            _arch: PhantomData,
        }
    }

    #[inline(always)]
    unsafe fn prepare_a(
        &mut self,
        _arch: Neon,
        src: *const half::f16,
        rows: usize,
        k: usize,
    ) -> *const f32 {
        let dst = self.a_buf.as_mut_ptr();
        // SAFETY: Caller guarantees pointer validity; scalar conversion is always correct.
        unsafe { convert_f16_to_f32_scalar(src, dst, rows * k) };
        self.a_buf[rows * k..].fill(0.0);
        dst
    }

    #[inline(always)]
    unsafe fn prepare_b(
        &mut self,
        _arch: Neon,
        src: *const half::f16,
        rows: usize,
        k: usize,
    ) -> *const f32 {
        let dst = self.b_buf.as_mut_ptr();
        // SAFETY: Caller guarantees pointer validity.
        unsafe { convert_f16_to_f32_scalar(src, dst, rows * k) };
        dst
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
        // Both A and B are prepared f32 — reuse the f32 micro-kernel.
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

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Scalar (emulated) f16 kernel implementation (8×2).
//!
//! Both A and B panels are converted from `half::f16` to `f32` in the
//! `prepare_a` / `prepare_b` hooks using [`CastFromSlice`]. All micro-kernel
//! work — both full-panel and remainder dispatch — delegates to
//! [`scalar_f32_microkernel`](super::super::f32::scalar::scalar_f32_microkernel).

use diskann_vector::conversion::CastFromSlice;
use diskann_wide::arch::Scalar;

use super::super::Kernel;
use super::super::f32::scalar::scalar_f32_microkernel;
use super::F16Kernel;

// ── Kernel<Scalar> ──────────────────────────────────────────────

// SAFETY: prepare_a / prepare_b convert f16→f32 into self-owned buffers sized
// to A_PANEL*k / B_PANEL*k (allocated in `new`). full_panel and
// remainder_dispatch delegate to scalar_f32_microkernel which reads within
// A_PANEL*k + B_PANEL*k prepared elements and writes A_PANEL scratch f32s —
// all within caller-guaranteed bounds.
unsafe impl Kernel<Scalar> for F16Kernel<8> {
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
        }
    }

    #[inline(always)]
    unsafe fn prepare_a(&mut self, _arch: Scalar, src: *const half::f16, _k: usize) -> *const f32 {
        // SAFETY: Caller guarantees src points to A_PANEL * k contiguous f16 values.
        let src_slice = unsafe { std::slice::from_raw_parts(src, self.a_buf.len()) };
        self.a_buf.as_mut_slice().cast_from_slice(src_slice);
        self.a_buf.as_ptr()
    }

    #[inline(always)]
    unsafe fn prepare_b(
        &mut self,
        _arch: Scalar,
        src: *const half::f16,
        rows: usize,
        k: usize,
    ) -> *const f32 {
        let count = rows * k;
        // SAFETY: Caller guarantees src points to rows * k contiguous f16 values.
        let src_slice = unsafe { std::slice::from_raw_parts(src, count) };
        self.b_buf[..count].cast_from_slice(src_slice);
        self.b_buf.as_ptr()
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

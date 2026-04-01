// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! V3 (AVX2+FMA) and V4 (AVX-512) f16 kernel implementations (16×4).
//!
//! Both A and B panels are converted from `half::f16` to `f32` in the
//! `prepare_a` / `prepare_b` hooks using hardware `_mm256_cvtph_ps` (F16C)
//! via the [`From<f16x8> for f32x8`] trait impl. All micro-kernel work —
//! both full-panel and remainder dispatch — delegates to
//! [`f32_microkernel`](super::super::f32::v3::f32_microkernel).

use std::marker::PhantomData;

use diskann_wide::SIMDVector;
use diskann_wide::arch::x86_64::{V3, V4};

use super::super::Kernel;
use super::super::f32::v3::f32_microkernel;
use super::F16Kernel;

diskann_wide::alias!(f32s = <V3>::f32x8);
diskann_wide::alias!(f16s = <V3>::f16x8);

// ── SIMD f16→f32 panel conversion ───────────────────────────────

/// Convert `count` f16 values to f32 using SIMD.
///
/// Processes 8 values at a time via `_mm256_cvtph_ps`. Handles a scalar tail
/// when `count` is not a multiple of 8.
///
/// # Safety
///
/// * `src` must point to `count` contiguous `half::f16` values.
/// * `dst` must point to writable space for `count` `f32` values.
#[inline(always)]
unsafe fn convert_f16_to_f32_simd(arch: V3, src: *const half::f16, dst: *mut f32, count: usize) {
    let full_chunks = count / 8;
    let tail = count % 8;

    for i in 0..full_chunks {
        let offset = i * 8;
        // SAFETY: offset + 8 <= count for full chunks.
        unsafe {
            let h = f16s::load_simd(arch, src.add(offset));
            let f: f32s = h.into();
            f.store_simd(dst.add(offset));
        }
    }

    // Scalar tail for remaining elements.
    let tail_start = full_chunks * 8;
    for i in 0..tail {
        // SAFETY: tail_start + i < count.
        unsafe {
            let val = src.add(tail_start + i).read_unaligned().to_f32();
            dst.add(tail_start + i).write(val);
        }
    }
}

// ── Kernel<V3> ──────────────────────────────────────────────────

// SAFETY: prepare_a / prepare_b convert f16→f32 into self-owned buffers sized
// to A_PANEL*k / B_PANEL*k (allocated in `new`). prepare_a zero-pads partial
// panels. full_panel and remainder_dispatch delegate to f32_microkernel which
// reads within A_PANEL*k + B_PANEL*k prepared elements and writes A_PANEL
// scratch f32s — all within caller-guaranteed bounds.
unsafe impl Kernel<V3> for F16Kernel<V3, 16> {
    type AElem = half::f16;
    type BElem = half::f16;
    type APrepared = f32;
    type BPrepared = f32;
    const A_PANEL: usize = 16;
    const B_PANEL: usize = 4;

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
        arch: V3,
        src: *const half::f16,
        rows: usize,
        k: usize,
    ) -> *const f32 {
        let dst = self.a_buf.as_mut_ptr();
        // SAFETY: Caller guarantees src has rows * k f16 values. self.a_buf
        // has A_PANEL * k f32 slots (rows <= A_PANEL).
        unsafe { convert_f16_to_f32_simd(arch, src, dst, rows * k) };
        // Zero-pad the remainder so the micro-kernel sees a full A_PANEL panel.
        self.a_buf[rows * k..].fill(0.0);
        dst
    }

    #[inline(always)]
    unsafe fn prepare_b(
        &mut self,
        arch: V3,
        src: *const half::f16,
        rows: usize,
        k: usize,
    ) -> *const f32 {
        let dst = self.b_buf.as_mut_ptr();
        // SAFETY: Caller guarantees src has rows * k f16 values. self.b_buf
        // has B_PANEL * k f32 slots (rows <= B_PANEL).
        unsafe { convert_f16_to_f32_simd(arch, src, dst, rows * k) };
        dst
    }

    #[inline(always)]
    unsafe fn full_panel(arch: V3, a: *const f32, b: *const f32, k: usize, r: *mut f32) {
        // A and B are already f32 after preparation — reuse the f32 micro-kernel.
        // SAFETY: Caller guarantees pointer validity per Kernel<V3> contract.
        unsafe { f32_microkernel::<{ Self::B_PANEL }>(arch, a, b, k, r) }
    }

    #[inline(always)]
    unsafe fn remainder_dispatch(
        arch: V3,
        remainder: usize,
        a: *const f32,
        b: *const f32,
        k: usize,
        r: *mut f32,
    ) {
        // Both A and B are prepared f32 — reuse the f32 micro-kernel.
        // SAFETY: Caller guarantees pointer validity per Kernel<V3> contract.
        unsafe {
            match remainder {
                1 => f32_microkernel::<1>(arch, a, b, k, r),
                2 => f32_microkernel::<2>(arch, a, b, k, r),
                3 => f32_microkernel::<3>(arch, a, b, k, r),
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

// ── Kernel<V4> delegation ───────────────────────────────────────

// SAFETY: V3 is a strict subset of V4. All V3 operations are valid on V4.
// Buffer ownership and bounds reasoning are identical to the V3 impl above.
unsafe impl Kernel<V4> for F16Kernel<V4, 16> {
    type AElem = half::f16;
    type BElem = half::f16;
    type APrepared = f32;
    type BPrepared = f32;
    const A_PANEL: usize = 16;
    const B_PANEL: usize = 4;

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
        arch: V4,
        src: *const half::f16,
        rows: usize,
        k: usize,
    ) -> *const f32 {
        let dst = self.a_buf.as_mut_ptr();
        // SAFETY: V3 ⊆ V4; caller guarantees pointer validity.
        unsafe { convert_f16_to_f32_simd(V3::from(arch), src, dst, rows * k) };
        self.a_buf[rows * k..].fill(0.0);
        dst
    }

    #[inline(always)]
    unsafe fn prepare_b(
        &mut self,
        arch: V4,
        src: *const half::f16,
        rows: usize,
        k: usize,
    ) -> *const f32 {
        let dst = self.b_buf.as_mut_ptr();
        // SAFETY: V3 ⊆ V4; caller guarantees pointer validity.
        unsafe { convert_f16_to_f32_simd(V3::from(arch), src, dst, rows * k) };
        dst
    }

    #[inline(always)]
    unsafe fn full_panel(arch: V4, a: *const f32, b: *const f32, k: usize, r: *mut f32) {
        // SAFETY: V3 ⊆ V4; caller guarantees pointer validity.
        unsafe { f32_microkernel::<{ Self::B_PANEL }>(V3::from(arch), a, b, k, r) }
    }

    #[inline(always)]
    unsafe fn remainder_dispatch(
        arch: V4,
        remainder: usize,
        a: *const f32,
        b: *const f32,
        k: usize,
        r: *mut f32,
    ) {
        // Both A and B are prepared f32 — reuse the f32 micro-kernel.
        // SAFETY: V3 ⊆ V4; caller guarantees pointer validity.
        unsafe {
            let arch = V3::from(arch);
            match remainder {
                1 => f32_microkernel::<1>(arch, a, b, k, r),
                2 => f32_microkernel::<2>(arch, a, b, k, r),
                3 => f32_microkernel::<3>(arch, a, b, k, r),
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

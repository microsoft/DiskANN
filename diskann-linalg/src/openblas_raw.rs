/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Raw CBLAS bindings for OpenBLAS. Used by the `openblas` feature gate to
//! provide `sgemm_aat_lower_openblas` — a real ssyrk that delivers the ~50%
//! FLOP savings faer's `triangular::matmul` failed to provide in our
//! microbench. Preferred over MKL because OpenBLAS lacks MKL's anti-AMD
//! CPUID dispatch.
//!
//! Only the subset of CBLAS we actually call is declared. Linker setup is in
//! `diskann-linalg/build.rs`.

#![allow(non_camel_case_types)]
#![allow(dead_code)]

use std::os::raw::c_int;

#[repr(C)]
pub enum CBLAS_LAYOUT {
    CblasRowMajor = 101,
    CblasColMajor = 102,
}

#[repr(C)]
pub enum CBLAS_TRANSPOSE {
    CblasNoTrans = 111,
    CblasTrans = 121,
    CblasConjTrans = 122,
}

#[repr(C)]
pub enum CBLAS_UPLO {
    CblasUpper = 121,
    CblasLower = 122,
}

unsafe extern "C" {
    pub unsafe fn cblas_sgemm(
        layout: CBLAS_LAYOUT,
        transa: CBLAS_TRANSPOSE,
        transb: CBLAS_TRANSPOSE,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: f32,
        c: *mut f32,
        ldc: c_int,
    );

    pub unsafe fn cblas_ssyrk(
        layout: CBLAS_LAYOUT,
        uplo: CBLAS_UPLO,
        trans: CBLAS_TRANSPOSE,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        beta: f32,
        c: *mut f32,
        ldc: c_int,
    );

    pub unsafe fn openblas_set_num_threads(num_threads: c_int);
}

// PiPNN calls cblas_ssyrk from rayon worker threads. Without pinning OpenBLAS
// to a single thread, every worker spins up OpenBLAS's own pool → N·N
// oversubscription. Pin once at first call.
static OPENBLAS_PIN: std::sync::Once = std::sync::Once::new();

#[inline]
fn ensure_openblas_single_threaded() {
    OPENBLAS_PIN.call_once(|| {
        // SAFETY: openblas_set_num_threads is exported by every recent OpenBLAS.
        unsafe { openblas_set_num_threads(1) };
    });
}

/// `C = A · Aᵀ` for `m × k` row-major A, writing only the LOWER triangle of C.
/// Uses OpenBLAS ssyrk with `CblasLower` + `CblasNoTrans`.
///
/// In row-major storage, the lower triangle of A·Aᵀ contains entries
/// `C[i][j]` for `j ≤ i`. CBLAS_ROW_MAJOR + CblasLower writes those.
pub fn sgemm_aat_lower_openblas(a: &[f32], m: usize, k: usize, c: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(c.len(), m * m);
    ensure_openblas_single_threaded();
    let m_i = m as c_int;
    let k_i = k as c_int;
    // SAFETY: pointers come from live slices; lda=k and ldc=m match row-major
    // stride; alpha/beta are plain scalars. OpenBLAS is linked via build.rs.
    unsafe {
        cblas_ssyrk(
            CBLAS_LAYOUT::CblasRowMajor,
            CBLAS_UPLO::CblasLower,
            CBLAS_TRANSPOSE::CblasNoTrans,
            m_i,
            k_i,
            1.0,
            a.as_ptr(),
            k_i,
            0.0,
            c.as_mut_ptr(),
            m_i,
        );
    }
}

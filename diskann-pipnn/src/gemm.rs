/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! GEMM abstraction using OpenBLAS (via cblas_sgemm) for maximum performance.
//!
//! Falls back to matrixmultiply if OpenBLAS is not available.

/// Compute C = A * B^T where A is m x k and B is n x k (both row-major).
/// Result C is m x n (row-major).
///
/// Uses OpenBLAS cblas_sgemm for near-peak FLOPS on AMD EPYC.
#[inline]
pub fn sgemm_abt(
    a: &[f32], m: usize, k: usize,
    b: &[f32], n: usize,
    c: &mut [f32],
) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), n * k);
    debug_assert_eq!(c.len(), m * n);

    // CblasRowMajor=101, CblasNoTrans=111, CblasTrans=112
    unsafe {
        cblas_sys::cblas_sgemm(
            cblas_sys::CBLAS_LAYOUT::CblasRowMajor,
            cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
            cblas_sys::CBLAS_TRANSPOSE::CblasTrans,
            m as i32,       // M: rows of A
            n as i32,       // N: rows of B (cols of C)
            k as i32,       // K: cols of A
            1.0,            // alpha
            a.as_ptr(),
            k as i32,       // lda
            b.as_ptr(),
            k as i32,       // ldb (row-major B, transposed)
            0.0,            // beta
            c.as_mut_ptr(),
            n as i32,       // ldc
        );
    }
}

/// Compute C = A * A^T where A is m x k (row-major).
/// Result C is m x m (row-major).
#[inline]
pub fn sgemm_aat(a: &[f32], m: usize, k: usize, c: &mut [f32]) {
    sgemm_abt(a, m, k, a, m, c);
}

extern "C" {
    fn openblas_set_num_threads(num_threads: i32);
}

/// Set OpenBLAS thread count at runtime.
/// Use num_threads > 1 for large single GEMM calls (e.g., top-level partition).
/// Use num_threads = 1 when outer parallelism (rayon) handles concurrency.
pub fn set_blas_threads(num_threads: usize) {
    unsafe {
        openblas_set_num_threads(num_threads as i32);
    }
}

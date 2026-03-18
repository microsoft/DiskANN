/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! GEMM abstraction using OpenBLAS (via cblas_sgemm) for maximum performance.

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgemm_abt_identity() {
        // A * I^T should equal A when B is the identity.
        // A = 2x3, I = 3x3 identity, result = 2x3.
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let identity = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let mut c = vec![0.0f32; 6]; // 2x3
        sgemm_abt(&a, 2, 3, &identity, 3, &mut c);

        for i in 0..6 {
            assert!(
                (c[i] - a[i]).abs() < 1e-6,
                "A*I^T != A at index {}: got {}, expected {}",
                i, c[i], a[i]
            );
        }
    }

    #[test]
    fn test_sgemm_abt_known() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        // A * B^T = [[1*5+2*6, 1*7+2*8], [3*5+4*6, 3*7+4*8]]
        //         = [[17, 23], [39, 53]]
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2
        let mut c = vec![0.0f32; 4];       // 2x2
        sgemm_abt(&a, 2, 2, &b, 2, &mut c);

        let expected = vec![17.0, 23.0, 39.0, 53.0];
        for i in 0..4 {
            assert!(
                (c[i] - expected[i]).abs() < 1e-5,
                "mismatch at {}: got {}, expected {}",
                i, c[i], expected[i]
            );
        }
    }

    #[test]
    fn test_sgemm_aat_symmetric() {
        // A * A^T should always be symmetric.
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; // 3x3
        let mut c = vec![0.0f32; 9]; // 3x3
        sgemm_aat(&a, 3, 3, &mut c);

        for i in 0..3 {
            for j in (i + 1)..3 {
                assert!(
                    (c[i * 3 + j] - c[j * 3 + i]).abs() < 1e-5,
                    "A*A^T not symmetric at ({},{}): {} vs {}",
                    i, j, c[i * 3 + j], c[j * 3 + i]
                );
            }
        }

        // Diagonal should be non-negative (sum of squares).
        for i in 0..3 {
            assert!(c[i * 3 + i] >= 0.0, "diagonal at ({},{}) is negative", i, i);
        }

        // Check known values: row 0 = [1,2,3]
        // (A*A^T)[0][0] = 1^2 + 2^2 + 3^2 = 14
        assert!((c[0] - 14.0).abs() < 1e-5, "got {}", c[0]);
    }

    #[test]
    fn test_sgemm_abt_rectangular() {
        // A = 2x3, B = 4x3 -> C = 2x4.
        // A = [[1,0,0],[0,1,0]], B = [[1,0,0],[0,1,0],[0,0,1],[1,1,0]]
        // A * B^T = [[1,0,0,1],[0,1,0,1]]
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let mut c = vec![0.0f32; 8]; // 2x4
        sgemm_abt(&a, 2, 3, &b, 4, &mut c);

        let expected = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        for i in 0..8 {
            assert!(
                (c[i] - expected[i]).abs() < 1e-6,
                "rectangular GEMM mismatch at {}: got {}, expected {}",
                i, c[i], expected[i]
            );
        }
    }

    #[test]
    fn test_sgemm_abt_large() {
        // 64x128 * 64x128 -> 64x64.
        // Fill with 1s: A * B^T where both are all-ones should give k=128 everywhere.
        let m = 64;
        let k = 128;
        let n = 64;
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; n * k];
        let mut c = vec![0.0f32; m * n];
        sgemm_abt(&a, m, k, &b, n, &mut c);

        for i in 0..(m * n) {
            assert!(
                (c[i] - k as f32).abs() < 1e-3,
                "large GEMM all-ones mismatch at {}: got {}, expected {}",
                i, c[i], k as f32
            );
        }
    }

    #[test]
    fn test_sgemm_abt_zeros() {
        // All-zero input should produce all-zero output.
        let m = 4;
        let k = 8;
        let n = 3;
        let a = vec![0.0f32; m * k];
        let b = vec![0.0f32; n * k];
        let mut c = vec![99.0f32; m * n]; // pre-fill with non-zero to verify overwrite
        sgemm_abt(&a, m, k, &b, n, &mut c);

        for i in 0..(m * n) {
            assert!(
                c[i].abs() < 1e-6,
                "all-zero GEMM should produce zero at {}: got {}",
                i, c[i]
            );
        }
    }

    #[test]
    fn test_sgemm_abt_negative() {
        // A = [[-1,-2],[-3,-4]], B = [[5,6],[7,8]]
        // A * B^T = [[-1*5+(-2)*6, -1*7+(-2)*8], [-3*5+(-4)*6, -3*7+(-4)*8]]
        //         = [[-17, -23], [-39, -53]]
        let a = vec![-1.0, -2.0, -3.0, -4.0]; // 2x2
        let b = vec![5.0, 6.0, 7.0, 8.0];     // 2x2
        let mut c = vec![0.0f32; 4];
        sgemm_abt(&a, 2, 2, &b, 2, &mut c);

        let expected = vec![-17.0, -23.0, -39.0, -53.0];
        for i in 0..4 {
            assert!(
                (c[i] - expected[i]).abs() < 1e-5,
                "negative GEMM mismatch at {}: got {}, expected {}",
                i, c[i], expected[i]
            );
        }
    }

    #[test]
    fn test_sgemm_abt_single_element() {
        // 1x1 * 1x1^T = product of the two scalars.
        let a = vec![3.0f32];
        let b = vec![5.0f32];
        let mut c = vec![0.0f32; 1];
        sgemm_abt(&a, 1, 1, &b, 1, &mut c);
        assert!((c[0] - 15.0).abs() < 1e-6);
    }
}

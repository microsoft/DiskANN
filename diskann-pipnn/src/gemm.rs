/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! GEMM abstraction using diskann-linalg (faer backend), consistent with DiskANN.

use diskann_linalg::Transpose;

/// Compute C = A * B^T where A is m x k and B is n x k (both row-major).
/// Result C is m x n (row-major).
///
/// Uses diskann-linalg's sgemm backed by faer, the same GEMM DiskANN uses internally.
#[inline]
pub fn sgemm_abt(a: &[f32], m: usize, k: usize, b: &[f32], n: usize, c: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), n * k);
    debug_assert_eq!(c.len(), m * n);

    diskann_linalg::sgemm(
        Transpose::None,
        Transpose::Ordinary,
        m,
        n,
        k,
        1.0,
        a,
        b,
        None,
        c,
    );
}

/// Compute C = A * A^T where A is m x k (row-major).
/// Result C is m x m (row-major).
#[inline]
pub fn sgemm_aat(a: &[f32], m: usize, k: usize, c: &mut [f32]) {
    sgemm_abt(a, m, k, a, m, c);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgemm_abt_identity() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let identity = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut c = vec![0.0f32; 6];
        sgemm_abt(&a, 2, 3, &identity, 3, &mut c);
        for i in 0..6 {
            assert!(
                (c[i] - a[i]).abs() < 1e-6,
                "A*I^T != A at {}: got {}, expected {}",
                i,
                c[i],
                a[i]
            );
        }
    }

    #[test]
    fn test_sgemm_abt_known() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f32; 4];
        sgemm_abt(&a, 2, 2, &b, 2, &mut c);
        let expected = vec![17.0, 23.0, 39.0, 53.0];
        for i in 0..4 {
            assert!(
                (c[i] - expected[i]).abs() < 1e-5,
                "mismatch at {}: got {}, expected {}",
                i,
                c[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_sgemm_aat_symmetric() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut c = vec![0.0f32; 9];
        sgemm_aat(&a, 3, 3, &mut c);
        for i in 0..3 {
            for j in (i + 1)..3 {
                assert!(
                    (c[i * 3 + j] - c[j * 3 + i]).abs() < 1e-5,
                    "not symmetric at ({},{})",
                    i,
                    j
                );
            }
        }
        assert!((c[0] - 14.0).abs() < 1e-5, "got {}", c[0]);
    }

    #[test]
    fn test_sgemm_abt_rectangular() {
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let mut c = vec![0.0f32; 8];
        sgemm_abt(&a, 2, 3, &b, 4, &mut c);
        let expected = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        for i in 0..8 {
            assert!(
                (c[i] - expected[i]).abs() < 1e-6,
                "rectangular mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn test_sgemm_abt_large() {
        let m = 64;
        let k = 128;
        let n = 64;
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; n * k];
        let mut c = vec![0.0f32; m * n];
        sgemm_abt(&a, m, k, &b, n, &mut c);
        for i in 0..(m * n) {
            assert!((c[i] - k as f32).abs() < 1e-3, "large mismatch at {}", i);
        }
    }

    #[test]
    fn test_sgemm_abt_zeros() {
        let m = 4;
        let k = 8;
        let n = 3;
        let a = vec![0.0f32; m * k];
        let b = vec![0.0f32; n * k];
        let mut c = vec![99.0f32; m * n];
        sgemm_abt(&a, m, k, &b, n, &mut c);
        for i in 0..(m * n) {
            assert!(c[i].abs() < 1e-6, "zeros mismatch at {}", i);
        }
    }

    #[test]
    fn test_sgemm_abt_negative() {
        let a = vec![-1.0, -2.0, -3.0, -4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f32; 4];
        sgemm_abt(&a, 2, 2, &b, 2, &mut c);
        let expected = vec![-17.0, -23.0, -39.0, -53.0];
        for i in 0..4 {
            assert!(
                (c[i] - expected[i]).abs() < 1e-5,
                "negative mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn test_sgemm_abt_single_element() {
        let a = vec![3.0f32];
        let b = vec![5.0f32];
        let mut c = vec![0.0f32; 1];
        sgemm_abt(&a, 1, 1, &b, 1, &mut c);
        assert!((c[0] - 15.0).abs() < 1e-6);
    }
}

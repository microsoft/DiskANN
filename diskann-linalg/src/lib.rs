/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub mod common;
pub use common::Transpose;

mod faer;
use faer::{random_distance_preserving_matrix_impl, sgemm_impl, svd_into_impl};
use rand::Rng;

// Make the reference implementation available for internal testing.
#[cfg(test)]
mod reference;

/// Matrix-matrix multiplication for implicit row-major matrices `a` and `b` using the
/// implicit row-major matrix `c` as the destination.
///
/// Performs one of the following operations:
/// ```ignore
/// 1. c = [beta * c] + alpha * a * b
/// 2. c = [beta * c] + alpha * a' * b
/// 3. c = [beta * c] + alpha * a * b'
/// 3. c = [beta * c] + alpha * a' * b'
/// ```
/// Where `x'` indicates the ordinary transpose of `x`.
///
/// If `beta` is `None`, the destination `c` is completely over-written.
///
/// * `atranspose`: Whether `a` should be interpreted as an in-place transpose.
/// * `btranspose`: Whether `b` should be interpreted as an in-place transpose.
/// * `m`: The number of rows in `c`. Additionally:
///     - If `!atranspose.is_transpose()`, this is the number of rows in `a`.
///     - If `atranspose.is_transpose()`, this is the number of rows in `a`.
/// * `n`: The number of columns `c`. Additionally:
///     - If `!btranspose.is_transpose()`, this is the number of columns in `b`.
///     - If `btranspose.is_transpose()`, this is the number of columns in `b`.
/// * `k`: The number of columns in matrix `a` and the number of rows in matrix `b`.
/// * `k`: Refer to the following:
///     - If `!atranspose.is_transpose()`, this is the number of columns in `a`.
///       Otherwise, this is the number of rows in `a`.
///     - If `!btranspose.is_transpose()`, this is the number of rows in `b`.
///       Otherwise, this is the number of columns in `b`.
/// * `alpha`: Scaling parameter for the operation `a * b`.
/// * `a`: The matrix `a` with dimension `m x k` (potentially after transposing).
/// * `b`: The matrix `b` with dimension `k x n` (potentially after transposing).
/// * `beta`: Optional scaling parameter for the matrix `c`. If `None`, then `c` will be
///   overwritten entirely.
/// * `c`: The output matrix with dimension `m x n`.
///
/// # Note
///
/// This inteface is a simplified version of the full cblas `sgemm` interface, namely that it
///
/// 1. Does not support column-major layouts
/// 2. Does not allow for arbitrary strides in the leading dimension of the matrices.
///
/// This is to support the common-case in DiskANN that uses a Row-Major layout and always
/// uses dense matrices.
///
/// If the more esoteric features of the cblas `sgemm` API are needed, we can provide
/// that as an interface extension.
///
/// # Panics
///
/// Panics if
/// * `a.len() != m * k`
/// * `b.len() != k * n`
/// * `c.len() != m * n`.
///
/// Additionally, if MKL is used, panics if any of `k, m, n` is not representable as a
/// signed 32-bit integer due to `cblas` limitations.
#[allow(clippy::too_many_arguments)]
pub fn sgemm(
    atranspose: Transpose,
    btranspose: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: Option<f32>,
    c: &mut [f32],
) {
    // Check size requirements.
    assert_eq!(
        a.len(),
        m * k,
        "expected {}x{} matrix `a` to have length {}, instead got {}",
        m,
        k,
        m * k,
        a.len()
    );
    assert_eq!(
        b.len(),
        k * n,
        "expected {}x{} matrix `b` to have length {}, instead got {}",
        k,
        n,
        k * n,
        b.len()
    );
    assert_eq!(
        c.len(),
        m * n,
        "expected {}x{} matrix `c` to have length {}, instead got {}",
        m,
        n,
        m * n,
        c.len()
    );

    // Invoke the actual implementation.
    sgemm_impl(atranspose, btranspose, m, n, k, alpha, a, b, beta, c)
}

/// Compute the SVD of the provided matrix implicit row-major matrix `data`.
///
/// * `m`: The number of rows in `a`.
/// * `n`: The number of columns in `a`.
/// * `a`: The data matrix to decompose with dimensiuon `m x n` stored in Row-Major order [Note 1].
/// * `singular_values`: Contains the singular values of `a` sorted so that
///   `singular_values[i] ≥ singular_values[i+1]`.
/// * `u`: Contains the `m x m` unitary matrix in Row-Major order.
/// * `vt`: Contains the `n x n` unitary matrix in Column-Major order [Note 2].
///
/// # Notes
///
/// 1. Due to the contract offered by `lapacke`, callers of this function must assume that
///    the contents of `a` are left in an undefined state after this function.
///
///    See: https://netlib.org/lapack/explore-html//df/d22/group__gesdd_gab9ffdde22b38f0cc442e44cbea23818f.html
///
/// 2. Similar to #1, the restriction that `vt` is transposed is a lapack byproduct.
///
/// # Panics
///
/// Panics if
///
/// * `a.len() != m * n`
/// * `singular_values.len() != min(m, n)`
/// * `u.len() != m * m`.
/// * `vt.len() != n * n`.
///
/// Additionally, if MKL is used, panics if any either `m` or `n` is not representable
/// as a signed 32-bit integer due to `cblas` limitations.
pub fn svd_into(
    m: usize,
    n: usize,
    a: &mut [f32],
    singular_values: &mut [f32],
    u: &mut [f32],
    vt: &mut [f32],
) -> Result<(), impl std::error::Error + 'static> {
    // Check size requirements.
    assert_eq!(a.len(), m * n);
    assert_eq!(singular_values.len(), m.min(n));
    assert_eq!(u.len(), m * m);
    assert_eq!(vt.len(), n * n);

    // Invoke the actual implementation.
    svd_into_impl(m, n, a, singular_values, u, vt)
}

/// Construct a random `dim x dim` distance preserving matrix.
///
/// Practically speaking, the returned matrix should be orthogonal with a determinant of
/// either +1 or -1.
pub fn random_distance_preserving_matrix<T: Rng + ?Sized>(dim: usize, rng: &mut T) -> Vec<f32> {
    random_distance_preserving_matrix_impl(dim, rng)
}

#[cfg(test)]
mod tests {
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use rand::{distr::Distribution, rngs::StdRng, SeedableRng};
    use rand_distr::StandardNormal;
    use serde::Deserialize;

    use super::*;
    use crate::reference;

    ////////////////////////
    // Simple SGEMM tests //
    ////////////////////////

    #[test]
    fn test_reference_implementation() {
        let problems = reference::test_sgemm_problems();
        for (i, problem) in problems.iter().enumerate() {
            let result = problem.check(sgemm);
            if let Err(err) = result {
                panic!("{} on iteration {}. Problem: {:?}", err, i, problem);
            }
        }
    }

    ///////////////
    // SVD Tests //
    ///////////////

    fn test_file_path(name: &str) -> String {
        format!("{}/test_data/{}", env!("CARGO_MANIFEST_DIR"), name)
    }

    /// The generate set of reference SVD input files.
    const SVD_INPUT_FILE: &str = "reference_svd_inputs.json";

    #[derive(Deserialize, Debug)]
    struct SVDTestCase {
        m: usize,
        n: usize,
        matrix: Vec<f32>,
        singular_values: Vec<f32>,
    }

    impl SVDTestCase {
        fn summary(&self) -> String {
            format!("svd test case with dimension {}x{}", self.m, self.n)
        }
    }

    struct SVDTolerance {
        absolute: f32,
        relative: f32,
    }

    impl SVDTolerance {
        fn check(&self, absolute: f32, relative: f32) -> bool {
            absolute <= self.absolute || relative <= self.relative
        }
    }

    fn materialize_singular_values(singular_values: &[f32], m: usize, n: usize) -> Vec<f32> {
        assert_eq!(singular_values.len(), m.min(n));
        let mut output = vec![0.0; m * n];

        for (i, &s) in singular_values.iter().enumerate() {
            output[n * i + i] = s;
        }
        output
    }

    fn test_svd(
        case: &SVDTestCase,
        singular_value_tolerance: &SVDTolerance,
        reconstructed_tolerance: &SVDTolerance,
        context: &dyn std::fmt::Display,
    ) {
        // Create the output matrices.
        let mut singular_values = vec![0.0; case.m.min(case.n)];
        let mut u = vec![0.0; case.m * case.m];
        let mut vt = vec![0.0; case.n * case.n];

        svd_into(
            case.m,
            case.n,
            &mut case.matrix.clone(),
            &mut singular_values,
            &mut u,
            &mut vt,
        )
        .unwrap();

        // Check the resulting singular values.
        for (i, (&got, &expected)) in
            std::iter::zip(singular_values.iter(), case.singular_values.iter()).enumerate()
        {
            let diff = (got - expected).abs();
            let relative = diff / expected;
            assert!(
                singular_value_tolerance.check(diff, relative),
                "got {} but expected {} (diff: {}, relative: {}) at position {}: {}",
                got,
                expected,
                diff,
                relative,
                i,
                context
            );
        }

        // Test the reconstruction.
        let full_singular_values = materialize_singular_values(&singular_values, case.m, case.n);
        let mut temp = vec![0.0; case.m * case.n];

        // Multiply `u * singular_values`.
        sgemm(
            Transpose::None,
            Transpose::None,
            case.m,
            case.n,
            case.m,
            1.0,
            &u,
            &full_singular_values,
            None,
            &mut temp,
        );

        let mut output = vec![0.0; case.m * case.n];
        sgemm(
            Transpose::None,
            Transpose::None,
            case.m,
            case.n,
            case.n,
            1.0,
            &temp,
            &vt,
            None,
            &mut output,
        );

        for row in 0..case.m {
            for col in 0..case.n {
                let got = output[case.n * row + col];
                let expected = case.matrix[case.n * row + col];
                let diff = (got - expected).abs();
                let relative = diff / expected;
                assert!(
                    reconstructed_tolerance.check(diff, relative),
                    "mismatch in reconstructed matrix at (row, col) = ({}, {}). \
                     Got {}, expected {} (diff: {}, relative: {}). {}",
                    row,
                    col,
                    got,
                    expected,
                    diff,
                    relative,
                    context
                );
            }
        }
    }

    #[test]
    fn test_svd_implementation() {
        let path = test_file_path(SVD_INPUT_FILE);
        let file = std::fs::File::open(path.clone())
            .unwrap_or_else(|_| panic!("failed to open file {path}"));

        let reader = std::io::BufReader::new(file);
        let cases: Vec<SVDTestCase> = serde_json::from_reader(reader).unwrap();

        let singular_values_tolerance = SVDTolerance {
            absolute: 2.0e-6,
            relative: 3.0e-6,
        };

        let reconstructed_tolerance = SVDTolerance {
            absolute: 5.0e-5,
            relative: 0.0,
        };

        for (i, case) in cases.iter().enumerate() {
            let context = format!(
                "while processing case {} of {}: {}",
                i + 1,
                cases.len(),
                case.summary()
            );
            test_svd(
                case,
                &singular_values_tolerance,
                &reconstructed_tolerance,
                &context,
            );
        }
    }

    ///////////////////////////
    // Rotation Matrix Tests //
    ///////////////////////////

    const EPSILON: f32 = 1e-5;

    fn test_distance_preserving_matrix_impl(dim: usize, rng: &mut StdRng) {
        // Construct the distance preserving matrix.
        let q = random_distance_preserving_matrix(dim, rng);

        // Check that `q * q'` is close to the identity matrix.
        let qm = ::faer::mat::MatRef::from_row_major_slice(&q, dim, dim);
        let m = qm * qm.transpose();

        for j in 0..dim {
            for i in 0..dim {
                if i == j {
                    assert_abs_diff_eq!(m[(i, j)], 1.0, epsilon = EPSILON);
                } else {
                    assert_abs_diff_eq!(m[(i, j)], 0.0, epsilon = EPSILON);
                }
            }
        }

        // Instead of explicitly checking the determinant, we sample using 100 randomly
        // generated vectors, verifying that the norms are unchanged.
        const RANDOM_TRIALS: usize = 100;
        let mut v = vec![0.0f32; dim];
        for _ in 0..RANDOM_TRIALS {
            v.iter_mut()
                .for_each(|i| *i = StandardNormal {}.sample(rng));
            let vm = ::faer::mat::MatRef::from_row_major_slice(&v, dim, 1);
            let v_norm = vm.squared_norm_l2();
            let t = qm * vm;
            let t_norm = t.squared_norm_l2();

            assert_relative_eq!(v_norm, t_norm, epsilon = EPSILON, max_relative = EPSILON);
            assert_ne!(vm, t);
        }
    }

    #[test]
    fn test_rotation_matrix() {
        let mut rng = StdRng::seed_from_u64(0xc0ff33);
        let num_trials = 5;
        for dim in [2, 100, 256] {
            for _ in 0..num_trials {
                test_distance_preserving_matrix_impl(dim, &mut rng);
            }
        }
    }
}

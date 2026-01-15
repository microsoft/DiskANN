/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use faer::{self, Par};
use rand::Rng;
use rand_distr::StandardNormal;

use super::common::Transpose;

/// See the documentation for `sgemm`.
///
/// The implementation may assume the the specified invariants hold for the sizes of the
/// intermediate arrays.
#[allow(clippy::too_many_arguments)]
pub(super) fn sgemm_impl(
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
    let a = atranspose.call(
        || faer::mat::MatRef::from_row_major_slice(a, m, k),
        || faer::mat::MatRef::from_row_major_slice(a, k, m).transpose(),
    );

    let b = btranspose.call(
        || faer::mat::MatRef::from_row_major_slice(b, k, n),
        || faer::mat::MatRef::from_row_major_slice(b, n, k).transpose(),
    );

    let mut c = faer::mat::MatMut::from_row_major_slice_mut(c, m, n);

    // Faer 0.22+ removed the option to scale by an arbitrary `beta`.
    // Instead, we need to manage it ourselves.
    let beta = match beta {
        Some(scale) => {
            if scale != 1.0 {
                c *= faer::Scale(scale);
            }
            faer::Accum::Add
        }
        None => faer::Accum::Replace,
    };

    faer::linalg::matmul::matmul(c, beta, a, b, alpha, Par::Seq)
}

/// See the documentation for `svd_into`.
///
/// The implementation may assume the the specified invariants hold for the sizes of the
/// intermediate arrays.
pub(super) fn svd_into_impl(
    m: usize,
    n: usize,
    a: &[f32],
    singular_values: &mut [f32],
    u: &mut [f32],
    vt: &mut [f32],
) -> Result<(), impl std::error::Error + 'static> {
    let a = faer::mat::MatRef::from_row_major_slice(a, m, n);
    let svd = a.svd().unwrap();

    // Singular Values
    let mut singular_values = faer::col::ColMut::from_slice_mut(singular_values);
    singular_values.copy_from(svd.S().column_vector());

    // u
    let mut u = faer::mat::MatMut::from_row_major_slice_mut(u, m, m);
    u.copy_from(svd.U());

    // v
    let mut vt = faer::mat::MatMut::from_row_major_slice_mut(vt, n, n).transpose_mut();
    vt.copy_from(svd.V());

    Ok::<(), std::convert::Infallible>(())
}

pub(super) fn random_distance_preserving_matrix_impl<R>(dim: usize, rng: &mut R) -> Vec<f32>
where
    R: Rng + ?Sized,
{
    let mut data: Vec<f32> = (0..dim * dim).map(|_| rng.sample(StandardNormal)).collect();

    // Construct the d x d matrix
    let mut a = faer::mat::MatMut::from_row_major_slice_mut(&mut data, dim, dim);

    // Compute the QR decomposition
    let qr = a.qr();
    let mut q = qr.compute_Q();
    let r = qr.R();

    // Adjust Q: for each column i, if the i-th diagonal of R is negative,
    // flip the sign of the i-th column of Q.
    for i in 0..dim {
        if r[(i, i)] < 0.0 {
            for j in 0..dim {
                q[(j, i)] = -q[(j, i)];
            }
        }
    }

    // Convert the Q matrix back to a flat vector.
    a.copy_from(q);
    data
}

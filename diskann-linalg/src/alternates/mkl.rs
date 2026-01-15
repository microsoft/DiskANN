/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// Declare the MKL crate as `extern` so its exposed C-functions become available for `cblas`.
extern crate intel_mkl_src;
use super::common::Transpose;

use thiserror::Error;

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
    // Check size requirements.
    // This is checked at the entry-point, but it doesn't hurt to check again.
    // Inlining should catch the double-checks and elide the second.
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    let m: i32 = m.try_into().unwrap();
    let n: i32 = n.try_into().unwrap();
    let k: i32 = k.try_into().unwrap();

    // SAFETY: We've done all wa can to ensure that the sizes of the intermediate matrices
    // are correct for the GEMM call.
    unsafe {
        cblas::sgemm(
            cblas::Layout::RowMajor,
            atranspose.forward(cblas::Transpose::None, cblas::Transpose::Ordinary),
            btranspose.forward(cblas::Transpose::None, cblas::Transpose::Ordinary),
            m,
            n,
            k,
            alpha,
            a,
            atranspose.forward(k, m),
            b,
            btranspose.forward(n, k),
            beta.unwrap_or(0.0),
            c,
            n,
        )
    }
}

#[derive(Debug, Error)]
#[error("lapacke::sgessd failed with return code {error_code}")]
struct SVDError {
    error_code: i32,
}

/// See the documentation for `svd_into`.
///
/// The implementation may assume the the specified invariants hold for the sizes of the
/// intermediate arrays.
pub(super) fn svd_into_impl(
    m: usize,
    n: usize,
    a: &mut [f32],
    singular_values: &mut [f32],
    u: &mut [f32],
    vt: &mut [f32],
) -> Result<(), impl std::error::Error + 'static> {
    // Check size requirements.
    // This is checked at the entry-point, but it doesn't hurt to check again.
    debug_assert_eq!(a.len(), m * n);
    debug_assert_eq!(singular_values.len(), m.min(n));
    debug_assert_eq!(u.len(), m * m);
    debug_assert_eq!(vt.len(), n * n);

    let m: i32 = m.try_into().unwrap();
    let n: i32 = n.try_into().unwrap();

    // SAFETY: We have checked the following:
    // * `a` has the appropriate length `m * n` with a leading dimension stride of `n`.
    // * `singular_values` has the length `min(m, n)`.
    // * `u` has the appropriate length `m * m` with a leading dimension stride of `m`.
    // * `v` has the appropriate length `n * n` with a leading dimension stride of `n`.
    let error_code = unsafe {
        lapacke::sgesdd(
            lapacke::Layout::RowMajor,
            b'A',
            m,
            n,
            a,
            n,
            singular_values,
            u,
            m,
            vt,
            n,
        )
    };

    match error_code {
        0 => Ok(()),
        error_code => Err(SVDError { error_code }),
    }
}

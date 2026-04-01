// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! f32 × f32 micro-kernel family for block-transposed multi-vector distance.
//!
//! Provides:
//!
//! - `F32Kernel<A, GROUP>` — marker type selecting the f32 micro-kernel for
//!   architecture `A` operating on `BlockTransposed<f32, GROUP>` data.
//! - [`chamfer_kernel`] — architecture- and GROUP-generic entry point for the
//!   reducing max-IP GEMM.
//! - [`MaxSim`](crate::multi_vector::distance::MaxSim) /
//!   [`Chamfer`](crate::multi_vector::distance::Chamfer) trait implementations,
//!   generic over GROUP.
//!
//! # Architecture-specific micro-kernels
//!
//! - `v3` (x86_64) — V3 (AVX2+FMA) 16×4 micro-kernel (GROUP=16), also used by V4.
//! - `scalar` — Emulated 8×2 micro-kernel (GROUP=8), also used by Neon (aarch64).

use std::marker::PhantomData;

use diskann_wide::Architecture;

use diskann_vector::{DistanceFunctionMut, PureDistanceFunction};

use super::Kernel;
use super::tiled_reduce::tiled_reduce;
use crate::multi_vector::distance::{Chamfer, MaxSim};
use crate::multi_vector::{BlockTransposedRef, MatRef, Standard};

mod scalar;
#[cfg(target_arch = "x86_64")]
mod v3;

// ── F32 kernel ───────────────────────────────────────────────────

/// Marker type selecting the f32 micro-kernel for architecture `A` operating
/// on block-transposed data with block size `GROUP`.
///
/// The actual SIMD micro-kernel body is provided by `Kernel<A>` implementations
/// in architecture-specific submodules (`v3`, `scalar`). Each implementation
/// sets `A_PANEL = GROUP`.
pub(crate) struct F32Kernel<A: Architecture, const GROUP: usize>(PhantomData<A>);

/// The natural block-transposed GROUP size for the f32 micro-kernel on the
/// current platform.
///
/// - x86_64 (V3/V4): 16 (two `f32x8` register tiles).
/// - Other platforms (Scalar/Neon): 8 (one `f32x8` register tile).
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)] // used in tests; callers will adopt
pub(crate) const F32_GROUP: usize = 16;

/// The natural block-transposed GROUP size for the f32 micro-kernel on the
/// current platform.
///
/// - x86_64 (V3/V4): 16 (two `f32x8` register tiles).
/// - Other platforms (Scalar/Neon): 8 (one `f32x8` register tile).
#[cfg(not(target_arch = "x86_64"))]
#[allow(dead_code)] // used in tests; callers will adopt
pub(crate) const F32_GROUP: usize = 8;

// ── Public entry point ───────────────────────────────────────────

#[inline(never)]
#[cold]
#[allow(clippy::panic)]
fn chamfer_kernel_panic() {
    panic!(
        "chamfer_kernel: precondition failed (scratch.len != available_rows or dimension mismatch)"
    );
}

/// Compute the reducing max-IP GEMM between a block-transposed A matrix and
/// a row-major B matrix, writing per-A-row max similarities into `scratch`.
///
/// This is a thin wrapper over the generic `tiled_reduce` loop using the
/// `F32Kernel<A, GROUP>` micro-kernel for the requested architecture.
///
/// # Arguments
///
/// * `arch` - Architecture token (e.g. V3, V4, Scalar, Neon).
/// * `a` - Block-transposed matrix view with block size `GROUP`.
/// * `b` - Row-major matrix view.
/// * `scratch` - Mutable buffer of length [`BlockTransposedRef::available_rows()`].
///   Must be initialized to `f32::MIN` before the first call. On return, `scratch[i]`
///   contains the maximum inner product between A row `i` and any B row.
///
/// # Panics
///
/// Panics if `scratch.len() != a.available_rows()` or `a.ncols() != b.vector_dim()`.
pub(crate) fn chamfer_kernel<A: Architecture, const GROUP: usize>(
    arch: A,
    a: BlockTransposedRef<'_, f32, GROUP>,
    b: MatRef<'_, Standard<f32>>,
    scratch: &mut [f32],
) where
    F32Kernel<A, GROUP>: Kernel<A, AElem = f32, BElem = f32>,
{
    if scratch.len() != a.available_rows() || a.ncols() != b.vector_dim() {
        chamfer_kernel_panic();
    }

    let k = a.ncols();
    let b_nrows = b.num_vectors();

    debug_assert_eq!(
        <F32Kernel<A, GROUP> as Kernel<A>>::A_PANEL,
        GROUP,
        "F32Kernel A_PANEL must equal GROUP for layout correctness"
    );

    // SAFETY:
    // - a.as_ptr() is valid for a.available_rows() * k elements of f32.
    // - MatRef<Standard<f32>> stores nrows * ncols contiguous f32 elements at as_raw_ptr().
    // - scratch.len() == a.available_rows() (checked above).
    // - a.available_rows() is always a multiple of GROUP, and the debug_assert above
    //   verifies A_PANEL == GROUP.
    unsafe {
        tiled_reduce::<A, F32Kernel<A, GROUP>>(
            arch,
            a.as_ptr(),
            a.available_rows(),
            b.as_slice().as_ptr(),
            b_nrows,
            k,
            scratch,
        );
    }
}

// ── MaxSim / Chamfer trait implementations ───────────────────────

impl<const GROUP: usize>
    DistanceFunctionMut<BlockTransposedRef<'_, f32, GROUP>, MatRef<'_, Standard<f32>>>
    for MaxSim<'_>
where
    F32Kernel<diskann_wide::arch::Current, GROUP>:
        Kernel<diskann_wide::arch::Current, AElem = f32, BElem = f32>,
{
    #[inline(always)]
    fn evaluate(
        &mut self,
        query: BlockTransposedRef<'_, f32, GROUP>,
        doc: MatRef<'_, Standard<f32>>,
    ) {
        assert!(
            self.size() == query.nrows(),
            "scores buffer not right size: {} != {}",
            self.size(),
            query.nrows()
        );

        if doc.num_vectors() == 0 {
            // No document vectors — fill with MAX (no similarity found).
            self.scores_mut().fill(f32::MAX);
            return;
        }

        let scratch = self.scores_mut();
        scratch.fill(f32::MIN);

        // Extend scratch to available_rows if needed (scratch may be smaller
        // than available_rows due to padding).
        let available = query.available_rows();
        let nq = query.nrows();

        if available == nq {
            // No padding — scratch is exactly the right size.
            chamfer_kernel(diskann_wide::ARCH, query, doc, scratch);
        } else {
            // Padding rows exist — need a larger scratch buffer.
            let mut padded_scratch = vec![f32::MIN; available];
            chamfer_kernel(diskann_wide::ARCH, query, doc, &mut padded_scratch);
            scratch.copy_from_slice(&padded_scratch[..nq]);
        }

        // The kernel wrote max inner products (positive = more similar).
        // DiskANN convention: negate so that lower = better (distance semantics).
        for s in scratch.iter_mut() {
            *s = -*s;
        }
    }
}

impl<const GROUP: usize>
    PureDistanceFunction<BlockTransposedRef<'_, f32, GROUP>, MatRef<'_, Standard<f32>>, f32>
    for Chamfer
where
    F32Kernel<diskann_wide::arch::Current, GROUP>:
        Kernel<diskann_wide::arch::Current, AElem = f32, BElem = f32>,
{
    #[inline(always)]
    fn evaluate(query: BlockTransposedRef<'_, f32, GROUP>, doc: MatRef<'_, Standard<f32>>) -> f32 {
        if doc.num_vectors() == 0 {
            return 0.0;
        }

        let available = query.available_rows();
        let nq = query.nrows();

        let mut scratch = vec![f32::MIN; available];
        chamfer_kernel(diskann_wide::ARCH, query, doc, &mut scratch);

        // Sum negated max similarities to get Chamfer distance.
        scratch.iter().take(nq).map(|&s| -s).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_vector::BlockTransposed;
    use crate::multi_vector::distance::QueryMatRef;

    use super::F32_GROUP as GROUP;

    /// Helper to create a MatRef from raw data.
    fn make_query_mat(data: &[f32], nrows: usize, ncols: usize) -> MatRef<'_, Standard<f32>> {
        MatRef::new(Standard::new(nrows, ncols).unwrap(), data).unwrap()
    }

    /// Generate deterministic test data.
    fn make_test_data(len: usize, ceil: usize, shift: usize) -> Vec<f32> {
        (0..len).map(|v| ((v + shift) % ceil) as f32).collect()
    }

    /// Test cases: (num_queries, num_docs, dim).
    const TEST_CASES: &[(usize, usize, usize)] = &[
        (1, 1, 4),   // Single query, single doc
        (1, 5, 8),   // Single query, multiple docs
        (5, 1, 8),   // Multiple queries, single doc
        (3, 4, 16),  // General case
        (7, 7, 32),  // Square case
        (2, 3, 128), // Larger dimension
        (16, 4, 64), // Exact A_PANEL on x86_64; two panels on aarch64
        (17, 4, 64), // One more than A_PANEL (remainder)
        (32, 5, 16), // Multiple full A-panels, remainder B-rows (5 % 4 = 1)
        (16, 6, 32), // Remainder B-rows (6 % 4 = 2)
        (16, 7, 32), // Remainder B-rows (7 % 4 = 3)
        (16, 8, 32), // No remainder B-rows (8 % 4 = 0)
    ];

    #[test]
    fn chamfer_matches_fallback() {
        for &(nq, nd, dim) in TEST_CASES {
            let query_data = make_test_data(nq * dim, dim, dim / 2);
            let doc_data = make_test_data(nd * dim, dim, dim);

            let query_mat = make_query_mat(&query_data, nq, dim);
            let doc = make_query_mat(&doc_data, nd, dim);

            // Reference: fallback Chamfer
            let simple_query: QueryMatRef<_> = query_mat.into();
            let expected = Chamfer::evaluate(simple_query, doc);

            // Block-transposed query
            let bt = BlockTransposed::<f32, GROUP>::from_matrix_view(query_mat.as_matrix_view());
            let actual = Chamfer::evaluate(bt.as_view(), doc);

            assert!(
                (actual - expected).abs() < 1e-2,
                "Chamfer mismatch for ({nq},{nd},{dim}): actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn max_sim_matches_fallback() {
        for &(nq, nd, dim) in TEST_CASES {
            let query_data = make_test_data(nq * dim, dim, dim / 2);
            let doc_data = make_test_data(nd * dim, dim, dim);

            let query_mat = make_query_mat(&query_data, nq, dim);
            let doc = make_query_mat(&doc_data, nd, dim);

            // Reference: fallback MaxSim
            let mut expected_scores = vec![0.0f32; nq];
            let simple_query: QueryMatRef<_> = query_mat.into();
            let _ = MaxSim::new(&mut expected_scores)
                .unwrap()
                .evaluate(simple_query, doc);

            // Block-transposed query
            let bt = BlockTransposed::<f32, GROUP>::from_matrix_view(query_mat.as_matrix_view());
            let mut actual_scores = vec![0.0f32; nq];
            MaxSim::new(&mut actual_scores)
                .unwrap()
                .evaluate(bt.as_view(), doc);

            for i in 0..nq {
                assert!(
                    (actual_scores[i] - expected_scores[i]).abs() < 1e-2,
                    "MaxSim[{i}] mismatch for ({nq},{nd},{dim}): actual={}, expected={}",
                    actual_scores[i],
                    expected_scores[i]
                );
            }
        }
    }

    #[test]
    fn chamfer_with_zero_docs_returns_zero() {
        let query_data = [1.0f32, 0.0, 0.0, 1.0];
        let query_mat = make_query_mat(&query_data, 2, 2);
        let bt = BlockTransposed::<f32, GROUP>::from_matrix_view(query_mat.as_matrix_view());

        let doc = make_query_mat(&[], 0, 2);
        let result = Chamfer::evaluate(bt.as_view(), doc);
        assert_eq!(result, 0.0);
    }

    #[test]
    #[should_panic(expected = "scores buffer not right size")]
    fn max_sim_panics_on_size_mismatch() {
        let query_data = [1.0f32, 2.0, 3.0, 4.0];
        let query_mat = make_query_mat(&query_data, 2, 2);
        let bt = BlockTransposed::<f32, GROUP>::from_matrix_view(query_mat.as_matrix_view());

        let doc = make_query_mat(&[1.0, 1.0], 1, 2);
        let mut scores = vec![0.0f32; 3]; // Wrong size
        MaxSim::new(&mut scores)
            .unwrap()
            .evaluate(bt.as_view(), doc);
    }
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! f32 micro-kernel family for block-transposed multi-vector distance.
//!
//! Provides:
//!
//! - `F32Kernel<GROUP>` — zero-sized marker type selecting the f32 micro-kernel
//!   for `BlockTransposed<f32, GROUP>` data.
//! - [`max_ip_kernel`] — architecture-, element-type-, and GROUP-generic entry point
//!   for the reducing max-IP GEMM. Accepts any element type `T` for which
//!   [`ConvertTo`](super::layouts::ConvertTo) impls exist (identity for f32,
//!   SIMD-accelerated f16→f32, etc.).
//!
//! # Architecture-specific micro-kernels
//!
//! - `v3` (x86_64) — V3 (AVX2+FMA) 16×4 micro-kernel (GROUP=16). V4 delegates to V3 at dispatch.
//! - `scalar` — Emulated 8×2 micro-kernel (GROUP=8). Neon delegates to Scalar at dispatch.

use diskann_wide::Architecture;

use super::Kernel;
use super::TileBudget;
use super::layouts::{self, DescribeLayout};
use super::tiled_reduce::tiled_reduce;
use crate::multi_vector::{BlockTransposedRef, MatRef, Standard};

mod scalar;
#[cfg(target_arch = "x86_64")]
mod v3;

// ── F32 kernel ───────────────────────────────────────────────────

/// Zero-sized kernel type for f32 micro-kernels with block size `GROUP`.
pub(crate) struct F32Kernel<const GROUP: usize>;

// ── Public entry point ───────────────────────────────────────────

#[inline(never)]
#[cold]
#[allow(clippy::panic)]
fn max_ip_kernel_panic(scratch_len: usize, padded_nrows: usize, a_ncols: usize, b_dim: usize) {
    panic!(
        "max_ip_kernel: precondition failed: \
         scratch.len()={scratch_len} (expected {padded_nrows}), \
         a.ncols()={a_ncols}, b.vector_dim()={b_dim}"
    );
}

/// Compute the reducing max-IP GEMM between a block-transposed A matrix and
/// a row-major B matrix, writing per-A-row max similarities into `scratch`.
///
/// Thin wrapper over [`tiled_reduce`] using `F32Kernel<GROUP>` for the
/// requested architecture. The element type `T` can be any `Copy` type with
/// matching [`ConvertTo`](super::layouts::ConvertTo) impls (zero-cost for
/// `T = f32`; SIMD f16→f32 conversion once per tile for `T = half::f16`).
///
/// `scratch` must have length [`BlockTransposedRef::padded_nrows()`] and be
/// initialized to `f32::MIN` before the first call. On return, `scratch[i]`
/// holds the maximum inner product between A row `i` and any B row.
///
/// # Panics
///
/// Panics if `scratch.len() != a.padded_nrows()` or `a.ncols() != b.vector_dim()`.
pub(super) fn max_ip_kernel<A: Architecture, T: Copy, const GROUP: usize>(
    arch: A,
    a: BlockTransposedRef<'_, T, GROUP>,
    b: MatRef<'_, Standard<T>>,
    scratch: &mut [f32],
    budget: TileBudget,
) where
    F32Kernel<GROUP>: Kernel<A>,
    layouts::BlockTransposed<T, GROUP>:
        layouts::ConvertTo<A, <F32Kernel<GROUP> as Kernel<A>>::Left> + layouts::Layout<Element = T>,
    layouts::RowMajor<T>: layouts::ConvertTo<A, <F32Kernel<GROUP> as Kernel<A>>::Right>
        + layouts::Layout<Element = T>,
{
    if scratch.len() != a.padded_nrows() || a.ncols() != b.vector_dim() {
        max_ip_kernel_panic(scratch.len(), a.padded_nrows(), a.ncols(), b.vector_dim());
    }

    let k = a.ncols();
    let b_nrows = b.num_vectors();

    // Compile-time: A_PANEL must equal GROUP for block-transposed layout correctness.
    const { assert!(<F32Kernel<GROUP> as Kernel<A>>::A_PANEL == GROUP) }

    let ca = a.layout();
    let cb = b.layout();

    // SAFETY:
    // - a.as_ptr() is valid for a.padded_nrows() * k elements of T.
    // - MatRef<Standard<T>> stores nrows * ncols contiguous T elements.
    // - scratch.len() == a.padded_nrows() (checked above).
    // - a.padded_nrows() is always a multiple of GROUP, and the const assert above
    //   verifies A_PANEL == GROUP at compile time.
    unsafe {
        tiled_reduce::<A, F32Kernel<GROUP>, _, _>(
            arch,
            &ca,
            &cb,
            a.as_ptr(),
            a.padded_nrows(),
            b.as_slice().as_ptr(),
            b_nrows,
            k,
            scratch,
            budget,
        );
    }
}

// ── Target3 dispatch ─────────────────────────────────────────────

impl<A, const GROUP: usize>
    diskann_wide::arch::Target3<
        A,
        (),
        BlockTransposedRef<'_, f32, GROUP>,
        MatRef<'_, Standard<f32>>,
        &mut [f32],
    > for F32Kernel<GROUP>
where
    A: Architecture,
    Self: Kernel<A>,
    layouts::BlockTransposed<f32, GROUP>:
        layouts::ConvertTo<A, <Self as Kernel<A>>::Left> + layouts::Layout<Element = f32>,
    layouts::RowMajor<f32>:
        layouts::ConvertTo<A, <Self as Kernel<A>>::Right> + layouts::Layout<Element = f32>,
{
    #[inline(always)]
    fn run(
        self,
        arch: A,
        lhs: BlockTransposedRef<'_, f32, GROUP>,
        rhs: MatRef<'_, Standard<f32>>,
        scratch: &mut [f32],
    ) {
        max_ip_kernel(arch, lhs, rhs, scratch, TileBudget::default());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_vector::{Chamfer, MaxSim, QueryComputer, QueryMatRef};
    use diskann_vector::{DistanceFunctionMut, PureDistanceFunction};

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
        (1, 1, 1),   // Degenerate single-element
        (1, 1, 2),   // Minimal non-trivial
        (1, 1, 4),   // Single query, single doc
        (1, 5, 8),   // Single query, multiple docs
        (5, 1, 8),   // Multiple queries, single doc
        (3, 2, 3),   // Prime k
        (3, 4, 16),  // General case
        (5, 3, 5),   // Prime k, A-remainder on aarch64
        (7, 7, 32),  // Square case
        (2, 3, 7),   // k not divisible by SIMD lanes
        (2, 3, 128), // Larger dimension
        (16, 4, 64), // Exact A_PANEL on x86_64; two panels on aarch64
        (17, 4, 64), // One more than A_PANEL (remainder)
        (32, 5, 16), // Multiple full A-panels, remainder B-rows (5 % 4 = 1)
        (48, 3, 16), // 3 A-tiles on x86_64; 6 on aarch64
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

            // QueryComputer-dispatched
            let computer = QueryComputer::<f32>::new(query_mat);
            let actual = computer.chamfer(doc);

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

            // QueryComputer-dispatched
            let computer = QueryComputer::<f32>::new(query_mat);
            let mut actual_scores = vec![0.0f32; nq];
            computer.max_sim(doc, &mut actual_scores);

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
        let computer = QueryComputer::<f32>::new(query_mat);

        let doc = make_query_mat(&[], 0, 2);
        let result = computer.chamfer(doc);
        assert_eq!(result, 0.0);
    }

    #[test]
    #[should_panic(expected = "scores buffer not right size")]
    fn max_sim_panics_on_size_mismatch() {
        let query_data = [1.0f32, 2.0, 3.0, 4.0];
        let query_mat = make_query_mat(&query_data, 2, 2);
        let computer = QueryComputer::<f32>::new(query_mat);

        let doc = make_query_mat(&[1.0, 1.0], 1, 2);
        let mut scores = vec![0.0f32; 3]; // Wrong size
        computer.max_sim(doc, &mut scores);
    }

    #[test]
    fn negative_values_propagate() {
        // Hand-crafted negative vectors: query = [[-1, -2], [-3, -4]], doc = [[-1, 0]]
        let query_data = [-1.0f32, -2.0, -3.0, -4.0];
        let doc_data = [-1.0f32, 0.0];

        let query_mat = make_query_mat(&query_data, 2, 2);
        let doc = make_query_mat(&doc_data, 1, 2);

        let simple_query: QueryMatRef<_> = query_mat.into();
        let expected = Chamfer::evaluate(simple_query, doc);

        let computer = QueryComputer::<f32>::new(query_mat);
        let actual = computer.chamfer(doc);

        assert!(
            (actual - expected).abs() < 1e-6,
            "Chamfer mismatch with negative values: actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn max_sim_with_zero_docs() {
        let query_data = [1.0f32, 0.0, 0.0, 1.0];
        let query_mat = make_query_mat(&query_data, 2, 2);
        let computer = QueryComputer::<f32>::new(query_mat);

        let doc = make_query_mat(&[], 0, 2);
        let mut scores = vec![0.0f32; 2];
        computer.max_sim(doc, &mut scores);

        // With zero docs the scores buffer is left untouched.
        for &s in &scores {
            assert_eq!(s, 0.0, "zero-doc MaxSim should leave scores untouched");
        }
    }
}

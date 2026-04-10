// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! f32 × f32 micro-kernel family for block-transposed multi-vector distance.
//!
//! Provides:
//!
//! - `F32Kernel<GROUP>` — zero-sized marker type selecting the f32 micro-kernel
//!   for `BlockTransposed<f32, GROUP>` data.
//! - [`max_ip_kernel`] — architecture- and GROUP-generic entry point for the
//!   reducing max-IP GEMM.
//!
//! # Architecture-specific micro-kernels
//!
//! - `v3` (x86_64) — V3 (AVX2+FMA) 16×4 micro-kernel (GROUP=16). V4 delegates to V3 at dispatch.
//! - `scalar` — Emulated 8×2 micro-kernel (GROUP=8). Neon delegates to Scalar at dispatch.

use diskann_wide::Architecture;

use super::Kernel;
use super::tiled_reduce::tiled_reduce;
use crate::multi_vector::{BlockTransposedRef, MatRef, Standard};

pub(super) mod scalar;
#[cfg(target_arch = "x86_64")]
pub(super) mod v3;

// ── F32 kernel ───────────────────────────────────────────────────

/// Zero-sized kernel type for f32 micro-kernels with block size `GROUP`.
///
/// Since `AElem == APrepared` and `BElem == BPrepared`, no staging buffers
/// are needed — [`prepare_a`](Kernel::prepare_a) and
/// [`prepare_b`](Kernel::prepare_b) return the source pointer directly.
/// The actual SIMD micro-kernel body is provided by `Kernel<A>`
/// implementations in architecture-specific submodules (`v3`, `scalar`).
/// Each implementation sets `A_PANEL = GROUP`.
pub(crate) struct F32Kernel<const GROUP: usize>;

// ── Public entry point ───────────────────────────────────────────

#[inline(never)]
#[cold]
#[allow(clippy::panic)]
fn max_ip_kernel_panic() {
    panic!(
        "max_ip_kernel: precondition failed (scratch.len != available_rows or dimension mismatch)"
    );
}

/// Compute the reducing max-IP GEMM between a block-transposed A matrix and
/// a row-major B matrix, writing per-A-row max similarities into `scratch`.
///
/// This is a thin wrapper over the generic `tiled_reduce` loop using the
/// `F32Kernel<GROUP>` micro-kernel for the requested architecture.
///
/// # Arguments
///
/// * `arch` - Architecture token (must satisfy the `Kernel` where-bound).
/// * `a` - Block-transposed matrix view with block size `GROUP`.
/// * `b` - Row-major matrix view.
/// * `scratch` - Mutable buffer of length [`BlockTransposedRef::available_rows()`].
///   Must be initialized to `f32::MIN` before the first call. On return, `scratch[i]`
///   contains the maximum inner product between A row `i` and any B row.
///
/// # Panics
///
/// Panics if `scratch.len() != a.available_rows()` or `a.ncols() != b.vector_dim()`.
pub(crate) fn max_ip_kernel<A: Architecture, const GROUP: usize>(
    arch: A,
    a: BlockTransposedRef<'_, f32, GROUP>,
    b: MatRef<'_, Standard<f32>>,
    scratch: &mut [f32],
) where
    F32Kernel<GROUP>: Kernel<A, AElem = f32, BElem = f32, APrepared = f32, BPrepared = f32>,
{
    if scratch.len() != a.available_rows() || a.ncols() != b.vector_dim() {
        max_ip_kernel_panic();
    }

    let k = a.ncols();
    let b_nrows = b.num_vectors();

    // Compile-time: A_PANEL must equal GROUP for block-transposed layout correctness.
    const { assert!(<F32Kernel<GROUP> as Kernel<A>>::A_PANEL == GROUP) }

    // SAFETY:
    // - a.as_ptr() is valid for a.available_rows() * k elements of f32.
    // - MatRef<Standard<f32>> stores nrows * ncols contiguous f32 elements at as_raw_ptr().
    // - scratch.len() == a.available_rows() (checked above).
    // - a.available_rows() is always a multiple of GROUP, and the const assert above
    //   verifies A_PANEL == GROUP at compile time.
    unsafe {
        tiled_reduce::<A, F32Kernel<GROUP>>(
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
    Self: Kernel<A, AElem = f32, BElem = f32, APrepared = f32, BPrepared = f32>,
{
    #[inline(always)]
    fn run(
        self,
        arch: A,
        lhs: BlockTransposedRef<'_, f32, GROUP>,
        rhs: MatRef<'_, Standard<f32>>,
        scratch: &mut [f32],
    ) {
        max_ip_kernel(arch, lhs, rhs, scratch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_vector::QueryComputer;
    use crate::multi_vector::distance::{Chamfer, MaxSim, QueryMatRef};
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

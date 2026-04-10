// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! f16 × f16 micro-kernel family for block-transposed multi-vector distance.
//!
//! Implements the [`Kernel<A>`](super::Kernel) trait with `AElem = BElem = f16`,
//! lazily unpacking both sides to f32 via the `prepare_a` / `prepare_b` hooks.
//! The micro-kernel body itself is the same f32 SIMD arithmetic used by the
//! f32 kernel family — zero additional micro-kernel code is required.
//!
//! # Architecture-specific implementations
//!
//! | Module   | Arch      | Geometry | Conversion       |
//! |----------|-----------|----------|------------------|
//! | `v3`     | V3        | 16 × 4   | [`CastFromSlice`]|
//! | `scalar` | Scalar    | 8 × 2    | [`CastFromSlice`]|
//!
//! [`CastFromSlice`] dispatches via `diskann_wide::ARCH` at compile time, so
//! the conversion is SIMD-accelerated on every platform that `diskann-wide`
//! supports (e.g. Neon on aarch64), regardless of the kernel's architecture
//! parameter.
//!
//! [`CastFromSlice`]: diskann_vector::conversion::CastFromSlice

use diskann_wide::Architecture;

use super::Kernel;
use super::tiled_reduce::tiled_reduce;
use crate::multi_vector::{BlockTransposedRef, MatRef, Standard};

mod scalar;
#[cfg(target_arch = "x86_64")]
mod v3;

// ── F16 kernel ───────────────────────────────────────────────────

/// Kernel type for f16 micro-kernels with block size `GROUP`.
///
/// Both A and B sides store [`half::f16`] values. The
/// [`prepare_a`](Kernel::prepare_a) / [`prepare_b`](Kernel::prepare_b)
/// hooks convert to f32 into caller-provided staging buffers allocated via
/// [`Kernel::new_buffers`]. The micro-kernel itself is the same f32 SIMD
/// arithmetic used by [`F32Kernel`](super::f32::F32Kernel).
pub(crate) struct F16Kernel<const GROUP: usize>;

// ── Public entry point ───────────────────────────────────────────

#[inline(never)]
#[cold]
#[allow(clippy::panic)]
fn max_ip_kernel_panic() {
    panic!(
        "max_ip_kernel: precondition failed (scratch.len != available_rows or dimension mismatch)"
    );
}

/// Compute the reducing max-IP GEMM between a block-transposed f16 A matrix
/// and a row-major f16 B matrix, writing per-A-row max similarities into
/// `scratch`.
///
/// Both A and B sides are lazily unpacked to f32 via the `prepare_a` /
/// `prepare_b` hooks. The micro-kernel itself operates in f32.
///
/// # Arguments
///
/// * `arch` - Architecture token (must satisfy the `Kernel` where-bound).
/// * `a` - Block-transposed f16 matrix view with block size `GROUP`.
/// * `b` - Row-major f16 matrix view.
/// * `scratch` - Mutable buffer of length [`BlockTransposedRef::available_rows()`].
///   Must be initialized to `f32::MIN` before the first call.
///
/// # Panics
///
/// Panics if `scratch.len() != a.available_rows()` or `a.ncols() != b.vector_dim()`.
pub(crate) fn max_ip_kernel<A: Architecture, const GROUP: usize>(
    arch: A,
    a: BlockTransposedRef<'_, half::f16, GROUP>,
    b: MatRef<'_, Standard<half::f16>>,
    scratch: &mut [f32],
) where
    F16Kernel<GROUP>:
        Kernel<A, AElem = half::f16, BElem = half::f16, APrepared = f32, BPrepared = f32>,
{
    if scratch.len() != a.available_rows() || a.ncols() != b.vector_dim() {
        max_ip_kernel_panic();
    }

    let k = a.ncols();
    let b_nrows = b.num_vectors();

    // Compile-time: A_PANEL must equal GROUP for block-transposed layout correctness.
    const { assert!(<F16Kernel<GROUP> as Kernel<A>>::A_PANEL == GROUP) }

    // SAFETY:
    // - a.as_ptr() is valid for a.available_rows() * k elements of f16.
    // - MatRef<Standard<f16>> stores nrows * ncols contiguous f16 elements.
    // - scratch.len() == a.available_rows() (checked above).
    // - a.available_rows() is always a multiple of GROUP, and the const assert above
    //   verifies A_PANEL == GROUP at compile time.
    unsafe {
        tiled_reduce::<A, F16Kernel<GROUP>>(
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
        BlockTransposedRef<'_, half::f16, GROUP>,
        MatRef<'_, Standard<half::f16>>,
        &mut [f32],
    > for F16Kernel<GROUP>
where
    A: Architecture,
    Self: Kernel<A, AElem = half::f16, BElem = half::f16, APrepared = f32, BPrepared = f32>,
{
    #[inline(always)]
    fn run(
        self,
        arch: A,
        lhs: BlockTransposedRef<'_, half::f16, GROUP>,
        rhs: MatRef<'_, Standard<half::f16>>,
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

    /// Helper to create a MatRef from raw f16 data.
    fn make_query_mat(
        data: &[half::f16],
        nrows: usize,
        ncols: usize,
    ) -> MatRef<'_, Standard<half::f16>> {
        MatRef::new(Standard::new(nrows, ncols).unwrap(), data).unwrap()
    }

    /// Generate deterministic test data as f16.
    fn make_test_data(len: usize, ceil: usize, shift: usize) -> Vec<half::f16> {
        (0..len)
            .map(|v| diskann_wide::cast_f32_to_f16(((v + shift) % ceil) as f32))
            .collect()
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

            // Reference: fallback Chamfer (works on f16 via InnerProduct)
            let simple_query: QueryMatRef<_> = query_mat.into();
            let expected = Chamfer::evaluate(simple_query, doc);

            // QueryComputer-dispatched
            let computer = QueryComputer::<half::f16>::new(query_mat);
            let actual = computer.chamfer(doc);

            assert!(
                (actual - expected).abs() < 1e-1,
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
            let computer = QueryComputer::<half::f16>::new(query_mat);
            let mut actual_scores = vec![0.0f32; nq];
            computer.max_sim(doc, &mut actual_scores);

            for i in 0..nq {
                assert!(
                    (actual_scores[i] - expected_scores[i]).abs() < 1e-1,
                    "MaxSim[{i}] mismatch for ({nq},{nd},{dim}): actual={}, expected={}",
                    actual_scores[i],
                    expected_scores[i]
                );
            }
        }
    }

    #[test]
    fn chamfer_with_zero_docs_returns_zero() {
        let query_data = [
            diskann_wide::cast_f32_to_f16(1.0),
            diskann_wide::cast_f32_to_f16(0.0),
            diskann_wide::cast_f32_to_f16(0.0),
            diskann_wide::cast_f32_to_f16(1.0),
        ];
        let query_mat = make_query_mat(&query_data, 2, 2);
        let computer = QueryComputer::<half::f16>::new(query_mat);

        let doc = make_query_mat(&[], 0, 2);
        let result = computer.chamfer(doc);
        assert_eq!(result, 0.0);
    }

    #[test]
    #[should_panic(expected = "scores buffer not right size")]
    fn max_sim_panics_on_size_mismatch() {
        let query_data = [
            diskann_wide::cast_f32_to_f16(1.0),
            diskann_wide::cast_f32_to_f16(2.0),
            diskann_wide::cast_f32_to_f16(3.0),
            diskann_wide::cast_f32_to_f16(4.0),
        ];
        let query_mat = make_query_mat(&query_data, 2, 2);
        let computer = QueryComputer::<half::f16>::new(query_mat);

        let doc_data = [
            diskann_wide::cast_f32_to_f16(1.0),
            diskann_wide::cast_f32_to_f16(1.0),
        ];
        let doc = make_query_mat(&doc_data, 1, 2);
        let mut scores = vec![0.0f32; 3]; // Wrong size
        computer.max_sim(doc, &mut scores);
    }

    #[test]
    fn negative_values_propagate() {
        // Hand-crafted negative vectors: query = [[-1, -2], [-3, -4]], doc = [[-1, 0]]
        let query_data = [
            diskann_wide::cast_f32_to_f16(-1.0),
            diskann_wide::cast_f32_to_f16(-2.0),
            diskann_wide::cast_f32_to_f16(-3.0),
            diskann_wide::cast_f32_to_f16(-4.0),
        ];
        let doc_data = [
            diskann_wide::cast_f32_to_f16(-1.0),
            diskann_wide::cast_f32_to_f16(0.0),
        ];

        let query_mat = make_query_mat(&query_data, 2, 2);
        let doc = make_query_mat(&doc_data, 1, 2);

        let simple_query: QueryMatRef<_> = query_mat.into();
        let expected = Chamfer::evaluate(simple_query, doc);

        let computer = QueryComputer::<half::f16>::new(query_mat);
        let actual = computer.chamfer(doc);

        assert!(
            (actual - expected).abs() < 1e-2,
            "Chamfer mismatch with negative values: actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn max_sim_with_zero_docs() {
        let query_data = [
            diskann_wide::cast_f32_to_f16(1.0),
            diskann_wide::cast_f32_to_f16(0.0),
            diskann_wide::cast_f32_to_f16(0.0),
            diskann_wide::cast_f32_to_f16(1.0),
        ];
        let query_mat = make_query_mat(&query_data, 2, 2);
        let computer = QueryComputer::<half::f16>::new(query_mat);

        let doc = make_query_mat(&[], 0, 2);
        let mut scores = vec![0.0f32; 2];
        computer.max_sim(doc, &mut scores);

        // With zero docs the kernel fills f32::MIN then negates → f32::MAX.
        for &s in &scores {
            assert_eq!(s, f32::MAX, "zero-doc MaxSim should produce f32::MAX");
        }
    }

    /// Verify f16 kernel precision with known exact inner products.
    ///
    /// Query: [[1, 2]], Doc: [[3, 4]]
    /// IP = 1*3 + 2*4 = 11. All values are exactly representable in f16.
    /// Chamfer returns negated distance: sum of (-IP) = -11.
    #[test]
    fn f16_known_exact_result() {
        let query_data = [
            diskann_wide::cast_f32_to_f16(1.0),
            diskann_wide::cast_f32_to_f16(2.0),
        ];
        let doc_data = [
            diskann_wide::cast_f32_to_f16(3.0),
            diskann_wide::cast_f32_to_f16(4.0),
        ];

        let query_mat = make_query_mat(&query_data, 1, 2);
        let doc = make_query_mat(&doc_data, 1, 2);

        let computer = QueryComputer::<half::f16>::new(query_mat);
        let actual = computer.chamfer(doc);

        // IP = 1*3 + 2*4 = 11. Chamfer returns the negated-IP distance = -11.
        // All intermediate values (1, 2, 3, 4, 11) are exactly representable in f16.
        assert!(
            (actual - (-11.0)).abs() < 1e-3,
            "f16 precision test: expected -11.0, got {actual}"
        );
    }
}

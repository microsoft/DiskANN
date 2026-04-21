// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! f16 dispatch adapter for block-transposed multi-vector distance.
//!
//! Reuses the f32 micro-kernel family with tile-level f16→f32 conversion
//! via [`ConvertTo`](super::layouts::ConvertTo). No f16-specific micro-kernel
//! code is needed — the [`F32Kernel`](super::f32::F32Kernel) does all the
//! SIMD work after conversion.
//!
//! Conversion from f16 to f32 is performed at tile granularity via
//! [`SliceCast`](diskann_vector::conversion::SliceCast), dispatched through
//! the runtime architecture token — the same SIMD level used by the
//! micro-kernel.

use diskann_wide::Architecture;

use super::Kernel;
use super::TileBudget;
use super::f32::{F32Kernel, max_ip_kernel};
use super::layouts;
use crate::multi_vector::{BlockTransposedRef, MatRef, Standard};

// ── F16 entry ────────────────────────────────────────────────────

pub(crate) struct F16Entry<const GROUP: usize>;

// ── Target3 dispatch ─────────────────────────────────────────────

impl<A, const GROUP: usize>
    diskann_wide::arch::Target3<
        A,
        (),
        BlockTransposedRef<'_, half::f16, GROUP>,
        MatRef<'_, Standard<half::f16>>,
        &mut [f32],
    > for F16Entry<GROUP>
where
    A: Architecture,
    F32Kernel<GROUP>: Kernel<A>,
    layouts::BlockTransposed<half::f16, GROUP>: layouts::ConvertTo<A, <F32Kernel<GROUP> as Kernel<A>>::Left>
        + layouts::Layout<Element = half::f16>,
    layouts::RowMajor<half::f16>: layouts::ConvertTo<A, <F32Kernel<GROUP> as Kernel<A>>::Right>
        + layouts::Layout<Element = half::f16>,
{
    #[inline(always)]
    fn run(
        self,
        arch: A,
        lhs: BlockTransposedRef<'_, half::f16, GROUP>,
        rhs: MatRef<'_, Standard<half::f16>>,
        scratch: &mut [f32],
    ) {
        max_ip_kernel(arch, lhs, rhs, scratch, TileBudget::default());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_vector::{BlockTransposed, Chamfer, MaxSim, QueryComputer, QueryMatRef};
    use diskann_vector::{DistanceFunctionMut, PureDistanceFunction};
    use diskann_wide::arch::Scalar;

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

        // With zero docs the scores buffer is left untouched.
        for &s in &scores {
            assert_eq!(s, 0.0, "zero-doc MaxSim should leave scores untouched");
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

    /// Direct test of f16 `max_ip_kernel` against a naive reference, bypassing
    /// `QueryComputer` to validate the kernel + ConvertTo pipeline directly.
    #[test]
    fn max_ip_kernel_matches_naive() {
        fn naive_max_ip_f16(
            a: &[half::f16],
            a_nrows: usize,
            b: &[half::f16],
            b_nrows: usize,
            k: usize,
        ) -> Vec<f32> {
            (0..a_nrows)
                .map(|i| {
                    (0..b_nrows)
                        .map(|j| {
                            (0..k)
                                .map(|d| a[i * k + d].to_f32() * b[j * k + d].to_f32())
                                .sum::<f32>()
                        })
                        .fold(f32::MIN, f32::max)
                })
                .collect()
        }

        let cases: &[(usize, usize, usize)] = &[
            (1, 1, 4),   // Single query, single doc
            (8, 3, 4),   // Single A-panel, B remainder
            (16, 5, 8),  // Two A-panels, B remainder
            (17, 4, 64), // A-panel remainder
            (16, 8, 32), // No B remainder
        ];

        for &(a_nrows, b_nrows, dim) in cases {
            let a_data = make_test_data(a_nrows * dim, dim.max(1), dim / 2);
            let b_data = make_test_data(b_nrows * dim, dim.max(1), dim);

            let a_mat = make_query_mat(&a_data, a_nrows, dim);
            let a_bt = BlockTransposed::<half::f16, 8>::from_matrix_view(a_mat.as_matrix_view());
            let b_mat = make_query_mat(&b_data, b_nrows, dim);

            let mut scratch = vec![f32::MIN; a_bt.padded_nrows()];
            max_ip_kernel::<Scalar, _, 8>(
                Scalar::new(),
                a_bt.as_view(),
                b_mat,
                &mut scratch,
                TileBudget::default(),
            );

            let expected = naive_max_ip_f16(&a_data, a_nrows, &b_data, b_nrows, dim);
            for i in 0..a_nrows {
                assert!(
                    (scratch[i] - expected[i]).abs() < 1e-1,
                    "row {i} mismatch for ({a_nrows},{b_nrows},{dim}): actual={}, expected={}",
                    scratch[i],
                    expected[i]
                );
            }
        }
    }
}

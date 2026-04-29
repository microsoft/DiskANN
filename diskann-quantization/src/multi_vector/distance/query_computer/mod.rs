// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Architecture-opaque query computer with runtime dispatch.
//!
//! [`QueryComputer`] wraps a block-transposed query and a captured
//! architecture token behind a trait-object vtable. CPU detection happens
//! once at construction; every subsequent distance call goes through
//! [`Architecture::run3`](diskann_wide::Architecture::run3) with full
//! `#[target_feature]` propagation — no re-dispatch and no enum matching
//! on the hot path.
//!
//! # Usage
//!
//! ```
//! use diskann_quantization::multi_vector::{
//!     QueryComputer, MatRef, Standard, Chamfer,
//! };
//! use diskann_vector::PureDistanceFunction;
//!
//! let query_data = [1.0f32, 0.0, 0.0, 1.0];
//! let doc_data = [1.0f32, 0.0, 0.0, 1.0];
//!
//! let query = MatRef::new(Standard::new(2, 2).unwrap(), &query_data).unwrap();
//! let doc = MatRef::new(Standard::new(2, 2).unwrap(), &doc_data).unwrap();
//!
//! // Build — runtime detects arch, picks optimal GROUP, captures both
//! let computer = QueryComputer::<f32>::new(query);
//!
//! // Distance — vtable → arch.run3 with target_feature propagation
//! let dist = Chamfer::evaluate(&computer, doc);
//! assert_eq!(dist, -2.0);
//! ```

mod f16;
mod f32;

use diskann_vector::{DistanceFunctionMut, PureDistanceFunction};

use crate::multi_vector::{BlockTransposed, MatRef, Standard};

use super::max_sim::{Chamfer, MaxSim};

/// Architecture-dispatched query computer for multi-vector distance.
pub struct QueryComputer<T: Copy> {
    inner: Box<dyn DynQueryComputer<T>>,
}

impl<T: Copy> QueryComputer<T> {
    /// Number of logical (non-padded) query vectors.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    /// Compute Chamfer distance (sum of per-query max similarities, negated).
    ///
    /// Returns `0.0` if the document has zero vectors.
    pub fn chamfer(&self, doc: MatRef<'_, Standard<T>>) -> f32 {
        let nq = self.nrows();
        if doc.num_vectors() == 0 {
            return 0.0;
        }
        let mut scores = vec![0.0f32; nq];
        self.max_sim(doc, &mut scores);
        scores.iter().sum()
    }

    /// Compute per-query-vector max similarities into `scores`.
    ///
    /// `scores` must have length equal to [`nrows()`](Self::nrows).
    /// Each entry is the negated max inner product for that query vector.
    ///
    /// # Panics
    ///
    /// Panics if `scores.len() != self.nrows()`.
    pub fn max_sim(&self, doc: MatRef<'_, Standard<T>>, scores: &mut [f32]) {
        let nq = self.nrows();
        assert_eq!(
            scores.len(),
            nq,
            "scores buffer not right size: {} != {}",
            scores.len(),
            nq
        );

        if doc.num_vectors() == 0 {
            return;
        }

        self.inner.compute_max_sim(doc, scores);
    }
}

trait DynQueryComputer<T: Copy> {
    fn compute_max_sim(&self, doc: MatRef<'_, Standard<T>>, scores: &mut [f32]);
    fn nrows(&self) -> usize;
}

struct Prepared<A, Q> {
    arch: A,
    prepared: Q,
}

fn build_prepared<T: Copy + Default, A, const GROUP: usize>(
    arch: A,
    query: MatRef<'_, Standard<T>>,
) -> Prepared<A, BlockTransposed<T, GROUP>> {
    let prepared = BlockTransposed::<T, GROUP>::from_matrix_view(query.as_matrix_view());
    Prepared { arch, prepared }
}

impl<T: Copy> PureDistanceFunction<&QueryComputer<T>, MatRef<'_, Standard<T>>, f32> for Chamfer {
    fn evaluate(query: &QueryComputer<T>, doc: MatRef<'_, Standard<T>>) -> f32 {
        query.chamfer(doc)
    }
}

impl<T: Copy> DistanceFunctionMut<&QueryComputer<T>, MatRef<'_, Standard<T>>> for MaxSim<'_> {
    fn evaluate(&mut self, query: &QueryComputer<T>, doc: MatRef<'_, Standard<T>>) {
        query.max_sim(doc, self.scores_mut());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_vector::{Chamfer, MaxSim, QueryMatRef};
    use diskann_vector::distance::InnerProduct;
    use diskann_vector::{DistanceFunctionMut, PureDistanceFunction};

    trait FromF32 {
        fn from_f32(v: f32) -> Self;
    }

    impl FromF32 for f32 {
        fn from_f32(v: f32) -> Self {
            v
        }
    }

    impl FromF32 for half::f16 {
        fn from_f32(v: f32) -> Self {
            diskann_wide::cast_f32_to_f16(v)
        }
    }

    fn make_mat<T: Copy>(data: &[T], nrows: usize, ncols: usize) -> MatRef<'_, Standard<T>> {
        MatRef::new(Standard::new(nrows, ncols).unwrap(), data).unwrap()
    }

    fn make_test_data<T: FromF32>(len: usize, ceil: usize, shift: usize) -> Vec<T> {
        (0..len)
            .map(|v| T::from_f32(((v + shift) % ceil) as f32))
            .collect()
    }

    /// Test cases: (num_queries, num_docs, dim).
    ///
    /// Sized to exercise:
    /// * degenerate single-element shapes,
    /// * `k` (dim) not divisible by SIMD lane count,
    /// * exact and off-by-one A_PANEL boundaries on both `GROUP=8` (Scalar/Neon)
    ///   and `GROUP=16` (V3/V4) configurations,
    /// * every B-row remainder class for the active `B_PANEL` (1, 2, 3 on V3;
    ///   1 on Scalar).
    ///
    /// Diverges from `kernels::tiled_reduce::tests::NAIVE_CASES`: the
    /// kernel-level matrix additionally covers zero-`k` / zero-`b_nrows`
    /// (kernel internal early-exit edges, with no public-API meaning —
    /// the API contracts for empty docs are pinned by the dedicated
    /// `chamfer_with_zero_docs` / `max_sim_with_zero_docs` tests) and a
    /// pair of Scalar-panel arithmetic edges already crossed by the
    /// shapes below.
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

    fn check_chamfer_matches<T: Copy + FromF32>(
        build: fn(MatRef<'_, Standard<T>>) -> QueryComputer<T>,
        tol: f32,
        label: &str,
    ) where
        InnerProduct: for<'a, 'b> PureDistanceFunction<&'a [T], &'b [T], f32>,
    {
        for &(nq, nd, dim) in TEST_CASES {
            let query_data = make_test_data::<T>(nq * dim, dim, dim / 2);
            let doc_data = make_test_data::<T>(nd * dim, dim, dim);

            let query = make_mat(&query_data, nq, dim);
            let doc = make_mat(&doc_data, nd, dim);

            let expected = Chamfer::evaluate(QueryMatRef::from(query), doc);
            let actual = build(query).chamfer(doc);

            assert!(
                (actual - expected).abs() < tol,
                "{label}Chamfer mismatch for ({nq},{nd},{dim}): actual={actual}, expected={expected}",
            );
        }
    }

    fn check_max_sim_matches<T: Copy + FromF32>(
        build: fn(MatRef<'_, Standard<T>>) -> QueryComputer<T>,
        tol: f32,
        label: &str,
    ) where
        InnerProduct: for<'a, 'b> PureDistanceFunction<&'a [T], &'b [T], f32>,
    {
        for &(nq, nd, dim) in TEST_CASES {
            let query_data = make_test_data::<T>(nq * dim, dim, dim / 2);
            let doc_data = make_test_data::<T>(nd * dim, dim, dim);

            let query = make_mat(&query_data, nq, dim);
            let doc = make_mat(&doc_data, nd, dim);

            let mut expected_scores = vec![0.0f32; nq];
            let _ = MaxSim::new(&mut expected_scores)
                .unwrap()
                .evaluate(QueryMatRef::from(query), doc);

            let computer = build(query);
            let mut actual_scores = vec![0.0f32; nq];
            computer.max_sim(doc, &mut actual_scores);

            for i in 0..nq {
                assert!(
                    (actual_scores[i] - expected_scores[i]).abs() < tol,
                    "{label}MaxSim[{i}] mismatch for ({nq},{nd},{dim}): actual={}, expected={}",
                    actual_scores[i],
                    expected_scores[i],
                );
            }
        }
    }

    #[test]
    fn query_computer_dimensions() {
        let data = vec![1.0f32; 5 * 8];
        let query = make_mat(&data, 5, 8);
        let computer = QueryComputer::<f32>::new(query);

        assert_eq!(computer.nrows(), 5);
    }

    #[test]
    fn query_computer_f16_dimensions() {
        let data = vec![diskann_wide::cast_f32_to_f16(1.0); 5 * 8];
        let query = make_mat(data.as_slice(), 5, 8);
        let computer = QueryComputer::<half::f16>::new(query);

        assert_eq!(computer.nrows(), 5);
    }

    #[test]
    fn chamfer_with_zero_docs() {
        let query = make_mat(&[1.0f32, 0.0, 0.0, 1.0], 2, 2);
        let computer = QueryComputer::<f32>::new(query);
        let doc = make_mat(&[], 0, 2);
        assert_eq!(computer.chamfer(doc), 0.0);
    }

    #[test]
    fn max_sim_with_zero_docs() {
        let query = make_mat(&[1.0f32, 0.0, 0.0, 1.0], 2, 2);
        let computer = QueryComputer::<f32>::new(query);
        let doc = make_mat::<f32>(&[], 0, 2);
        let mut scores = vec![0.0f32; 2];
        computer.max_sim(doc, &mut scores);
        // With zero docs the scores buffer is left untouched.
        for &s in &scores {
            assert_eq!(s, 0.0, "zero-doc MaxSim should leave scores untouched");
        }
    }

    #[test]
    #[should_panic(expected = "scores buffer not right size")]
    fn max_sim_panics_on_size_mismatch() {
        let query = make_mat(&[1.0f32, 2.0, 3.0, 4.0], 2, 2);
        let computer = QueryComputer::<f32>::new(query);
        let doc = make_mat(&[1.0, 1.0], 1, 2);
        let mut scores = vec![0.0f32; 3]; // Wrong size
        computer.max_sim(doc, &mut scores);
    }

    macro_rules! test_matches_fallback {
        ($mod_name:ident, $ty:ty, $tol:expr, $label:literal) => {
            mod $mod_name {
                use super::*;

                #[test]
                fn chamfer_matches_fallback() {
                    check_chamfer_matches(QueryComputer::<$ty>::new, $tol, $label);
                }

                #[test]
                fn max_sim_matches_fallback() {
                    check_max_sim_matches(QueryComputer::<$ty>::new, $tol, $label);
                }
            }
        };
    }

    test_matches_fallback!(f32, f32, 1e-2, "f32 ");
    test_matches_fallback!(f16, half::f16, 1e-1, "f16 ");
}

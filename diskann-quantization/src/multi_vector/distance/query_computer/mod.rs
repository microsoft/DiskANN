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

use crate::multi_vector::block_transposed::BlockTransposed;
use crate::multi_vector::matrix::{MatRef, Standard};

/// An architecture-optimized query computer for multi-vector distance.
///
/// The architecture token and block-transposed GROUP are captured behind a
/// vtable at construction time. Distance calls dispatch through
/// [`Architecture::run3`](diskann_wide::Architecture::run3) with full
/// `#[target_feature]` propagation.
pub struct QueryComputer<T: Copy> {
    inner: Box<dyn DynQueryComputer<T>>,
}

impl<T: Copy> QueryComputer<T> {
    /// Number of logical (non-padded) query vectors.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    /// Dimensionality of each query vector.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    /// Total available rows (padded to GROUP boundary).
    #[inline]
    pub fn available_rows(&self) -> usize {
        self.inner.available_rows()
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
        assert!(
            scores.len() == nq,
            "scores buffer not right size: {} != {}",
            scores.len(),
            nq
        );

        if doc.num_vectors() == 0 {
            return;
        }

        let mut scratch = vec![f32::MIN; self.available_rows()];
        self.inner.raw_kernel(doc, &mut scratch);

        for (dst, &src) in scores.iter_mut().zip(&scratch[..nq]) {
            *dst = -src;
        }
    }
}

trait DynQueryComputer<T: Copy> {
    /// Run the SIMD kernel, writing max inner products into `scratch`.
    ///
    /// Values are positive (higher = more similar); the caller negates.
    fn raw_kernel(&self, doc: MatRef<'_, Standard<T>>, scratch: &mut [f32]);
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn available_rows(&self) -> usize;
}

struct Prepared<A, Q> {
    arch: A,
    prepared: Q,
    nrows: usize,
    ncols: usize,
    available_rows: usize,
}

/// Helper to build a [`Prepared`] from a [`MatRef`] and architecture token.
fn build_prepared<T: Copy + Default, A, const GROUP: usize>(
    arch: A,
    query: MatRef<'_, Standard<T>>,
) -> Prepared<A, BlockTransposed<T, GROUP>> {
    let nrows = query.num_vectors();
    let ncols = query.vector_dim();
    let prepared = BlockTransposed::<T, GROUP>::from_matrix_view(query.as_matrix_view());
    let available_rows = prepared.available_rows();
    Prepared {
        arch,
        prepared,
        nrows,
        ncols,
        available_rows,
    }
}

use diskann_vector::{DistanceFunctionMut, PureDistanceFunction};

use super::max_sim::{Chamfer, MaxSim};

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
    use crate::multi_vector::distance::{Chamfer, MaxSim, QueryMatRef};
    use diskann_vector::{DistanceFunctionMut, PureDistanceFunction};

    fn make_query_mat(data: &[f32], nrows: usize, ncols: usize) -> MatRef<'_, Standard<f32>> {
        MatRef::new(Standard::new(nrows, ncols).unwrap(), data).unwrap()
    }

    fn make_test_data(len: usize, ceil: usize, shift: usize) -> Vec<f32> {
        (0..len).map(|v| ((v + shift) % ceil) as f32).collect()
    }

    fn make_f16_mat(
        data: &[half::f16],
        nrows: usize,
        ncols: usize,
    ) -> MatRef<'_, Standard<half::f16>> {
        MatRef::new(Standard::new(nrows, ncols).unwrap(), data).unwrap()
    }

    fn make_f16_test_data(len: usize, ceil: usize, shift: usize) -> Vec<half::f16> {
        (0..len)
            .map(|v| diskann_wide::cast_f32_to_f16(((v + shift) % ceil) as f32))
            .collect()
    }

    const TEST_CASES: &[(usize, usize, usize)] = &[
        (1, 1, 4),   // Minimal
        (3, 4, 16),  // General case
        (7, 7, 32),  // Square case
        (17, 4, 64), // One more than A_PANEL (remainder)
    ];

    #[test]
    fn query_computer_dimensions() {
        let data = vec![1.0f32; 5 * 8];
        let query = make_query_mat(&data, 5, 8);
        let computer = QueryComputer::<f32>::new(query);

        assert_eq!(computer.nrows(), 5);
        assert_eq!(computer.ncols(), 8);
        assert!(computer.available_rows() >= 5);
        assert_eq!(computer.available_rows() % 8, 0);
    }

    #[test]
    fn query_computer_f16_dimensions() {
        let data = vec![diskann_wide::cast_f32_to_f16(1.0); 5 * 8];
        let query = MatRef::new(Standard::new(5, 8).unwrap(), data.as_slice()).unwrap();
        let computer = QueryComputer::<half::f16>::new(query);

        assert_eq!(computer.nrows(), 5);
        assert_eq!(computer.ncols(), 8);
        assert!(computer.available_rows() >= 5);
        assert_eq!(computer.available_rows() % 8, 0);
    }

    #[test]
    fn query_computer_single_vector() {
        let data = vec![1.0f32; 4];
        let query = make_query_mat(&data, 1, 4);
        let computer = QueryComputer::<f32>::new(query);

        assert_eq!(computer.nrows(), 1);
        assert_eq!(computer.ncols(), 4);
        assert!(computer.available_rows() >= 1);
    }

    #[test]
    fn chamfer_with_zero_docs() {
        let query = make_query_mat(&[1.0f32, 0.0, 0.0, 1.0], 2, 2);
        let computer = QueryComputer::<f32>::new(query);
        let doc = make_query_mat(&[], 0, 2);
        assert_eq!(computer.chamfer(doc), 0.0);
    }

    #[test]
    fn max_sim_with_zero_docs() {
        let query = make_query_mat(&[1.0f32, 0.0, 0.0, 1.0], 2, 2);
        let computer = QueryComputer::<f32>::new(query);
        let doc = make_query_mat(&[], 0, 2);
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
        let query = make_query_mat(&[1.0f32, 2.0, 3.0, 4.0], 2, 2);
        let computer = QueryComputer::<f32>::new(query);
        let doc = make_query_mat(&[1.0, 1.0], 1, 2);
        let mut scores = vec![0.0f32; 3]; // Wrong size
        computer.max_sim(doc, &mut scores);
    }

    macro_rules! test_matches_fallback {
        ($mod_name:ident, $make_data:ident, $make_mat:ident, $ty:ty, $tol:expr, $label:literal) => {
            mod $mod_name {
                use super::*;

                #[test]
                fn chamfer_matches_fallback() {
                    for &(nq, nd, dim) in TEST_CASES {
                        let query_data = $make_data(nq * dim, dim, dim / 2);
                        let doc_data = $make_data(nd * dim, dim, dim);

                        let query = $make_mat(&query_data, nq, dim);
                        let doc = $make_mat(&doc_data, nd, dim);

                        let expected = Chamfer::evaluate(QueryMatRef::from(query), doc);
                        let actual = QueryComputer::<$ty>::new(query).chamfer(doc);

                        assert!(
                            (actual - expected).abs() < $tol,
                            "{}Chamfer mismatch for ({},{},{}): actual={}, expected={}",
                            $label,
                            nq,
                            nd,
                            dim,
                            actual,
                            expected
                        );
                    }
                }

                #[test]
                fn max_sim_matches_fallback() {
                    for &(nq, nd, dim) in TEST_CASES {
                        let query_data = $make_data(nq * dim, dim, dim / 2);
                        let doc_data = $make_data(nd * dim, dim, dim);

                        let query = $make_mat(&query_data, nq, dim);
                        let doc = $make_mat(&doc_data, nd, dim);

                        let mut expected_scores = vec![0.0f32; nq];
                        let _ = MaxSim::new(&mut expected_scores)
                            .unwrap()
                            .evaluate(QueryMatRef::from(query), doc);

                        let computer = QueryComputer::<$ty>::new(query);
                        let mut actual_scores = vec![0.0f32; nq];
                        computer.max_sim(doc, &mut actual_scores);

                        for i in 0..nq {
                            assert!(
                                (actual_scores[i] - expected_scores[i]).abs() < $tol,
                                "{}MaxSim[{}] mismatch for ({},{},{}): actual={}, expected={}",
                                $label,
                                i,
                                nq,
                                nd,
                                dim,
                                actual_scores[i],
                                expected_scores[i]
                            );
                        }
                    }
                }
            }
        };
    }

    test_matches_fallback!(f32, make_test_data, make_query_mat, f32, 1e-2, "f32 ");
    test_matches_fallback!(
        f16,
        make_f16_test_data,
        make_f16_mat,
        half::f16,
        1e-1,
        "f16 "
    );
}

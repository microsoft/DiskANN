/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Fallback kernel implementation of multi-vector distance computation.

use std::ops::Deref;

use diskann_utils::views::rowmajor::{self, Matrix};
use diskann_vector::distance::InnerProduct;
use diskann_vector::{DistanceFunctionMut, PureDistanceFunction};

use super::max_sim::{Chamfer, MaxSim};
use super::projected_eigen::ProjectedEigen;
use crate::multi_vector::MaxSimError;

///////////
// Query //
///////////

/// A query matrix view for asymmetric distance functions.
///
/// This wrapper distinguishes query matrices from document matrices
/// at compile time, preventing accidental argument swapping in asymmetric
/// distance computations like [`MaxSim`] and [`Chamfer`].
///
/// # Example
///
/// ```
/// use diskann_utils::views::rowmajor::{Ref as MatRef};
/// use diskann_quantization::multi_vector::distance::Query;
///
/// let data = [1.0f32, 2.0, 3.0, 4.0];
/// let view = MatRef::try_from_data(&data, 2, 2).unwrap();
/// let query = Query(view);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Query<M>(pub M);

/// Deref so that we can transparently access the `MatRef` in distance functions.
impl<M> Deref for Query<M> {
    type Target = M;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

////////////////////
// FallbackKernel //
////////////////////

/// Fallback double-loop kernel to compute max-sim distances over multi-vectors.
///
/// This kernel performs a simple double-loop over the rows of `query`
/// and the `doc` and dispatches to [`InnerProduct`] to compute the similarity.
pub struct FallbackKernel;

impl FallbackKernel {
    /// Core kernel for computing per-query-vector max similarities (min negated inner-product).
    ///
    /// For each `query` vector, computes the maximum similarity (negated inner product)
    /// to any document vector, then calls `f(index, score)` with the result.
    /// If there are no vectors in the `doc`, the score is `f32::MAX`.
    ///
    /// The callback can be used to aggregate or set scores as needed - as is the
    /// case with [`MaxSim`] and [`Chamfer`].
    ///
    /// # Arguments
    ///
    /// * `query` - The query multi-vector (wrapped as [`QueryMatRef`])
    /// * `doc` - The document multi-vector
    /// * `f` - Callback invoked with `(query_index, similarity)` for each query vector
    #[inline]
    pub(crate) fn max_sim_kernel<F, T>(
        query: Query<rowmajor::Ref<'_, T>>,
        doc: rowmajor::Ref<'_, T>,
        mut f: F,
    ) where
        F: FnMut(usize, f32),
        InnerProduct: for<'a, 'b> PureDistanceFunction<&'a [T], &'b [T], f32>,
    {
        for (i, q_vec) in query.rows().enumerate() {
            // `InnerProduct::evaluate` returns negated inner product
            let mut min_dist = f32::MAX;

            for d_vec in doc.rows() {
                let dist = InnerProduct::evaluate(q_vec, d_vec);
                min_dist = min_dist.min(dist);
            }

            f(i, min_dist);
        }
    }

    /// Core kernel for computing per-query-vector projected-eigen scores.
    ///
    /// For each `query` vector, sums the negated squared inner product
    /// against every document vector, then calls `f(index, score)` with the
    /// result. If there are no vectors in the `doc`, the kernel returns
    /// immediately.
    ///
    /// The callback can be used to aggregate scores as needed - as is the
    /// case with [`ProjectedEigen`].
    ///
    /// # Arguments
    ///
    /// * `query` - The query multi-vector (wrapped as [`QueryMatRef`])
    /// * `doc` - The document multi-vector
    /// * `f` - Callback invoked with `(query_index, score)` for each query vector
    #[inline]
    pub(crate) fn projected_eigen_kernel<F, T>(
        query: Query<rowmajor::Ref<'_, T>>,
        doc: rowmajor::Ref<'_, T>,
        mut f: F,
    ) where
        F: FnMut(usize, f32),
        InnerProduct: for<'a, 'b> PureDistanceFunction<&'a [T], &'b [T], f32>,
    {
        // Early exit if no doc vectors - callback should never be invoked
        if doc.nrows() == 0 {
            return;
        }

        for (i, q_vec) in query.rows().enumerate() {
            let mut sum = 0.0f32;

            for d_vec in doc.rows() {
                // `InnerProduct::evaluate` returns the negated inner product;
                // squaring discards the sign, so negate the squared value to
                // obtain `-IP(q, d)²`.
                let ip = InnerProduct::evaluate(q_vec, d_vec);
                sum += -(ip * ip);
            }

            f(i, sum);
        }
    }
}

////////////
// MaxSim //
////////////

impl<T>
    DistanceFunctionMut<Query<rowmajor::Ref<'_, T>>, rowmajor::Ref<'_, T>, Result<(), MaxSimError>>
    for MaxSim<'_>
where
    InnerProduct: for<'a, 'b> PureDistanceFunction<&'a [T], &'b [T], f32>,
{
    #[inline(always)]
    fn evaluate(
        &mut self,
        query: Query<rowmajor::Ref<'_, T>>,
        doc: rowmajor::Ref<'_, T>,
    ) -> Result<(), MaxSimError> {
        let size = self.size();
        let n_queries = query.nrows();

        if self.size() != query.nrows() {
            return Err(MaxSimError::InvalidBufferLength(size, n_queries));
        }

        if query.ncols() != doc.ncols() {
            return Err(MaxSimError::UnequalDim(doc.ncols(), query.ncols()));
        }

        FallbackKernel::max_sim_kernel(query, doc, |i, score| {
            // SAFETY: We asserted that self.size() == query.num_vectors(),
            // and i < query.num_vectors() due to the kernel loop bound.
            unsafe { *self.scores.get_unchecked_mut(i) = score };
        });

        Ok(())
    }
}

/////////////
// Chamfer //
/////////////

impl<T> PureDistanceFunction<Query<rowmajor::Ref<'_, T>>, rowmajor::Ref<'_, T>, f32> for Chamfer
where
    InnerProduct: for<'a, 'b> PureDistanceFunction<&'a [T], &'b [T], f32>,
{
    #[inline(always)]
    fn evaluate(query: Query<rowmajor::Ref<'_, T>>, doc: rowmajor::Ref<'_, T>) -> f32 {
        let mut sum = 0.0f32;

        FallbackKernel::max_sim_kernel(query, doc, |_i, score| {
            sum += score;
        });

        sum
    }
}

/////////////////////
// ProjectedEigen //
/////////////////////

impl<T: Copy> PureDistanceFunction<Query<rowmajor::Ref<'_, T>>, rowmajor::Ref<'_, T>, f32>
    for ProjectedEigen
where
    InnerProduct: for<'a, 'b> PureDistanceFunction<&'a [T], &'b [T], f32>,
{
    #[inline(always)]
    fn evaluate(query: Query<rowmajor::Ref<'_, T>>, doc: rowmajor::Ref<'_, T>) -> f32 {
        let mut sum = 0.0f32;

        FallbackKernel::projected_eigen_kernel(query, doc, |_i, score| {
            sum += score;
        });

        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a QueryMatRef from raw data
    fn make_query(data: &[f32], nrows: usize, ncols: usize) -> Query<rowmajor::Ref<'_, f32>> {
        Query(make_doc(data, nrows, ncols))
    }

    /// Helper to create a MatRef from raw data
    fn make_doc(data: &[f32], nrows: usize, ncols: usize) -> rowmajor::Ref<'_, f32> {
        rowmajor::Ref::try_from_data(data, nrows, ncols).unwrap()
    }

    /// Naive implementation of max-sim for a single query vector against all doc vectors.
    fn naive_max_sim_single(query_vec: &[f32], doc: rowmajor::Ref<'_, f32>) -> f32 {
        doc.rows()
            .map(|d_vec| {
                let ip: f32 = query_vec.iter().zip(d_vec.iter()).map(|(a, b)| a * b).sum();
                -ip
            })
            .fold(f32::MAX, f32::min)
    }

    /// Naive implementation of projected-eigen for a single query vector
    /// against all doc vectors: `\sum_{j} -IP(q, d_{j})^2`.
    fn naive_projected_eigen_single(query_vec: &[f32], doc: rowmajor::Ref<'_, f32>) -> f32 {
        doc.rows()
            .map(|d_vec| {
                let ip: f32 = query_vec.iter().zip(d_vec.iter()).map(|(a, b)| a * b).sum();
                -(ip * ip)
            })
            .sum()
    }

    /// Generate deterministic test data.
    fn make_test_data(len: usize, ceil: usize, shift: usize) -> Vec<f32> {
        (0..len).map(|v| ((v + shift) % ceil) as f32).collect()
    }

    mod query_mat_ref {
        use super::*;

        #[test]
        fn from_mat_ref_and_deref() {
            let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
            let query = make_query(&data, 2, 3);

            // Deref access works
            assert_eq!(query.nrows(), 2);
            assert_eq!(query.ncols(), 3);
            assert_eq!(query.get_row(0), Some(&[1.0f32, 2.0, 3.0][..]));
        }

        #[test]
        fn is_copy() {
            let data = [1.0f32, 2.0];
            let query = make_query(&data, 1, 2);
            let copy = query;
            let _ = (query, copy); // Both usable
        }
    }

    mod distance_functions {
        use diskann_utils::Reborrow;

        use super::*;

        #[test]
        fn max_sim_panics_on_size_mismatch() {
            let query = make_query(&[1.0, 2.0, 3.0, 4.0], 2, 2);
            let doc = make_doc(&[1.0, 1.0], 1, 2);

            let mut scores = vec![0.0f32; 3]; // Wrong size
            let r = MaxSim::new(&mut scores).evaluate(query, doc);
            assert!(r.is_err());
        }

        /// Tests both MaxSim and Chamfer against naive implementations across
        /// various matrix sizes including edge cases (single row/col).
        #[test]
        fn matches_naive_implementation() {
            let test_cases = [
                (1, 1, 4),   // Single query, single doc
                (1, 5, 8),   // Single query, multiple docs
                (5, 1, 8),   // Multiple queries, single doc
                (3, 4, 16),  // General case
                (7, 7, 32),  // Square case
                (2, 3, 128), // Larger dimension
            ];

            for (nq, nd, dim) in test_cases.iter() {
                let query_data = make_test_data(nq * dim, *dim, dim / 2);
                let doc_data = make_test_data(nd * dim, *dim, *dim);

                let query = make_query(&query_data, *nq, *dim);
                let doc = make_doc(&doc_data, *nd, *dim);

                // Test MaxSim
                let mut scores = vec![0.0f32; *nq];
                let r = MaxSim::new(&mut scores).evaluate(query, doc);
                assert!(r.is_ok());

                let expected_scores: Vec<f32> = query
                    .rows()
                    .map(|q_vec| naive_max_sim_single(q_vec, doc))
                    .collect();

                for i in 0..*nq {
                    assert!(
                        (scores[i] - expected_scores[i]).abs() < 1e-10,
                        "MaxSim mismatch at {} for ({},{},{})",
                        i,
                        nq,
                        nd,
                        dim
                    );
                }

                // Check that FallbackKernel produces the same values as the naive reference.
                FallbackKernel::max_sim_kernel(query, doc, |i, score| {
                    assert!((expected_scores[i] - score).abs() <= 1e-10)
                });

                // Test Chamfer
                let chamfer = Chamfer::evaluate(query, doc);
                let expected_chamfer: f32 = expected_scores.iter().sum();

                assert!(
                    (chamfer - expected_chamfer).abs() < 1e-10,
                    "Chamfer mismatch for ({},{},{})",
                    nq,
                    nd,
                    dim
                );

                // Test ProjectedEigen
                let projected = ProjectedEigen::evaluate(query, doc);
                let expected_projected: f32 = query
                    .rows()
                    .map(|q_vec| naive_projected_eigen_single(q_vec, doc))
                    .sum();

                assert!(
                    (projected - expected_projected).abs()
                        < 1e-6 * expected_projected.abs().max(1.0),
                    "ProjectedEigen mismatch for ({},{},{})",
                    nq,
                    nd,
                    dim
                );
            }
        }

        #[test]
        fn chamfer_with_zero_queries_returns_zero() {
            let query = make_query(&[], 0, 2);
            let doc = make_doc(&[1.0, 0.0, 0.0, 1.0], 2, 2);

            let result = Chamfer::evaluate(query, doc);

            // No query vectors means sum is 0
            assert_eq!(result, 0.0);

            let result = Chamfer::evaluate(Query(doc), query.deref().reborrow());

            assert_eq!(result, f32::INFINITY);
        }

        #[test]
        fn projected_eigen_with_zero_docs_returns_zero() {
            let query = make_query(&[1.0, 0.0, 0.0, 1.0], 2, 2);
            let doc = make_doc(&[], 0, 2);

            // No document vectors means no pairs contribute, so the sum is 0.
            let result = ProjectedEigen::evaluate(query, doc);
            assert_eq!(result, 0.0);
        }
    }
}

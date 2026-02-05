// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Simple kernel implementation of multi-vector distance computation.

use std::ops::Deref;

use diskann_vector::distance::InnerProduct;
use diskann_vector::{DistanceFunctionMut, PureDistanceFunction};

use super::max_sim::{Chamfer, MaxSim};
use crate::multi_vector::{MatRef, MaxSimError, Repr, Standard};

/////////////////
// QueryMatRef //
/////////////////

/// A query matrix view for asymmetric distance functions.
///
/// This wrapper distinguishes query matrices from document matrices
/// at compile time, preventing accidental argument swapping in asymmetric
/// distance computations like [`MaxSim`] and [`Chamfer`].
///
/// # Example
///
/// ```
/// use diskann_quantization::multi_vector::{MatRef, Standard};
/// use diskann_quantization::multi_vector::distance::QueryMatRef;
///
/// let data = [1.0f32, 2.0, 3.0, 4.0];
/// let view = MatRef::new(Standard::new(2, 2), &data).unwrap();
/// let query: QueryMatRef<_> = view.into();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct QueryMatRef<'a, T: Repr>(pub MatRef<'a, T>);

impl<'a, T: Repr> From<MatRef<'a, T>> for QueryMatRef<'a, T> {
    fn from(view: MatRef<'a, T>) -> Self {
        Self(view)
    }
}

/// Deref so that we can transparently access the `MatRef` in distance functions.
impl<'a, T: Repr> Deref for QueryMatRef<'a, T> {
    type Target = MatRef<'a, T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

//////////////////
// SimpleKernel //
//////////////////

/// Simple double-loop kernel to compute max-sim distances over multi-vectors.
///
/// This kernel performs a simple double-loop over the rows of `query`
/// and the `doc` and dispatches to [`InnerProduct`] to compute the similarity.
pub struct SimpleKernel;

impl SimpleKernel {
    /// Core kernel for computing per-query-vector max similarities (min negated inner-product).
    ///
    /// For each `query` vector, computes the maximum similarity (negated inner product)
    /// to any document vector, then calls `f(index, score)` with the result. If
    /// there are no vectors in the `doc`, the kernel returns immediately.
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
    pub(crate) fn max_sim_kernel<F, T: Copy>(
        query: MatRef<'_, Standard<T>>,
        doc: MatRef<'_, Standard<T>>,
        mut f: F,
    ) where
        F: FnMut(usize, f32),
        InnerProduct: for<'a, 'b> PureDistanceFunction<&'a [T], &'b [T], f32>,
    {
        // Early exit if no doc vectors - callback should never be invoked
        if doc.num_vectors() == 0 {
            return;
        }

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
}

////////////
// MaxSim //
////////////

impl<T: Copy>
    DistanceFunctionMut<
        QueryMatRef<'_, Standard<T>>,
        MatRef<'_, Standard<T>>,
        Result<(), MaxSimError>,
    > for MaxSim<'_>
where
    InnerProduct: for<'a, 'b> PureDistanceFunction<&'a [T], &'b [T], f32>,
{
    #[inline(always)]
    fn evaluate(
        &mut self,
        query: QueryMatRef<'_, Standard<T>>,
        doc: MatRef<'_, Standard<T>>,
    ) -> Result<(), MaxSimError> {
        let size = self.size();
        let n_queries = query.num_vectors();

        if self.size() != query.num_vectors() {
            return Err(MaxSimError::InvalidBufferLength(size, n_queries));
        }

        SimpleKernel::max_sim_kernel(*query, doc, |i, score| {
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

impl<T: Copy> PureDistanceFunction<MatRef<'_, Standard<T>>, MatRef<'_, Standard<T>>, f32>
    for Chamfer
where
    InnerProduct: for<'a, 'b> PureDistanceFunction<&'a [T], &'b [T], f32>,
{
    #[inline(always)]
    fn evaluate(query: MatRef<'_, Standard<T>>, doc: MatRef<'_, Standard<T>>) -> f32 {
        let mut sum = 0.0f32;

        SimpleKernel::max_sim_kernel(query, doc, |_i, score| {
            sum += score;
        });

        sum
    }
}

impl<T: Copy> PureDistanceFunction<QueryMatRef<'_, Standard<T>>, MatRef<'_, Standard<T>>, f32>
    for Chamfer
where
    Self: for<'a, 'b> PureDistanceFunction<MatRef<'a, Standard<T>>, MatRef<'b, Standard<T>>, f32>,
{
    #[inline(always)]
    fn evaluate(query: QueryMatRef<'_, Standard<T>>, doc: MatRef<'_, Standard<T>>) -> f32 {
        Self::evaluate(*query, doc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a QueryMatRef from raw data
    fn make_query(data: &[f32], nrows: usize, ncols: usize) -> QueryMatRef<'_, Standard<f32>> {
        MatRef::new(Standard::new(nrows, ncols), data)
            .unwrap()
            .into()
    }

    /// Helper to create a MatRef from raw data
    fn make_doc(data: &[f32], nrows: usize, ncols: usize) -> MatRef<'_, Standard<f32>> {
        MatRef::new(Standard::new(nrows, ncols), data).unwrap()
    }

    /// Naive implementation of max-sim for a single query vector against all doc vectors.
    fn naive_max_sim_single(query_vec: &[f32], doc: &MatRef<'_, Standard<f32>>) -> f32 {
        doc.rows()
            .map(|d_vec| {
                let ip: f32 = query_vec.iter().zip(d_vec.iter()).map(|(a, b)| a * b).sum();
                -ip
            })
            .fold(f32::MAX, f32::min)
    }

    /// Generate a vector of random f32 values in [-1, 1] for testing
    fn make_test_data(len: usize, ceil: usize, shift: usize) -> Vec<f32> {
        (0..len).map(|v| ((v + shift) % ceil) as f32).collect()
    }

    mod query_mat_ref {
        use super::*;

        #[test]
        fn from_mat_ref_and_deref() {
            let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
            let view = MatRef::new(Standard::new(2, 3), &data).unwrap();
            let query: QueryMatRef<_> = view.into();

            // Deref access works
            assert_eq!(query.num_vectors(), 2);
            assert_eq!(query.vector_dim(), 3);
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
            let r = MaxSim::new(&mut scores).unwrap().evaluate(query, doc);
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
                let r = MaxSim::new(&mut scores).unwrap().evaluate(query, doc);
                assert!(r.is_ok());

                let expected_scores: Vec<f32> = query
                    .rows()
                    .map(|q_vec| naive_max_sim_single(q_vec, &doc))
                    .collect();

                for i in 0..*nq {
                    assert!(
                        (scores[i] - expected_scores[i]).abs() < 1e-5,
                        "MaxSim mismatch at {} for ({},{},{})",
                        i,
                        nq,
                        nd,
                        dim
                    );
                }

                // Check that SimpleKernel is also correct.
                SimpleKernel::max_sim_kernel(*query, doc, |i, score| {
                    assert!((scores[i] - score).abs() <= 1e-6)
                });

                // Test Chamfer
                let chamfer = Chamfer::evaluate(query, doc);
                let expected_chamfer: f32 = expected_scores.iter().sum();

                assert!(
                    (chamfer - expected_chamfer).abs() < 1e-4,
                    "Chamfer mismatch for ({},{},{})",
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

            let result = Chamfer::evaluate(doc, query.deref().reborrow());

            assert_eq!(result, 0.0);
        }
    }
}

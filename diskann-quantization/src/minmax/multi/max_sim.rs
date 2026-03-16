// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Distance implementations for MinMax quantized multi-vectors.

use diskann_vector::{DistanceFunctionMut, PureDistanceFunction};

use super::super::vectors::{DataRef, MinMaxIP};
use super::meta::MinMaxMeta;
use crate::bits::{Representation, Unsigned};
use crate::distances::{self, UnequalLengths};
use crate::multi_vector::distance::QueryMatRef;
use crate::multi_vector::{Chamfer, MatRef, MaxSim};

//////////////////
// MinMaxKernel //
//////////////////

/// Kernel for computing [`MaxSim`] and [`Chamfer`] distance using MinMax quantized vectors.
///
/// Uses a simple double-iteration strategy and computes pairwise inner-products between
/// query vectors and document vectors using [`MinMaxIP`].
pub struct MinMaxKernel;

impl MinMaxKernel {
    /// Core kernel for computing per-query-vector max similarities using MinMax.
    ///
    /// For each query vector, computes the maximum similarity (min distance using
    /// MinMax inner product) to any document vector, then calls `f(index, score)` with the result.
    ///
    /// # Arguments
    ///
    /// * `query` - The query MinMax multi-vector (wrapped as [`QueryMatRef`])
    /// * `doc` - The document MinMax multi-vector
    /// * `f` - Callback invoked with `(query_index, min_distance)` for each query vector
    #[inline(always)]
    pub(crate) fn max_sim_kernel<const N: usize, const M: usize, F>(
        query: QueryMatRef<'_, MinMaxMeta<N>>,
        doc: MatRef<'_, MinMaxMeta<M>>,
        mut f: F,
    ) -> Result<(), UnequalLengths>
    where
        Unsigned: Representation<N> + Representation<M>,
        distances::InnerProduct: for<'x, 'y> PureDistanceFunction<
                crate::bits::BitSlice<'x, N, Unsigned>,
                crate::bits::BitSlice<'y, M, Unsigned>,
                distances::MathematicalResult<u32>,
            >,
        F: FnMut(usize, f32),
    {
        for (i, q_ref) in query.rows().enumerate() {
            // Find min distance (IP returns negated, so min = max similarity)
            let mut min_distance = f32::MAX;

            for d_ref in doc.rows() {
                // Use MinMaxIP to compute negated inner product as distance
                let dist = <MinMaxIP as PureDistanceFunction<
                    DataRef<'_, N>,
                    DataRef<'_, M>,
                    distances::Result<f32>,
                >>::evaluate(q_ref, d_ref)?;

                min_distance = min_distance.min(dist);
            }

            f(i, min_distance);
        }

        Ok(())
    }
}

////////////
// MaxSim //
////////////

impl<const N: usize, const M: usize>
    DistanceFunctionMut<QueryMatRef<'_, MinMaxMeta<N>>, MatRef<'_, MinMaxMeta<M>>> for MaxSim<'_>
where
    Unsigned: Representation<N> + Representation<M>,
    distances::InnerProduct: for<'x, 'y> PureDistanceFunction<
            crate::bits::BitSlice<'x, N, Unsigned>,
            crate::bits::BitSlice<'y, M, Unsigned>,
            distances::MathematicalResult<u32>,
        >,
{
    #[inline(always)]
    fn evaluate(&mut self, query: QueryMatRef<'_, MinMaxMeta<N>>, doc: MatRef<'_, MinMaxMeta<M>>) {
        assert!(
            self.size() == query.num_vectors(),
            "scores buffer not right size : {} != {}",
            self.size(),
            query.num_vectors()
        );

        let _ = MinMaxKernel::max_sim_kernel(query, doc, |i, score| {
            // SAFETY: We asserted that self.size() == query.num_vectors(),
            // and i < query.num_vectors() due to the kernel loop bound.
            let _ = self.set(i, score);
        });
    }
}

/////////////
// Chamfer //
/////////////

impl<const N: usize, const M: usize>
    PureDistanceFunction<QueryMatRef<'_, MinMaxMeta<N>>, MatRef<'_, MinMaxMeta<M>>, f32> for Chamfer
where
    Unsigned: Representation<N> + Representation<M>,
    distances::InnerProduct: for<'a, 'b> PureDistanceFunction<
            crate::bits::BitSlice<'a, N, Unsigned>,
            crate::bits::BitSlice<'b, M, Unsigned>,
            distances::MathematicalResult<u32>,
        >,
{
    #[inline(always)]
    fn evaluate(query: QueryMatRef<'_, MinMaxMeta<N>>, doc: MatRef<'_, MinMaxMeta<M>>) -> f32 {
        let mut sum = 0.0f32;

        let _ = MinMaxKernel::max_sim_kernel(query, doc, |_i, score| {
            sum += score;
        });

        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CompressInto;
    use crate::algorithms::Transform;
    use crate::algorithms::transforms::NullTransform;
    use crate::bits::{Representation, Unsigned};
    use crate::minmax::{Data, MinMaxQuantizer};
    use crate::multi_vector::{Defaulted, Mat, Standard};
    use crate::num::Positive;
    use diskann_utils::ReborrowMut;
    use std::num::NonZeroUsize;

    macro_rules! expand_to_bitrates {
        ($name:ident, $func:ident) => {
            #[test]
            fn $name() {
                // Homogeneous
                $func::<1, 1>();
                $func::<2, 2>();
                $func::<4, 4>();
                $func::<8, 8>();
                // Heterogeneous
                $func::<8, 4>();
                $func::<8, 2>();
                $func::<8, 1>();
            }
        };
    }

    /// Test cases: (num_queries, num_docs, dim)
    const TEST_CASES: &[(usize, usize, usize)] = &[
        (1, 1, 4),   // Single query, single doc
        (1, 5, 8),   // Single query, multiple docs
        (5, 1, 8),   // Multiple queries, single doc
        (3, 4, 16),  // General case
        (7, 7, 32),  // Square case
        (2, 3, 128), // Larger dimension
    ];

    fn make_quantizer(dim: usize) -> MinMaxQuantizer {
        MinMaxQuantizer::new(
            Transform::Null(NullTransform::new(NonZeroUsize::new(dim).unwrap())),
            Positive::new(1.0).unwrap(),
        )
    }

    fn generate_input_mat(n: usize, dim: usize, offset: usize) -> Vec<f32> {
        (0..n * dim)
            .map(|idx| {
                let i = idx / dim;
                let j = idx % dim;
                ((i + offset) * dim + j) as f32 * 0.1
            })
            .collect()
    }

    fn compress_mat<const NBITS: usize>(
        quantizer: &MinMaxQuantizer,
        input: &[f32],
        n: usize,
        dim: usize,
    ) -> Mat<MinMaxMeta<NBITS>>
    where
        Unsigned: Representation<NBITS>,
    {
        let input_mat = MatRef::new(Standard::<f32>::new(n, dim).unwrap(), input).unwrap();
        let mut output: Mat<MinMaxMeta<NBITS>> =
            Mat::new(MinMaxMeta::new(n, dim), Defaulted).unwrap();
        quantizer
            .compress_into(input_mat, output.reborrow_mut())
            .unwrap();
        output
    }

    /// Naive max-sim for one query vector: min distance to any doc vector.
    fn naive_max_sim_single<const N: usize, const M: usize>(
        query: DataRef<'_, N>,
        doc: &MatRef<'_, MinMaxMeta<M>>,
    ) -> f32
    where
        Unsigned: Representation<N> + Representation<M>,
        distances::InnerProduct: for<'x, 'y> PureDistanceFunction<
                crate::bits::BitSlice<'x, N, Unsigned>,
                crate::bits::BitSlice<'y, M, Unsigned>,
                distances::MathematicalResult<u32>,
            >,
    {
        doc.rows()
            .map(|d| {
                <MinMaxIP as PureDistanceFunction<
                    DataRef<'_, N>,
                    DataRef<'_, M>,
                    distances::Result<f32>,
                >>::evaluate(query, d)
                .unwrap()
            })
            .fold(f32::MAX, f32::min)
    }

    fn test_matches_naive<const N: usize, const M: usize>()
    where
        Unsigned: Representation<N> + Representation<M>,
        distances::InnerProduct: for<'x, 'y> PureDistanceFunction<
                crate::bits::BitSlice<'x, N, Unsigned>,
                crate::bits::BitSlice<'y, M, Unsigned>,
                distances::MathematicalResult<u32>,
            >,
    {
        for &(nq, nd, dim) in TEST_CASES {
            let quantizer = make_quantizer(dim);

            let query_data = generate_input_mat(nq, dim, 0);
            let doc_data = generate_input_mat(nd, dim, nq);

            let query_mat = compress_mat::<N>(&quantizer, &query_data, nq, dim);
            let doc_mat = compress_mat::<M>(&quantizer, &doc_data, nd, dim);

            let query: QueryMatRef<_> = query_mat.as_view().into();
            let doc = doc_mat.as_view();

            // Test MaxSim matches naive
            let expected: Vec<f32> = query
                .rows()
                .map(|q| naive_max_sim_single(q, &doc))
                .collect();

            let mut scores = vec![0.0f32; nq];
            MaxSim::new(&mut scores).unwrap().evaluate(query, doc);

            for (i, (&got, &exp)) in scores.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-5,
                    "({N},{M}) ({nq},{nd},{dim}) MaxSim[{i}]: {got} != {exp}"
                );
            }

            // Test kernel matches MaxSim
            let mut kernel_scores = vec![0.0f32; nq];
            MinMaxKernel::max_sim_kernel(query, doc, |i, s| kernel_scores[i] = s).unwrap();
            assert_eq!(
                scores, kernel_scores,
                "({N},{M}) ({nq},{nd},{dim}) kernel mismatch"
            );

            // Test Chamfer equals sum of MaxSim
            let chamfer = Chamfer::evaluate(query, doc);
            let sum: f32 = scores.iter().sum();
            assert!(
                (chamfer - sum).abs() < 1e-4,
                "({N},{M}) ({nq},{nd},{dim}) Chamfer {chamfer} != sum {sum}"
            );
        }
    }

    expand_to_bitrates!(matches_naive, test_matches_naive);

    #[test]
    #[should_panic(expected = "scores buffer not right size")]
    fn max_sim_panics_on_size_mismatch() {
        let dim = 4;
        let row_bytes = Data::<8>::canonical_bytes(dim);
        let query_data = vec![0u8; 2 * row_bytes];
        let doc_data = vec![0u8; 3 * row_bytes];

        let query: QueryMatRef<_> = MatRef::new(MinMaxMeta::<8>::new(2, dim), &query_data)
            .unwrap()
            .into();
        let doc = MatRef::new(MinMaxMeta::<8>::new(3, dim), &doc_data).unwrap();

        let mut scores = vec![0.0f32; 5]; // Wrong size
        MaxSim::new(&mut scores).unwrap().evaluate(query, doc);
    }
}

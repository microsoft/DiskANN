// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Distance implementations for MinMax quantized multi-vectors.

use diskann_vector::{DistanceFunctionMut, PureDistanceFunction};

use super::super::vectors::{DataRef, FullQueryMeta, MinMaxIP};
use super::meta::MinMaxMeta;
use crate::bits::{Representation, Unsigned};
use crate::distances::{self, UnequalLengths};
use crate::multi_vector::distance::QueryMatRef;
use crate::multi_vector::{Chamfer, MatRef, MaxSim, Repr, SliceMatRepr};

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
    /// The query [`Repr`] can differ from the document repr, enabling asymmetric distance
    /// (e.g. full-precision queries against quantized documents) as long as [`MinMaxIP`]
    /// is implemented between the query row type and [`DataRef`].
    ///
    /// # Arguments
    ///
    /// * `query` - The query multi-vector (wrapped as [`QueryMatRef`])
    /// * `doc` - The document MinMax multi-vector
    /// * `f` - Callback invoked with `(query_index, min_distance)` for each query vector
    #[inline(always)]
    pub(crate) fn max_sim_kernel<'q, 'd, const NBITS: usize, Q, F>(
        query: QueryMatRef<'q, Q>,
        doc: MatRef<'d, MinMaxMeta<NBITS>>,
        mut f: F,
    ) -> Result<(), UnequalLengths>
    where
        Q: Repr + 'q,
        Q::Row<'q>: Copy,
        Unsigned: Representation<NBITS>,
        MinMaxIP: PureDistanceFunction<Q::Row<'q>, DataRef<'d, NBITS>, distances::Result<f32>>,
        F: FnMut(usize, f32),
    {
        for (i, q_ref) in query.rows().enumerate() {
            // Find min distance (IP returns negated, so min = max similarity)
            let mut min_distance = f32::MAX;

            for d_ref in doc.rows() {
                let dist = MinMaxIP::evaluate(q_ref, d_ref)?;
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

impl<'q, 'd, const NBITS: usize, Q>
    DistanceFunctionMut<QueryMatRef<'q, Q>, MatRef<'d, MinMaxMeta<NBITS>>> for MaxSim<'_>
where
    Q: Repr + 'q,
    Q::Row<'q>: Copy,
    Unsigned: Representation<NBITS>,
    MinMaxIP: PureDistanceFunction<Q::Row<'q>, DataRef<'d, NBITS>, distances::Result<f32>>,
{
    #[inline(always)]
    fn evaluate(&mut self, query: QueryMatRef<'q, Q>, doc: MatRef<'d, MinMaxMeta<NBITS>>) {
        assert!(
            self.size() == query.num_vectors(),
            "scores buffer not right size : {} != {}",
            self.size(),
            query.num_vectors()
        );

        let _ = MinMaxKernel::max_sim_kernel(query, doc, |i, score| {
            // SAFETY: We asserted that self.size() == query.num_vectors(),
            // and i < query.num_vectors() due to the kernel loop bound.
            unsafe { *self.scores.get_unchecked_mut(i) = score };
        });
    }
}

/////////////
// Chamfer //
/////////////

impl<'q, 'd, const NBITS: usize, Q>
    PureDistanceFunction<QueryMatRef<'q, Q>, MatRef<'d, MinMaxMeta<NBITS>>, f32> for Chamfer
where
    Q: Repr + 'q,
    Q::Row<'q>: Copy,
    Unsigned: Representation<NBITS>,
    MinMaxIP: PureDistanceFunction<Q::Row<'q>, DataRef<'d, NBITS>, distances::Result<f32>>,
{
    #[inline(always)]
    fn evaluate(query: QueryMatRef<'q, Q>, doc: MatRef<'d, MinMaxMeta<NBITS>>) -> f32 {
        let mut sum = 0.0f32;

        let _ = MinMaxKernel::max_sim_kernel(query, doc, |_i, score| {
            sum += score;
        });

        sum
    }
}

/////////////////////////////////
// Asymmetric: FullQuery alias //
/////////////////////////////////

/// Type alias for a full-precision query matrix view using [`SliceMatRepr`].
///
/// Each row is a [`FullQueryRef`](super::super::vectors::FullQueryRef) containing
/// the full-precision vector elements and precomputed metadata (sum, norm²).
/// This enables asymmetric distance computation where queries remain in full
/// precision while documents are MinMax-quantized.
pub type FullQueryMatRef<'a> = QueryMatRef<'a, SliceMatRepr<f32, FullQueryMeta>>;

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
                $func::<1>();
                $func::<2>();
                $func::<4>();
                $func::<8>();
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

    /// Verify MaxSim, kernel, and Chamfer all agree with a naive per-row computation
    /// for an arbitrary query repr `Q`.
    fn verify_case<'q, 'd, const NBITS: usize, Q>(
        query: QueryMatRef<'q, Q>,
        doc: MatRef<'d, MinMaxMeta<NBITS>>,
        nq: usize,
        nd: usize,
        dim: usize,
        label: &str,
    ) where
        Q: Repr + 'q,
        Q::Row<'q>: Copy,
        Unsigned: Representation<NBITS>,
        MinMaxIP: PureDistanceFunction<Q::Row<'q>, DataRef<'d, NBITS>, distances::Result<f32>>,
    {
        // Naive max-sim: for each query row, find min distance to any doc row.
        let expected: Vec<f32> = query
            .rows()
            .map(|q| {
                doc.rows()
                    .map(|d| MinMaxIP::evaluate(q, d).unwrap())
                    .fold(f32::MAX, f32::min)
            })
            .collect();

        // Test MaxSim matches naive
        let mut scores = vec![0.0f32; nq];
        MaxSim::new(&mut scores).unwrap().evaluate(query, doc);

        for (i, (&got, &exp)) in scores.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "NBITS={NBITS} ({nq},{nd},{dim}) {label} MaxSim[{i}]: {got} != {exp}"
            );
        }

        // Test kernel matches MaxSim
        let mut kernel_scores = vec![0.0f32; nq];
        MinMaxKernel::max_sim_kernel(query, doc, |i, s| kernel_scores[i] = s).unwrap();
        assert_eq!(
            scores, kernel_scores,
            "NBITS={NBITS} ({nq},{nd},{dim}) {label} kernel mismatch"
        );

        // Test Chamfer equals sum of MaxSim
        let chamfer = Chamfer::evaluate(query, doc);
        let sum: f32 = scores.iter().sum();
        assert!(
            (chamfer - sum).abs() < 1e-4,
            "NBITS={NBITS} ({nq},{nd},{dim}) {label} Chamfer {chamfer} != sum {sum}"
        );
    }

    fn test_matches_naive<const NBITS: usize>()
    where
        Unsigned: Representation<NBITS>,
        distances::InnerProduct: for<'x, 'y> PureDistanceFunction<
                crate::bits::BitSlice<'x, NBITS, Unsigned>,
                crate::bits::BitSlice<'y, NBITS, Unsigned>,
                distances::MathematicalResult<u32>,
            >,
    {
        for &(nq, nd, dim) in TEST_CASES {
            let quantizer = make_quantizer(dim);

            let query_data = generate_input_mat(nq, dim, 0);
            let doc_data = generate_input_mat(nd, dim, nq);

            let query_mat = compress_mat::<NBITS>(&quantizer, &query_data, nq, dim);
            let doc_mat = compress_mat::<NBITS>(&quantizer, &doc_data, nd, dim);

            let query: QueryMatRef<_> = query_mat.as_view().into();
            let doc = doc_mat.as_view();

            verify_case(query, doc, nq, nd, dim, "");
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

    /////////////////////////////
    // Asymmetric (full query) //
    /////////////////////////////

    /// Build an owned `Mat<SliceMatRepr<f32, FullQueryMeta>>` from raw f32 query vectors.
    ///
    /// Each row gets a `FullQueryMeta { sum, norm_squared }` computed from its elements.
    fn build_full_query_mat(
        data: &[f32],
        nrows: usize,
        ncols: usize,
    ) -> Mat<SliceMatRepr<f32, FullQueryMeta>> {
        let repr = SliceMatRepr::<f32, FullQueryMeta>::new(nrows, ncols).unwrap();
        let mut mat = Mat::new(repr, Defaulted).unwrap();

        for i in 0..nrows {
            let src = &data[i * ncols..(i + 1) * ncols];
            let sum: f32 = src.iter().sum();
            let norm_squared: f32 = src.iter().map(|x| x * x).sum();

            let mut row = mat.get_row_mut(i).unwrap();
            *row.meta_mut() = FullQueryMeta { sum, norm_squared };
            row.vector_mut().copy_from_slice(src);
        }

        mat
    }

    fn test_full_query_matches_naive<const NBITS: usize>()
    where
        Unsigned: Representation<NBITS>,
        distances::InnerProduct: for<'x, 'y> PureDistanceFunction<
                &'x [f32],
                crate::bits::BitSlice<'y, NBITS, Unsigned>,
                distances::MathematicalResult<f32>,
            >,
    {
        for &(nq, nd, dim) in TEST_CASES {
            let quantizer = make_quantizer(dim);

            let query_data = generate_input_mat(nq, dim, 0);
            let doc_data = generate_input_mat(nd, dim, nq);

            let query_mat = build_full_query_mat(&query_data, nq, dim);
            let doc_mat = compress_mat::<NBITS>(&quantizer, &doc_data, nd, dim);

            let query: FullQueryMatRef<'_> = query_mat.as_view().into();
            let doc = doc_mat.as_view();

            verify_case(query, doc, nq, nd, dim, "FullQuery");
        }
    }

    expand_to_bitrates!(full_query_matches_naive, test_full_query_matches_naive);

    #[test]
    #[should_panic(expected = "scores buffer not right size")]
    fn full_query_max_sim_panics_on_size_mismatch() {
        let dim = 4;
        let query_data = vec![1.0f32; 2 * dim];
        let query_mat = build_full_query_mat(&query_data, 2, dim);
        let query: FullQueryMatRef<'_> = query_mat.as_view().into();

        let quantizer = make_quantizer(dim);
        let doc_data = vec![1.0f32; 3 * dim];
        let doc_mat = compress_mat::<8>(&quantizer, &doc_data, 3, dim);
        let doc = doc_mat.as_view();

        let mut scores = vec![0.0f32; 5]; // Wrong size (should be 2)
        MaxSim::new(&mut scores).unwrap().evaluate(query, doc);
    }

    #[test]
    fn full_query_chamfer_with_zero_queries_returns_zero() {
        let dim = 4;
        let query_mat = build_full_query_mat(&[], 0, dim);
        let query: FullQueryMatRef<'_> = query_mat.as_view().into();

        let quantizer = make_quantizer(dim);
        let doc_data = vec![1.0f32; 2 * dim];
        let doc_mat = compress_mat::<8>(&quantizer, &doc_data, 2, dim);
        let doc = doc_mat.as_view();

        let result = Chamfer::evaluate(query, doc);
        assert_eq!(result, 0.0);
    }
}

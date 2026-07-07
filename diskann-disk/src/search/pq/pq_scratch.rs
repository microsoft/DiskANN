/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
//! Aligned allocator

use diskann::{ANNError, ANNResult};

use diskann_quantization::alloc::{AlignedAllocator, Poly};

#[derive(Debug)]
/// PQ scratch
pub struct PQScratch {
    /// Aligned pq table distance scratch, the length must be at least [256 * NCHUNKS]. 256 is the number of PQ centroids.
    /// This is used to store the distance between each chunk in the query vector to each centroid, which is why the length is num of centroids * num of chunks
    pub aligned_pqtable_dist_scratch: Poly<[f32], AlignedAllocator>,

    /// Aligned dist scratch, must be at least diskann MAX_DEGREE
    /// This is used to temporarily save the pq distance between query vector to the candidate vectors.
    pub aligned_dist_scratch: Poly<[f32], AlignedAllocator>,

    /// Aligned pq coord scratch, must be at least [N_CHUNKS * MAX_DEGREE]
    /// This is used to store the pq coordinates of the candidate vectors.
    pub aligned_pq_coord_scratch: Poly<[u8], AlignedAllocator>,

    /// Query scratch buffer stored as `f32`, sized by the PQ table's logical dimension.
    /// `set` populates it from a caller-provided `&[f32]`; `PQTable::preprocess_query` can
    /// then rotate or otherwise preprocess it.
    pub query_scratch: Vec<f32>,
}

impl PQScratch {
    /// Create a new pq scratch.
    ///
    /// `dim` is the PQ table's logical dimension (`PQData::get_dim()`); the
    /// internal `query_scratch` buffer is sized to exactly this many `f32` slots.
    pub fn new(
        graph_degree: usize,
        dim: usize,
        num_pq_chunks: usize,
        num_centers: usize,
    ) -> ANNResult<Self> {
        let aligned_pq_coord_scratch =
            Poly::broadcast(0u8, graph_degree * num_pq_chunks, AlignedAllocator::A128)
                .map_err(ANNError::log_index_error)?;
        let aligned_pqtable_dist_scratch =
            Poly::broadcast(0f32, num_centers * num_pq_chunks, AlignedAllocator::A128)
                .map_err(ANNError::log_index_error)?;
        let aligned_dist_scratch = Poly::broadcast(0f32, graph_degree, AlignedAllocator::A128)
            .map_err(ANNError::log_index_error)?;
        let query_scratch = vec![0.0f32; dim];

        Ok(Self {
            aligned_pqtable_dist_scratch,
            aligned_dist_scratch,
            aligned_pq_coord_scratch,
            query_scratch,
        })
    }

    /// Copy `query` into `query_scratch`.
    ///
    /// `query` must already be in full-precision `f32` representation; quantized
    /// inputs (e.g. `MinMaxElement`) should be decoded via `VectorRepr::as_f32`
    /// at the caller boundary before invoking this method.
    ///
    /// Returns `DimensionMismatchError` if `query.len() != query_scratch.len()`.
    pub fn set(&mut self, query: &[f32]) -> ANNResult<()> {
        let dim = self.query_scratch.len();
        if query.len() != dim {
            return Err(ANNError::log_dimension_mismatch_error(format!(
                "PQScratch::set: expected query of length {dim}, got {}",
                query.len()
            )));
        }
        self.query_scratch.copy_from_slice(query);
        Ok(())
    }

    /// Return the largest number of PQ vectors whose distances can be computed using this
    /// scratch data structure.
    pub(crate) fn max_vectors(&self) -> usize {
        self.aligned_dist_scratch.len()
    }
}

#[cfg(test)]
mod tests {
    use diskann_quantization::num::PowerOfTwo;
    use rstest::rstest;

    use super::PQScratch;

    #[rstest]
    #[case(512, 8, 128, 256)] // default test case
    #[case(59, 16, 37, 41)] // not multiple of 256
    fn test_pq_scratch(
        #[case] graph_degree: usize,
        #[case] dim: usize,
        #[case] num_pq_chunks: usize,
        #[case] num_centers: usize,
    ) {
        let mut pq_scratch: PQScratch =
            PQScratch::new(graph_degree, dim, num_pq_chunks, num_centers).unwrap();

        assert_eq!(
            (pq_scratch.aligned_pqtable_dist_scratch.as_ptr() as usize) % PowerOfTwo::V128.raw(),
            0
        );
        assert_eq!(
            (pq_scratch.aligned_dist_scratch.as_ptr() as usize) % PowerOfTwo::V128.raw(),
            0
        );
        assert_eq!(
            (pq_scratch.aligned_pq_coord_scratch.as_ptr() as usize) % PowerOfTwo::V128.raw(),
            0
        );

        assert_eq!(pq_scratch.max_vectors(), graph_degree);

        // Test set() method
        let query: Vec<f32> = (1..=dim).map(|i| i as f32).collect();
        pq_scratch.set(&query).unwrap();

        (0..query.len()).for_each(|i| {
            assert_eq!(pq_scratch.query_scratch[i], query[i]);
        });
    }

    #[test]
    fn test_pq_scratch_set_rejects_short_query() {
        let dim = 16;
        let mut pq_scratch = PQScratch::new(64, dim, 4, 256).unwrap();

        // Query shorter than dim should fail
        let short_query: Vec<f32> = (1..dim).map(|i| i as f32).collect(); // dim-1 elements
        let err = pq_scratch.set(&short_query).unwrap_err();
        assert_eq!(err.kind(), diskann::ANNErrorKind::DimensionMismatchError);
        assert!(err.to_string().contains("expected query of length"));
    }

    #[test]
    fn test_pq_scratch_set_rejects_oversized_query() {
        let dim = 8;
        let mut pq_scratch = PQScratch::new(64, dim, 4, 256).unwrap();

        // Query longer than dim should fail
        let long_query: Vec<f32> = (1..=dim + 10).map(|i| i as f32).collect();
        let err = pq_scratch.set(&long_query).unwrap_err();
        assert_eq!(err.kind(), diskann::ANNErrorKind::DimensionMismatchError);
        assert!(err.to_string().contains("expected query of length"));
    }
}

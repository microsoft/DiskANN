/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
//! Aligned allocator

use diskann::{error::IntoANNResult, utils::VectorRepr, ANNError, ANNResult};

use diskann_providers::common::{AlignedSlice, aligned_alloc};

#[derive(Debug)]
/// PQ scratch
pub struct PQScratch {
    /// Aligned pq table distance scratch, the length must be at least [256 * NCHUNKS]. 256 is the number of PQ centroids.
    /// This is used to store the distance between each chunk in the query vector to each centroid, which is why the length is num of centroids * num of chunks
    pub aligned_pqtable_dist_scratch: AlignedSlice<f32>,

    /// Aligned dist scratch, must be at least diskann MAX_DEGREE
    /// This is used to temporarily save the pq distance between query vector to the candidate vectors.
    pub aligned_dist_scratch: AlignedSlice<f32>,

    /// Aligned pq coord scratch, must be at least [N_CHUNKS * MAX_DEGREE]
    /// This is used to store the pq coordinates of the candidate vectors.
    pub aligned_pq_coord_scratch: AlignedSlice<u8>,

    /// Query scratch buffer stored as `f32`. `set` initializes it by copying/converting the
    /// raw query values; `PQTable.PreprocessQuery` can then rotate or otherwise preprocess it.
    pub rotated_query: Vec<f32>,
}

impl PQScratch {
    /// 128 bytes alignment to optimize for the L2 Adjacent Cache Line Prefetcher.
    const ALIGNED_ALLOC_128: usize = 128;

    /// Create a new pq scratch
    pub fn new(
        graph_degree: usize,
        dim: usize,
        num_pq_chunks: usize,
        num_centers: usize,
    ) -> ANNResult<Self> {
        let aligned_pq_coord_scratch =
            aligned_alloc(graph_degree * num_pq_chunks, PQScratch::ALIGNED_ALLOC_128)?;
        let aligned_pqtable_dist_scratch =
            aligned_alloc(num_centers * num_pq_chunks, PQScratch::ALIGNED_ALLOC_128)?;
        let aligned_dist_scratch = aligned_alloc(graph_degree, PQScratch::ALIGNED_ALLOC_128)?;
        let rotated_query = vec![0.0f32; dim];

        Ok(Self {
            aligned_pqtable_dist_scratch,
            aligned_dist_scratch,
            aligned_pq_coord_scratch,
            rotated_query,
        })
    }

    /// Copy `query` into `rotated_query`, converting to `f32`.
    ///
    /// `dim` is the element count in the `T` representation. The decompressed
    /// `f32` length returned by `T::as_f32` may differ (e.g. `MinMaxElement`
    /// expands to more `f32`s than its raw element count), so the destination
    /// slice is sized by that actual length.
    ///
    /// Returns `DimensionMismatchError` if `dim > query.len()` or the
    /// decompressed vector does not fit in `rotated_query`.
    pub fn set<T: VectorRepr>(&mut self, dim: usize, query: &[T]) -> ANNResult<()> {
        if dim > query.len() {
            return Err(ANNError::log_dimension_mismatch_error(format!(
                "PQScratch::set: expected query of length >= {dim}, got {}",
                query.len()
            )));
        }
        let query = T::as_f32(&query[..dim]).into_ann_result()?;
        if query.len() > self.rotated_query.len() {
            return Err(ANNError::log_dimension_mismatch_error(format!(
                "PQScratch::set: decompressed query of length {} does not fit rotated_query buffer of length {}",
                query.len(),
                self.rotated_query.len()
            )));
        }
        self.rotated_query[..query.len()].copy_from_slice(&query);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
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

        // Check alignment of the AlignedSlice buffers.
        assert_eq!(
            (pq_scratch.aligned_pqtable_dist_scratch.as_ptr() as usize)
                % PQScratch::ALIGNED_ALLOC_128,
            0
        );
        assert_eq!(
            (pq_scratch.aligned_dist_scratch.as_ptr() as usize) % PQScratch::ALIGNED_ALLOC_128,
            0
        );
        assert_eq!(
            (pq_scratch.aligned_pq_coord_scratch.as_ptr() as usize) % PQScratch::ALIGNED_ALLOC_128,
            0
        );

        // Test set() method
        let query: Vec<u8> = (1..=dim).map(|i| i as u8).collect();
        pq_scratch.set::<u8>(query.len(), &query).unwrap();

        (0..query.len()).for_each(|i| {
            assert_eq!(pq_scratch.rotated_query[i], query[i] as f32);
        });
    }
}

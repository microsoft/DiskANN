/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
//! Aligned allocator

use std::mem::size_of;

use diskann::{ANNResult, error::IntoANNResult, utils::VectorRepr};

use crate::common::AlignedBoxWithSlice;

#[derive(Debug)]
/// PQ scratch
pub struct PQScratch {
    /// Aligned pq table distance scratch, the length must be at least [256 * NCHUNKS]. 256 is the number of PQ centroids.
    /// This is used to store the distance between each chunk in the query vector to each centroid, which is why the length is num of centroids * num of chunks
    pub aligned_pqtable_dist_scratch: AlignedBoxWithSlice<f32>,

    /// Aligned dist scratch, must be at least diskann MAX_DEGREE
    /// This is used to temporarily save the pq distance between query vector to the candidate vectors.
    pub aligned_dist_scratch: AlignedBoxWithSlice<f32>,

    /// Aligned pq coord scratch, must be at least [N_CHUNKS * MAX_DEGREE]
    /// This is used to store the pq coordinates of the candidate vectors.
    pub aligned_pq_coord_scratch: AlignedBoxWithSlice<u8>,

    /// Rotated query. It is initialized as the normalized query vector. Use PQTable.PreprocessQuery to rotate it.
    pub rotated_query: AlignedBoxWithSlice<f32>,

    /// Aligned query float. The query vector is normalized with "norm" and stored here.
    pub aligned_query_float: AlignedBoxWithSlice<f32>,
}

impl PQScratch {
    /// 128 bytes alignment to optimize for the L2 Adjacent Cache Line Prefetcher.
    const ALIGNED_ALLOC_128: usize = 128;

    /// Create a new pq scratch
    pub fn new(
        graph_degree: usize,
        aligned_dim: usize,
        num_pq_chunks: usize,
        num_centers: usize,
    ) -> ANNResult<Self> {
        let aligned_pq_coord_scratch =
            AlignedBoxWithSlice::new(graph_degree * num_pq_chunks, PQScratch::ALIGNED_ALLOC_128)?;
        let aligned_pqtable_dist_scratch =
            AlignedBoxWithSlice::new(num_centers * num_pq_chunks, PQScratch::ALIGNED_ALLOC_128)?;
        let aligned_dist_scratch =
            AlignedBoxWithSlice::new(graph_degree, PQScratch::ALIGNED_ALLOC_128)?;
        let aligned_query_float = AlignedBoxWithSlice::new(aligned_dim, 8 * size_of::<f32>())?;
        let rotated_query = AlignedBoxWithSlice::new(aligned_dim, 8 * size_of::<f32>())?;

        Ok(Self {
            aligned_pqtable_dist_scratch,
            aligned_dist_scratch,
            aligned_pq_coord_scratch,
            rotated_query,
            aligned_query_float,
        })
    }

    /// Set rotated_query and aligned_query_float values
    pub fn set<T>(&mut self, dim: usize, query: &[T], norm: f32) -> ANNResult<()>
    where
        T: VectorRepr + Copy,
    {
        let query = &T::as_f32(&query[..dim]).into_ann_result()?;

        for (d, item) in query.iter().enumerate() {
            let query_val = *item;
            if (norm - 1.0).abs() > f32::EPSILON {
                self.rotated_query[d] = query_val / norm;
                self.aligned_query_float[d] = query_val / norm;
            } else {
                self.rotated_query[d] = query_val;
                self.aligned_query_float[d] = query_val;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::model::PQScratch;

    #[rstest]
    #[case(512, 8, 128, 256)] // default test case
    #[case(59, 16, 37, 41)] // not multiple of 256
    fn test_pq_scratch(
        #[case] graph_degree: usize,
        #[case] aligned_dim: usize,
        #[case] num_pq_chunks: usize,
        #[case] num_centers: usize,
    ) {
        let mut pq_scratch: PQScratch =
            PQScratch::new(graph_degree, aligned_dim, num_pq_chunks, num_centers).unwrap();

        // Check alignment
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
        assert_eq!((pq_scratch.rotated_query.as_ptr() as usize) % 32, 0);
        assert_eq!((pq_scratch.aligned_query_float.as_ptr() as usize) % 32, 0);

        // Test set() method
        let query: Vec<u8> = (1..=aligned_dim).map(|i| i as u8).collect();
        let norm = 2.0f32;
        pq_scratch.set::<u8>(query.len(), &query, norm).unwrap();

        (0..query.len()).for_each(|i| {
            assert_eq!(pq_scratch.rotated_query[i], query[i] as f32 / norm);
            assert_eq!(pq_scratch.aligned_query_float[i], query[i] as f32 / norm);
        });
    }
}

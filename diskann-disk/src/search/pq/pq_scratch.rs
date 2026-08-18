/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
//! Aligned allocator

use diskann::ANNResult;

use diskann_quantization::alloc::{AlignedAllocator, Poly};
use diskann_vector::PreprocessedDistanceFunction;

use crate::error::{diskann_error, ErrorKind};

/// Preprocessed query-to-centroid distances for PQ codes.
#[derive(Debug)]
pub struct PQQueryComputer {
    aligned_pqtable_dist_scratch: Poly<[f32], AlignedAllocator>,
    query_scratch: Vec<f32>,
    num_pq_chunks: usize,
    num_centers: usize,
}

impl PQQueryComputer {
    /// Create an empty query computer for the given PQ schema.
    pub(crate) fn new(dim: usize, num_pq_chunks: usize, num_centers: usize) -> ANNResult<Self> {
        let aligned_pqtable_dist_scratch =
            Poly::broadcast(0f32, num_centers * num_pq_chunks, AlignedAllocator::A128)
                .map_err(|e| diskann_error!(ErrorKind::IndexError, e))?;
        Ok(Self {
            aligned_pqtable_dist_scratch,
            query_scratch: vec![0.0; dim],
            num_pq_chunks,
            num_centers,
        })
    }

    /// Copy a full-precision query into the preprocessing buffer.
    pub(crate) fn set(&mut self, query: &[f32]) -> ANNResult<()> {
        let dim = self.query_scratch.len();
        if query.len() != dim {
            return Err(diskann_error!(
                ErrorKind::DimensionMismatchError,
                "PQQueryComputer::set: expected query of length {dim}, got {}",
                query.len()
            ));
        }
        self.query_scratch.copy_from_slice(query);
        Ok(())
    }

    pub(crate) fn lookup_table(&self) -> &[f32] {
        &self.aligned_pqtable_dist_scratch
    }

    pub(super) fn preprocessing_buffers(&mut self) -> (&[f32], &mut [f32]) {
        (&self.query_scratch, &mut self.aligned_pqtable_dist_scratch)
    }
}

impl PreprocessedDistanceFunction<&[u8], f32> for PQQueryComputer {
    fn evaluate_similarity(&self, code: &[u8]) -> f32 {
        assert_eq!(
            code.len(),
            self.num_pq_chunks,
            "PQ code has the wrong number of chunks",
        );
        code.iter()
            .enumerate()
            .map(|(chunk, &center)| {
                self.aligned_pqtable_dist_scratch[chunk * self.num_centers + center as usize]
            })
            .sum()
    }
}

#[derive(Debug)]
/// PQ scratch
pub struct PQScratch {
    /// Preprocessed query state shared by graph and flat PQ search.
    pub query_computer: PQQueryComputer,

    /// Aligned dist scratch, must be at least diskann MAX_DEGREE
    /// This is used to temporarily save the pq distance between query vector to the candidate vectors.
    pub aligned_dist_scratch: Poly<[f32], AlignedAllocator>,

    /// Aligned pq coord scratch, must be at least [N_CHUNKS * MAX_DEGREE]
    /// This is used to store the pq coordinates of the candidate vectors.
    pub aligned_pq_coord_scratch: Poly<[u8], AlignedAllocator>,
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
                .map_err(|e| diskann_error!(ErrorKind::IndexError, e))?;
        let aligned_dist_scratch = Poly::broadcast(0f32, graph_degree, AlignedAllocator::A128)
            .map_err(|e| diskann_error!(ErrorKind::IndexError, e))?;

        Ok(Self {
            query_computer: PQQueryComputer::new(dim, num_pq_chunks, num_centers)?,
            aligned_dist_scratch,
            aligned_pq_coord_scratch,
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
        self.query_computer.set(query)
    }

    /// Return the largest number of PQ vectors that fit in the batch scratch.
    pub(crate) fn max_vectors(&self) -> usize {
        self.aligned_dist_scratch.len()
    }
}

#[cfg(test)]
mod tests {
    use diskann_quantization::num::PowerOfTwo;
    use diskann_vector::PreprocessedDistanceFunction;
    use rstest::rstest;

    use super::{PQQueryComputer, PQScratch};

    use crate::error::{error_kind, ErrorKind};

    #[test]
    fn query_computer_scores_pq_code() {
        let mut computer = PQQueryComputer::new(2, 2, 3).unwrap();
        computer
            .aligned_pqtable_dist_scratch
            .copy_from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

        assert_eq!(computer.evaluate_similarity(&[1, 2]), 6.0);
    }

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
            (pq_scratch.query_computer.lookup_table().as_ptr() as usize) % PowerOfTwo::V128.raw(),
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
            assert_eq!(pq_scratch.query_computer.query_scratch[i], query[i]);
        });
    }

    #[test]
    fn test_pq_scratch_set_rejects_short_query() {
        let dim = 16;
        let mut pq_scratch = PQScratch::new(64, dim, 4, 256).unwrap();

        // Query shorter than dim should fail
        let short_query: Vec<f32> = (1..dim).map(|i| i as f32).collect(); // dim-1 elements
        let err = pq_scratch.set(&short_query).unwrap_err();
        assert_eq!(error_kind(&err), ErrorKind::DimensionMismatchError);
        assert!(err.to_string().contains("expected query of length"));
    }

    #[test]
    fn test_pq_scratch_set_rejects_oversized_query() {
        let dim = 8;
        let mut pq_scratch = PQScratch::new(64, dim, 4, 256).unwrap();

        // Query longer than dim should fail
        let long_query: Vec<f32> = (1..=dim + 10).map(|i| i as f32).collect();
        let err = pq_scratch.set(&long_query).unwrap_err();
        assert_eq!(error_kind(&err), ErrorKind::DimensionMismatchError);
        assert!(err.to_string().contains("expected query of length"));
    }
}

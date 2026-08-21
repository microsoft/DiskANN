/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
//! Aligned allocator

use diskann::ANNResult;

use diskann_quantization::alloc::{AlignedAllocator, Poly};
use diskann_utils::object_pool::{ObjectPool, PoolOption, TryAsPooled};
use diskann_vector::PreprocessedDistanceFunction;
use std::sync::Arc;

use crate::error::{diskann_error, ErrorKind};

#[derive(Clone, Copy, Debug)]
pub(crate) struct PQQueryComputerArgs {
    dim: usize,
    num_pq_chunks: usize,
    num_centers: usize,
}

impl PQQueryComputerArgs {
    pub(crate) fn new(dim: usize, num_pq_chunks: usize, num_centers: usize) -> Self {
        Self {
            dim,
            num_pq_chunks,
            num_centers,
        }
    }
}

#[derive(Debug)]
pub(crate) struct PQQueryComputerStorage {
    aligned_pqtable_dist_scratch: Poly<[f32], AlignedAllocator>,
    query_scratch: Vec<f32>,
    num_pq_chunks: usize,
    num_centers: usize,
}

impl TryAsPooled<PQQueryComputerArgs> for PQQueryComputerStorage {
    type Error = diskann::ANNError;

    fn try_create(args: PQQueryComputerArgs) -> Result<Self, Self::Error> {
        let aligned_pqtable_dist_scratch = Poly::broadcast(
            0f32,
            args.num_centers * args.num_pq_chunks,
            AlignedAllocator::A128,
        )
        .map_err(|e| diskann_error!(ErrorKind::IndexError, e))?;

        Ok(Self {
            aligned_pqtable_dist_scratch,
            query_scratch: vec![0.0; args.dim],
            num_pq_chunks: args.num_pq_chunks,
            num_centers: args.num_centers,
        })
    }

    fn try_modify(&mut self, args: PQQueryComputerArgs) -> Result<(), Self::Error> {
        if self.query_scratch.len() != args.dim
            || self.num_pq_chunks != args.num_pq_chunks
            || self.num_centers != args.num_centers
        {
            *self = Self::try_create(args)?;
        }
        Ok(())
    }
}

/// Opaque preprocessed query state created internally by disk search strategies.
#[derive(Debug)]
pub struct PQQueryComputer {
    storage: PoolOption<PQQueryComputerStorage>,
}

impl PQQueryComputer {
    /// Create an empty query computer for the given PQ schema.
    #[cfg(test)]
    pub(crate) fn new(dim: usize, num_pq_chunks: usize, num_centers: usize) -> ANNResult<Self> {
        Ok(Self {
            storage: PoolOption::try_non_pooled_create(PQQueryComputerArgs::new(
                dim,
                num_pq_chunks,
                num_centers,
            ))?,
        })
    }

    pub(crate) fn pooled(
        pool: &Arc<ObjectPool<PQQueryComputerStorage>>,
        args: PQQueryComputerArgs,
    ) -> ANNResult<Self> {
        Ok(Self {
            storage: PoolOption::try_pooled(pool, args)?,
        })
    }

    /// Copy a full-precision query into the preprocessing buffer.
    pub(crate) fn set(&mut self, query: &[f32]) -> ANNResult<()> {
        let dim = self.storage.query_scratch.len();
        if query.len() != dim {
            return Err(diskann_error!(
                ErrorKind::DimensionMismatchError,
                "PQQueryComputer::set: expected query of length {dim}, got {}",
                query.len()
            ));
        }
        self.storage.query_scratch.copy_from_slice(query);
        Ok(())
    }

    pub(crate) fn lookup_table(&self) -> &[f32] {
        &self.storage.aligned_pqtable_dist_scratch
    }

    pub(super) fn preprocessing_buffers(&mut self) -> (&[f32], &mut [f32]) {
        let storage = &mut *self.storage;
        (
            &storage.query_scratch,
            &mut storage.aligned_pqtable_dist_scratch,
        )
    }
}

impl PreprocessedDistanceFunction<&[u8], f32> for PQQueryComputer {
    fn evaluate_similarity(&self, code: &[u8]) -> f32 {
        assert_eq!(
            code.len(),
            self.storage.num_pq_chunks,
            "PQ code has the wrong number of chunks",
        );
        code.iter()
            .enumerate()
            .map(|(chunk, &center)| {
                self.storage.aligned_pqtable_dist_scratch
                    [chunk * self.storage.num_centers + center as usize]
            })
            .sum()
    }
}

#[derive(Debug)]
/// PQ scratch
pub struct PQScratch {
    /// Aligned PQ table distance scratch.
    pub aligned_pqtable_dist_scratch: Poly<[f32], AlignedAllocator>,

    /// Aligned dist scratch, must be at least diskann MAX_DEGREE
    /// This is used to temporarily save the pq distance between query vector to the candidate vectors.
    pub aligned_dist_scratch: Poly<[f32], AlignedAllocator>,

    /// Aligned pq coord scratch, must be at least [N_CHUNKS * MAX_DEGREE]
    /// This is used to store the pq coordinates of the candidate vectors.
    pub aligned_pq_coord_scratch: Poly<[u8], AlignedAllocator>,

    /// Query scratch buffer stored as `f32`.
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
                .map_err(|e| diskann_error!(ErrorKind::IndexError, e))?;
        let aligned_dist_scratch = Poly::broadcast(0f32, graph_degree, AlignedAllocator::A128)
            .map_err(|e| diskann_error!(ErrorKind::IndexError, e))?;
        let aligned_pqtable_dist_scratch =
            Poly::broadcast(0f32, num_centers * num_pq_chunks, AlignedAllocator::A128)
                .map_err(|e| diskann_error!(ErrorKind::IndexError, e))?;

        Ok(Self {
            aligned_pqtable_dist_scratch,
            aligned_dist_scratch,
            aligned_pq_coord_scratch,
            query_scratch: vec![0.0; dim],
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
            return Err(diskann_error!(
                ErrorKind::DimensionMismatchError,
                "PQScratch::set: expected query of length {dim}, got {}",
                query.len()
            ));
        }
        self.query_scratch.copy_from_slice(query);
        Ok(())
    }

    /// Return the largest number of PQ vectors that fit in the batch scratch.
    #[cfg(test)]
    pub(crate) fn max_vectors(&self) -> usize {
        self.aligned_dist_scratch.len()
    }
}

#[derive(Debug)]
pub(crate) struct PQBatchScratch {
    pub(crate) aligned_dist_scratch: Poly<[f32], AlignedAllocator>,
    pub(crate) aligned_pq_coord_scratch: Poly<[u8], AlignedAllocator>,
}

impl PQBatchScratch {
    pub(crate) fn new(graph_degree: usize, num_pq_chunks: usize) -> ANNResult<Self> {
        let aligned_pq_coord_scratch =
            Poly::broadcast(0u8, graph_degree * num_pq_chunks, AlignedAllocator::A128)
                .map_err(|e| diskann_error!(ErrorKind::IndexError, e))?;
        let aligned_dist_scratch = Poly::broadcast(0f32, graph_degree, AlignedAllocator::A128)
            .map_err(|e| diskann_error!(ErrorKind::IndexError, e))?;
        Ok(Self {
            aligned_dist_scratch,
            aligned_pq_coord_scratch,
        })
    }

    pub(crate) fn max_vectors(&self) -> usize {
        self.aligned_dist_scratch.len()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use diskann_quantization::num::PowerOfTwo;
    use diskann_utils::object_pool::ObjectPool;
    use diskann_vector::PreprocessedDistanceFunction;
    use rstest::rstest;

    use super::{PQQueryComputer, PQQueryComputerArgs, PQQueryComputerStorage, PQScratch};

    use crate::error::{error_kind, ErrorKind};

    #[test]
    fn query_computer_scores_pq_code() {
        let mut computer = PQQueryComputer::new(2, 2, 3).unwrap();
        computer
            .preprocessing_buffers()
            .1
            .copy_from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

        assert_eq!(computer.evaluate_similarity(&[1, 2]), 6.0);
    }

    #[test]
    fn query_computer_reuses_pooled_storage() {
        let args = PQQueryComputerArgs::new(2, 2, 3);
        let pool = Arc::new(ObjectPool::<PQQueryComputerStorage>::try_new(args, 0, None).unwrap());
        assert!(pool.is_empty());

        let first_ptr;
        {
            let computer = PQQueryComputer::pooled(&pool, args).unwrap();
            assert!(pool.is_empty());
            first_ptr = computer.lookup_table().as_ptr();
        }
        assert_eq!(pool.len(), 1);

        let second_ptr;
        {
            let computer = PQQueryComputer::pooled(&pool, args).unwrap();
            assert!(pool.is_empty());
            second_ptr = computer.lookup_table().as_ptr();
        }
        assert_eq!(pool.len(), 1);

        assert_eq!(first_ptr, second_ptr);
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

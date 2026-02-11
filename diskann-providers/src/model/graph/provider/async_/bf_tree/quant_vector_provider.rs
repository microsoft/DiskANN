/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Bf-Tree quant vector provider.

use std::sync::Arc;

use bf_tree::{BfTree, Config};
use bytemuck::bytes_of;
use diskann::{
    ANNError, ANNErrorKind, ANNResult,
    error::IntoANNResult,
    utils::{VectorRepr, object_pool::ObjectPool},
};
use diskann_quantization::CompressInto;
use diskann_vector::distance::Metric;
use thiserror::Error;

use super::super::common::TestCallCount;
use super::ConfigError;
use crate::{
    model::{
        distance::common::distance_table_pool,
        pq::{self, FixedChunkPQTable},
    },
    utils::BridgeErr,
};

pub struct QuantVectorProvider {
    quant_vector_index: BfTree,
    max_vectors: usize,
    num_start_points: usize,
    pub pq_chunk_table: Arc<FixedChunkPQTable>,
    metric: Metric,
    pub(super) num_get_calls: TestCallCount,

    vec_pool: Arc<ObjectPool<Vec<f32>>>,
}

type DistanceComputer = pq::distance::DistanceComputer<Arc<FixedChunkPQTable>>;
type QueryComputer = pq::distance::QueryComputer<Arc<FixedChunkPQTable>>;

impl QuantVectorProvider {
    pub fn new_with_config(
        dist_metric: Metric,
        max_vectors: usize,
        num_start_points: usize,
        pq_chunk_table: FixedChunkPQTable,
        config: Config,
    ) -> ANNResult<Self> {
        let quant_vector_index = BfTree::with_config(config, None).map_err(ConfigError)?;
        let vec_pool = Arc::new(distance_table_pool(&pq_chunk_table));

        Ok(Self {
            max_vectors,
            num_start_points,
            quant_vector_index,
            pq_chunk_table: Arc::new(pq_chunk_table),
            metric: dist_metric,
            num_get_calls: TestCallCount::default(),
            vec_pool,
        })
    }

    /// Return the metric associated with this provider
    pub(crate) fn metric(&self) -> Metric {
        self.metric
    }

    /// Access the BfTree config
    pub(crate) fn config(&self) -> &Config {
        self.quant_vector_index.config()
    }

    /// Create a snapshot of the quant vector index
    ///
    pub fn snapshot(&self) {
        self.quant_vector_index.snapshot();
    }

    /// Create a new instance from an existing BfTree (for loading from snapshot)
    ///
    pub(crate) fn new_from_bftree(
        dist_metric: Metric,
        max_vectors: usize,
        num_start_points: usize,
        pq_chunk_table: FixedChunkPQTable,
        quant_vector_index: BfTree,
    ) -> Self {
        let vec_pool = Arc::new(distance_table_pool(&pq_chunk_table));
        Self {
            max_vectors,
            num_start_points,
            quant_vector_index,
            pq_chunk_table: Arc::new(pq_chunk_table),
            metric: dist_metric,
            num_get_calls: TestCallCount::default(),
            vec_pool,
        }
    }

    /// Return the total number of points including starting points
    #[inline(always)]
    pub fn total(&self) -> usize {
        self.max_vectors + self.num_start_points
    }

    /// Return the dimension of the full-precision data associated with this provider
    pub fn full_dim(&self) -> usize {
        self.pq_chunk_table.get_dim()
    }

    /// Return the number of PQ chunks in the underlying PQ schema
    pub fn pq_chunks(&self) -> usize {
        self.pq_chunk_table.get_num_chunks()
    }

    /// Create a query computer for the provided query vector
    pub fn query_computer<T>(&self, query: &[T]) -> ANNResult<QueryComputer>
    where
        T: Copy + VectorRepr,
    {
        QueryComputer::new(
            self.pq_chunk_table.clone(),
            self.metric,
            &T::as_f32(query).into_ann_result()?,
            Some(self.vec_pool.clone()),
        )
    }

    /// Create a distance computer for the underlying schema
    pub fn distance_computer(
        &self,
    ) -> Result<DistanceComputer, pq::distance::dynamic::DistanceComputerConstructionError> {
        DistanceComputer::new(self.pq_chunk_table.clone(), self.metric)
    }

    pub(crate) fn get_vector_into(&self, i: usize, buffer: &mut [u8]) -> ANNResult<()> {
        let expected = buffer.len();
        if buffer.len() != expected {
            #[derive(Debug, Error)]
            #[error("expected a buffer with dim {0}, instead got {1}")]
            struct WrongDim(usize, usize);

            return Err(ANNError::new(
                ANNErrorKind::IndexError,
                WrongDim(expected, buffer.len()),
            ));
        }

        self.num_get_calls.increment();
        match self.quant_vector_index.read(bytes_of(&i), buffer) {
            bf_tree::LeafReadResult::Found(read_size) => {
                if read_size as usize != expected {
                    return ANNResult::Err(ANNError::log_index_error(format!(
                        "The bf-tree entry for vector id {} is marked as found but has size {} instead of the expected size {}",
                        i, read_size, expected,
                    )));
                }
            }
            bf_tree::LeafReadResult::Deleted => {
                return ANNResult::Err(ANNError::log_index_error(format!(
                    "The bf-tree entry for vector id {} is marked as deleted",
                    i,
                )));
            }
            bf_tree::LeafReadResult::InvalidKey => {
                return ANNResult::Err(ANNError::log_index_error(format!(
                    "The bf-tree entry for vector id {} is marked as invalid",
                    i,
                )));
            }
            bf_tree::LeafReadResult::NotFound => {
                return ANNResult::Err(ANNError::log_index_error(format!(
                    "The bf-tree entry for vector id {} is marked as not found",
                    i,
                )));
            }
        };

        Ok(())
    }

    /// Return the quant vector at index `i`.
    pub(crate) fn get_vector_sync(&self, i: usize) -> ANNResult<Vec<u8>> {
        let mut value = vec![0u8; self.pq_chunks()];
        self.get_vector_into(i, &mut value)?;
        Ok(value)
    }

    /// Compress the vector, `v`, and set the compressed quant vector with Id, `i`, to it
    ///
    /// Errors if:
    ///
    /// * `i > self.total()`: `i` must be in bounds.
    /// * `v.dim() != self.full_dim()`: The slice must have the proper length.
    /// * PQ compression encounters an error (such as the presence of `NaN`s).
    pub(crate) fn set_vector_sync<T>(&self, i: usize, v: &[T]) -> ANNResult<()>
    where
        T: Copy + VectorRepr,
    {
        if i >= self.total() {
            return Err(ANNError::log_index_error(
                "Vector id is out of boundary in the dataset.",
            ));
        }

        let vf32: &[f32] = &T::as_f32(v).into_ann_result()?;

        if vf32.len() != self.full_dim() {
            return Err(ANNError::log_index_error(
                "Vector f32 dimension is not equal to the expected dimension.",
            ));
        }

        // Serialize the key into a byte string, &[u8]
        let key = bytes_of::<usize>(&i);

        // Quantize the full vector and de-serialize it as byte string
        let dim = self.pq_chunk_table.get_num_chunks();
        let quant_vector = &mut vec![0u8; dim];

        self.pq_chunk_table
            .compress_into(vf32, quant_vector)
            .bridge_err()?;

        self.quant_vector_index.insert(key, quant_vector);

        Ok(())
    }

    /// Set the quant vecotr with Id, `i``, to `v`
    ///
    /// Errors if:
    ///
    /// * `i >= self.total()`: `i` must be in bounds.
    /// * `v.len() != self.pq_chunks()`: `v` must have the right length.
    #[cfg(test)]
    pub(crate) fn set_quant_vector(&self, i: usize, v: &[u8]) -> ANNResult<()> {
        if i >= self.total() {
            return Err(ANNError::log_index_error(
                "Vector id is out of boundary in the dataset.",
            ));
        }
        if v.len() != self.pq_chunks() {
            return Err(ANNError::log_index_error(
                "Vector dimension is not equal to the expected dimension.",
            ));
        }

        // Update pq vector with id = i to v
        let key = bytes_of::<usize>(&i);

        self.quant_vector_index.insert(key, v);

        Ok(())
    }
}

///////////
// Tests //
///////////

/// These unit tests target the functionality of Bf-Tree quant vector provider alone
#[cfg(test)]
mod tests {
    use diskann::ANNErrorKind;
    use diskann_vector::{DistanceFunction, PreprocessedDistanceFunction, distance::Metric};
    use tokio::task::JoinSet;

    use super::*;

    /// Test edges cases of the Bf-Tree quant vector provider
    #[tokio::test]
    async fn common_errors() {
        let dim = 5;
        let centroid = vec![0.0; dim];
        let offsets = vec![0, dim];
        let full_pivot_data = vec![0.0; 256 * dim];

        let pq_chunk_table = FixedChunkPQTable::new(
            dim,
            full_pivot_data.into(),
            centroid.into(),
            offsets.into(),
            None,
        )
        .unwrap();

        let bf_tree_config = Config::default();
        let provider =
            QuantVectorProvider::new_with_config(Metric::L2, 10, 1, pq_chunk_table, bf_tree_config)
                .unwrap();

        // try to set an out of bounds vector
        let result = provider.set_quant_vector(20, &[]).unwrap_err();
        assert_eq!(result.kind(), ANNErrorKind::IndexError);

        // SAFETY: We have exclusive ownership of `provider`
        let result = provider.set_vector_sync::<f32>(20, &[]).unwrap_err();
        assert_eq!(result.kind(), ANNErrorKind::IndexError);

        // try to set a vector with the wrong dimension
        let result = provider.set_quant_vector(0, &[]).unwrap_err();
        assert_eq!(result.kind(), ANNErrorKind::IndexError);
    }

    fn create_test_provider() -> QuantVectorProvider {
        let num_points = 3;
        let frozen_points = 2;
        let dim = 2;

        // We can create a really simple, 1 chunk PQ table with known entries to allow
        // us to easily verify results.
        let table = FixedChunkPQTable::new(
            dim,
            Box::new([0.0, 0.0, 1.0, 1.0, 2.0, 2.0]),
            Box::new([0.0, 0.0]),
            Box::new([0, dim]),
            None,
        )
        .unwrap();

        let bf_tree_config = Config::default();
        let provider = QuantVectorProvider::new_with_config(
            Metric::L2,
            num_points,
            frozen_points,
            table,
            bf_tree_config,
        )
        .unwrap();

        assert_eq!(provider.total(), num_points + frozen_points);
        assert_eq!(provider.full_dim(), dim);

        // Set Vector.
        provider.set_vector_sync(0, &[-1.5, -1.5]).unwrap();
        provider.set_vector_sync(1, &[-0.5, -0.5]).unwrap();
        provider.set_vector_sync(2, &[0.5, 0.5]).unwrap();
        provider.set_vector_sync(3, &[1.5, 1.5]).unwrap();
        provider.set_vector_sync(4, &[2.5, 2.5]).unwrap();
        provider
    }

    /// Test the similarity functions of the provider
    #[tokio::test]
    async fn test_similarity_function() {
        let provider = create_test_provider();

        // Get Vector.
        assert_eq!(provider.get_vector_sync(0).unwrap(), &[0]);
        assert_eq!(provider.get_vector_sync(1).unwrap(), &[0]);
        assert_eq!(provider.get_vector_sync(2).unwrap(), &[0]);
        assert_eq!(provider.get_vector_sync(3).unwrap(), &[1]);
        assert_eq!(provider.get_vector_sync(4).unwrap(), &[2]);

        // Error checking.
        assert!(provider.set_vector_sync(5, &[0.0, 0.0]).is_err());
        assert!(provider.set_vector_sync(2, &[0.0]).is_err());

        // Query Computer.
        let c = provider.query_computer(&[-0.5, -0.5]).unwrap();
        let expected: f32 = 1.5 * 1.5 * 2.0;
        assert_eq!(
            c.evaluate_similarity(&provider.get_vector_sync(3).unwrap()),
            expected
        );

        // Distance Computer.
        let d = provider.distance_computer().unwrap();
        assert_eq!(
            d.evaluate_similarity(
                &provider.get_vector_sync(0).unwrap(),
                &provider.get_vector_sync(3).unwrap()
            ),
            2.0
        );

        let slice: &[f32] = &[-0.5, -0.5];
        assert_eq!(
            d.evaluate_similarity(slice, &provider.get_vector_sync(3).unwrap()),
            expected,
        );
    }

    /// Test the interleaved and parallell traversal of the Bf-Tree
    /// by invoking the async accessors of the quant vector provider
    #[tokio::test(flavor = "multi_thread", worker_threads = 5)]
    async fn test_parallel_tree_traversal() {
        let dim = 2;
        let centroid = vec![0.0; dim];
        let offsets = vec![0, dim];
        let full_pivot_data = vec![0.0; 256 * dim];
        let pq_chunk_table = FixedChunkPQTable::new(
            dim,
            full_pivot_data.into(),
            centroid.into(),
            offsets.into(),
            None,
        )
        .unwrap();

        let bf_tree_config = Config::default();
        let provider = Arc::new(
            QuantVectorProvider::new_with_config(Metric::L2, 10, 1, pq_chunk_table, bf_tree_config)
                .unwrap(),
        );
        let mut set = JoinSet::new();
        for i in 0..11 {
            let vector = vec![i as f32, (i + 1) as f32];
            let provider_clone = Arc::clone(&provider);
            set.spawn(async move {
                // One tokio task per vector insertion
                provider_clone.set_vector_sync(i as usize, &vector).unwrap()
            });
        }

        while let Some(res) = set.join_next().await {
            res.unwrap();
        }

        let dim = provider.pq_chunk_table.get_num_chunks();
        let mut quant_vector: Vec<u8> = vec![0; dim];
        let quant_vector_ref: &mut [u8] = &mut quant_vector;

        for i in 0..11 {
            // SAFETY: We're only accessing one at a time.
            let quant_vector = provider.get_vector_sync(i as usize).unwrap();
            match provider
                .pq_chunk_table
                .compress_into(&[(i as f32), (i + 1) as f32], quant_vector_ref)
            {
                Ok(_) => {}
                Err(e) => {
                    panic!("{}", e)
                }
            };
            assert_eq!(&quant_vector_ref, &quant_vector);
        }
    }
}

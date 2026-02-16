/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Bf-Tree vector provider

use std::marker::PhantomData;

use bf_tree::{BfTree, Config};
use bytemuck::{bytes_of, cast_slice};
use diskann::{
    ANNError, ANNErrorKind, ANNResult,
    utils::{ErrorToVectorId, TryIntoVectorId, VectorId, VectorRepr},
};
use thiserror::Error;

use super::super::common::TestCallCount;
use super::ConfigError;

pub struct VectorProvider<T: VectorRepr, I: VectorId = u32> {
    dim: usize,
    pub max_vectors: usize,
    pub num_start_points: usize,
    vector_index: BfTree,
    pub(super) num_get_calls: TestCallCount,
    _phantom: PhantomData<(T, I)>,
}

impl<T: VectorRepr, I: VectorId> VectorProvider<T, I> {
    /// Create a new instance based on bf-tree Config directly
    pub fn new_with_config(
        max_vectors: usize,
        dim: usize,
        num_start_points: usize,
        config: Config,
    ) -> ANNResult<Self> {
        let vector_index = BfTree::with_config(config, None).map_err(ConfigError)?;

        Ok(Self {
            dim,
            max_vectors,
            num_start_points,
            vector_index,
            num_get_calls: TestCallCount::default(),
            _phantom: PhantomData,
        })
    }

    /// Create a new instance from an existing BfTree (for loading from snapshot)
    ///
    #[inline(always)]
    pub fn new_from_bftree(
        max_vectors: usize,
        dim: usize,
        num_start_points: usize,
        vector_index: BfTree,
    ) -> Self {
        Self {
            dim,
            max_vectors,
            num_start_points,
            vector_index,
            num_get_calls: TestCallCount::default(),
            _phantom: PhantomData,
        }
    }

    /// Return the total number of points including starting points
    ///
    #[inline(always)]
    pub fn total(&self) -> usize {
        self.max_vectors + self.num_start_points
    }

    // /// Return vector Id range for the starting points
    // ///
    // #[inline(always)]
    // pub fn start_point_range(&self) -> std::ops::Range<usize> {
    //     self.max_vectors..self.total()
    // }

    /// Return the vector dimension
    ///
    #[inline(always)]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Return a vector of vector Ids of the starting points
    ///
    #[inline(always)]
    pub fn starting_points(&self) -> Result<Vec<I>, ErrorToVectorId<usize, I>> {
        (self.max_vectors..self.total())
            .map(|i| i.try_into_vector_id())
            .collect()
    }

    /// Access the BfTree config
    pub(crate) fn config(&self) -> &Config {
        self.vector_index.config()
    }

    /// Create a snapshot of the vector index
    ///
    #[inline(always)]
    pub fn snapshot(&self) {
        self.vector_index.snapshot();
    }

    /// Set vector with Id, `i``, to `v`
    ///
    /// Errors if:
    ///
    /// * `i > self.total()`: `i` must be in bounds.
    /// * `v.dim() != self.dim()`: The slice must have the proper length
    #[inline(always)]
    pub(crate) fn set_vector_sync(&self, i: usize, v: &[T]) -> ANNResult<()> {
        if v.len() != self.dim {
            return Err(ANNError::log_index_error(
                "Vector dimension is not equal to the expected dimension.",
            ));
        }
        if i >= self.total() {
            return Err(ANNError::log_index_error(
                "Vector id is out of boundary in the dataset.",
            ));
        }

        // Serialize the key, vector_id, into a byte string, &[u8]
        let key = bytes_of::<usize>(&i);
        let value = cast_slice::<T, u8>(v);

        self.vector_index.insert(key, value);

        Ok(())
    }

    pub(crate) fn get_vector_into(&self, i: usize, buffer: &mut [T]) -> ANNResult<()> {
        if buffer.len() != self.dim {
            #[derive(Debug, Error)]
            #[error("expected a buffer with dim {0}, instead got {1}")]
            struct WrongDim(usize, usize);

            return Err(ANNError::new(
                ANNErrorKind::IndexError,
                WrongDim(self.dim(), buffer.len()),
            ));
        }

        self.num_get_calls.increment();
        match self
            .vector_index
            .read(bytes_of(&i), bytemuck::must_cast_slice_mut::<_, u8>(buffer))
        {
            bf_tree::LeafReadResult::Found(read_size) => {
                let vector_size = std::mem::size_of::<T>() * self.dim;
                if read_size as usize != vector_size {
                    return Err(ANNError::log_index_error(format!(
                        "The bf-tree entry for vector id {} is marked as found but has size {} instead of the expected size {}",
                        i, read_size, vector_size,
                    )));
                }
            }
            bf_tree::LeafReadResult::Deleted => {
                return Err(ANNError::log_index_error(format!(
                    "The bf-tree entry for vector id {} is marked as deleted",
                    i
                )));
            }
            bf_tree::LeafReadResult::InvalidKey => {
                return Err(ANNError::log_index_error(format!(
                    "The bf-tree entry for vector id {} is marked as invalid",
                    i
                )));
            }
            bf_tree::LeafReadResult::NotFound => {
                return Err(ANNError::log_index_error(format!(
                    "The bf-tree entry for vector id {} is marked as not found",
                    i
                )));
            }
        };

        Ok(())
    }

    /// Return the vector at index `i`
    #[inline(always)]
    pub(crate) fn get_vector_sync(&self, i: usize) -> ANNResult<Vec<T>> {
        // Search for the corresponding vector
        let mut vector = vec![T::default(); self.dim];
        self.get_vector_into(i, &mut vector)?;
        Ok(vector)
    }
}

///////////
// Tests //
///////////

/// These unit tests target the functionality of Bf-Tree vector provider alone
///
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use diskann::utils::vecid_from_usize;
    use tokio::task::JoinSet;

    use super::*;

    /// Test the interleaved and parallell traversal of the Bf-Tree
    /// by invoking the async accessors of the vector provider
    ///
    #[tokio::test(flavor = "multi_thread", worker_threads = 5)]
    async fn test_parallel_tree_traversal() {
        let num_points = 100;
        let bf_tree_config = Config::default();
        let vector_provider = Arc::new(
            VectorProvider::<f32>::new_with_config(num_points, 3, 2, bf_tree_config).unwrap(),
        );
        let mut set = JoinSet::new();
        for i in 0..num_points {
            let vector = vec![i as f32, (i + 1) as f32, (i + 2) as f32];
            let vector_provider_clone = Arc::clone(&vector_provider);
            set.spawn(async move {
                // One tokio task per vector insertion
                vector_provider_clone
                    .set_vector_sync(vecid_from_usize(i).unwrap(), &vector)
                    .unwrap()
            });
        }

        while let Some(res) = set.join_next().await {
            res.unwrap();
        }

        for i in 0..num_points {
            // SAFETY: We're only accessing one at a time.
            let vector = vector_provider
                .get_vector_sync(vecid_from_usize(i).unwrap())
                .unwrap();
            assert_eq!(&vector, &vec![(i as f32), (i + 1) as f32, (i + 2) as f32]);
        }
        assert_eq!(vector_provider.num_get_calls.get(), num_points);
    }

    /// Test other methods and edge cases of the vector provider and sycrhnoization mechanism of Bf-Tree
    ///
    #[tokio::test(flavor = "multi_thread", worker_threads = 5)]
    async fn test_parallel_vector_access() {
        let num_points = 3;
        let frozen_points = 2;
        let dim = 3;
        let bf_tree_config = Config::default();

        let provider = Arc::new(
            VectorProvider::<f32>::new_with_config(num_points, dim, frozen_points, bf_tree_config)
                .unwrap(),
        );

        let mut set = JoinSet::new();
        for _ in 0..5 {
            let provider_ref = Arc::clone(&provider);
            set.spawn(async move {
                provider_ref.set_vector_sync(0, &[0.0, 0.0, 0.0]).unwrap();
                provider_ref.set_vector_sync(1, &[1.0, 1.0, 1.0]).unwrap();
                provider_ref.set_vector_sync(2, &[2.0, 2.0, 2.0]).unwrap();
                provider_ref.set_vector_sync(3, &[3.0, 3.0, 3.0]).unwrap();
                provider_ref.set_vector_sync(4, &[4.0, 4.0, 4.0]).unwrap();

                assert_eq!(provider_ref.get_vector_sync(4).unwrap(), &[4.0, 4.0, 4.0]);
                assert_eq!(provider_ref.get_vector_sync(3).unwrap(), &[3.0, 3.0, 3.0]);
                assert_eq!(provider_ref.get_vector_sync(2).unwrap(), &[2.0, 2.0, 2.0]);
                assert_eq!(provider_ref.get_vector_sync(1).unwrap(), &[1.0, 1.0, 1.0]);
                assert_eq!(provider_ref.get_vector_sync(0).unwrap(), &[0.0, 0.0, 0.0]);

                // Error checking.
                assert!(provider_ref.set_vector_sync(5, &[0.0, 0.0, 0.0]).is_err());
                assert!(provider_ref.set_vector_sync(2, &[0.0, 0.0]).is_err());
            });
        }

        while let Some(res) = set.join_next().await {
            res.unwrap();
        }
    }

    /// Test new_from_bftree constructor
    #[tokio::test]
    async fn test_new_from_bftree() {
        let bftree = BfTree::with_config(Config::default(), None).expect("Failed to create BfTree");
        let provider = VectorProvider::<f32>::new_from_bftree(100, 3, 2, bftree);

        // Verify fields are set correctly
        assert_eq!(provider.dim(), 3);
        assert_eq!(provider.max_vectors, 100);
        assert_eq!(provider.num_start_points, 2);
        assert_eq!(provider.total(), 102);

        // Verify the provider is functional
        provider.set_vector_sync(0, &[1.0, 2.0, 3.0]).unwrap();
        let result = provider.get_vector_sync(0).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }
}

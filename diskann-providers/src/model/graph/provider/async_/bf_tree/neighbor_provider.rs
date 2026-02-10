/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Bf-Tree neighbor list provider.

use std::marker::PhantomData;

use bf_tree::{BfTree, Config};
use bytemuck::{bytes_of, cast_slice, cast_slice_mut};
use diskann::{
    ANNError, ANNResult,
    graph::AdjacencyList,
    provider::HasId,
    utils::{IntoUsize, TryIntoVectorId, VectorId},
};

use super::super::common::TestCallCount;
use super::ConfigError;

pub struct NeighborProvider<I: VectorId> {
    adjacency_list_index: BfTree,
    dim: usize, // Max number of neighbors in a neighbor list + 1 for the neighbor count
    pub num_get_calls: TestCallCount,
    _phantom: PhantomData<I>,
}

impl<I: VectorId> HasId for NeighborProvider<I> {
    type Id = I;
}

impl<I: VectorId> NeighborProvider<I> {
    /// Create a new instance based on bf-tree Config directly
    pub fn new_with_config(max_degree: u32, config: Config) -> ANNResult<Self> {
        let adj_list_index = BfTree::with_config(config, None).map_err(ConfigError)?;

        Ok(Self::new(max_degree, adj_list_index))
    }

    fn new(max_degree: u32, adjacency_list_index: BfTree) -> Self {
        Self {
            adjacency_list_index,
            dim: 1 + max_degree.into_usize(),
            num_get_calls: TestCallCount::default(),
            _phantom: PhantomData,
        }
    }

    /// Access the BfTree config
    pub(crate) fn config(&self) -> &Config {
        self.adjacency_list_index.config()
    }

    /// Create a snapshot of the adjacency list index
    ///
    pub fn snapshot(&self) {
        self.adjacency_list_index.snapshot();
    }

    /// Return the maximum degree (number of neighbors per vector)
    ///
    pub fn max_degree(&self) -> u32 {
        (self.dim - 1) as u32
    }

    /// Create a new instance from an existing BfTree (for loading from snapshot)
    ///
    pub(crate) fn new_from_bftree(max_degree: u32, adjacency_list_index: BfTree) -> Self {
        Self::new(max_degree, adjacency_list_index)
    }

    /// Retrieve the neighbor list of a vector
    /// `neighbors` is cleared first upon each invocation
    /// One data copy is involved which copies the data from bf-tree to `neighbors`
    pub fn get_neighbors(&self, vector_id: I, neighbors: &mut AdjacencyList<I>) -> ANNResult<()> {
        #[cfg(test)]
        self.num_get_calls.increment();

        // Resize 'neighbors' to hold any full-size neighbor list
        let mut guard = neighbors.resize(self.dim);

        // Serialize the key, vector_id, into a byte string, &[u8]
        let i = vector_id.into_usize();
        let key = bytes_of::<usize>(&i);

        // Search and retrieve the corresponding neighbor list data as a byte string, &[u8], in the format of
        // |VectorId|VectorId|...|Invalid|Invalid|VectorId (list length)|
        // Where list length is the full list length and 'Invalid' indicates unfilled empty slots in the list
        let value = cast_slice_mut::<I, u8>(&mut guard);
        match self.adjacency_list_index.read(key, value) {
            bf_tree::LeafReadResult::Found(read_size) => {
                // If found, then re-construct the neighbor list (valid neighbors only)
                if read_size > 0 {
                    // A non-empty neighbor list should at least contain one entry for the list length
                    if (read_size as usize) < std::mem::size_of::<I>() {
                        return Err(ANNError::log_index_error(
                            "Retrieved neighbor list is shorter than a single VectorID",
                        ));
                    }

                    // A retrieved neighbor list should not be longer than the max degree
                    if (read_size as usize) > (std::mem::size_of::<I>() * self.dim) {
                        return Err(ANNError::log_index_error(
                            "Retrieved neighbor list is longer than the max degree",
                        ));
                    }

                    // Retrieved data length must be in the multiple of VectorID
                    if !(read_size as usize).is_multiple_of(std::mem::size_of::<I>()) {
                        return Err(ANNError::log_index_error(
                            "Retrieved neighbor list length is not in the multiple of VectorID",
                        ));
                    }

                    // The last entry in the retrieved data is neighbor length
                    let nbr_count =
                        guard[(read_size as usize) / std::mem::size_of::<I>() - 1].into_usize();

                    // The specified list length must be smaller than the retrieved data length
                    if (read_size as usize) < (std::mem::size_of::<I>() * (nbr_count + 1)) {
                        return Err(ANNError::log_index_error(
                            "The length of the retrieved neighbor list is shorter than the specified length",
                        ));
                    }

                    guard.finish(nbr_count);
                }
            }
            bf_tree::LeafReadResult::Deleted => {
                return Err(ANNError::log_index_error(
                    "The bf-tree entry for the vector is marked as deleted",
                ));
            }
            bf_tree::LeafReadResult::InvalidKey => {
                return Err(ANNError::log_index_error(
                    "The bf-tree entry for the vector key is marked as invalid",
                ));
            }
            bf_tree::LeafReadResult::NotFound => {
                return Err(ANNError::log_index_error(
                    "The bf-tree entry for the vector key is marked as not found",
                ));
            }
        };

        Ok(())
    }

    /// Insert a neighbor list of a vector in bf-tree as a (K, V) pair
    /// K: vector id
    /// V: |VectorId|VectorId|...|Invalid|Invalid|VectorId (list length)|
    /// Where list length is the full list length and 'Invalid' indicates unfilled empty slots in the list
    /// Note: assuming all neighbors in the input list, 'neighbors', are valid
    /// Two data copies are involved: 1) Copy from the immutable `neighbors` to the proper byte array with neighbor length
    /// 2) Copy from the byte array to bf-tree
    #[allow(clippy::expect_used)]
    pub fn set_neighbors(&self, vector_id: I, neighbors: &[I]) -> ANNResult<()> {
        #[cfg(test)]
        self.num_get_calls.increment();

        if neighbors.len() > self.dim - 1 {
            return Err(ANNError::log_index_error(
                "The provided neighbor list is longer than the max degree",
            ));
        }

        // Serialize the key, vector_id, into a byte string, &[u8]
        let i = vector_id.into_usize();
        let key = bytes_of::<usize>(&i);

        // Serialize the value, neighbor list, into a byte string, &u[8]
        let neighbor_list_edges_in_byte = cast_slice::<I, u8>(neighbors);

        let neighbor_list_len = neighbors
            .len()
            .try_into_vector_id()
            .expect("Fail to convert #neighbors as neighbor vec Id");
        let neighbor_list_len_in_byte = bytes_of::<I>(&neighbor_list_len);

        // Format
        // |VectorId|...|VectorId|VectorId (list length)|
        let value: &[u8] = &[neighbor_list_edges_in_byte, neighbor_list_len_in_byte].concat();

        // Insert the assembled (K, V) pair into bf-tree
        self.adjacency_list_index.insert(key, value);

        Ok(())
    }

    /// Append unique vectors into a neighbor list
    /// The newly appended neighbor list will always be extended to 'dim' long to avoid frequent mem copy in bf-tree
    /// Note: assuming all neighbors in the input list, 'new_neighbor_ids', are valid
    /// Three data copies: 1) get_neighbors 2) copy new neighbors to the neighbor list 3) copy the new neighbor list to bf-tree
    #[allow(clippy::expect_used)]
    pub fn append_vector(&self, vector_id: I, new_neighbor_ids: &[I]) -> ANNResult<()> {
        // Retrieve existing neighborlist
        let mut neighbor_list = AdjacencyList::with_capacity(self.dim);
        self.get_neighbors(vector_id, &mut neighbor_list)?;

        // Append the new neighbors
        let mut new_neighbor_added = false;
        for new_neighbor_id in new_neighbor_ids {
            if neighbor_list.len() == self.dim - 1 {
                break;
            }
            new_neighbor_added |= neighbor_list.push(*new_neighbor_id);
        }

        // If unique new neighbors are appended, then upsert the new neighbor list back into the tree
        if new_neighbor_added {
            let nbr_count = neighbor_list.len();
            let mut neighbor_list: Vec<_> = neighbor_list.into();
            neighbor_list.resize(self.dim, I::default());
            neighbor_list[self.dim - 1] =
                I::from_usize(nbr_count).expect("Fails to cast usize as VectorId");

            // Given that we already have a full sized neighbor list ready to be directly saved in bf-tree
            // We avoid one data copy by directly writing to bf-tree instead of invoking set_neighbor()
            // Also avoid a bunch of unnecssary checks
            let i = vector_id.into_usize();
            let key = bytes_of::<usize>(&i);
            let value = cast_slice::<I, u8>(&neighbor_list);
            self.adjacency_list_index.insert(key, value);
        }

        Ok(())
    }

    pub fn delete_vector(&self, vector_id: I) -> ANNResult<()> {
        // Serialize the key, vector_id, into a byte string, &[u8]
        let i = vector_id.into_usize();
        let key = bytes_of::<usize>(&i);

        self.adjacency_list_index.delete(key);
        Ok(())
    }
}

///////////
// Tests //
///////////

/// These unit tests target the functionality of Bf-Tree neighbor list provider alone
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio::task::JoinSet;

    use super::*;

    /// Test corner cases of appending to neighbor list
    #[tokio::test]
    async fn test_neighbor_accessors() {
        let bf_tree_config = Config::default();
        let neighbor_provider =
            NeighborProvider::<u32>::new_with_config(6, bf_tree_config).unwrap();

        // Set the neighbor list of a vector
        let adj_list = vec![1, 2, 3];
        neighbor_provider.set_neighbors(1, &adj_list).unwrap();

        let mut result = AdjacencyList::with_capacity(10);
        neighbor_provider.get_neighbors(1, &mut result).unwrap();
        assert_eq!(&*adj_list, &*result);

        // Append two neighbors, one of which is a duplicate
        let mut new_neighbors = vec![9, 2, 9];
        neighbor_provider.append_vector(1, &new_neighbors).unwrap();

        neighbor_provider.get_neighbors(1, &mut result).unwrap();

        let mut adj_list_new = vec![1, 2, 3, 9];
        assert_eq!(&*adj_list_new, &*result);

        // Append three more neighbors, and the last one should be ignored due to max degree
        new_neighbors = vec![5, 6, 7];
        neighbor_provider.append_vector(1, &new_neighbors).unwrap();

        neighbor_provider.get_neighbors(1, &mut result).unwrap();

        adj_list_new = vec![1, 2, 3, 9, 5, 6];
        assert_eq!(&*adj_list_new, &*result);

        // Overwrite the neighbor list of the vector to empty
        new_neighbors = vec![];
        neighbor_provider.set_neighbors(1, &new_neighbors).unwrap();
        neighbor_provider.get_neighbors(1, &mut result).unwrap();

        assert_eq!(&*new_neighbors, &*result);

        // Append to an emptied neighbor list
        new_neighbors = vec![3, 4, 5];
        neighbor_provider.append_vector(1, &new_neighbors).unwrap();

        neighbor_provider.get_neighbors(1, &mut result).unwrap();

        assert_eq!(&*new_neighbors, &*result);

        neighbor_provider.delete_vector(1).unwrap();

        assert!(neighbor_provider.get_neighbors(1, &mut result).is_err());

        new_neighbors = vec![];
        assert_eq!(&*new_neighbors, &*result);
    }

    /// Test the interleaved and parallell traversal of the Bf-Tree
    /// by invoking the async accessors of the neighbor list provider
    #[tokio::test(flavor = "multi_thread", worker_threads = 5)]
    async fn test_parallel_tree_traversal() {
        let bf_tree_config = Config::default();
        let neighbor_provider =
            Arc::new(NeighborProvider::<u32>::new_with_config(120, bf_tree_config).unwrap());

        let mut set = JoinSet::new();
        for i in 0..100 {
            let neighbor_list = vec![i as u32, (i + 1) as u32, (i + 2) as u32];
            let neighbor_provider_clone = Arc::clone(&neighbor_provider);
            set.spawn(async move {
                // One tokio task per neighbor list insertion
                neighbor_provider_clone
                    .set_neighbors(i as u32, &neighbor_list)
                    .unwrap()
            });
        }

        while let Some(res) = set.join_next().await {
            res.unwrap();
        }

        let mut result = AdjacencyList::with_capacity(neighbor_provider.dim);
        for i in 0..100 {
            // SAFETY: We're only accessing one at a time.
            neighbor_provider
                .get_neighbors(i as u32, &mut result)
                .unwrap();

            let neighbor_list = vec![i as u32, (i + 1) as u32, (i + 2) as u32];
            assert_eq!(&*neighbor_list, &*result);
        }
    }

    /// Test that snapshot can be called without errors
    #[tokio::test]
    async fn test_snapshot() {
        // Use a temporary directory that gets cleaned up when dropped
        let temp_dir = tempfile::tempdir().unwrap();
        let snapshot_path = temp_dir.path().join("test_neighbor_snapshot.bftree");

        let mut bf_tree_config = Config::new(&snapshot_path, 16384 * 16);
        bf_tree_config.storage_backend(bf_tree::StorageBackend::Std);

        let neighbor_provider =
            NeighborProvider::<u32>::new_with_config(6, bf_tree_config).unwrap();

        // Set some neighbor lists
        neighbor_provider.set_neighbors(1, &[2, 3, 4]).unwrap();
        neighbor_provider.set_neighbors(2, &[1, 3, 5]).unwrap();

        // Call snapshot - should not panic
        neighbor_provider.snapshot();

        // Verify data is still accessible after snapshot
        let mut result = AdjacencyList::with_capacity(10);
        neighbor_provider.get_neighbors(1, &mut result).unwrap();
        assert_eq!(&[2, 3, 4], &*result);

        neighbor_provider.get_neighbors(2, &mut result).unwrap();
        assert_eq!(&[1, 3, 5], &*result);

        // temp_dir is automatically cleaned up when it goes out of scope
    }

    /// Test that max_degree returns the correct value
    #[tokio::test]
    async fn test_max_degree() {
        let bf_tree_config = Config::default();

        // Test with various max_degree values
        let neighbor_provider =
            NeighborProvider::<u32>::new_with_config(6, bf_tree_config.clone()).unwrap();
        assert_eq!(neighbor_provider.max_degree(), 6);

        let neighbor_provider =
            NeighborProvider::<u32>::new_with_config(120, bf_tree_config.clone()).unwrap();
        assert_eq!(neighbor_provider.max_degree(), 120);

        let neighbor_provider =
            NeighborProvider::<u32>::new_with_config(1, bf_tree_config).unwrap();
        assert_eq!(neighbor_provider.max_degree(), 1);
    }

    /// Test new_from_bftree constructor
    #[tokio::test]
    async fn test_new_from_bftree() {
        let bftree = BfTree::with_config(Config::default(), None).expect("Failed to create BfTree");
        let neighbor_provider = NeighborProvider::<u32>::new_from_bftree(10, bftree);

        assert_eq!(neighbor_provider.max_degree(), 10);

        // Verify the provider is functional
        neighbor_provider.set_neighbors(1, &[2, 3]).unwrap();
        let mut result = AdjacencyList::with_capacity(11);
        neighbor_provider.get_neighbors(1, &mut result).unwrap();
        assert_eq!(&[2, 3], &*result);
    }

    /// Test other methods and edge cases of the vector provider and sycrhnoization mechanism of Bf-Tree
    #[tokio::test(flavor = "multi_thread", worker_threads = 5)]
    async fn test_parallel_neighbor_access() {
        let bf_tree_config = Config::default();
        let neighbor_provider =
            Arc::new(NeighborProvider::<u32>::new_with_config(120, bf_tree_config).unwrap());

        let mut set = JoinSet::new();
        for _ in 0..5 {
            let neighbor_provider_clone = Arc::clone(&neighbor_provider);
            set.spawn(async move {
                for i in 0..5 {
                    neighbor_provider_clone
                        .set_neighbors(i as u32, &[1, 2, 3, 4, 5])
                        .unwrap();
                }

                let mut result = AdjacencyList::with_capacity(neighbor_provider_clone.dim);
                for i in 0..5 {
                    neighbor_provider_clone
                        .get_neighbors(i as u32, &mut result)
                        .unwrap();

                    assert_eq!(&[1, 2, 3, 4, 5], &*result);
                }
            });
        }

        while let Some(res) = set.join_next().await {
            res.unwrap();
        }
    }
}

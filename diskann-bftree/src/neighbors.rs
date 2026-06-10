/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Bf-Tree neighbor list provider.

use std::marker::PhantomData;

use crate::AsKey;
use bf_tree::{BfTree, Config};
use bytemuck::{cast_slice, cast_slice_mut};
use diskann::{
    graph::AdjacencyList,
    provider::{self, HasId},
    utils::{IntoUsize, VectorId},
    ANNError, ANNResult,
};

use super::ConfigError;
use crate::TestCallCount;

pub struct NeighborProvider<I: VectorId + AsKey> {
    adjacency_list_index: BfTree,
    dim: usize, // Max number of neighbors in a neighbor list + 1 for the neighbor count
    #[allow(dead_code)]
    pub(crate) num_get_calls: TestCallCount,
    _phantom: PhantomData<I>,
}

impl<I: VectorId + AsKey> HasId for NeighborProvider<I> {
    type Id = I;
}

impl<I: VectorId + AsKey> NeighborProvider<I> {
    /// Create a new instance based on bf-tree Config directly
    pub fn new_with_config(max_degree: u32, config: Config) -> ANNResult<Self> {
        let adj_list_index = BfTree::with_config(config, None).map_err(ConfigError)?;

        Self::new(max_degree, adj_list_index)
    }

    fn new(max_degree: u32, adjacency_list_index: BfTree) -> ANNResult<Self> {
        let dim = 1 + max_degree.into_usize();

        Ok(Self {
            adjacency_list_index,
            dim,
            num_get_calls: TestCallCount::default(),
            _phantom: PhantomData,
        })
    }

    /// Access the BfTree config
    pub(crate) fn config(&self) -> &Config {
        self.adjacency_list_index.config()
    }

    /// Access the underlying BfTree
    pub(crate) fn bftree(&self) -> &BfTree {
        &self.adjacency_list_index
    }

    /// Return the maximum degree (number of neighbors per vector)
    ///
    pub fn max_degree(&self) -> u32 {
        (self.dim - 1) as u32
    }

    /// Create a new instance from an existing BfTree (for loading from snapshot)
    ///
    pub(crate) fn new_from_bftree(
        max_degree: u32,
        adjacency_list_index: BfTree,
    ) -> ANNResult<Self> {
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
        let key = vector_id.as_key();

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

                    // The last entry in the retrieved data is the neighbor count,
                    // encoded as a little-endian u32 in the low 4 bytes (see `len_cell`).
                    let nbr_count =
                        read_len(&guard[(read_size as usize) / std::mem::size_of::<I>() - 1]);

                    // The specified list length must be smaller than the retrieved data length
                    if (read_size as usize) < (std::mem::size_of::<I>() * (nbr_count as usize + 1))
                    {
                        return Err(ANNError::log_index_error(
                            "The length of the retrieved neighbor list is shorter than the specified length",
                        ));
                    }

                    guard.finish(nbr_count as usize);
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
    /// V: |VectorId|...|VectorId|Invalid|...|count (u32 LE)|
    /// Where count is the neighbor list length and 'Invalid' indicates unfilled empty slots
    /// Note: assuming all neighbors in the input list, 'neighbors', are valid
    pub fn set_neighbors(&self, vector_id: I, neighbors: &[I], buf: &mut [u8]) -> ANNResult<()> {
        #[cfg(test)]
        self.num_get_calls.increment();

        if neighbors.len() > self.dim - 1 {
            return Err(ANNError::log_index_error(
                "The provided neighbor list is longer than the max degree",
            ));
        }

        // Serialize the key, vector_id, into a byte string, &[u8]
        let key = vector_id.as_key();

        // Serialize the value into the reusable buffer.
        // Format: |VectorId|...|VectorId|length cell|, where the trailing cell is one
        // `I`-sized slot whose low 4 bytes hold the count as a little-endian u32 (see
        // `len_cell`).
        let neighbor_bytes = cast_slice::<I, u8>(neighbors);
        let total_len = neighbor_bytes.len() + std::mem::size_of::<I>();

        buf[..neighbor_bytes.len()].copy_from_slice(neighbor_bytes);
        let cell: I = len_cell(neighbors.len() as u32);
        buf[neighbor_bytes.len()..total_len].copy_from_slice(bytemuck::bytes_of(&cell));

        self.adjacency_list_index.insert(key, &buf[..total_len]);

        Ok(())
    }

    /// Append unique vectors into a neighbor list
    /// The newly appended neighbor list will always be extended to 'dim' long to avoid frequent mem copy in bf-tree
    /// Note: assuming all neighbors in the input list, 'new_neighbor_ids', are valid
    pub fn append_vector(
        &self,
        vector_id: I,
        new_neighbor_ids: &[I],
        buf: &mut [u8],
    ) -> ANNResult<()> {
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

        // If unique new neighbors are appended, write back using the reusable buffer
        if new_neighbor_added {
            let nbr_count = neighbor_list.len();
            let key = vector_id.as_key();

            // Build the value into the scratch buffer:
            // |neighbor_0|...|neighbor_n|padding(I::default)|...|length cell|
            // Total size is always self.dim elements to avoid bf-tree page fragmentation.
            // The trailing cell is a length cell (see `len_cell`); intermediate padding
            // slots are zeroed.
            let id_size = std::mem::size_of::<I>();
            let total_len = self.dim * id_size;

            buf[..total_len].fill(0);

            // Copy existing neighbors into the buffer
            let neighbors_bytes = cast_slice::<I, u8>(&neighbor_list);
            buf[..neighbors_bytes.len()].copy_from_slice(neighbors_bytes);

            // Write the length cell at the last slot
            let cell: I = len_cell(nbr_count as u32);
            let count_offset = (self.dim - 1) * id_size;
            buf[count_offset..count_offset + id_size].copy_from_slice(bytemuck::bytes_of(&cell));

            self.adjacency_list_index.insert(key, &buf[..total_len]);
        }

        Ok(())
    }

    pub fn delete_vector(&self, vector_id: I) -> ANNResult<()> {
        // Serialize the key, vector_id, into a byte string, &[u8]
        let key = vector_id.as_key();

        self.adjacency_list_index.delete(key);
        Ok(())
    }

    pub(crate) fn scratch(&self) -> NeighborAccessor<'_, I> {
        let buf_size = self.dim * std::mem::size_of::<I>();
        NeighborAccessor {
            provider: self,
            buf: vec![0u8; buf_size],
        }
    }
}

pub struct NeighborAccessor<'a, I>
where
    I: VectorId + AsKey,
{
    provider: &'a NeighborProvider<I>,
    buf: Vec<u8>,
}

impl<'a, I> NeighborAccessor<'a, I>
where
    I: VectorId + AsKey,
{
    pub fn write_neighbors(&mut self, id: I, neighbors: &[I]) -> ANNResult<()> {
        self.provider.set_neighbors(id, neighbors, &mut self.buf)
    }
    pub fn write_append(&mut self, id: I, neighbors: &[I]) -> ANNResult<()> {
        self.provider.append_vector(id, neighbors, &mut self.buf)
    }
}

impl<'a, I> HasId for NeighborAccessor<'a, I>
where
    I: VectorId + AsKey,
{
    type Id = I;
}

impl<'a, I> provider::NeighborAccessor for NeighborAccessor<'a, I>
where
    I: VectorId + AsKey,
{
    fn get_neighbors(
        self,
        id: Self::Id,
        neighbors: &mut AdjacencyList<Self::Id>,
    ) -> impl std::future::Future<Output = ANNResult<Self>> + Send {
        let result = self.provider.get_neighbors(id, neighbors);
        std::future::ready(result.map(|()| self))
    }
}

impl<'a, I> provider::NeighborAccessorMut for NeighborAccessor<'a, I>
where
    I: VectorId + AsKey,
{
    fn set_neighbors(
        mut self,
        id: Self::Id,
        neighbors: &[Self::Id],
    ) -> impl std::future::Future<Output = ANNResult<Self>> + Send {
        let result = self.provider.set_neighbors(id, neighbors, &mut self.buf);
        std::future::ready(result.map(|()| self))
    }
    fn append_vector(
        mut self,
        id: Self::Id,
        neighbors: &[Self::Id],
    ) -> impl std::future::Future<Output = ANNResult<Self>> + Send {
        let result = self.provider.append_vector(id, neighbors, &mut self.buf);
        std::future::ready(result.map(|()| self))
    }
}

/// Construct an `I`-typed cell whose low 4 bytes hold `len` as a little-endian `u32`.
///
/// This is the encoding used for the trailing length cell in the neighbor-list value;
/// see [`NeighborProvider::set_neighbors`] for the full layout. The decode side is
/// [`read_len`].
fn len_cell<I: bytemuck::Pod>(len: u32) -> I {
    const { assert!(std::mem::size_of::<I>() >= 4) };

    let mut cell = I::zeroed();
    bytemuck::bytes_of_mut(&mut cell)[..4].copy_from_slice(&len.to_le_bytes());
    cell
}

/// Decode a length cell produced by [`len_cell`], returning the little-endian `u32`
/// stored in its low 4 bytes.
fn read_len<I: bytemuck::Pod>(cell: &I) -> u32 {
    const { assert!(std::mem::size_of::<I>() >= 4) };

    bytemuck::pod_read_unaligned(&bytemuck::bytes_of(cell)[..4])
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
        let mut scratch = neighbor_provider.scratch();

        // Set the neighbor list of a vector
        let adj_list = vec![1, 2, 3];
        scratch.write_neighbors(1, &adj_list).unwrap();

        let mut result = AdjacencyList::with_capacity(10);
        neighbor_provider.get_neighbors(1, &mut result).unwrap();
        assert_eq!(&*adj_list, &*result);

        // Append two neighbors, one of which is a duplicate
        let new_neighbors = vec![9, 2, 9];
        scratch.write_append(1, &new_neighbors).unwrap();

        neighbor_provider.get_neighbors(1, &mut result).unwrap();

        let adj_list_new = vec![1, 2, 3, 9];
        assert_eq!(&*adj_list_new, &*result);

        // Append three more neighbors, and the last one should be ignored due to max degree
        let new_neighbors = vec![5, 6, 7];
        scratch.write_append(1, &new_neighbors).unwrap();

        neighbor_provider.get_neighbors(1, &mut result).unwrap();

        let adj_list_new = vec![1, 2, 3, 9, 5, 6];
        assert_eq!(&*adj_list_new, &*result);

        // Overwrite the neighbor list of the vector to empty
        scratch.write_neighbors(1, &[]).unwrap();
        neighbor_provider.get_neighbors(1, &mut result).unwrap();
        assert!(result.is_empty());

        // Append to an emptied neighbor list
        let new_neighbors = vec![3, 4, 5];
        scratch.write_append(1, &new_neighbors).unwrap();

        neighbor_provider.get_neighbors(1, &mut result).unwrap();
        assert_eq!(&*new_neighbors, &*result);

        neighbor_provider.delete_vector(1).unwrap();
        assert!(neighbor_provider.get_neighbors(1, &mut result).is_err());
    }

    /// Test the interleaved and parallel traversal of the Bf-Tree
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
                let mut scratch = neighbor_provider_clone.scratch();
                scratch.write_neighbors(i as u32, &neighbor_list).unwrap();
            });
        }

        while let Some(res) = set.join_next().await {
            res.unwrap();
        }

        let mut result = AdjacencyList::with_capacity(121);
        for i in 0..100 {
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
        let mut scratch = neighbor_provider.scratch();

        // Set some neighbor lists
        scratch.write_neighbors(1, &[2, 3, 4]).unwrap();
        scratch.write_neighbors(2, &[1, 3, 5]).unwrap();

        // Call snapshot - should not panic
        neighbor_provider.adjacency_list_index.snapshot();

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
        let neighbor_provider = NeighborProvider::<u32>::new_from_bftree(10, bftree).unwrap();

        assert_eq!(neighbor_provider.max_degree(), 10);

        // Verify the provider is functional
        let mut scratch = neighbor_provider.scratch();
        scratch.write_neighbors(1, &[2, 3]).unwrap();
        let mut result = AdjacencyList::with_capacity(11);
        neighbor_provider.get_neighbors(1, &mut result).unwrap();
        assert_eq!(&[2, 3], &*result);
    }

    /// Test other methods and edge cases of the vector provider and synchronization mechanism of Bf-Tree
    #[tokio::test(flavor = "multi_thread", worker_threads = 5)]
    async fn test_parallel_neighbor_access() {
        let bf_tree_config = Config::default();
        let neighbor_provider =
            Arc::new(NeighborProvider::<u32>::new_with_config(120, bf_tree_config).unwrap());

        let mut set = JoinSet::new();
        for _ in 0..5 {
            let neighbor_provider_clone = Arc::clone(&neighbor_provider);
            set.spawn(async move {
                let mut scratch = neighbor_provider_clone.scratch();
                for i in 0..5 {
                    scratch.write_neighbors(i as u32, &[1, 2, 3, 4, 5]).unwrap();
                }

                let mut result = AdjacencyList::with_capacity(121);
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

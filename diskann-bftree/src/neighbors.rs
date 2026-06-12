/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Bf-Tree neighbor list provider.

use std::marker::PhantomData;

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

pub struct NeighborProvider<I: VectorId> {
    adjacency_list_index: BfTree,
    dim: usize, // Max number of neighbors in a neighbor list + 1 for the neighbor count
    #[allow(dead_code)]
    pub(crate) num_get_calls: TestCallCount,
    _phantom: PhantomData<I>,
}

impl<I: VectorId> HasId for NeighborProvider<I> {
    type Id = I;
}

impl<I: VectorId> NeighborProvider<I> {
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

        // Resize 'neighbors' to hold any full-size neighbor list + I-cell for length
        let mut guard = neighbors.resize(self.dim);

        // Serialize the key, vector_id, into a byte string, &[u8]
        let key = bytemuck::bytes_of(&vector_id);

        // Search and retrieve the corresponding neighbor list data as a byte string,
        // &[u8], in the format described in [`Self::set_neighbors_internal`].
        let value = cast_slice_mut::<I, u8>(&mut guard);
        match self.adjacency_list_index.read(key, value) {
            bf_tree::LeafReadResult::Found(read_size) => {
                // If found, then re-construct the neighbor list (valid neighbors only)
                if read_size > 0 {
                    // A retrieved neighbor list should be exactly max_degree + 1 length
                    if read_size != (self.max_degree() + 1) * std::mem::size_of::<I>() as u32 {
                        return Err(ANNError::log_index_error(
                            "Retrieved neighbor list is not expected length = max degree + 1",
                        ));
                    }

                    // The last I-cell stores the neighbor count (see `cell_to_len`).
                    //
                    // SAFETY: we know this will not panic since
                    // read_size == self.dim * std::mem::size_of::<I>().
                    let count = cell_to_len(&guard[self.dim - 1]);

                    // The specified list length must be smaller than the retrieved data length
                    if count > self.max_degree() {
                        return Err(ANNError::log_index_error(
                            "Size of retrieved neighbor list is shorter than the stored neighbor count",
                        ));
                    }

                    guard.finish(count as usize);
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

    /// Internal function for setting the neighbors for a vector id.
    ///
    /// Each key is a `VectorId`, written in bytes. The value is a neighbor list
    /// of length exactly `self.max_degree() + 1`. Specifically, an array of
    /// exactly `n` neighbors is written as :  
    /// ```text
    /// |  I0  | ... |  In  | padding |  ...  | [n; 0] |
    ///     
    /// -----------------------------------------------
    ///                     |
    ///            self.max_degree() + 1
    /// ```
    /// where `[n; 0]` represents the u32 count `n` byte-packed into a type `I`,
    /// and 'padding' indicates unfilled empty slots.
    ///
    /// Note: assuming all neighbors in the input list, 'neighbors', are valid.
    fn set_neighbors_internal(
        &self,
        vector_id: I,
        neighbors: &[I],
        buf: &mut [I],
    ) -> ANNResult<()> {
        #[cfg(test)]
        self.num_get_calls.increment();

        if buf.len() < self.dim {
            return Err(ANNError::log_index_error(
                "The provided buffer is not long enough",
            ));
        }

        // Serialize the value into the reusable buffer.
        buf[..neighbors.len()].copy_from_slice(neighbors);
        buf[self.dim - 1] = len_to_cell(neighbors.len() as u32);

        let key = bytemuck::bytes_of(&vector_id);
        let value = cast_slice::<I, u8>(&buf[..self.dim]);

        self.adjacency_list_index.insert(key, value);

        Ok(())
    }

    /// Insert a neighbor list of a vector in bf-tree as a (K, V) pair.
    ///
    /// The list of neighbors and their length is written into the passed buffer
    /// `buf` before writing the leading `self.dim * std::mem::size_of::<I>()`
    /// bytes of it into the bf-tree.
    ///
    /// # Errors
    ///  
    ///  - Neighbor length is larger than `self.max_degree()`
    ///  - Buffer length (in `I` cells) is smaller than `max_degree() + 1`
    pub fn set_neighbors(&self, vector_id: I, neighbors: &[I], buf: &mut [I]) -> ANNResult<()> {
        if neighbors.len() > self.max_degree() as usize {
            return Err(ANNError::log_index_error(
                "The provided neighbor list is longer than the max degree",
            ));
        };

        self.set_neighbors_internal(vector_id, neighbors, buf)
    }

    /// Append unique vectors into a neighbor list.
    ///
    /// The newly appended neighbor list will always be extended to 'dim'
    /// long to avoid frequent mem copy in bf-tree.
    ///
    /// Note: assuming all neighbors in the input list, 'new_neighbor_ids', are valid
    pub fn append_vector(
        &self,
        vector_id: I,
        new_neighbor_ids: &[I],
        buf: &mut [I],
    ) -> ANNResult<()> {
        // Retrieve existing neighborlist
        let mut neighbor_list = AdjacencyList::with_capacity(self.dim);
        self.get_neighbors(vector_id, &mut neighbor_list)?;

        // Append the new neighbors
        let mut new_neighbor_added = false;
        for new_neighbor_id in new_neighbor_ids {
            if neighbor_list.len() == self.max_degree() as usize {
                break;
            }
            new_neighbor_added |= neighbor_list.push(*new_neighbor_id);
        }

        // If unique new neighbors are appended, write back using the reusable buffer
        if new_neighbor_added {
            self.set_neighbors_internal(vector_id, &neighbor_list, buf)?;
        }

        Ok(())
    }

    pub fn delete_vector(&self, vector_id: I) -> ANNResult<()> {
        // Serialize the key, vector_id, into a byte string, &[u8]
        let key = bytemuck::bytes_of(&vector_id);

        self.adjacency_list_index.delete(key);
        Ok(())
    }

    pub(crate) fn scratch(&self) -> NeighborAccessor<'_, I> {
        NeighborAccessor {
            provider: self,
            buf: vec![I::zeroed(); self.dim],
        }
    }
}

pub struct NeighborAccessor<'a, I>
where
    I: VectorId,
{
    provider: &'a NeighborProvider<I>,
    buf: Vec<I>,
}

impl<'a, I> NeighborAccessor<'a, I>
where
    I: VectorId,
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
    I: VectorId,
{
    type Id = I;
}

impl<'a, I> provider::NeighborAccessor for NeighborAccessor<'a, I>
where
    I: VectorId,
{
    fn get_neighbors(
        &mut self,
        id: Self::Id,
        neighbors: &mut AdjacencyList<Self::Id>,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send {
        std::future::ready(self.provider.get_neighbors(id, neighbors))
    }
}

impl<'a, I> provider::NeighborAccessorMut for NeighborAccessor<'a, I>
where
    I: VectorId,
{
    fn set_neighbors(
        &mut self,
        id: Self::Id,
        neighbors: &[Self::Id],
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send {
        std::future::ready(self.provider.set_neighbors(id, neighbors, &mut self.buf))
    }
    fn append_vector(
        &mut self,
        id: Self::Id,
        neighbors: &[Self::Id],
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send {
        std::future::ready(self.provider.append_vector(id, neighbors, &mut self.buf))
    }
}

/// Const for size in bytes of a `u32`
const FOUR: usize = std::mem::size_of::<u32>();

/// Encode a neighbor-list length as an `I`-sized cell.
///
/// The count is stored as a little-endian `u32` in the low 4 bytes of the cell; any
/// remaining bytes are zero. The compile-time assertion guarantees that `I` is wide
/// enough.
fn len_to_cell<I: bytemuck::Pod>(len: u32) -> I {
    const { assert!(std::mem::size_of::<I>() >= FOUR) };
    let mut cell = I::zeroed();
    bytemuck::bytes_of_mut(&mut cell)[..FOUR].copy_from_slice(&len.to_le_bytes());
    cell
}

/// Decode a neighbor-list length from a cell produced by [`len_to_cell`].
fn cell_to_len<I: bytemuck::Pod>(cell: &I) -> u32 {
    const { assert!(std::mem::size_of::<I>() >= FOUR) };

    let mut low = [0u8; FOUR];
    low.copy_from_slice(&bytemuck::bytes_of(cell)[..FOUR]);

    u32::from_le_bytes(low)
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

    /// Length round-trips through an `I`-sized cell .
    #[test]
    fn len_cell_round_trip() {
        for len in [0u32, 1, 42, u32::MAX] {
            assert_eq!(cell_to_len::<u32>(&len_to_cell::<u32>(len)), len);
            assert_eq!(cell_to_len::<u64>(&len_to_cell::<u64>(len)), len);
        }
    }

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

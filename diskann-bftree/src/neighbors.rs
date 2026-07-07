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
use crate::locks::StripedLocks;
use crate::{bftree_insert, TestCallCount};

pub struct NeighborProvider<I: VectorId + IntoUsize> {
    adjacency_list_index: BfTree,
    dim: usize, // Max number of neighbors in a neighbor list + 1 for the neighbor count
    #[allow(dead_code)]
    pub(crate) num_get_calls: TestCallCount,
    _phantom: PhantomData<I>,
}

impl<I: VectorId + IntoUsize> HasId for NeighborProvider<I> {
    type Id = I;
}

impl<I: VectorId + IntoUsize> NeighborProvider<I> {
    /// Create a new instance based on bf-tree Config directly.
    pub fn new_with_config(max_degree: u32, config: Config) -> ANNResult<Self> {
        let key_size = std::mem::size_of::<u32>();
        let value_size = (max_degree as usize + 1) * std::mem::size_of::<u32>();
        crate::validate_record_size("neighbor_provider", &config, key_size, value_size)?;

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

    /// Retrieve the neighbor list of a vector.
    ///
    /// Does not acquire any lock. Callers must ensure appropriate synchronization.
    ///
    /// `neighbors` is cleared first upon each invocation. A vector with no
    /// stored neighbor list yet (bf-tree `NotFound`) yields an empty list rather
    /// than an error, so neighbor lists are created lazily on first write.
    /// Involves one data copy, from the bf-tree into `neighbors`.
    pub fn get_neighbors(&self, vector_id: I, neighbors: &mut AdjacencyList<I>) -> ANNResult<()> {
        #[cfg(test)]
        self.num_get_calls.increment();

        // Clear the output before any early returns so callers always see
        // a deterministic empty state on error.
        neighbors.clear();

        self.get_neighbors_unlocked(vector_id, neighbors)
    }

    /// Retrieve the neighbor list without acquiring any lock.
    ///
    /// Callers must ensure they already hold the appropriate lock.
    fn get_neighbors_unlocked(
        &self,
        vector_id: I,
        neighbors: &mut AdjacencyList<I>,
    ) -> ANNResult<()> {
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
                    // A retrieved neighbor list should be exactly dim length
                    if read_size as usize != self.dim * std::mem::size_of::<I>() {
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
                // The vertex has no stored neighbor list yet. bf-tree natively
                // distinguishes "absent" from "present but empty", so treat
                // absence as an empty adjacency list rather than an error. This
                // lets neighbor lists be created lazily on first write, avoiding
                // an O(max_points) eager initialization at construction.
                //
                // `guard` is dropped at the end of this match without `finish`,
                // which clears `neighbors` to the empty list (identical to the
                // `Found(0)` case above).
            }
        };

        Ok(())
    }

    /// Internal function for setting the neighbors for a vector id.
    ///
    /// Each key is a `VectorId`, written in bytes. The value is a
    /// neighbor list of length exactly `self.dim`. Specifically,
    /// an array of exactly `n` neighbors is written as :
    /// ```text
    /// |  I0  | ... |  In  | padding |  ...  | [n; 0] |
    ///
    /// -----------------------------------------------
    ///                     |
    ///                 self.dim
    /// ```
    /// where `[n; 0]` represents the u32 count `n` byte-packed
    /// into a type `I`, and 'padding' indicates unfilled empty slots.
    ///
    /// Note: assuming all neighbors in the input list, 'neighbors',
    /// are valid.
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

        bftree_insert(&self.adjacency_list_index, key, value)?;

        Ok(())
    }

    /// Insert a neighbor list of a vector in bf-tree as a (K, V) pair.
    ///
    /// Does not acquire any lock. Callers must ensure appropriate synchronization.
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
        if neighbors.len() > self.dim - 1 {
            return Err(ANNError::log_index_error(
                "The provided neighbor list is longer than the max degree",
            ));
        };

        self.set_neighbors_internal(vector_id, neighbors, buf)
    }

    /// Append unique vectors into a neighbor list.
    ///
    /// Does not acquire any lock. Callers must ensure appropriate synchronization
    /// for the entire read-modify-write cycle.
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
        self.get_neighbors_unlocked(vector_id, &mut neighbor_list)?;

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
            self.set_neighbors_internal(vector_id, &neighbor_list, buf)?;
        }

        Ok(())
    }

    pub fn delete_vector(&self, vector_id: I) -> ANNResult<()> {
        let key = bytemuck::bytes_of(&vector_id);
        self.adjacency_list_index.delete(key);
        Ok(())
    }

    pub(crate) fn scratch<'a>(&'a self, locks: &'a StripedLocks) -> NeighborAccessor<'a, I> {
        NeighborAccessor {
            provider: self,
            locks,
            buf: vec![I::zeroed(); self.dim],
        }
    }
}

pub struct NeighborAccessor<'a, I>
where
    I: VectorId + IntoUsize,
{
    provider: &'a NeighborProvider<I>,
    locks: &'a StripedLocks,
    buf: Vec<I>,
}

impl<'a, I> NeighborAccessor<'a, I>
where
    I: VectorId + IntoUsize,
{
    pub fn write_neighbors(&mut self, id: I, neighbors: &[I]) -> ANNResult<()> {
        let _guard = self.locks.lock(id.into_usize());
        self.provider.set_neighbors(id, neighbors, &mut self.buf)
    }
    pub fn write_append(&mut self, id: I, neighbors: &[I]) -> ANNResult<()> {
        let _guard = self.locks.lock(id.into_usize());
        self.provider.append_vector(id, neighbors, &mut self.buf)
    }
}

impl<'a, I> HasId for NeighborAccessor<'a, I>
where
    I: VectorId + IntoUsize,
{
    type Id = I;
}

impl<'a, I> provider::NeighborAccessor for NeighborAccessor<'a, I>
where
    I: VectorId + IntoUsize,
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
    I: VectorId + IntoUsize,
{
    fn set_neighbors(
        &mut self,
        id: Self::Id,
        neighbors: &[Self::Id],
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send {
        let _guard = self.locks.lock(id.into_usize());
        std::future::ready(self.provider.set_neighbors(id, neighbors, &mut self.buf))
    }
    fn append_vector(
        &mut self,
        id: Self::Id,
        neighbors: &[Self::Id],
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send {
        let _guard = self.locks.lock(id.into_usize());
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

    /// Number of concurrent worker tasks for stress tests, scaled to the host's
    /// available parallelism with a floor so contention is guaranteed even on
    /// low-core CI runners. Falls back to the floor if parallelism is unknown.
    fn stress_thread_count() -> u32 {
        std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(8)
            .max(8)
    }

    /// Build a `NeighborProvider<u32>` with a default bf-tree config.
    fn new_provider(max_degree: u32) -> NeighborProvider<u32> {
        NeighborProvider::<u32>::new_with_config(max_degree, Config::default()).unwrap()
    }

    /// Build a shared `NeighborProvider<u32>` with a default bf-tree config.
    fn new_shared_provider(max_degree: u32) -> Arc<NeighborProvider<u32>> {
        Arc::new(new_provider(max_degree))
    }

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
        let locks = Arc::new(StripedLocks::new());
        let neighbor_provider = new_provider(6);
        let mut scratch = neighbor_provider.scratch(&locks);

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

    /// Lazy init: reading ids that were never written must yield an empty list
    /// rather than erroring or panicking. This hammers a wide sweep of unwritten
    /// ids (including some interleaved with written ones) to confirm the
    /// `NotFound` -> empty mapping holds and never crashes.
    #[test]
    fn test_get_unwritten_ids_returns_empty() {
        let locks = Arc::new(StripedLocks::new());
        let neighbor_provider = new_provider(64);
        let mut scratch = neighbor_provider.scratch(&locks);

        // Write neighbor lists for a sparse set of ids, leaving large gaps.
        let written: [u32; 3] = [10, 1_000, 100_000];
        for &id in &written {
            scratch.write_neighbors(id, &[1, 2, 3]).unwrap();
        }

        let mut result = AdjacencyList::with_capacity(64);

        // Sweep a wide id range, including the written ids and the gaps between
        // and beyond them. The buffer is reused across calls to confirm it is
        // cleared on every read.
        for id in 0u32..200_000 {
            neighbor_provider.get_neighbors(id, &mut result).unwrap();
            if written.contains(&id) {
                assert_eq!(&[1, 2, 3], &*result, "written id {id} lost its list");
            } else {
                assert!(result.is_empty(), "unwritten id {id} should be empty");
            }
        }

        // Ids far beyond anything ever written are still just empty, not errors.
        for id in [u32::MAX - 1, u32::MAX] {
            neighbor_provider.get_neighbors(id, &mut result).unwrap();
            assert!(result.is_empty());
        }
    }

    /// Lazy init under concurrency: many threads simultaneously hammering reads
    /// of never-written ids must all observe empty lists with no panics, lost
    /// updates, or errors.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_concurrent_reads_of_unwritten_ids() {
        let neighbor_provider = new_shared_provider(64);
        let num_threads = stress_thread_count();

        let mut set = JoinSet::new();
        for t in 0..num_threads {
            let neighbor_provider = Arc::clone(&neighbor_provider);
            set.spawn(async move {
                let mut result = AdjacencyList::with_capacity(64);
                // Overlapping id ranges across threads to maximize contention on
                // the shared bf-tree read path for absent keys.
                for id in (t * 1_000)..(t * 1_000 + 10_000) {
                    neighbor_provider.get_neighbors(id, &mut result).unwrap();
                    assert!(result.is_empty());
                }
            });
        }

        while let Some(res) = set.join_next().await {
            res.unwrap();
        }
    }

    /// Test the interleaved and parallel traversal of the Bf-Tree
    /// by invoking the async accessors of the neighbor list provider
    #[tokio::test(flavor = "multi_thread", worker_threads = 5)]
    async fn test_parallel_tree_traversal() {
        let locks = Arc::new(StripedLocks::new());
        let neighbor_provider = new_shared_provider(120);

        let mut set = JoinSet::new();
        for i in 0..100 {
            let neighbor_list = vec![i as u32, (i + 1) as u32, (i + 2) as u32];
            let neighbor_provider_clone = Arc::clone(&neighbor_provider);
            let locks = locks.clone();
            set.spawn(async move {
                let mut scratch = neighbor_provider_clone.scratch(&locks);
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

    /// Test that cpr_snapshot can be called without errors
    #[tokio::test]
    async fn test_snapshot() {
        let locks = Arc::new(StripedLocks::new());
        // Use a temporary directory that gets cleaned up when dropped
        let temp_dir = tempfile::tempdir().unwrap();
        let snapshot_path = temp_dir.path().join("test_neighbor_snapshot.bftree");
        let snapshot_output = temp_dir.path().join("test_neighbor_snapshot_out.bftree");

        let mut bf_tree_config = Config::new(&snapshot_path, 16384 * 16);
        bf_tree_config.storage_backend(bf_tree::StorageBackend::Std);
        bf_tree_config.use_snapshot(true);

        let neighbor_provider =
            NeighborProvider::<u32>::new_with_config(6, bf_tree_config).unwrap();
        let mut scratch = neighbor_provider.scratch(&locks);

        // Set some neighbor lists
        scratch.write_neighbors(1, &[2, 3, 4]).unwrap();
        scratch.write_neighbors(2, &[1, 3, 5]).unwrap();

        // Call cpr_snapshot - should not panic
        neighbor_provider
            .adjacency_list_index
            .cpr_snapshot(&snapshot_output);

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
        // Test with various max_degree values
        let neighbor_provider = new_provider(6);
        assert_eq!(neighbor_provider.max_degree(), 6);

        let neighbor_provider = new_provider(120);
        assert_eq!(neighbor_provider.max_degree(), 120);

        let neighbor_provider = new_provider(1);
        assert_eq!(neighbor_provider.max_degree(), 1);
    }

    /// Test new_from_bftree constructor
    #[tokio::test]
    async fn test_new_from_bftree() {
        let locks = Arc::new(StripedLocks::new());
        let bftree = BfTree::with_config(Config::default(), None).expect("Failed to create BfTree");
        let neighbor_provider = NeighborProvider::<u32>::new_from_bftree(10, bftree).unwrap();

        assert_eq!(neighbor_provider.max_degree(), 10);

        // Verify the provider is functional
        let mut scratch = neighbor_provider.scratch(&locks);
        scratch.write_neighbors(1, &[2, 3]).unwrap();
        let mut result = AdjacencyList::with_capacity(11);
        neighbor_provider.get_neighbors(1, &mut result).unwrap();
        assert_eq!(&[2, 3], &*result);
    }

    /// Test other methods and edge cases of the vector provider and synchronization mechanism of Bf-Tree
    #[tokio::test(flavor = "multi_thread", worker_threads = 5)]
    async fn test_parallel_neighbor_access() {
        let locks = Arc::new(StripedLocks::new());
        let neighbor_provider = new_shared_provider(120);

        let mut set = JoinSet::new();
        for _ in 0..5 {
            let neighbor_provider_clone = Arc::clone(&neighbor_provider);
            let locks = locks.clone();
            set.spawn(async move {
                let mut scratch = neighbor_provider_clone.scratch(&locks);
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

    /// Stress test: many threads concurrently append to the SAME vertex.
    ///
    /// Without per-vertex locking, this test would non-deterministically lose edges
    /// due to the TOCTOU race in append_vector's read-modify-write cycle. With the
    /// per-vertex write lock, every appended edge must be present in the final list.
    ///
    /// Lost-update races are probabilistic, so a single pass may not surface a
    /// regression. The scenario is therefore repeated over many outer iterations,
    /// and the writer count scales with the host's available parallelism (with a
    /// floor) to keep contention meaningful on CI boxes of any core count. The
    /// `multi_thread` runtime is left unsized so tokio scales its worker pool to
    /// the number of CPUs.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_concurrent_append_no_lost_edges() {
        let num_threads = stress_thread_count();
        let edges_per_thread = 20u32;
        // Size the degree to the workload so every distinct edge can be retained.
        let max_degree = num_threads * edges_per_thread;

        // Repeat the whole scenario many times: each iteration is a fresh chance
        // to interleave threads such that a broken lock would drop an edge.
        let outer_iterations = 100;
        for _ in 0..outer_iterations {
            let locks = Arc::new(StripedLocks::new());
            let provider = new_shared_provider(max_degree);

            // Initialize vertex 0 with an empty neighbor list.
            let mut scratch = provider.scratch(&locks);
            scratch.write_neighbors(0, &[]).unwrap();

            // Each thread appends a unique set of neighbor IDs to vertex 0.
            // Thread t appends IDs: [t*edges+1, ..., t*edges+edges]
            let mut set = JoinSet::new();
            for t in 0..num_threads {
                let provider_clone = Arc::clone(&provider);
                let locks = locks.clone();
                set.spawn(async move {
                    let mut scratch = provider_clone.scratch(&locks);
                    let base = t * edges_per_thread + 1;
                    for offset in 0..edges_per_thread {
                        let neighbor_id = base + offset;
                        scratch.write_append(0, &[neighbor_id]).unwrap();
                    }
                });
            }

            while let Some(res) = set.join_next().await {
                res.unwrap();
            }

            // Verify ALL edges from all threads are present.
            let mut result = AdjacencyList::with_capacity(max_degree as usize + 1);
            provider.get_neighbors(0, &mut result).unwrap();

            let expected_count = (num_threads * edges_per_thread) as usize;
            assert_eq!(
                result.len(),
                expected_count,
                "Expected {} edges but found {} — edges were lost!",
                expected_count,
                result.len()
            );

            // Verify every expected ID is present.
            for t in 0..num_threads {
                let base = t * edges_per_thread + 1;
                for offset in 0..edges_per_thread {
                    let expected_id = base + offset;
                    assert!(
                        result.contains(expected_id),
                        "Missing edge {} from thread {}",
                        expected_id,
                        t
                    );
                }
            }
        }
    }

    /// Stress test: concurrent appends to DIFFERENT vertices (no contention).
    ///
    /// Validates that the lock table doesn't introduce false sharing or deadlocks
    /// when independent vertices are mutated in parallel. The writer count scales
    /// with the host's available parallelism (with a floor), and the `multi_thread`
    /// runtime is left unsized so tokio scales its worker pool to the CPU count.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_concurrent_append_independent_vertices() {
        let locks = Arc::new(StripedLocks::new());
        let max_degree = 64u32;
        let num_threads = stress_thread_count();
        let vertices_per_thread = 50u32;
        let num_vertices = num_threads * vertices_per_thread; // disjoint ranges, no overlap
        let provider = new_shared_provider(max_degree);

        // Initialize all vertices with empty neighbor lists.
        {
            let mut scratch = provider.scratch(&locks);
            for v in 0..num_vertices {
                scratch.write_neighbors(v, &[]).unwrap();
            }
        }

        let mut set = JoinSet::new();

        for t in 0..num_threads {
            let provider_clone = Arc::clone(&provider);
            let locks = locks.clone();
            set.spawn(async move {
                let mut scratch = provider_clone.scratch(&locks);
                // Each thread owns a disjoint range of vertices
                let base_vertex = t * vertices_per_thread;
                for i in 0..vertices_per_thread {
                    let vertex = base_vertex + i;
                    let neighbor_id = t * 1000 + i;
                    scratch.write_append(vertex, &[neighbor_id]).unwrap();
                }
            });
        }

        while let Some(res) = set.join_next().await {
            res.unwrap();
        }

        // Verify total edge count across all vertices matches expectation.
        let mut total_edges = 0usize;
        let mut result = AdjacencyList::with_capacity(max_degree as usize + 1);
        for v in 0..num_vertices {
            provider.get_neighbors(v, &mut result).unwrap();
            total_edges += result.len();
        }

        let expected_total = (num_threads * vertices_per_thread) as usize;
        assert_eq!(
            total_edges, expected_total,
            "Expected {} total edges but found {} — edges were lost or duplicated!",
            expected_total, total_edges
        );
    }

    /// Stress test: mixed concurrent reads and writes on the same vertex.
    ///
    /// Writers continuously append while readers verify the neighbor list is
    /// always in a consistent state (length matches actual content, no torn reads).
    ///
    /// The read/write phase is repeated over many outer iterations to surface
    /// torn-read races more reliably, the writer/reader counts scale with the
    /// host's available parallelism (with a floor), and the `multi_thread`
    /// runtime is left unsized so tokio scales its worker pool to the CPU count.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_concurrent_read_write_consistency() {
        let parallelism = stress_thread_count();
        let num_writers = parallelism;
        let num_readers = parallelism;
        let edges_per_writer = 20u32;
        // Size the degree to the workload so every distinct edge can be retained.
        let max_degree = num_writers * edges_per_writer;
        // Largest neighbor ID any writer can append (IDs run 1..=num_writers*edges).
        let max_valid_id = num_writers * edges_per_writer;

        let outer_iterations = 100;
        for _ in 0..outer_iterations {
            let locks = Arc::new(StripedLocks::new());
            let provider = new_shared_provider(max_degree);

            // Initialize vertex 0.
            let mut scratch = provider.scratch(&locks);
            scratch.write_neighbors(0, &[]).unwrap();

            let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let writers_remaining = Arc::new(std::sync::atomic::AtomicU32::new(num_writers));

            // Spawn writer threads that append edges.
            let mut set = JoinSet::new();

            for t in 0..num_writers {
                let provider_clone = Arc::clone(&provider);
                let done_clone = Arc::clone(&done);
                let remaining_clone = Arc::clone(&writers_remaining);
                let locks = locks.clone();
                set.spawn(async move {
                    let mut scratch = provider_clone.scratch(&locks);
                    let base = t * edges_per_writer + 1;
                    for offset in 0..edges_per_writer {
                        scratch.write_append(0, &[base + offset]).unwrap();
                        tokio::task::yield_now().await;
                    }
                    // Signal done only when ALL writers have finished.
                    if remaining_clone.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1 {
                        done_clone.store(true, std::sync::atomic::Ordering::Release);
                    }
                });
            }

            // Spawn reader threads that continuously read and validate consistency.
            for _ in 0..num_readers {
                let provider_clone = Arc::clone(&provider);
                let done_clone = Arc::clone(&done);
                set.spawn(async move {
                    let mut result = AdjacencyList::with_capacity(max_degree as usize + 1);
                    let mut iterations = 0u64;
                    // Validate at least once before observing `done`. A reader that is
                    // first scheduled only after every writer has already finished would
                    // otherwise see `done == true` immediately, read nothing, and trip the
                    // liveness assertion below — a scheduling artifact, not a real defect.
                    // Reading the fully-written state is still a valid, consistent
                    // observation, so the invariants hold regardless of timing.
                    loop {
                        provider_clone.get_neighbors(0, &mut result).unwrap();
                        // The list must never exceed max_degree
                        assert!(
                            result.len() <= max_degree as usize,
                            "Neighbor list exceeded max_degree"
                        );
                        // Every ID must be a real appended edge: non-zero and within
                        // the range any writer could have produced. Anything else
                        // signals a torn or garbage read.
                        for &id in result.iter() {
                            assert!(id > 0, "Found invalid zero ID in neighbor list");
                            assert!(
                                id <= max_valid_id,
                                "Found out-of-bounds ID {id} (max valid is {max_valid_id})"
                            );
                        }
                        iterations += 1;
                        if done_clone.load(std::sync::atomic::Ordering::Acquire) {
                            break;
                        }
                        tokio::task::yield_now().await;
                    }
                    assert!(
                        iterations > 0,
                        "Reader never got a chance to run — test is not validating anything"
                    );
                });
            }

            while let Some(res) = set.join_next().await {
                res.unwrap();
            }

            // Final check: all edges from all writers present.
            let mut result = AdjacencyList::with_capacity(max_degree as usize + 1);
            provider.get_neighbors(0, &mut result).unwrap();
            let expected = (num_writers * edges_per_writer) as usize;
            assert_eq!(
                result.len(),
                expected,
                "Expected {} edges but found {}",
                expected,
                result.len()
            );
        }
    }
}

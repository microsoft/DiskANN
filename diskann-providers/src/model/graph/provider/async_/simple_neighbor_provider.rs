/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::RwLock;

use crate::storage::{StorageReadProvider, StorageWriteProvider};
use diskann::{graph::AdjacencyList, provider::HasId, ANNError, ANNResult};
use diskann_vector::contains::ContainsSimd;
use tracing::trace;

use super::common::{AlignedMemoryVectorStore, TestCallCount};
use crate::storage::{
    self, AsyncIndexMetadata, AsyncQuantLoadContext, DiskGraphOnly, LoadWith, SaveWith,
};

pub struct SimpleNeighborProviderAsync {
    // Each adjacency list is stored in a fixed size slice of size max_degree * graph_slack_factor + 1.
    // The length of the list is stored in the extra element at the end as a u32.
    graph: AlignedMemoryVectorStore<u32>,
    locks: Vec<RwLock<()>>,
    num_start_points: usize,

    pub num_get_calls: TestCallCount,
}

impl SimpleNeighborProviderAsync {
    pub fn new(
        max_points: usize,
        num_start_points: usize,
        max_degree: u32,
        graph_slack_factor: f32,
    ) -> Self {
        let size = max_points + num_start_points;
        let graph = AlignedMemoryVectorStore::with_capacity(
            size,
            (max_degree as f32 * graph_slack_factor) as usize + 1,
        );
        let locks = (0..size).map(|_| RwLock::new(())).collect::<Vec<_>>();

        Self {
            graph,
            locks,
            num_start_points,
            num_get_calls: TestCallCount::default(),
        }
    }

    /// Return the neighbor list for `index` as a slice.
    ///
    /// SAFETY:
    ///
    /// This function will never read out of bounds, but it does not synchronize access to
    /// the data. It must be called while holding the corresponding lock at `self.locks[index]`.
    unsafe fn get_slice(&self, index: usize) -> &[u32] {
        // SAFETY: This function must be called while the corresponding lock for this slot
        // is held.
        let s = unsafe { self.graph.get_slice(index) };

        let len = s[self.graph.dim() - 1] as usize;
        &s[0..len]
    }

    pub fn set_neighbors_sync(&self, id: usize, neighbors: &[u32]) -> ANNResult<()> {
        assert!(
            neighbors.len() < self.graph.dim(),
            "neighbors ({}) exceeded max adjacency list size ({})",
            neighbors.len(),
            self.graph.dim() - 1,
        );

        // Lint: We don't have a good way of recovering from lock poisoning anyways.
        #[allow(clippy::unwrap_used)]
        let _guard = self.locks[id].write().unwrap();

        // SAFETY: We are holding the write lock for this id.
        let list = unsafe { self.graph.get_mut_slice(id) };
        list[0..neighbors.len()].copy_from_slice(neighbors);

        // The assertion above guarantees `neighbors.len() < self.graph.dim()`, which
        // means it fits in a `u32` (graph dim is sized in `u32` anyway).
        list[self.graph.dim() - 1] = neighbors.len() as u32;
        Ok(())
    }

    pub fn get_neighbors_sync(
        &self,
        id: usize,
        neighbors: &mut AdjacencyList<u32>,
    ) -> ANNResult<()> {
        #[cfg(test)]
        self.num_get_calls.increment();

        // Lint: We don't have a good way of recovering from lock poisoning anyways.
        #[allow(clippy::unwrap_used)]
        let _guard = self.locks[id].read().unwrap();

        // SAFETY: We are holding the read lock for `id`.
        let list = unsafe { self.get_slice(id) };
        neighbors.overwrite_trusted(list);
        Ok(())
    }

    pub fn append_vector_sync(&self, id: usize, new_neighbor_ids: &[u32]) -> ANNResult<()> {
        // Lint: We don't have a good way of recovering from lock poisoning anyways.
        #[allow(clippy::unwrap_used)]
        let _guard = self.locks[id].write().unwrap();

        // SAFETY: We took the write lock for `id` above.
        let list_raw = unsafe { self.graph.get_mut_slice(id) };
        let len = list_raw[self.graph.dim() - 1] as usize;
        let mut new_len = len;
        let mut list = &mut list_raw[0..len];

        for new_neighbor_id in new_neighbor_ids {
            if u32::contains_simd(list, *new_neighbor_id) {
                trace!("append_vector: new neighbor already exists");
                continue;
            }

            if new_len < self.graph.dim() - 1 {
                list_raw[new_len] = *new_neighbor_id;
                new_len += 1;
                list = &mut list_raw[0..new_len];
            } else {
                trace!("append_vector: some new neighbors discarded; adjacency list full");
                break;
            }
        }

        // `new_len < self.graph.dim()` is enforced by the loop above, so the cast is safe.
        list_raw[self.graph.dim() - 1] = new_len as u32;
        Ok(())
    }
}

impl HasId for SimpleNeighborProviderAsync {
    type Id = u32;
}

impl SimpleNeighborProviderAsync {
    /// Load the graph directly from a canonical DiskANN graph storage at path `path`.
    ///
    /// See also: [`storage::bin::load_graph`].
    pub fn load_direct<P>(provider: &P, path: &str) -> ANNResult<Self>
    where
        P: StorageReadProvider,
    {
        storage::bin::load_graph(
            provider,
            path,
            |num_points, max_degree, num_start_points| {
                // The value `num_points` is the total number of vectors discovered in the
                // source file, including start points.
                //
                // Work backwards from this value to determine the internal `max_points`.
                let max_points = num_points.checked_sub(num_start_points).ok_or_else(|| {
                    ANNError::log_index_error(format_args!(
                        "expected {} start points but the on-disk dataset only has {} total points",
                        num_start_points, num_points,
                    ))
                })?;

                // The provided `max_degree` here is the observed maximum degree in the input
                // file. Therefore, we don't need to apply a slack factor to it.
                Ok(Self::new(
                    max_points,
                    num_start_points,
                    max_degree as u32,
                    1.0,
                ))
            },
        )
    }

    /// Save `self` directly to a canonical DiskANN graph storage at path `path`.
    ///
    /// See also: [`storage::bin::save_graph`].
    pub fn save_direct<P>(&self, provider: &P, start_point: u32, path: &str) -> ANNResult<usize>
    where
        P: StorageWriteProvider,
    {
        storage::bin::save_graph(self, provider, start_point, path)
    }
}

/// This is an adaptor for compatibility with the async index serialization.
///
/// The parameter consists of `(start_point, prefix)` because the index start point is not
/// saved within `SimpleNeighborPRoviderAsync`.
impl SaveWith<(u32, AsyncIndexMetadata)> for SimpleNeighborProviderAsync {
    type Ok = usize;
    type Error = ANNError;

    async fn save_with<P>(
        &self,
        provider: &P,
        (start_point, metadata): &(u32, AsyncIndexMetadata),
    ) -> ANNResult<usize>
    where
        P: StorageWriteProvider,
    {
        self.save_direct(provider, *start_point, metadata.prefix())
    }
}

/// This implementation handles the conversion between async index and disk index format.
/// Parameters:
/// - `start_point`: The vector ID used during async index building (exceed max_point bounds)
/// - `actual_start_point`: The real vector ID with identical vector values as `start_point`
/// - `prefix`: Path prefix for the disk index files
///
/// The substitution of `start_point` with `actual_start_point` ensures compatibility
/// with the on-disk format while preserving the correct entry point information.
impl SaveWith<(u32, u32, DiskGraphOnly)> for SimpleNeighborProviderAsync {
    type Ok = usize;
    type Error = ANNError;

    async fn save_with<P>(
        &self,
        provider: &P,
        (imem_start_point, actual_start_point, metadata): &(u32, u32, DiskGraphOnly),
    ) -> Result<Self::Ok, Self::Error>
    where
        P: StorageWriteProvider,
    {
        storage::bin::save_graph_with_remapped_start(
            self,
            provider,
            *imem_start_point,
            *actual_start_point,
            metadata.prefix(),
        )
    }
}

/// This is an adaptor for compatibility with the async index serialization.
impl LoadWith<AsyncIndexMetadata> for SimpleNeighborProviderAsync {
    type Error = ANNError;

    async fn load_with<P>(provider: &P, metadata: &AsyncIndexMetadata) -> ANNResult<Self>
    where
        P: StorageReadProvider,
    {
        Self::load_direct(provider, metadata.prefix())
    }
}

/// This is an adaptor for compatibility with the async index serialization.
impl LoadWith<AsyncQuantLoadContext> for SimpleNeighborProviderAsync {
    type Error = ANNError;

    async fn load_with<P>(provider: &P, ctx: &AsyncQuantLoadContext) -> ANNResult<Self>
    where
        P: StorageReadProvider,
    {
        Self::load_with(provider, &ctx.metadata).await
    }
}

////////////////////////////////////////////
// SetAdjacencyList and GetAdjacencyList //
///////////////////////////////////////////

/// Hook into [`storage::bin::load_graph`] by implementing [`storage::bin::SetAdjacencyList`].
impl storage::bin::SetAdjacencyList for SimpleNeighborProviderAsync {
    type Item = u32;
    fn set_adjacency_list(&mut self, i: usize, element: &[u32]) -> ANNResult<()> {
        self.set_neighbors_sync(i, element)?;
        Ok(())
    }
}

/// Hook into [`storage::bin::save_graph`] by implementing [`storage::bin::GetAdjacencyList`].
impl storage::bin::GetAdjacencyList for SimpleNeighborProviderAsync {
    type Element = u32;
    type Item<'a> = AdjacencyList<u32>;

    fn get_adjacency_list(&self, i: usize) -> ANNResult<Self::Item<'_>> {
        let mut list = AdjacencyList::new();
        self.get_neighbors_sync(i, &mut list)?;
        Ok(list)
    }

    fn total(&self) -> usize {
        self.locks.len()
    }

    fn additional_points(&self) -> u64 {
        self.num_start_points as u64
    }

    fn max_degree(&self) -> Option<u32> {
        Some((self.graph.dim() - 1) as u32)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use crate::storage::VirtualStorageProvider;

    use super::*;

    #[test]
    fn test_neighbor_provider() {
        let neighbor_provider = SimpleNeighborProviderAsync::new(10, 1, 5, 1.0);

        let adj_list = vec![1, 2, 3];
        neighbor_provider.set_neighbors_sync(1, &adj_list).unwrap();

        let mut result = AdjacencyList::new();
        neighbor_provider
            .get_neighbors_sync(1, &mut result)
            .unwrap();

        assert_eq!(&adj_list, &*result);

        let new_adj_list = AdjacencyList::from_iter_untrusted([4, 5, 6]);
        neighbor_provider
            .set_neighbors_sync(1, &new_adj_list)
            .unwrap();

        neighbor_provider
            .get_neighbors_sync(1, &mut result)
            .unwrap();

        assert_eq!(new_adj_list, result);
    }

    #[tokio::test]
    async fn test_save_load() {
        let max_degree = 5;
        let max_points = 8;
        let additional_points = 2;

        let provider =
            SimpleNeighborProviderAsync::new(max_points, additional_points, max_degree, 1.0);

        // Setup a virtual storage provider with memory filesystem
        let storage = VirtualStorageProvider::new_memory();

        // Fill the graph, each node i will have neighbors [i+1, i+2, i+3]
        for i in 0..max_points + additional_points {
            let neighbors: Vec<u32> = (1..4).map(|j| i as u32 + j).collect();
            provider.set_neighbors_sync(i, &neighbors).unwrap();
        }

        let prefix = AsyncIndexMetadata::new("/resumable_test");
        // Test SaveWith implementation
        let start_point = 0;
        let result = provider
            .save_with(&storage, &(start_point, prefix.clone()))
            .await;
        assert!(result.is_ok(), "Failed to save with resumable context");

        // Verify the file was created
        let expected_path = prefix.prefix();
        assert!(
            storage.exists(expected_path),
            "Resumable graph file was not created"
        );

        let receiver = SimpleNeighborProviderAsync::load_direct(&storage, prefix.prefix()).unwrap();

        for i in 0..max_points + additional_points {
            let mut result = AdjacencyList::new();
            let mut loaded_result = AdjacencyList::new();
            provider.get_neighbors_sync(i, &mut result).unwrap();
            receiver.get_neighbors_sync(i, &mut loaded_result).unwrap();
            assert_eq!(
                result, loaded_result,
                "Adjacency list for node {} doesn't match after loading",
                i
            );
        }
    }
}

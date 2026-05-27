/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::RwLock;

use crate::storage::{StorageReadProvider, StorageWriteProvider};
use diskann::{
    ANNError, ANNResult,
    graph::AdjacencyList,
    provider::HasId,
    utils::{TryIntoVectorId, VectorId},
};
use tracing::trace;

use super::common::{AlignedMemoryVectorStore, TestCallCount};
use crate::storage::{
    self, AsyncIndexMetadata, AsyncQuantLoadContext, DiskGraphOnly, LoadWith, SaveWith,
};

pub struct SimpleNeighborProviderAsync<I: VectorId> {
    // Each adjacency list is stored in a fixed size slice of size max_degree * graph_slack_factor + 1.
    // The length of the list is stored in the extra element at the end.
    graph: AlignedMemoryVectorStore<I>,
    locks: Vec<RwLock<()>>,
    num_start_points: usize,

    pub num_get_calls: TestCallCount,
}

impl<I: VectorId> SimpleNeighborProviderAsync<I> {
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
    unsafe fn get_slice(&self, index: usize) -> &[I] {
        // SAFETY: This function must be called while the corresponding lock for this slot
        // is held.
        let s = unsafe { self.graph.get_slice(index) };

        let len = s[self.graph.dim() - 1].into_usize();
        &s[0..len]
    }

    pub fn set_neighbors_sync(&self, id: usize, neighbors: &[I]) -> ANNResult<()> {
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

        // Lint: neighbor lists won't overflow the VectorIdType
        #[allow(clippy::unwrap_used)]
        {
            list[self.graph.dim() - 1] = neighbors.len().try_into_vector_id().unwrap();
        }
        Ok(())
    }

    pub fn get_neighbors_sync(&self, id: usize, neighbors: &mut AdjacencyList<I>) -> ANNResult<()> {
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

    pub fn append_vector_sync(&self, id: usize, new_neighbor_ids: &[I]) -> ANNResult<()> {
        // Lint: We don't have a good way of recovering from lock poisoning anyways.
        #[allow(clippy::unwrap_used)]
        let _guard = self.locks[id].write().unwrap();

        // SAFETY: We took the write lock for `id` above.
        let list_raw = unsafe { self.graph.get_mut_slice(id) };
        let len = list_raw[self.graph.dim() - 1].into_usize();
        let mut new_len = len;
        let mut list = &mut list_raw[0..len];

        for new_neighbor_id in new_neighbor_ids {
            if I::contains_simd(list, *new_neighbor_id) {
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

        // Lint: adjacency list sizes will not overflow VectorId
        #[allow(clippy::unwrap_used)]
        {
            list_raw[self.graph.dim() - 1] = new_len.try_into_vector_id().unwrap();
        }
        Ok(())
    }
}

impl<I: VectorId> HasId for SimpleNeighborProviderAsync<I> {
    type Id = I;
}

impl SimpleNeighborProviderAsync<u32> {
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
impl SaveWith<(u32, AsyncIndexMetadata)> for SimpleNeighborProviderAsync<u32> {
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
impl SaveWith<(u32, u32, DiskGraphOnly)> for SimpleNeighborProviderAsync<u32> {
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
        let graph = DiskAdaptor {
            provider: self,
            inmem_start_point: *imem_start_point,
            actual_start_point: *actual_start_point,
        };

        storage::bin::save_graph(&graph, provider, *actual_start_point, metadata.prefix())
    }
}

/// This is an adaptor for compatibility with the async index serialization.
impl LoadWith<AsyncIndexMetadata> for SimpleNeighborProviderAsync<u32> {
    type Error = ANNError;

    async fn load_with<P>(provider: &P, metadata: &AsyncIndexMetadata) -> ANNResult<Self>
    where
        P: StorageReadProvider,
    {
        Self::load_direct(provider, metadata.prefix())
    }
}

/// This is an adaptor for compatibility with the async index serialization.
impl LoadWith<AsyncQuantLoadContext> for SimpleNeighborProviderAsync<u32> {
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
impl storage::bin::SetAdjacencyList for SimpleNeighborProviderAsync<u32> {
    type Item = u32;
    fn set_adjacency_list(&mut self, i: usize, element: &[u32]) -> ANNResult<()> {
        self.set_neighbors_sync(i, element)?;
        Ok(())
    }
}

/// Hook into [`storage::bin::save_graph`] by implementing [`storage::bin::GetAdjacencyList`].
impl storage::bin::GetAdjacencyList for SimpleNeighborProviderAsync<u32> {
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

/// This adaptor translates between the in-memory async index representation
/// and the on-disk index format during serialization.
///
/// Key differences between the formats:
/// 1. Disk format requires a valid vector ID as start point, while async index uses a
///    virtual ID (max_points + 1) that exceeds the valid dataset range
/// 2. In-memory index appends the virtual start point at the end of adjacency lists
/// 3. Disk format expects additional_points = 0, while async index uses additional_points = 1
///
/// This adaptor handles these differences by:
/// - Substituting the virtual start point ID with an actual dataset ID when found in adjacency lists
/// - Excluding the virtual point from the total count (subtracting 1 from length)
/// - Setting additional_points to 0 as required by the disk format specification
///
/// Used with [`storage::bin::save_graph`] to persist an async index in standard DiskANN format.
struct DiskAdaptor<'a> {
    provider: &'a SimpleNeighborProviderAsync<u32>,
    inmem_start_point: u32,
    actual_start_point: u32,
}

impl storage::bin::GetAdjacencyList for DiskAdaptor<'_> {
    type Element = u32;
    type Item<'item>
        = Vec<u32>
    where
        Self: 'item;

    fn get_adjacency_list(&self, i: usize) -> ANNResult<Self::Item<'_>> {
        let mut list = AdjacencyList::new();
        self.provider.get_neighbors_sync(i, &mut list)?;

        // Need to change to a `Vec` because remapping the start point can cause duplicates,
        // and changing the logic to not have duplicates changes the exact nature of the
        // graph and breaks integration tests for the disk index builder.
        let mut list: Vec<_> = list.into();
        for i in list.iter_mut() {
            if *i == self.inmem_start_point {
                *i = self.actual_start_point;
            }
        }

        Ok(list)
    }

    fn total(&self) -> usize {
        // Don't include any start points at the end.
        self.provider.locks.len() - self.provider.num_start_points
    }

    /// Fixed to 0 for the disk format
    fn additional_points(&self) -> u64 {
        0
    }

    fn max_degree(&self) -> Option<u32> {
        None
    }
}

//////////////////////////////////
// diskann-record Save/Load     //
//////////////////////////////////
//
// The adjacency-list bytes are streamed to a side-car file (`graph.bin`) via the
// existing `save_direct` / `load_direct` helpers; the manifest itself only carries the
// resulting [`diskann_record::save::Handle`] under the `"graph"` key.

/// On-disk filename used for the adjacency-list blob inside the manifest directory.
const GRAPH_ARTIFACT: &str = "graph.bin";

impl diskann_record::save::Save for SimpleNeighborProviderAsync<u32> {
    const VERSION: diskann_record::Version = diskann_record::Version::new(0, 0, 0);

    fn save(
        &self,
        context: diskann_record::save::Context<'_>,
    ) -> diskann_record::save::Result<diskann_record::save::Record<'_>> {
        let mut writer = context.write(GRAPH_ARTIFACT)?;
        {
            let shim = crate::storage::SingleUseWriteProvider::new(GRAPH_ARTIFACT, &mut writer);
            // The canonical graph file format records a `start_point` in its header that
            // `load_direct` reads but discards. The graph's start point is tracked
            // separately by `StartPoints` and persisted at the `DefaultProvider` level, so
            // we write `0` here as a placeholder. Round-tripped `SimpleNeighborProviderAsync`
            // values do not carry a start point on their own.
            self.save_direct(&shim, 0, GRAPH_ARTIFACT)
                .map_err(diskann_record::save::Error::new)?;
        }
        let handle = writer.finish()?;
        let mut record = diskann_record::save::Record::empty();
        record.insert("graph", handle)?;
        Ok(record)
    }
}

impl diskann_record::load::Load<'_> for SimpleNeighborProviderAsync<u32> {
    const VERSION: diskann_record::Version = diskann_record::Version::new(0, 0, 0);

    fn load(object: diskann_record::load::Object<'_>) -> diskann_record::load::Result<Self> {
        diskann_record::load_fields!(object, [graph: diskann_record::save::Handle]);
        let mut reader = object.read(&graph)?;
        let shim = crate::storage::SingleUseReadProvider::new(GRAPH_ARTIFACT, &mut reader)
            .map_err(diskann_record::load::Error::new)?;
        Self::load_direct(&shim, GRAPH_ARTIFACT).map_err(diskann_record::load::Error::new)
    }

    fn load_legacy(
        _object: diskann_record::load::Object<'_>,
    ) -> diskann_record::load::Result<Self> {
        Err(diskann_record::load::error::Kind::UnknownVersion.into())
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
        let neighbor_provider = SimpleNeighborProviderAsync::<u32>::new(10, 1, 5, 1.0);

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
            SimpleNeighborProviderAsync::<u32>::new(max_points, additional_points, max_degree, 1.0);

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

        let receiver =
            SimpleNeighborProviderAsync::<u32>::load_direct(&storage, prefix.prefix()).unwrap();

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

    /////////////////////////////////
    // diskann-record round-trips //
    /////////////////////////////////

    type TestNeighborProvider = SimpleNeighborProviderAsync<u32>;

    fn round_trip_helper(provider: &TestNeighborProvider) -> TestNeighborProvider {
        let dir = tempfile::tempdir().expect("tempdir");
        let manifest = dir.path().join("manifest.json");
        diskann_record::save::save_to_disk(provider, dir.path(), &manifest).expect("save_to_disk");
        diskann_record::load::load_from_disk::<TestNeighborProvider>(&manifest, dir.path())
            .expect("load_from_disk")
    }

    fn assert_adjacency_lists_match(
        left: &TestNeighborProvider,
        right: &TestNeighborProvider,
        total: usize,
    ) {
        for i in 0..total {
            let mut l = AdjacencyList::new();
            let mut r = AdjacencyList::new();
            left.get_neighbors_sync(i, &mut l).unwrap();
            right.get_neighbors_sync(i, &mut r).unwrap();
            assert_eq!(
                l, r,
                "adjacency list for node {} differs after round-trip",
                i
            );
        }
    }

    #[test]
    fn simple_neighbor_provider_round_trips_default_initialised() {
        let max_points = 4;
        let additional_points = 1;
        let provider = TestNeighborProvider::new(max_points, additional_points, 5, 1.0);
        let restored = round_trip_helper(&provider);
        assert_adjacency_lists_match(&provider, &restored, max_points + additional_points);
    }

    #[test]
    fn simple_neighbor_provider_round_trips_populated() {
        let max_points = 8;
        let additional_points = 2;
        let max_degree = 5;
        let provider = TestNeighborProvider::new(max_points, additional_points, max_degree, 1.0);
        for i in 0..max_points + additional_points {
            let neighbors: Vec<u32> = (1..4).map(|j| i as u32 + j).collect();
            provider.set_neighbors_sync(i, &neighbors).unwrap();
        }
        let restored = round_trip_helper(&provider);
        assert_adjacency_lists_match(&provider, &restored, max_points + additional_points);
    }

    #[test]
    fn simple_neighbor_provider_round_trips_preserves_row_width() {
        // The on-disk format records `max_degree = dim - 1`, so a non-unit slack factor
        // round-trips through `save_direct` / `load_direct` even though the loader uses
        // `slack = 1.0`.
        let max_points = 3;
        let additional_points = 1;
        let max_degree = 4;
        let slack = 1.5;
        let provider = TestNeighborProvider::new(max_points, additional_points, max_degree, slack);
        // dim = (max_degree * slack) as usize + 1 = 7
        let expected_dim = (max_degree as f32 * slack) as usize + 1;
        // Fill each row to its inflated capacity to make sure the wider row width matters.
        for i in 0..max_points + additional_points {
            let neighbors: Vec<u32> = (0..(expected_dim - 1) as u32)
                .map(|j| (i as u32 * 10) + j)
                .collect();
            provider.set_neighbors_sync(i, &neighbors).unwrap();
        }
        let restored = round_trip_helper(&provider);
        assert_adjacency_lists_match(&provider, &restored, max_points + additional_points);
    }

    #[test]
    fn simple_neighbor_provider_round_trips_with_jagged_adjacency_lists() {
        let max_points = 6;
        let additional_points = 1;
        let provider = TestNeighborProvider::new(max_points, additional_points, 5, 1.0);
        // Mix of empty, short, and full adjacency lists.
        provider.set_neighbors_sync(0, &[]).unwrap();
        provider.set_neighbors_sync(1, &[10]).unwrap();
        provider.set_neighbors_sync(2, &[20, 21]).unwrap();
        provider.set_neighbors_sync(3, &[30, 31, 32, 33]).unwrap();
        // Leave id=4, 5, 6 with the default-empty adjacency lists.
        let restored = round_trip_helper(&provider);
        assert_adjacency_lists_match(&provider, &restored, max_points + additional_points);
    }
}

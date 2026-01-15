/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::{cmp::min, collections::VecDeque, sync::Arc, time::Instant};

use diskann::{utils::TryIntoVectorId, ANNError, ANNResult};
use diskann_providers::{common::AlignedBoxWithSlice, model::graph::traits::GraphDataType};
use hashbrown::HashSet;
use tracing::info;

use crate::{
    data_model::{Cache, CachingStrategy, GraphHeader},
    search::{
        provider::{
            cached_disk_vertex_provider::CachedDiskVertexProvider,
            disk_vertex_provider::DiskVertexProvider,
        },
        traits::{VertexProvider, VertexProviderFactory},
    },
    utils::aligned_file_reader::{
        traits::{AlignedFileReader, AlignedReaderFactory},
        AlignedRead,
    },
};

const DEFAULT_DISK_SECTOR_LEN: usize = 4096;
const BEAM_WIDTH_FOR_BFS: usize = 32;

/// DiskVertexProviderFactory. This is one of the implementations for the `VertexProviderFactory` trait.
pub struct DiskVertexProviderFactory<
    Data: GraphDataType<VectorIdType = u32>,
    ReaderFactory: AlignedReaderFactory,
> {
    pub aligned_reader_factory: ReaderFactory,
    pub caching_strategy: CachingStrategy,
    pub cache: Option<Arc<Cache<Data>>>,
}

/// DiskVertexProviderFactory. This is one of the implementations for the `VertexProviderFactory` trait, for which the associated graph data is read from disk.
impl<Data, ReaderFactory> VertexProviderFactory<Data>
    for DiskVertexProviderFactory<Data, ReaderFactory>
where
    ReaderFactory: AlignedReaderFactory,
    Data: GraphDataType<VectorIdType = u32>,
{
    type VertexProviderType = CachedDiskVertexProvider<Data, ReaderFactory::AlignedReaderType>;

    fn get_header(&self) -> ANNResult<GraphHeader> {
        // Here we still need the hardcoded len, because the length of the read_buf needs to be the multiple times of DEFAULT_DISK_SECTOR_LEN.
        // since this is the implementation for the disk vertex provider, there're only two kinds of sector lengths: 4096 and 512.
        // it's okay to hardcoded at this place.
        let buffer_len = GraphHeader::get_size().next_multiple_of(DEFAULT_DISK_SECTOR_LEN);
        let mut read_buf = AlignedBoxWithSlice::<u8>::new(buffer_len, buffer_len)?;
        let aligned_read = AlignedRead::new(0_u64, read_buf.as_mut_slice())?;
        self.aligned_reader_factory
            .build()?
            .read(&mut [aligned_read])?;

        // Create a GraphHeader from the buffer.
        GraphHeader::try_from(&read_buf.as_slice()[8..])
    }

    fn create_vertex_provider(
        &self,
        max_batch_size: usize,
        header: &GraphHeader,
    ) -> ANNResult<Self::VertexProviderType> {
        let sector_reader = self.aligned_reader_factory.build()?;
        match self.caching_strategy {
            CachingStrategy::StaticCacheWithBfsNodes(_) => match self.cache {
                Some(ref cache) => CachedDiskVertexProvider::new(
                    header,
                    max_batch_size,
                    sector_reader,
                    cache.clone(),
                ),
                None => Err(ANNError::log_index_error(
                    "Cache must be initialised for StaticCacheWithBfsNodes caching strategy",
                )),
            },
            CachingStrategy::None => CachedDiskVertexProvider::new(
                header,
                max_batch_size,
                sector_reader,
                Arc::new(Cache::new(0, 0)?),
            ),
        }
    }
}

impl<Data: GraphDataType<VectorIdType = u32>, ReaderFactory: AlignedReaderFactory>
    DiskVertexProviderFactory<Data, ReaderFactory>
{
    /// Creates a DiskVertexProviderFactory instance.
    pub fn new(
        aligned_reader_factory: ReaderFactory,
        caching_strategy: CachingStrategy,
    ) -> ANNResult<Self> {
        let mut disk_vertex_provider_factory = DiskVertexProviderFactory {
            aligned_reader_factory,
            caching_strategy,
            cache: None,
        };

        if disk_vertex_provider_factory.caching_strategy != CachingStrategy::None {
            disk_vertex_provider_factory.setup_cache()?;
        }

        Ok(disk_vertex_provider_factory)
    }

    fn create_disk_vertex_provider(
        &self,
        max_batch_size: usize,
        header: &GraphHeader,
    ) -> ANNResult<DiskVertexProvider<Data, ReaderFactory::AlignedReaderType>> {
        DiskVertexProvider::new(header, max_batch_size, self.aligned_reader_factory.build()?)
    }

    fn setup_cache(&mut self) -> ANNResult<()> {
        let timer = Instant::now();

        match self.caching_strategy {
            CachingStrategy::StaticCacheWithBfsNodes(mut num_nodes_to_cache) => {
                if num_nodes_to_cache == 0 {
                    ANNError::log_index_error(
                        "num_nodes_to_cache should be greater than 0 for StaticCacheWithBfsNodes caching strategy",
                    );
                }

                let graph_metadata = self.get_header()?;
                let graph_metadata = graph_metadata.metadata();
                let memory_aligned_dimension = graph_metadata.dims.next_multiple_of(8);

                if num_nodes_to_cache > graph_metadata.num_pts as usize {
                    info!(
                        "Reducing nodes to cache from: {} to: {} (total no. of nodes)",
                        num_nodes_to_cache, graph_metadata.num_pts
                    );
                    num_nodes_to_cache = graph_metadata.num_pts as usize;
                }

                let start_node = graph_metadata.medoid as u32;
                self.cache = Some(Arc::new(self.build_cache_via_bfs(
                    start_node,
                    num_nodes_to_cache,
                    memory_aligned_dimension,
                )?));
            }
            CachingStrategy::None => {}
        }

        info!("Cache setup took: {} ms", timer.elapsed().as_millis());
        Ok(())
    }

    fn build_cache_via_bfs(
        &self,
        start_node: u32,
        num_nodes_to_cache: usize,
        dimension: usize,
    ) -> ANNResult<Cache<Data>> {
        info!("Building cache with {} nodes via BFS.", num_nodes_to_cache);
        let mut cache = Cache::new(dimension, num_nodes_to_cache)?;
        let mut vertex_provider =
            self.create_disk_vertex_provider(BEAM_WIDTH_FOR_BFS, &self.get_header()?)?;

        let mut visited = HashSet::with_capacity(num_nodes_to_cache);
        let mut queue = VecDeque::with_capacity(num_nodes_to_cache);
        let mut nodes_in_a_batch = Vec::with_capacity(BEAM_WIDTH_FOR_BFS);

        queue.push_back(start_node);
        visited.insert(start_node);

        while (!queue.is_empty()) && cache.len() < num_nodes_to_cache {
            nodes_in_a_batch.clear();
            let batch_size = min(queue.len(), BEAM_WIDTH_FOR_BFS);
            for _ in 0..batch_size {
                let node = queue.pop_front().ok_or_else(|| {
                    ANNError::log_index_error("Error while caching Nodes via BFS: Queue is empty")
                })?;
                nodes_in_a_batch.push(node.try_into_vector_id().map_err(ANNError::from)?);
            }

            vertex_provider.load_vertices(&nodes_in_a_batch)?;

            for (idx, node) in nodes_in_a_batch.iter().enumerate() {
                Self::insert_in_cache(node, idx, &mut vertex_provider, &mut cache)?;
                let adjacency_list = cache.get_adjacency_list(node).ok_or_else(|| {
                    ANNError::log_index_error(format!("Error while caching Nodes via BFS: Adjacency List not found for inserted node {} in cache.", node))
                })?;
                for neighbor_id in adjacency_list.iter() {
                    if !visited.contains(neighbor_id) {
                        queue.push_back(*neighbor_id);
                        visited.insert(*neighbor_id);
                    }
                }
                if cache.len() >= num_nodes_to_cache {
                    break;
                }
            }
        }

        ANNResult::Ok(cache)
    }

    fn insert_in_cache<AlignedReaderType>(
        node: &Data::VectorIdType,
        idx: usize,
        vertex_provider: &mut DiskVertexProvider<Data, AlignedReaderType>,
        cache: &mut Cache<Data>,
    ) -> ANNResult<()>
    where
        AlignedReaderType: AlignedFileReader,
    {
        vertex_provider.process_loaded_node(node, idx)?;
        let vector = vertex_provider.get_vector(node)?;
        let adjacency_list = vertex_provider.get_adjacency_list(node)?;
        let associated_data = vertex_provider.get_associated_data(node)?;

        cache.insert(node, vector, adjacency_list.clone(), *associated_data)
    }
}

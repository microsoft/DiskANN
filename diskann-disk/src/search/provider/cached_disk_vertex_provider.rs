/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::Arc;

use diskann::{ANNError, ANNResult};
use diskann_providers::model::graph::{graph_data_model::AdjacencyList, traits::GraphDataType};

use crate::utils::aligned_file_reader::traits::AlignedFileReader;
use hashbrown::HashMap;

use crate::{
    data_model::{Cache, GraphHeader},
    search::{provider::disk_vertex_provider::DiskVertexProvider, traits::VertexProvider},
};

pub struct CachedDiskVertexProvider<Data, AlignedReaderType>
where
    Data: GraphDataType<VectorIdType = u32>,
    AlignedReaderType: AlignedFileReader,
{
    // Global shared static cache.
    cache: Arc<Cache<Data>>,

    // The underlying disk vertex provider to read from disk.
    vector_provider: DiskVertexProvider<Data, AlignedReaderType>,

    // Maintains the mapping of local index of the uncached vertices to index in filtered list
    // after removing cached nodes.
    nodes_to_fetch_local_idx_to_filtered_idx: HashMap<usize, usize>,

    // The number of vertices loaded by this provider.
    vertices_loaded_count: u32,
}

impl<Data, AlignedReaderType> VertexProvider<Data>
    for CachedDiskVertexProvider<Data, AlignedReaderType>
where
    Data: GraphDataType<VectorIdType = u32>,
    AlignedReaderType: AlignedFileReader,
{
    fn get_vector(
        &self,
        vertex_id: &Data::VectorIdType,
    ) -> ANNResult<&[<Data as GraphDataType>::VectorDataType]> {
        match self.cache.get_vector(vertex_id) {
            Some(vector) => Ok(vector),
            None => self.vector_provider.get_vector(vertex_id),
        }
    }

    fn get_adjacency_list(
        &self,
        vertex_id: &Data::VectorIdType,
    ) -> ANNResult<&AdjacencyList<Data::VectorIdType>> {
        match self.cache.get_adjacency_list(vertex_id) {
            Some(adj_list) => Ok(adj_list),
            None => self.vector_provider.get_adjacency_list(vertex_id),
        }
    }

    fn get_associated_data(
        &self,
        vertex_id: &Data::VectorIdType,
    ) -> ANNResult<&Data::AssociatedDataType> {
        match self.cache.get_associated_data(vertex_id) {
            Some(associated_data) => Ok(associated_data),
            None => self.vector_provider.get_associated_data(vertex_id),
        }
    }

    fn load_vertices(&mut self, vertex_ids: &[Data::VectorIdType]) -> ANNResult<()> {
        self.clear_before_next_read();

        self.vertices_loaded_count += vertex_ids.len() as u32;

        let uncached_vertex_ids = self.filter_cached_nodes(vertex_ids)?;
        if uncached_vertex_ids.is_empty() {
            return ANNResult::Ok(());
        }

        self.vector_provider.load_vertices(&uncached_vertex_ids)
    }

    fn process_loaded_node(
        &mut self,
        vertex_id: &Data::VectorIdType,
        mut idx: usize,
    ) -> Result<(), ANNError> {
        if self.cache.contains(vertex_id) {
            return Ok(());
        }
        idx = self.nodes_to_fetch_local_idx_to_filtered_idx[&idx];
        self.vector_provider.process_loaded_node(vertex_id, idx)
    }

    fn io_operations(&self) -> u32 {
        self.vector_provider.io_operations()
    }

    fn clear(&mut self) {
        self.clear_before_next_read();
        self.vector_provider.clear();
        self.vertices_loaded_count = 0;
    }

    fn vertices_loaded_count(&self) -> u32 {
        self.vertices_loaded_count
    }
}

impl<Data, AlignedReaderType> CachedDiskVertexProvider<Data, AlignedReaderType>
where
    Data: GraphDataType<VectorIdType = u32>,
    AlignedReaderType: AlignedFileReader,
{
    /// Create new CachedDiskVertexProvider instance.
    pub fn new(
        header: &GraphHeader,
        max_batch_size: usize,
        sector_reader: AlignedReaderType,
        cache: Arc<Cache<Data>>,
    ) -> ANNResult<Self> {
        let vector_provider = DiskVertexProvider::new(header, max_batch_size, sector_reader)?;
        Ok(CachedDiskVertexProvider {
            cache,
            vector_provider,
            nodes_to_fetch_local_idx_to_filtered_idx: HashMap::new(),
            vertices_loaded_count: 0,
        })
    }

    // Filter cached nodes from the list of vertex ids leaving uncached nodes.
    fn filter_cached_nodes(
        &mut self,
        vertex_ids: &[Data::VectorIdType],
    ) -> ANNResult<Vec<Data::VectorIdType>> {
        let mut uncached_vertex_ids = Vec::new();

        for (idx, vertex_id) in vertex_ids.iter().enumerate() {
            if !self.cache.contains(vertex_id) {
                self.nodes_to_fetch_local_idx_to_filtered_idx
                    .insert(idx, uncached_vertex_ids.len());
                uncached_vertex_ids.push(*vertex_id);
            }
        }

        Ok(uncached_vertex_ids)
    }

    fn clear_before_next_read(&mut self) {
        self.nodes_to_fetch_local_idx_to_filtered_idx.clear();
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_struct_definition() {
        // Struct definition is verified at compile time
    }
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::ptr;

use byteorder::{ByteOrder, LittleEndian};
use diskann::{ANNError, ANNResult};
use diskann_providers::{
    common::AlignedBoxWithSlice,
    model::{
        graph::{graph_data_model::AdjacencyList, traits::GraphDataType},
        FP_VECTOR_MEM_ALIGN,
    },
};
use hashbrown::HashMap;

use crate::{
    data_model::GraphHeader,
    search::{provider::disk_sector_graph::DiskSectorGraph, traits::VertexProvider},
    utils::aligned_file_reader::traits::AlignedFileReader,
};

struct Offsets {
    vec_idx: usize,
    idx: usize,
}

pub struct DiskVertexProvider<Data, AlignedReaderType>
where
    Data: GraphDataType<VectorIdType = u32>,
    AlignedReaderType: AlignedFileReader,
{
    /// Centroid vertex id.
    pub centroid_vertex_id: u64,

    /// Memory-aligned dimension.  In-memory the vectors should be this size.
    memory_aligned_dimension: usize,

    /// the len of fp vector
    fp_vector_len: u64,

    // sector graph
    sector_graph: DiskSectorGraph<AlignedReaderType>,

    // Aligned fp vector cache
    aligned_vector_buf: AlignedBoxWithSlice<Data::VectorDataType>,

    // The cached adjacency list.
    cached_adjacency_list: Vec<AdjacencyList<Data::VectorIdType>>,

    // The cached associated data.
    cached_associated_data: Vec<Data::AssociatedDataType>,

    // A hashmap containing the loaded vertex_ids to local data offsets.
    loaded_nodes: HashMap<Data::VectorIdType, Offsets>,

    /// The size of the data associated with each vertex.
    associated_data_size: usize,

    // The length of a node.
    node_len: u64,

    // No of IO operations performed via this provider.
    io_operations: u32,

    // Max nodes that can be loaded in a single batch.
    max_batch_size: usize,
}

impl<Data, AlignedReaderType> VertexProvider<Data> for DiskVertexProvider<Data, AlignedReaderType>
where
    Data: GraphDataType<VectorIdType = u32>,
    AlignedReaderType: AlignedFileReader,
{
    fn get_vector(
        &self,
        vertex_id: &Data::VectorIdType,
    ) -> ANNResult<&[<Data as GraphDataType>::VectorDataType]> {
        match self.loaded_nodes.get(vertex_id) {
            Some(local_offset) => Ok(&self.aligned_vector_buf[local_offset.idx
                * self.memory_aligned_dimension
                ..(local_offset.idx * self.memory_aligned_dimension)
                    + self.memory_aligned_dimension]),
            None => Err(ANNError::log_get_vertex_data_error(
                vertex_id.to_string(),
                "Vector".to_string(),
            )),
        }
    }

    fn get_adjacency_list(
        &self,
        vertex_id: &Data::VectorIdType,
    ) -> ANNResult<&AdjacencyList<Data::VectorIdType>> {
        match self.loaded_nodes.get(vertex_id) {
            Some(local_offset) => Ok(&self.cached_adjacency_list[local_offset.vec_idx]),
            None => Err(ANNError::log_get_vertex_data_error(
                vertex_id.to_string(),
                "AdjacencyList".to_string(),
            )),
        }
    }

    fn get_associated_data(
        &self,
        vertex_id: &Data::VectorIdType,
    ) -> ANNResult<&Data::AssociatedDataType> {
        match self.loaded_nodes.get(vertex_id) {
            Some(local_offset) => Ok(&self.cached_associated_data[local_offset.vec_idx]),
            None => Err(ANNError::log_get_vertex_data_error(
                vertex_id.to_string(),
                "AssociatedData".to_string(),
            )),
        }
    }

    fn load_vertices(&mut self, vertex_ids: &[Data::VectorIdType]) -> ANNResult<()> {
        self.clear_before_next_read();
        self.fetch_nodes(vertex_ids)?;
        self.io_operations += vertex_ids.len() as u32;
        Ok(())
    }

    fn process_loaded_node(
        &mut self,
        vertex_id: &Data::VectorIdType,
        idx: usize,
    ) -> Result<(), ANNError> {
        let fp_vector_buf =
            &self.sector_graph.node_disk_buf(idx, *vertex_id)[..self.fp_vector_len as usize];

        // memcpy from fp_vector_buf to the aligned buffer..
        // The safe condition is met here since the dimension of the vector in fp_vector_buffer is the same with aligned_vector_buffer.
        // fp_vector_buf and aligned_vector_buffer.as_mut_ptr() are guaranteed to be non-overlapping.
        unsafe {
            ptr::copy_nonoverlapping(
                fp_vector_buf.as_ptr(),
                self.aligned_vector_buf[idx * self.memory_aligned_dimension
                    ..(idx * self.memory_aligned_dimension) + self.memory_aligned_dimension]
                    .as_mut_ptr() as *mut u8,
                fp_vector_buf.len(),
            );
        }
        let neighbor_and_data_buf =
            &self.sector_graph.node_disk_buf(idx, *vertex_id)[self.fp_vector_len as usize..];
        let num_neighbors = LittleEndian::read_u32(&neighbor_and_data_buf[0..4]) as usize;
        let neighbor_buf = &neighbor_and_data_buf[..4 + num_neighbors * 4];
        let adjacency_list = AdjacencyList::try_from(neighbor_buf)?;
        self.cached_adjacency_list.push(adjacency_list);

        let data_end: usize = (self.node_len - self.fp_vector_len) as usize;
        let associated_data = bincode::deserialize(
            &neighbor_and_data_buf[data_end - self.associated_data_size..data_end],
        )
        .map_err(|err| {
            ANNError::log_serde_error(
                "Error deserializing associated data from bytes".to_string(),
                *err,
            )
        })?;
        self.cached_associated_data.push(associated_data);

        // due to async nature i/o operation finished: vec_idx is different from request issue index: idx
        let vec_idx = self.loaded_nodes.len();
        self.loaded_nodes
            .insert(*vertex_id, Offsets { vec_idx, idx });

        Ok(())
    }

    fn io_operations(&self) -> u32 {
        self.io_operations
    }

    fn clear(&mut self) {
        self.clear_before_next_read();
        self.io_operations = 0;
    }

    fn vertices_loaded_count(&self) -> u32 {
        self.io_operations
    }
}

impl<Data, AlignedReaderType> DiskVertexProvider<Data, AlignedReaderType>
where
    Data: GraphDataType<VectorIdType = u32>,
    AlignedReaderType: AlignedFileReader,
{
    /// Create DiskGraph instance.
    pub fn new(
        header: &GraphHeader,
        max_batch_size: usize,
        sector_reader: AlignedReaderType,
    ) -> ANNResult<Self> {
        let metadata = header.metadata();
        let memory_aligned_dimension = metadata.dims.next_multiple_of(8);
        Ok(Self {
            centroid_vertex_id: metadata.medoid,
            memory_aligned_dimension,
            fp_vector_len: (metadata.dims * std::mem::size_of::<Data::VectorDataType>()) as u64,
            sector_graph: DiskSectorGraph::new(sector_reader, header, max_batch_size)?,

            aligned_vector_buf: AlignedBoxWithSlice::new(
                max_batch_size * memory_aligned_dimension,
                FP_VECTOR_MEM_ALIGN,
            )?,
            cached_adjacency_list: Vec::with_capacity(max_batch_size),
            cached_associated_data: Vec::with_capacity(max_batch_size),
            loaded_nodes: HashMap::with_capacity(max_batch_size),
            associated_data_size: metadata.associated_data_length,
            node_len: metadata.node_len,
            io_operations: 0,
            max_batch_size,
        })
    }

    fn reconfigure(&mut self, max_batch_size: usize) -> ANNResult<()> {
        if max_batch_size > self.max_batch_size {
            self.clear_before_next_read();
            self.sector_graph.reconfigure(max_batch_size)?;
            self.cached_adjacency_list.reserve(max_batch_size);
            self.cached_associated_data.reserve(max_batch_size);
            self.loaded_nodes.reserve(max_batch_size);
            self.aligned_vector_buf = AlignedBoxWithSlice::new(
                max_batch_size * self.memory_aligned_dimension,
                FP_VECTOR_MEM_ALIGN,
            )?;
            self.max_batch_size = max_batch_size;
        }
        Ok(())
    }

    /// Fetch nodes from disk index
    fn fetch_nodes(&mut self, nodes_to_fetch: &[Data::VectorIdType]) -> ANNResult<()> {
        self.reconfigure(nodes_to_fetch.len())?;
        let sectors_to_fetch: Vec<u64> = nodes_to_fetch
            .iter()
            .map(|&vertex_id| self.sector_graph.node_sector_index(vertex_id))
            .collect();

        self.sector_graph.read_graph(&sectors_to_fetch)?;

        Ok(())
    }

    /// Reset graph
    fn clear_before_next_read(&mut self) {
        self.sector_graph.reset();
        self.loaded_nodes.clear();
        self.cached_adjacency_list.clear();
        self.cached_associated_data.clear();
    }

    pub fn memory_aligned_dimension(&self) -> usize {
        self.memory_aligned_dimension
    }
}

#[cfg(test)]
mod disk_vertex_provider_tests {
    use std::sync::Arc;

    use diskann::{graph::config, utils::ONE};
    use diskann_providers::storage::{
        StorageReadProvider, StorageWriteProvider, VirtualStorageProvider,
    };
    use diskann_providers::{
        model::{graph::traits::GraphDataType, IndexConfiguration},
        storage::get_disk_index_file,
        test_utils::graph_data_type_utils::GraphDataF32VectorU32Data,
        utils::load_metadata_from_file,
    };
    use diskann_utils::test_data_root;
    use vfs::OverlayFS;

    use crate::{
        build::builder::build::DiskIndexBuilder,
        data_model::{CachingStrategy, GraphHeader},
        disk_index_build_parameter::{
            DiskIndexBuildParameters, MemoryBudget, NumPQChunks, DISK_SECTOR_LEN,
        },
        search::{
            provider::disk_vertex_provider_factory::DiskVertexProviderFactory,
            traits::{VertexProvider, VertexProviderFactory},
        },
        storage::DiskIndexWriter,
        utils::VirtualAlignedReaderFactory,
        QuantizationType,
    };

    fn generate_disk_index_with_associated_data<StorageProviderType>(
        storage_provider: &StorageProviderType,
        index_path_prefix: &str,
    ) where
        StorageProviderType: StorageReadProvider + StorageWriteProvider,
        <StorageProviderType as StorageReadProvider>::Reader: std::marker::Send,
        StorageProviderType: 'static,
    {
        let max_degree = 4;
        let l_build = 50;
        let data_path = "/disk_index_search/disk_index_siftsmall_learn_256pts_data.fbin";
        let associated_data_path = "/sift/siftsmall_learn_256pts_u32_associated_data.fbin";

        let metadata = load_metadata_from_file(storage_provider, data_path).unwrap();

        let memory_budget = MemoryBudget::try_from_gb(1.0).unwrap();
        let num_pq_chunks = NumPQChunks::new_with(128, metadata.ndims).unwrap();

        let disk_index_build_parameters =
            DiskIndexBuildParameters::new(memory_budget, QuantizationType::FP, num_pq_chunks);

        let config = config::Builder::new_with(
            max_degree,
            config::MaxDegree::default_slack(),
            l_build,
            diskann_vector::distance::Metric::L2.into(),
            |b| {
                b.saturate_after_prune(true);
            },
        )
        .build()
        .unwrap();

        let config = IndexConfiguration::new(
            diskann_vector::distance::Metric::L2,
            metadata.ndims,
            metadata.npoints,
            ONE,
            1,
            config,
        );
        let disk_index_writer = DiskIndexWriter::new(
            data_path.to_string(),
            index_path_prefix.to_string(),
            Some(associated_data_path.to_string()),
            DISK_SECTOR_LEN,
        )
        .unwrap();

        let mut disk_index: DiskIndexBuilder<GraphDataF32VectorU32Data, StorageProviderType> =
            DiskIndexBuilder::<GraphDataF32VectorU32Data, StorageProviderType>::new(
                storage_provider,
                disk_index_build_parameters,
                config,
                disk_index_writer,
            )
            .unwrap();

        let mem_index_file_path = format!("{}_mem.index.data", index_path_prefix);
        let mem_index_associated_data_path =
            format!("{}_mem.index.associated_data", index_path_prefix);

        if storage_provider.exists(&mem_index_file_path) {
            storage_provider
                .delete(&mem_index_file_path)
                .expect("Failed to delete mem index file");
        }

        if storage_provider.exists(&mem_index_associated_data_path) {
            storage_provider
                .delete(&mem_index_associated_data_path)
                .expect("Failed to delete mem index associated data file");
        }

        disk_index.build().unwrap();

        // Assert that all data was kept in memory and no files were written to the disk.
        assert!(!storage_provider.exists(&mem_index_file_path));
        assert!(!storage_provider.exists(&mem_index_associated_data_path));

        storage_provider
            .delete(&format!("{}_pq_pivots.bin", index_path_prefix))
            .expect("Failed to delete file");
        storage_provider
            .delete(&format!("{}_pq_compressed.bin", index_path_prefix))
            .expect("Failed to delete file");
    }

    #[test]
    fn test_disk_index_with_associated_data() {
        let storage_provider = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));
        let index_path_prefix = "/disk_index_search/disk_index_sift_learn_R4_L50_A1.2_test_disk_index_with_associated_data";
        generate_disk_index_with_associated_data(storage_provider.as_ref(), index_path_prefix);

        // These extra braces are important for passing this test on windows as without them the bellow remove file function call will fail.
        {
            let vertex_provider_factory = DiskVertexProviderFactory::new(
                VirtualAlignedReaderFactory::new(
                    get_disk_index_file(index_path_prefix).to_string(),
                    storage_provider.clone(),
                ),
                CachingStrategy::None,
            )
            .unwrap();
            let (mut vertex_provider, header) =
                create_disk_provider::<GraphDataF32VectorU32Data>(&vertex_provider_factory);

            let nodes = (0..256).map(|i| i as u32).collect::<Vec<u32>>();
            VertexProvider::load_vertices(&mut vertex_provider, nodes.as_slice()).unwrap();

            for (idx, vertex_id) in nodes.iter().enumerate() {
                VertexProvider::process_loaded_node(&mut vertex_provider, vertex_id, idx).unwrap();
            }

            for vertex_id in 0..header.metadata().num_pts {
                let test_vertex_id = vertex_id as u32;
                let associated_data =
                    VertexProvider::get_associated_data(&vertex_provider, &test_vertex_id).unwrap();
                assert_eq!(
                    test_vertex_id,
                    { *associated_data },
                    "vertex_id: {}, associated_data {}",
                    vertex_id,
                    { *associated_data }
                );
            }

            assert_eq!(vertex_provider.io_operations(), 256);
            assert_eq!(vertex_provider.vertices_loaded_count(), 256);
        }

        storage_provider
            .delete(&get_disk_index_file(index_path_prefix))
            .expect("Failed to delete file");
    }

    // Test is broken until proper 64-bit support is implemented.
    //
    // #[test]
    // fn test_disk_index_with_associated_data_u64_vertex_id() {
    //     let storage_provider = VirtualStorageProvider::new_overlay();
    //     let index_path_prefix = "/tests/data/disk_index_search/disk_index_sift_learn_R4_L50_A1.2_test_disk_index_with_associated_data_u64_vid";
    //     generate_disk_index_with_associated_data(storage_provider, index_path_prefix);
    //     {
    //         let reader =
    //             crate::model::build_aligned_file_reader(&get_disk_index_file(index_path_prefix))
    //                 .unwrap();
    //         let (mut vertex_provider, header) = create_disk_provider_factory::<
    //             GraphDataF32WithU64IdVectorU32Data,
    //             AlignedFileReaderType,
    //         >(index_path_prefix, &reader);

    //         let nodes = (0..256).map(|i| i as u64).collect::<Vec<u64>>();
    //         let _ = VertexProvider::load_vertices(&mut vertex_provider, nodes.as_slice());

    //         for (idx, vertex_id) in nodes.iter().enumerate() {
    //             VertexProvider::process_loaded_node(&mut vertex_provider, vertex_id, idx).unwrap();
    //         }

    //         for vertex_id in 0..header.metadata().num_pts {
    //             let test_vertex_id = vertex_id;
    //             let associated_data =
    //                 VertexProvider::get_associated_data(&vertex_provider, &test_vertex_id).unwrap();
    //             assert_eq!(
    //                 test_vertex_id, *associated_data as u64,
    //                 "vertex_id: {}, associated_data {}",
    //                 vertex_id, *associated_data as u64
    //             );
    //         }
    //     }

    //     storage_provider.delete(&get_disk_index_file(index_path_prefix)).expect("Failed to delete file");
    // }

    fn create_disk_provider<Data: GraphDataType<VectorIdType = u32>>(
        vertex_provider_factory: &DiskVertexProviderFactory<Data, VirtualAlignedReaderFactory<OverlayFS>>,
    ) -> (
        <DiskVertexProviderFactory<Data, VirtualAlignedReaderFactory<OverlayFS>> as VertexProviderFactory<
            Data,
        >>::VertexProviderType,
        GraphHeader,
    ){
        let header = vertex_provider_factory.get_header().unwrap();

        let vertex_provider = vertex_provider_factory
            .create_vertex_provider(256, &header)
            .unwrap();

        (vertex_provider, header)
    }
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::{
    marker::PhantomData,
    mem::{self, size_of},
};

use crate::data_model::GraphDataType;
use diskann::{utils::VectorRepr, ANNResult};
use diskann_providers::storage::{StorageReadProvider, StorageWriteProvider};
use diskann_providers::{
    model::{IndexConfiguration, MAX_PQ_TRAINING_SET_SIZE},
    utils::{
        load_metadata_from_file, RayonThreadPoolRef, SampleVectorReader, SamplingDensity,
        READ_WRITE_BLOCK_SIZE,
    },
};
use diskann_utils::io::read_bin;
use rand::seq::SliceRandom;
use tracing::info;

use crate::{
    build::builder::{inmem_builder::build_inmem_index, quantizer::BuildQuantizer},
    storage::{CachedReader, CachedWriter, DiskIndexWriter},
    utils::instrumentation::{BuildMergedVamanaIndexCheckpoint, PerfLogger},
    utils::partition_with_ram_budget,
    DiskIndexBuildParameters,
};

/// Number of nearest shards each vector is assigned to during partitioning.
const PARTITION_ASSIGNMENTS_PER_VECTOR: usize = 2;

/// Builds a merged Vamana index from overlapping dataset shards.
pub(super) struct MergedVamanaIndexBuilder<'a, Data, StorageProvider>
where
    Data: GraphDataType<VectorIdType = u32>,
    StorageProvider: StorageReadProvider + StorageWriteProvider,
{
    index_configuration: &'a IndexConfiguration,
    disk_build_param: &'a DiskIndexBuildParameters,
    index_writer: &'a DiskIndexWriter,
    build_quantizer: &'a BuildQuantizer,
    storage_provider: &'a StorageProvider,
    rng: diskann_providers::utils::StandardRng,
    _phantom: PhantomData<Data>,
}

impl<'a, Data, StorageProvider> MergedVamanaIndexBuilder<'a, Data, StorageProvider>
where
    Data: GraphDataType<VectorIdType = u32>,
    Data::VectorDataType: VectorRepr,
    StorageProvider: StorageReadProvider + StorageWriteProvider + 'static,
    <StorageProvider as StorageReadProvider>::Reader: Send,
{
    pub(super) fn new(
        index_configuration: &'a IndexConfiguration,
        disk_build_param: &'a DiskIndexBuildParameters,
        index_writer: &'a DiskIndexWriter,
        build_quantizer: &'a BuildQuantizer,
        storage_provider: &'a StorageProvider,
    ) -> Self {
        Self {
            index_configuration,
            disk_build_param,
            index_writer,
            build_quantizer,
            storage_provider,
            rng: diskann_providers::utils::create_rnd_from_optional_seed(
                index_configuration.random_seed,
            ),
            _phantom: PhantomData,
        }
    }

    pub(super) async fn build(mut self, pool: RayonThreadPoolRef<'_>) -> ANNResult<()> {
        let mut logger = PerfLogger::new_disk_index_build_logger();
        let dataset_file = self.index_writer.get_dataset_file();
        let merged_index_prefix = self.index_writer.get_merged_index_prefix();
        let output_vamana = self.index_writer.get_mem_index_file();
        let max_degree = self.index_configuration.config.pruned_degree_u32().get();

        let num_parts =
            self.partition_data(&dataset_file, &merged_index_prefix, max_degree, pool)?;
        logger.log_checkpoint(BuildMergedVamanaIndexCheckpoint::PartitionData);

        for shard_id in 0..num_parts {
            self.build_shard_index(&dataset_file, &merged_index_prefix, shard_id)
                .await?;
        }
        logger.log_checkpoint(BuildMergedVamanaIndexCheckpoint::BuildIndicesOnShards);

        self.merge_and_cleanup(&merged_index_prefix, num_parts, max_degree, output_vamana)?;
        logger.log_checkpoint(BuildMergedVamanaIndexCheckpoint::MergeIndices);

        Ok(())
    }

    fn create_shard_index_config(&self, shard_base_file: &str) -> ANNResult<IndexConfiguration> {
        let base_config = self.index_configuration;
        let storage_provider = self.storage_provider;

        let search_list_size = base_config.config.l_build().get();
        let pruned_degree = base_config.config.pruned_degree().get();

        let low_degree_params = diskann::graph::config::Builder::new(
            2 * pruned_degree / 3,
            diskann::graph::config::MaxDegree::default_slack(),
            search_list_size,
            base_config.dist_metric.into(),
        )
        .build()?;

        let metadata = load_metadata_from_file(storage_provider, shard_base_file)?;

        let mut index_config = (*base_config).clone();
        index_config.max_points = metadata.npoints();
        index_config.config = low_degree_params;

        Ok(index_config)
    }

    fn retrieve_shard_data_from_ids<T>(
        &self,
        dataset_file: &str,
        shard_ids_file: &str,
        shard_base_file: &str,
    ) -> ANNResult<()>
    where
        T: Default + bytemuck::Pod,
    {
        let storage_provider = self.storage_provider;
        let shard_ids = read_bin::<u32>(&mut storage_provider.open_reader(shard_ids_file)?)?;
        let shard_size = shard_ids.nrows();
        info!("Loaded {} shard ids from {}", shard_size, shard_ids_file);
        let max_id = shard_ids.as_slice().iter().max().copied().unwrap_or(0);
        let sampling_rate = shard_ids.as_slice().len() as f64 / (max_id + 1) as f64;

        let mut dataset_reader: SampleVectorReader<T, _> = SampleVectorReader::new(
            dataset_file,
            SamplingDensity::from_sample_rate(sampling_rate),
            storage_provider,
        )?;

        let (_npts, dim) = dataset_reader.get_dataset_headers();

        let mut shard_base_cached_writer = CachedWriter::<StorageProvider>::new(
            shard_base_file,
            READ_WRITE_BLOCK_SIZE,
            storage_provider.create_for_write(shard_base_file)?,
        )?;

        let dummy_size: u32 = 0;
        shard_base_cached_writer.write(&dummy_size.to_le_bytes())?;
        shard_base_cached_writer.write(&dim.to_le_bytes())?;

        let mut num_written: u32 = 0;
        dataset_reader.read_vectors(shard_ids.as_slice().iter().copied(), |vector_t| {
            // Casting Pod type to bytes always succeeds (u8 has alignment of 1)
            let vector_bytes: &[u8] = bytemuck::must_cast_slice(vector_t);
            shard_base_cached_writer.write(vector_bytes)?;
            num_written += 1;
            Ok(())
        })?;

        info!(
            "Written file: {} with {} points",
            shard_base_file, num_written
        );

        shard_base_cached_writer.flush()?;
        shard_base_cached_writer.reset()?;
        shard_base_cached_writer.write(&num_written.to_le_bytes())?;

        Ok(())
    }

    async fn build_shard_index(
        &self,
        dataset_file: &str,
        merged_index_prefix: &str,
        shard_id: usize,
    ) -> ANNResult<()> {
        let shard_base_file =
            DiskIndexWriter::get_merged_index_subshard_data_file(merged_index_prefix, shard_id);
        let shard_ids_file =
            DiskIndexWriter::get_merged_index_subshard_id_map_file(merged_index_prefix, shard_id);
        self.retrieve_shard_data_from_ids::<Data::VectorDataType>(
            dataset_file,
            &shard_ids_file,
            &shard_base_file,
        )?;
        info!("Generated data for shard {}", shard_id);

        let index_config = self.create_shard_index_config(&shard_base_file)?;
        let shard_index_file = DiskIndexWriter::get_merged_index_subshard_mem_index_file(
            merged_index_prefix,
            shard_id,
        );

        build_inmem_index::<Data::VectorDataType, _>(
            index_config,
            self.build_quantizer,
            &shard_base_file,
            &shard_index_file,
            self.storage_provider,
        )
        .await
    }

    fn merge_shards(
        &mut self,
        merged_index_prefix: &str,
        num_parts: usize,
        max_degree: u32,
        output_vamana: String,
    ) -> ANNResult<()> {
        // Read ID maps
        let mut vamana_names = vec![String::new(); num_parts];
        let mut id_maps: Vec<Vec<u32>> = vec![Vec::new(); num_parts];
        for shard in 0..num_parts {
            vamana_names[shard] = DiskIndexWriter::get_merged_index_subshard_mem_index_file(
                merged_index_prefix,
                shard,
            );

            let id_maps_file =
                DiskIndexWriter::get_merged_index_subshard_id_map_file(merged_index_prefix, shard);
            id_maps[shard] = self.read_idmap(id_maps_file)?;
        }

        // find max node id
        let num_nodes: u32 = *id_maps.iter().flatten().max().unwrap_or(&0) + 1;
        let num_elements: u32 = id_maps.iter().map(|idmap| idmap.len() as u32).sum();
        info!("# nodes: {}, max degree: {}", num_nodes, max_degree);

        // compute inverse map: node -> shards
        let mut node_shard: Vec<(u32, u32)> = Vec::with_capacity(num_elements as usize);
        for (shard, id_map) in id_maps.iter().enumerate() {
            info!("Creating inverse map -- shard #{}", shard);
            node_shard.extend(id_map.iter().map(|node_id| (*node_id, shard as u32)));
        }
        node_shard.sort_unstable_by(|left, right| {
            left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1))
        });

        info!("Finished computing node -> shards map");

        // create cached vamana readers
        let mut vamana_readers = Vec::new();
        for name in &vamana_names {
            let reader = CachedReader::<StorageProvider>::new(
                name,
                READ_WRITE_BLOCK_SIZE,
                self.storage_provider,
            )?;
            vamana_readers.push(reader);
        }

        // create cached vamana writers
        let mut merged_vamana_cached_writer = CachedWriter::<StorageProvider>::new(
            &output_vamana,
            READ_WRITE_BLOCK_SIZE,
            self.storage_provider.create_for_write(&output_vamana)?,
        )?;

        // expected file size + max degree + medoid_id + frozen_point info
        let vamana_metadata_size =
            size_of::<u64>() + size_of::<u32>() + size_of::<u32>() + size_of::<u64>();

        // we initialize the size of the merged index to the metadata size
        // we will overwrite the index size at the end
        let mut merged_index_size: u64 = vamana_metadata_size as u64;
        merged_vamana_cached_writer.write(&merged_index_size.to_le_bytes())?;

        let mut read_buf_8_bytes = [0u8; 8];

        // get max input width
        let mut max_input_width = 0;
        // read width from each vamana to advance buffer by sizeof(uint32_t) bytes
        for reader in &mut vamana_readers {
            reader.read(&mut read_buf_8_bytes)?;
            let _expected_file_size: u64 = u64::from_le_bytes(read_buf_8_bytes);
            let input_width = reader.read_u32()?;
            max_input_width = input_width.max(max_input_width);
        }

        // write max_degree to merged_vamana_index
        let output_width: u32 = max_degree;
        info!(
            "Max input width: {}, output width: {}",
            max_input_width, output_width
        );

        merged_vamana_cached_writer.write(&output_width.to_le_bytes())?;

        // write medoid to merged_vamana_index
        for shard in 0..num_parts {
            // read medoid
            let mut medoid: u32 = vamana_readers[shard].read_u32()?;
            vamana_readers[shard].read(&mut read_buf_8_bytes)?;
            let vamana_index_frozen: u64 = u64::from_le_bytes(read_buf_8_bytes);
            debug_assert_eq!(vamana_index_frozen, 0);

            // rename medoid
            medoid = id_maps[shard][medoid as usize];

            // write renamed medoid
            if shard == (num_parts - 1) {
                // uncomment if running hierarchical
                merged_vamana_cached_writer.write(&medoid.to_le_bytes())?;
            }
        }

        let vamana_index_frozen: u64 = 0; // as of now the functionality to merge many overlapping vamana
                                          // indices is supported only for bulk indices without frozen point.
                                          // Hence the final index will also not have any frozen points.
        merged_vamana_cached_writer.write(&vamana_index_frozen.to_le_bytes())?;

        info!("Starting merge");

        let mut nbr_set = vec![false; num_nodes as usize];
        let mut final_nbrs: Vec<u32> = Vec::new();
        let mut cur_id = 0;
        for pair in &node_shard {
            let (node_id, shard_id) = *pair;
            if cur_id < node_id {
                final_nbrs.shuffle(&mut self.rng);

                let nnbrs: u32 = std::cmp::min(final_nbrs.len() as u32, max_degree);
                merged_vamana_cached_writer.write(&nnbrs.to_le_bytes())?;

                let bytes = final_nbrs
                    .iter()
                    .take(nnbrs as usize)
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<u8>>();
                merged_vamana_cached_writer.write(&bytes)?;

                merged_index_size += (size_of::<u32>() + nnbrs as usize * size_of::<u32>()) as u64;
                if cur_id % 499999 == 1 {
                    print!(".");
                }
                cur_id = node_id;

                final_nbrs.iter().for_each(|p| nbr_set[*p as usize] = false);
                final_nbrs.clear();
            }

            // read num of neighbors from vamana index
            let num_nbrs = vamana_readers[shard_id as usize].read_u32()?;

            if num_nbrs == 0 {
                info!(
                    "WARNING: shard #{}, node_id {} has 0 nbrs",
                    shard_id, node_id
                );
            } else {
                let mut nbrs_bytes = vec![0u8; num_nbrs as usize * mem::size_of::<u32>()];
                vamana_readers[shard_id as usize].read(&mut nbrs_bytes)?;
                let nbrs: &[u32] = bytemuck::cast_slice(&nbrs_bytes);

                // rename nodes
                for j in 0..num_nbrs {
                    let nbr = nbrs[j as usize];
                    let renamed_node = id_maps[shard_id as usize][nbr as usize];
                    if !nbr_set[renamed_node as usize] {
                        nbr_set[renamed_node as usize] = true;
                        final_nbrs.push(renamed_node);
                    }
                }
            }
        }

        // write the last node, to be refactored...
        final_nbrs.shuffle(&mut self.rng);

        let nnbrs: u32 = std::cmp::min(final_nbrs.len() as u32, max_degree);
        merged_vamana_cached_writer.write(&nnbrs.to_le_bytes())?;

        let bytes = final_nbrs
            .iter()
            .take(nnbrs as usize)
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>();
        merged_vamana_cached_writer.write(&bytes)?;

        merged_index_size += (size_of::<u32>() + nnbrs as usize * size_of::<u32>()) as u64;

        nbr_set.clear();
        final_nbrs.clear();

        info!("Expected size: {}", merged_index_size);
        merged_vamana_cached_writer.reset()?;
        merged_vamana_cached_writer.write(&merged_index_size.to_le_bytes())?;

        info!("Finished merge");
        Ok(())
    }

    fn read_idmap(&self, idmaps_path: String) -> Result<Vec<u32>, diskann_utils::io::ReadBinError> {
        let data = read_bin::<u32>(&mut self.storage_provider.open_reader(&idmaps_path)?)?;
        Ok(data.into_inner().into_vec())
    }

    fn partition_data(
        &mut self,
        dataset_file: &str,
        merged_index_prefix: &str,
        max_degree: u32,
        pool: RayonThreadPoolRef<'_>,
    ) -> ANNResult<usize> {
        let sampling_rate = MAX_PQ_TRAINING_SET_SIZE / self.index_configuration.max_points as f64;
        let ram_budget_in_bytes = self.disk_build_param.build_memory_limit().in_bytes() as f64;

        partition_with_ram_budget::<Data::VectorDataType, _, _>(
            dataset_file,
            self.index_configuration.dim,
            sampling_rate,
            ram_budget_in_bytes,
            PARTITION_ASSIGNMENTS_PER_VECTOR,
            merged_index_prefix,
            self.storage_provider,
            &mut self.rng,
            pool,
            |num_points, dim| {
                let datasize = std::mem::size_of::<Data::VectorDataType>() as u64;
                let graph_degree = 2 * max_degree / 3;
                estimate_build_index_ram_usage(
                    num_points,
                    dim,
                    datasize,
                    graph_degree as u64,
                    self.disk_build_param.build_quantization(),
                )
            },
        )
    }

    fn merge_and_cleanup(
        &mut self,
        merged_index_prefix: &str,
        num_parts: usize,
        max_degree: u32,
        output_vamana: String,
    ) -> ANNResult<()> {
        // merge all in-memory indices into one
        self.merge_shards(merged_index_prefix, num_parts, max_degree, output_vamana)?;

        // delete tempFiles
        for p in 0..num_parts {
            let shard_base_file =
                DiskIndexWriter::get_merged_index_subshard_data_file(merged_index_prefix, p);
            let shard_ids_file =
                DiskIndexWriter::get_merged_index_subshard_id_map_file(merged_index_prefix, p);
            let shard_index_file =
                DiskIndexWriter::get_merged_index_subshard_mem_index_file(merged_index_prefix, p);

            self.storage_provider.delete(&shard_base_file)?;
            self.storage_provider.delete(&shard_ids_file)?;
            self.storage_provider.delete(&shard_index_file)?;
        }

        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod disk_index_builder_tests {
    use std::{io::Read, sync::Arc, time::Instant};

    use crate::test_utils::{GraphDataF32VectorU32Data, GraphDataF32VectorUnitData};
    use diskann::{
        graph::config,
        utils::{IntoUsize, VectorRepr, ONE},
        ANNResult,
    };
    use diskann_providers::storage::VirtualStorageProvider;
    use diskann_providers::storage::{
        get_compressed_pq_file, get_disk_index_file, get_pq_pivot_file,
    };

    use diskann_utils::test_data_root;
    use diskann_vector::{
        distance::Metric::{self, L2},
        DistanceFunction,
    };
    use rstest::rstest;
    use vfs::OverlayFS;

    use super::*;
    use crate::{
        build::builder::build::DiskIndexBuilder,
        data_model::{CachingStrategy, GraphHeader},
        disk_index_build_parameter::{DiskIndexBuildParameters, MemoryBudget, NumPQChunks},
        search::provider::{
            aligned_file_reader::VirtualAlignedReaderFactory, disk_provider::DiskIndexSearcher,
            disk_vertex_provider_factory::DiskVertexProviderFactory,
        },
        storage::disk_index_reader::DiskIndexReader,
        utils::QueryStatistics,
        QuantizationType,
    };
    const DEFAULT_DISK_SECTOR_LEN: usize = 4096;
    pub const TEST_DATA_FILE: &str = "/sift/siftsmall_learn_256pts.fbin";
    /// We can use the same index prefix for all tests since we use virtual storage provider
    const INDEX_PATH_PREFIX: &str = "/disk_index_build/sift_learn_test_disk_index_build";
    const TRUTH_INDEX_PATH_PREFIX_R4_L50: &str = "/disk_index_build/truth_sift_learn_R4_L50";

    pub struct TestParams {
        pub dim: usize,
        pub full_dim: usize,
        pub max_degree: u32,
        pub num_pq_chunks: usize,
        pub build_quantization_type: QuantizationType,
        pub l_build: u32,
        pub data_path: String,
        pub index_path_prefix: String,
        pub associated_data_path: Option<String>,
        pub index_build_ram_gb: f64,
        pub data_compression_chunk_vector_count: Option<usize>,
        pub num_threads: usize,
        pub metric: Metric,
        pub block_size: usize,
    }

    impl Default for TestParams {
        fn default() -> Self {
            Self {
                dim: 128, // D
                full_dim: 128,
                max_degree: 4, // R
                num_pq_chunks: 128,
                build_quantization_type: QuantizationType::FP, // No quantization, i.e. QuantizationType::FP
                l_build: 50,
                data_path: TEST_DATA_FILE.to_string(),
                index_path_prefix: INDEX_PATH_PREFIX.to_string(),
                associated_data_path: None,
                index_build_ram_gb: 1.0,
                data_compression_chunk_vector_count: None,
                num_threads: 1,
                metric: L2,
                block_size: DEFAULT_DISK_SECTOR_LEN,
            }
        }
    }

    impl TestParams {
        /// Returns the appropriate truth index path prefix for build comparison.
        fn truth_index_path_prefix(&self) -> &str {
            match (self.max_degree, self.l_build, self.index_build_ram_gb) {
                (4, 50, 1.0) => TRUTH_INDEX_PATH_PREFIX_R4_L50,
                (max_degree, l_build, index_build_ram_gb) => panic!(
                    "Truth index path not found for max_degree={}, l_build={}, index_build_ram_gb={}",
                    max_degree, l_build, index_build_ram_gb
                ),
            }
        }
        pub fn truth_pq_compressed_path(&self) -> String {
            let prefix = match self.num_pq_chunks {
                128 => TRUTH_INDEX_PATH_PREFIX_R4_L50,
                num_pq_chunks => panic!(
                    "Truth pq compressed path not found for num_pq_chunks={}",
                    num_pq_chunks,
                ),
            };
            get_compressed_pq_file(prefix)
        }

        pub fn pq_compressed_path(&self) -> String {
            get_compressed_pq_file(&self.index_path_prefix)
        }
    }

    pub fn new_vfs() -> VirtualStorageProvider<OverlayFS> {
        VirtualStorageProvider::new_overlay(test_data_root())
    }

    pub struct IndexBuildFixture<StorageProvider: StorageReadProvider + StorageWriteProvider> {
        pub storage_provider: Arc<StorageProvider>,
        pub params: TestParams,
    }

    impl<StorageProvider: StorageReadProvider + StorageWriteProvider + 'static>
        IndexBuildFixture<StorageProvider>
    {
        pub fn new(storage_provider: StorageProvider, params: TestParams) -> ANNResult<Self> {
            Ok(Self {
                storage_provider: Arc::new(storage_provider),
                params,
            })
        }

        pub fn build<T>(&self) -> ANNResult<()>
        where
            T: GraphDataType<VectorIdType = u32>,
            StorageProvider::Reader: std::marker::Send + Read,
        {
            // Create disk index build parameters
            let mut disk_index_build_parameters = DiskIndexBuildParameters::new(
                MemoryBudget::try_from_gb(self.params.index_build_ram_gb)?,
                self.params.build_quantization_type,
                NumPQChunks::new_with(self.params.num_pq_chunks, self.params.full_dim)?,
            );
            if let Some(chunk_vector_count) = self.params.data_compression_chunk_vector_count {
                disk_index_build_parameters = disk_index_build_parameters
                    .with_data_compression_chunk_vector_count(chunk_vector_count);
            }

            let config = config::Builder::new_with(
                self.params.max_degree.into_usize(),
                config::MaxDegree::default_slack(),
                self.params.l_build.into_usize(),
                self.params.metric.into(),
                |b| {
                    b.saturate_after_prune(true);
                },
            )
            .build()?;

            let metadata =
                load_metadata_from_file(self.storage_provider.as_ref(), &self.params.data_path)
                    .unwrap();

            assert_eq!(
                self.params.dim,
                metadata.ndims(),
                "Parameters dimension {} and data dimension {} are not equal",
                self.params.dim,
                metadata.ndims(),
            );

            let config = IndexConfiguration::new(
                self.params.metric,
                self.params.dim,
                metadata.npoints(),
                ONE,
                self.params.num_threads,
                config,
            )
            .with_pseudo_rng_from_seed(100);

            let disk_index_writer = DiskIndexWriter::new(
                self.params.data_path.clone(),
                self.params.index_path_prefix.clone(),
                self.params.associated_data_path.clone(),
                self.params.block_size,
            )?;

            let mut disk_index = DiskIndexBuilder::<T, _>::new(
                self.storage_provider.as_ref(),
                disk_index_build_parameters,
                config,
                disk_index_writer,
            )?;

            let timer = Instant::now();
            disk_index.build()?;
            println!("Indexing time: {} seconds", timer.elapsed().as_secs_f64());

            Ok(())
        }

        pub fn compare_pq_compressed_files(&self) {
            self.compare_files(
                &self.params.pq_compressed_path(),
                &self.params.truth_pq_compressed_path(),
            );
        }

        pub fn assert_index_max_degree<T: GraphDataType>(&self) -> ANNResult<()> {
            let index_file_path = get_disk_index_file(&self.params.index_path_prefix);
            let file_data = load_file_to_vec(self.storage_provider.as_ref(), &index_file_path);
            let graph_header = GraphHeader::try_from(&file_data[8..])?;
            let max_degree = graph_header.max_degree::<T::VectorDataType>()?;
            assert_eq!(
                max_degree, self.params.max_degree as usize,
                "Max degree mismatch: expected {}, got {}",
                self.params.max_degree, max_degree
            );

            Ok(())
        }

        fn compare_disk_index_with_associated_data(
            &self,
            pivot_file_prefix_test: &str,
            pivot_file_prefix_expected: &str,
            index_file_suffix: &str,
        ) {
            let pq_pivot_path = pivot_file_prefix_test.to_string() + index_file_suffix;
            let pq_pivot_path_truth = pivot_file_prefix_expected.to_string() + index_file_suffix;
            let file1 = load_file_to_vec(self.storage_provider.as_ref(), &pq_pivot_path);
            let file2 = load_file_to_vec(self.storage_provider.as_ref(), &pq_pivot_path_truth);
            compare_disk_index_graphs(&file1, &file2)
        }

        pub fn compare_files(&self, file_path1: &str, file_path2: &str) {
            let file1 = load_file_to_vec(self.storage_provider.as_ref(), file_path1);
            let file2 = load_file_to_vec(self.storage_provider.as_ref(), file_path2);

            assert_eq!(file1.len(), file2.len());
            assert_eq!(file1, file2)
        }
    }

    /// Common helper function for one-shot async index build tests
    fn run_one_shot_test<F>(index_path_prefix: String, params_customizer: F)
    where
        F: FnOnce(TestParams) -> TestParams,
    {
        let l_build = 64;
        let max_degree = 16;
        let top_k = 10;
        let search_l = 32;

        let base_params = TestParams {
            l_build,
            max_degree,
            index_path_prefix,
            ..TestParams::default()
        };

        let params = params_customizer(base_params);

        let fixture = IndexBuildFixture::new(new_vfs(), params).unwrap();
        fixture.build::<GraphDataF32VectorUnitData>().unwrap();

        // Validate search recall against ground truth for async tests
        verify_search_result_with_ground_truth::<GraphDataF32VectorUnitData>(
            &fixture.params,
            top_k,
            search_l,
            &fixture.storage_provider,
        )
        .unwrap();

        fixture
            .assert_index_max_degree::<GraphDataF32VectorUnitData>()
            .unwrap();

        // Assert that all data was kept in memory and no files were written to the disk.
        let mem_index_file_path = format!("{}_mem.index.data", fixture.params.index_path_prefix);
        assert!(!fixture.storage_provider.exists(&mem_index_file_path));
    }

    #[rstest]
    fn test_build_from_iter_one_shot_with_metric(
        #[values(Metric::L2, Metric::InnerProduct, Metric::Cosine)] metric: Metric,
    ) {
        let index_path_prefix = format!("{}_metric_{:?}", INDEX_PATH_PREFIX, metric);

        run_one_shot_test(index_path_prefix, |params| TestParams { metric, ..params });
    }

    /// Forces the multi-sector-per-node layout: with a 512-byte block, a 128-d f32 vector
    /// (512 B) plus its neighbor list exceeds one sector, so each node spans multiple
    /// sectors (`node_len > block_size`). The default 4096-byte sector packs these small
    /// nodes many-per-sector, leaving the crossing path otherwise untested; 512 is one of
    /// the two block sizes the disk format supports.
    #[test]
    fn test_build_multi_sector_per_node() {
        let params = TestParams {
            block_size: 512,
            max_degree: 16,
            l_build: 64,
            index_path_prefix: format!("{}_multi_sector", INDEX_PATH_PREFIX),
            ..TestParams::default()
        };
        let fixture = IndexBuildFixture::new(new_vfs(), params).unwrap();
        fixture.build::<GraphDataF32VectorUnitData>().unwrap();
        verify_search_result_with_ground_truth::<GraphDataF32VectorUnitData>(
            &fixture.params,
            10,
            32,
            &fixture.storage_provider,
        )
        .unwrap();
    }

    #[test]
    fn test_build_from_iter_one_shot_with_associated_data() {
        // Set up test data
        let params = TestParams {
            associated_data_path: Some(
                "/sift/siftsmall_learn_256pts_u32_associated_data.fbin".to_string(),
            ),
            ..TestParams::default()
        };

        // Create fixture with virtual storage provider
        let fixture = IndexBuildFixture::new(new_vfs(), params).unwrap();

        // Build the index with the associated data
        fixture.build::<GraphDataF32VectorU32Data>().unwrap();

        // Assert that all data was kept in memory and no files were written to the disk.
        let mem_index_file_path = format!("{}_mem.index.data", fixture.params.index_path_prefix);
        let mem_index_associated_data_path = format!(
            "{}_mem.index.associated_data",
            fixture.params.index_path_prefix
        );
        assert!(!fixture.storage_provider.exists(&mem_index_file_path));
        assert!(!fixture
            .storage_provider
            .exists(&mem_index_associated_data_path));

        // assert index files are expected.
        fixture.compare_disk_index_with_associated_data(
            &fixture.params.index_path_prefix,
            fixture.params.truth_index_path_prefix(),
            "_disk.index",
        );
    }

    #[test]
    fn test_build_from_iter_merged_index() {
        // Use the same parameters from [test_sift_build_and_search] in diskann_index
        let l_build = 64;
        let max_degree = 16;
        let top_k = 10;
        let search_l = 32;

        let index_path_prefix =
            "/disk_index_build/disk_index_sift_learn_test_disk_index_build_merged".to_string();
        let params = TestParams {
            l_build,
            max_degree,
            index_path_prefix,
            index_build_ram_gb: 0.0001, // small enough to trigger merged index build
            ..TestParams::default()
        };

        let fixture = IndexBuildFixture::new(new_vfs(), params).unwrap();

        fixture.build::<GraphDataF32VectorUnitData>().unwrap();

        verify_search_result_with_ground_truth::<GraphDataF32VectorUnitData>(
            &fixture.params,
            top_k,
            search_l,
            &fixture.storage_provider,
        )
        .unwrap();

        fixture
            .assert_index_max_degree::<GraphDataF32VectorUnitData>()
            .unwrap();
    }

    #[rstest]
    #[case(QuantizationType::SQ { nbits: 2, standard_deviation: None }, "SQ quantization is only supported for 1 bit")]
    fn test_build_quantization_type_failure_cases(
        #[case] build_quantization_type: QuantizationType,
        #[case] error_message: &str,
    ) {
        let storage_provider = VirtualStorageProvider::new_overlay(test_data_root());
        let disk_index_builder = create_disk_index_builder(
            1000, // num_points
            128,  // dim
            128,  // num_pq_chunks
            &storage_provider,
            build_quantization_type,
        );

        let err = disk_index_builder.err().unwrap();
        assert!(err.to_string().contains(error_message));
    }

    fn load_file_to_vec<StorageType: StorageReadProvider>(
        storage_provider: &StorageType,
        file_path: &str,
    ) -> Vec<u8> {
        let mut file = storage_provider.open_reader(file_path).unwrap();
        let mut buffer = vec![];
        file.read_to_end(&mut buffer).unwrap();
        buffer
    }

    /// Verifies that search results exactly match the ground truth of nearest neighbors
    ///
    /// This function performs validation of search results by:
    /// 1. Running searches on the index using actual data points from the dataset as queries
    /// 2. Computing the exact ground truth results using direct distance calculations
    /// 3. Verifying that the search engine returns precisely the same results as the ground truth
    pub(crate) fn verify_search_result_with_ground_truth<
        G: GraphDataType<VectorIdType = u32, AssociatedDataType = ()>,
    >(
        params: &TestParams,
        top_k: usize,
        search_l: u32,
        storage_provider: &Arc<VirtualStorageProvider<OverlayFS>>,
    ) -> ANNResult<()> {
        let pq_pivot_path = get_pq_pivot_file(&params.index_path_prefix);
        let pq_compressed_path = get_compressed_pq_file(&params.index_path_prefix);
        let index_file_path = get_disk_index_file(&params.index_path_prefix);

        let index_reader =
            DiskIndexReader::new(pq_pivot_path, pq_compressed_path, storage_provider.as_ref())?;

        let vertex_provider_factory = DiskVertexProviderFactory::new(
            VirtualAlignedReaderFactory::new(index_file_path, Arc::clone(storage_provider)),
            CachingStrategy::None,
        )?;

        let search_engine = DiskIndexSearcher::<G, DiskVertexProviderFactory<G, _>>::new(
            1,
            u32::MAX as usize,
            &index_reader,
            vertex_provider_factory,
            params.metric,
            None,
        )?;

        let data =
            read_bin::<G::VectorDataType>(&mut storage_provider.open_reader(&params.data_path)?)?;
        let dim = data.ncols();
        let distance = <G::VectorDataType>::distance(params.metric, Some(dim));

        // Here, we use elements of the dataset to search the dataset itself.
        //
        // We do this for each query, computing the expected ground truth and verifying
        // that our simple graph search matches.
        //
        // Because this dataset is small, we can expect exact equality.
        for (q, query_data) in data.row_iter().enumerate() {
            let gt =
                diskann_providers::test_utils::groundtruth(data.as_view(), query_data, |a, b| {
                    distance.evaluate_similarity(a, b)
                });

            let mut query_stats = QueryStatistics::default();

            let mut indices = vec![0u32; top_k];
            let mut distances = vec![0f32; top_k];
            let mut associated_data = vec![(); top_k];

            _ = search_engine.search_internal(
                query_data,
                top_k,
                search_l,
                None, // beam_width
                &mut query_stats,
                &mut indices,
                &mut distances,
                &mut associated_data,
                &crate::search::search_mode::SearchMode::graph(),
            );

            diskann_providers::test_utils::assert_top_k_exactly_match(
                q, &gt, &indices, &distances, top_k,
            );
        }

        Ok(())
    }

    // Compare that the index built in test is the same as the truth index. The truth index doesn't have associated data, we are only comparing the vector and neighbor data.
    pub fn compare_disk_index_graphs(graph_data: &[u8], truth_graph_data: &[u8]) {
        let graph_header = GraphHeader::try_from(&graph_data[8..]).unwrap();
        let truth_graph_header = GraphHeader::try_from(&truth_graph_data[8..]).unwrap();

        let test_node_per_block = graph_header.metadata().num_nodes_per_block;
        let test_max_node_length = graph_header.metadata().node_len;

        let truth_node_per_block = truth_graph_header.metadata().num_nodes_per_block;
        let truth_max_node_length = truth_graph_header.metadata().node_len;

        assert_eq!(
            graph_header.metadata().num_pts,
            truth_graph_header.metadata().num_pts
        );

        assert_eq!(
            graph_header.metadata().dims,
            truth_graph_header.metadata().dims
        );

        let num_pts = graph_header.metadata().num_pts as usize;
        let dim = graph_header.metadata().dims;

        for idx in 0..num_pts {
            let test_node_id_offset = node_data_offset(
                idx,
                test_max_node_length as usize,
                test_node_per_block as usize,
                DEFAULT_DISK_SECTOR_LEN,
            );

            let truth_node_id_offset = node_data_offset(
                idx,
                truth_max_node_length as usize,
                truth_node_per_block as usize,
                DEFAULT_DISK_SECTOR_LEN,
            );

            // Assert that the vector data is the same between the test and truth graphs for this node.
            assert_eq!(
                &graph_data
                    [test_node_id_offset..test_node_id_offset + dim * std::mem::size_of::<f32>()],
                &truth_graph_data
                    [truth_node_id_offset..truth_node_id_offset + dim * std::mem::size_of::<f32>()]
            );

            // Assert that the neighbor count is the same between the test and truth graphs for this node.
            let test_nbr_cnt_offset = test_node_id_offset + dim * std::mem::size_of::<f32>();
            let truth_nbr_cnt_offset = truth_node_id_offset + dim * std::mem::size_of::<f32>();

            let test_nbr_count = u32::from_le_bytes([
                graph_data[test_nbr_cnt_offset],
                graph_data[test_nbr_cnt_offset + 1],
                graph_data[test_nbr_cnt_offset + 2],
                graph_data[test_nbr_cnt_offset + 3],
            ]);

            let truth_nbr_count = u32::from_le_bytes([
                truth_graph_data[truth_nbr_cnt_offset],
                truth_graph_data[truth_nbr_cnt_offset + 1],
                truth_graph_data[truth_nbr_cnt_offset + 2],
                truth_graph_data[truth_nbr_cnt_offset + 3],
            ]);

            assert_eq!(test_nbr_count, truth_nbr_count);

            // Assert the neighbors (u32) are the same between the test and truth graphs for this node.
            let test_nbr_offset = test_nbr_cnt_offset + 4;
            let truth_nbr_offset = truth_nbr_cnt_offset + 4;
            assert_eq!(
                graph_data[test_nbr_offset..test_nbr_offset + test_nbr_count as usize * 4],
                truth_graph_data[truth_nbr_offset..truth_nbr_offset + truth_nbr_count as usize * 4]
            );
        }
    }

    pub fn node_data_offset(
        node_id: usize,
        node_length: usize,
        nodes_per_block: usize,
        block_size: usize,
    ) -> usize {
        let block_id = node_id / nodes_per_block;
        let node_id_in_block = node_id % nodes_per_block;
        let offset = block_id * block_size + node_id_in_block * node_length;
        offset + block_size
    }

    fn create_disk_index_builder(
        num_points: usize,
        dim: usize,
        num_of_pq_chunks: usize,
        storage_provider: &VirtualStorageProvider<OverlayFS>,
        build_quantization_type: QuantizationType,
    ) -> ANNResult<
        DiskIndexBuilder<'_, GraphDataF32VectorUnitData, VirtualStorageProvider<OverlayFS>>,
    > {
        let memory_budget = MemoryBudget::try_from_gb(1.0)?;
        let num_pq_chunks = NumPQChunks::new_with(num_of_pq_chunks, dim)?;

        let build_parameters =
            DiskIndexBuildParameters::new(memory_budget, build_quantization_type, num_pq_chunks);

        let index_configuration = IndexConfiguration::new(
            L2,
            dim,
            num_points,
            ONE,
            1,
            config::Builder::new_with(4, config::MaxDegree::default_slack(), 50, L2.into(), |b| {
                b.saturate_after_prune(true);
            })
            .build()?,
        );

        let disk_index_writer = DiskIndexWriter::new(
            "data_path".to_string(),
            "index_path_prefix".to_string(),
            None,
            DEFAULT_DISK_SECTOR_LEN,
        )?;

        DiskIndexBuilder::<GraphDataF32VectorUnitData, VirtualStorageProvider<OverlayFS>>::new(
            storage_provider,
            build_parameters,
            index_configuration,
            disk_index_writer,
        )
    }
}

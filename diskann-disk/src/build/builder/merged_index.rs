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
    model::{IndexConfiguration, GRAPH_SLACK_FACTOR, MAX_PQ_TRAINING_SET_SIZE},
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
    DiskIndexBuildParameters, QuantizationType,
};

/// Overhead factor for RAM estimation during index build (10% buffer).
const OVERHEAD_FACTOR: f64 = 1.1f64;

/// Number of nearest shards each vector is assigned to during partitioning.
const PARTITION_ASSIGNMENTS_PER_VECTOR: usize = 2;

/// Estimate RAM usage in bytes for building an index.
#[inline]
pub(super) fn estimate_build_index_ram_usage(
    num_points: u64,
    dim: u64,
    datasize: u64,
    graph_degree: u64,
    build_quantization_type: &QuantizationType,
) -> f64 {
    let graph_size =
        (num_points * graph_degree * mem::size_of::<u32>() as u64) as f64 * GRAPH_SLACK_FACTOR;

    let single_vec_size = match *build_quantization_type {
        QuantizationType::FP => dim.next_multiple_of(8u64) * datasize,
        // We can skip PQ pivots data as it is very small(~3MB) for even large datasets like OAI-3072.
        QuantizationType::PQ { num_chunks } => num_chunks as u64,
        // `+ std::mem::size_of::<f32>()` for f32 compensation metadata for the scalar quantizer.
        QuantizationType::SQ { nbits, .. } => {
            (nbits as u64 * dim).div_ceil(8) + std::mem::size_of::<f32>() as u64
        }
    };

    OVERHEAD_FACTOR * (graph_size + (single_vec_size * num_points) as f64)
}

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

                merged_vamana_cached_writer
                    .write(bytemuck::must_cast_slice(&final_nbrs[..nnbrs as usize]))?;

                merged_index_size += (size_of::<u32>() + nnbrs as usize * size_of::<u32>()) as u64;
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

        merged_vamana_cached_writer
            .write(bytemuck::must_cast_slice(&final_nbrs[..nnbrs as usize]))?;

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
mod ram_estimation_tests {
    use rstest::rstest;

    use super::*;
    use crate::QuantizationType;

    #[rstest]
    #[case(QuantizationType::FP)]
    #[case(QuantizationType::PQ { num_chunks: 15 })]
    #[case(QuantizationType::SQ { nbits: 1, standard_deviation: None })]
    fn test_estimate_build_index_ram_usage(#[case] build_quantization_type: QuantizationType) {
        let num_points = 1000;
        let dim = 128;
        let size_of_t = std::mem::size_of::<f32>() as u64;
        let graph_degree = 50;

        let single_vec_size = match build_quantization_type {
            QuantizationType::FP => dim * size_of_t,
            QuantizationType::PQ { num_chunks } => num_chunks as u64,
            QuantizationType::SQ { nbits, .. } => {
                (nbits as u64 * dim).div_ceil(8) + std::mem::size_of::<f32>() as u64
            }
        };
        let mut expected_ram_usage = (num_points as f64)
            * (graph_degree as f64)
            * (std::mem::size_of::<u32>() as f64)
            * GRAPH_SLACK_FACTOR
            + (num_points * single_vec_size) as f64;
        expected_ram_usage *= OVERHEAD_FACTOR;

        let actual_ram_usage = estimate_build_index_ram_usage(
            num_points,
            dim,
            size_of_t,
            graph_degree,
            &build_quantization_type,
        );

        assert_eq!(actual_ram_usage, expected_ram_usage);
    }
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Disk index builder implementation.
use std::{marker::PhantomData, mem};

use crate::data_model::GraphDataType;
use diskann::{utils::VectorRepr, ANNResult};
use diskann_providers::storage::{StorageReadProvider, StorageWriteProvider};
use diskann_providers::{
    model::{IndexConfiguration, MAX_PQ_TRAINING_SET_SIZE, NUM_KMEANS_REPS_PQ, NUM_PQ_CENTROIDS},
    storage::PQStorage,
    utils::{create_thread_pool, RayonThreadPoolRef},
};
use tracing::info;

use crate::{
    build::builder::{
        inmem_builder::build_inmem_index,
        merged_index::{estimate_build_index_ram_usage, MergedVamanaIndexBuilder},
        quantizer::BuildQuantizer,
        tokio::create_runtime,
    },
    disk_index_build_parameter::BYTES_IN_GB,
    storage::{
        quant::{PQGeneration, PQGenerationContext, QuantDataGenerator},
        DiskIndexWriter,
    },
    utils::instrumentation::{DiskIndexBuildCheckpoint, PerfLogger},
    DiskIndexBuildParameters, QuantizationType,
};

pub struct DiskIndexBuilder<'a, Data, StorageProvider>
where
    Data: GraphDataType<VectorIdType = u32>,
    StorageProvider: StorageReadProvider + StorageWriteProvider,
{
    index_writer: DiskIndexWriter,
    pq_storage: PQStorage,
    disk_build_param: DiskIndexBuildParameters,
    index_configuration: IndexConfiguration,
    storage_provider: &'a StorageProvider,
    build_quantizer: BuildQuantizer,
    _phantom: PhantomData<Data>,
}

impl<'a, Data, StorageProvider> DiskIndexBuilder<'a, Data, StorageProvider>
where
    Data: GraphDataType<VectorIdType = u32>,
    Data::VectorDataType: VectorRepr,
    StorageProvider: StorageReadProvider + StorageWriteProvider + 'static,
    <StorageProvider as StorageReadProvider>::Reader: std::marker::Send,
{
    pub fn new(
        storage_provider: &'a StorageProvider,
        disk_build_param: DiskIndexBuildParameters,
        index_configuration: IndexConfiguration,
        index_writer: DiskIndexWriter,
    ) -> ANNResult<Self> {
        let pq_storage = PQStorage::new(
            &(index_writer.get_index_path_prefix() + "_pq_pivots.bin"),
            &(index_writer.get_index_path_prefix() + "_pq_compressed.bin"),
            Some(&index_writer.get_dataset_file()),
        );

        info!(
            "Training quantizer for {} quantized builds.",
            disk_build_param.build_quantization().to_string()
        );
        let build_quantizer = BuildQuantizer::train::<Data, _>(
            disk_build_param.build_quantization(),
            &index_writer.get_index_path_prefix(),
            &index_configuration,
            &pq_storage,
            storage_provider,
        )?;

        Ok(Self {
            disk_build_param,
            index_configuration,
            index_writer,
            storage_provider,
            pq_storage,
            build_quantizer,
            _phantom: PhantomData,
        })
    }

    pub fn build(&mut self) -> ANNResult<()> {
        let runtime = create_runtime(self.index_configuration.num_threads)?;
        runtime.block_on(self.build_internal())
    }

    async fn build_internal(&mut self) -> ANNResult<()> {
        let mut logger = PerfLogger::new_disk_index_build_logger();

        let pool = create_thread_pool(self.index_configuration.num_threads)?;

        info!(
            "Starting index build: R={} L={} Indexing RAM budget={} T={}",
            self.index_configuration.config.pruned_degree(),
            self.index_configuration.config.l_build(),
            self.disk_build_param.build_memory_limit().in_bytes(),
            self.index_configuration.num_threads
        );

        self.generate_compressed_data(pool.as_ref())?;
        logger.log_checkpoint(DiskIndexBuildCheckpoint::PqConstruction);

        self.build_inmem_index(pool.as_ref()).await?;
        logger.log_checkpoint(DiskIndexBuildCheckpoint::InmemIndexBuild);

        // Use physical file to pass the memory index to the disk writer
        self.create_disk_layout()?;
        logger.log_checkpoint(DiskIndexBuildCheckpoint::DiskLayout);

        Ok(())
    }

    fn generate_compressed_data(&mut self, pool: RayonThreadPoolRef<'_>) -> ANNResult<()> {
        let num_points = self.index_configuration.max_points;
        let num_chunks = self.disk_build_param.search_pq_chunks();

        let storage_provider = self.storage_provider;

        info!(
            "Compressing data into {} bytes per vector for disk search",
            num_chunks.get()
        );

        let quantizer_context = PQGenerationContext {
            pq_storage: self.pq_storage.clone(),
            num_chunks: num_chunks.get(),
            max_kmeans_reps: NUM_KMEANS_REPS_PQ,
            num_centers: NUM_PQ_CENTROIDS,
            seed: self.index_configuration.random_seed,
            p_val: MAX_PQ_TRAINING_SET_SIZE / (num_points as f64),
            storage_provider,
            pool,
            dim: self.index_configuration.dim,
            metric: self.index_configuration.dist_metric,
        };

        let generator = QuantDataGenerator::<
            Data::VectorDataType,
            PQGeneration<Data::VectorDataType, StorageProvider>,
        >::new(
            self.index_writer.get_dataset_file(),
            self.pq_storage.get_compressed_data_path().into(),
            &quantizer_context,
        )?;
        generator.generate_data(
            storage_provider,
            pool,
            self.disk_build_param.data_compression_chunk_vector_count(),
        )
    }

    async fn build_inmem_index(&mut self, pool: RayonThreadPoolRef<'_>) -> ANNResult<()> {
        match determine_build_strategy::<Data>(
            &self.index_configuration,
            self.disk_build_param.build_memory_limit().in_bytes() as f64,
            self.disk_build_param.build_quantization(),
        ) {
            IndexBuildStrategy::Merged => {
                MergedVamanaIndexBuilder::<Data, _>::new(
                    &self.index_configuration,
                    &self.disk_build_param,
                    &self.index_writer,
                    &self.build_quantizer,
                    self.storage_provider,
                )
                .build(pool)
                .await
            }
            IndexBuildStrategy::OneShot => {
                build_inmem_index::<Data::VectorDataType, _>(
                    self.index_configuration.clone(),
                    &self.build_quantizer,
                    &self.index_writer.get_dataset_file(),
                    &self.index_writer.get_mem_index_file(),
                    self.storage_provider,
                )
                .await
            }
        }
    }

    fn create_disk_layout(&mut self) -> ANNResult<()> {
        self.index_writer
            .create_disk_layout::<Data, StorageProvider>(self.storage_provider)?;
        self.index_writer
            .index_build_cleanup(self.storage_provider)?;

        Ok(())
    }
}

pub(crate) enum IndexBuildStrategy {
    OneShot,
    Merged,
}

pub(crate) fn determine_build_strategy<Data: GraphDataType>(
    index_configuration: &IndexConfiguration,
    index_build_ram_limit_in_bytes: f64,
    build_quantization_type: &QuantizationType,
) -> IndexBuildStrategy {
    let estimated_index_ram_in_bytes = estimate_build_index_ram_usage(
        index_configuration.max_points as u64,
        index_configuration.dim as u64,
        mem::size_of::<Data::VectorDataType>() as u64,
        index_configuration.config.max_degree().get() as u64,
        build_quantization_type,
    );

    info!(
        "Estimated index RAM usage: {} GB, index_build_ram_limit={} GB",
        estimated_index_ram_in_bytes / BYTES_IN_GB,
        index_build_ram_limit_in_bytes / BYTES_IN_GB
    );

    if estimated_index_ram_in_bytes >= index_build_ram_limit_in_bytes {
        info!(
            "Insufficient memory budget for index build in one shot, index_build_ram_limit={} GB estimated_index_ram={} GB",
            index_build_ram_limit_in_bytes / BYTES_IN_GB,
            estimated_index_ram_in_bytes / BYTES_IN_GB,
        );
        IndexBuildStrategy::Merged
    } else {
        info!(
            "Full index fits in RAM budget, should consume at most {} GBs, so building in one shot",
            estimated_index_ram_in_bytes / BYTES_IN_GB
        );
        IndexBuildStrategy::OneShot
    }
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Async disk index builder implementation.
use std::{
    num::NonZeroUsize,
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex},
};

use crate::data_model::GraphDataType;
use diskann::{
    utils::{async_tools, VectorRepr, ONE},
    ANNError, ANNResult,
};
use diskann_providers::storage::{StorageReadProvider, StorageWriteProvider};
use diskann_providers::{
    model::{
        graph::provider::async_::inmem::DefaultProviderParameters, IndexConfiguration,
        MAX_PQ_TRAINING_SET_SIZE, NUM_KMEANS_REPS_PQ, NUM_PQ_CENTROIDS,
    },
    storage::{DiskGraphOnly, PQStorage},
    utils::{
        create_thread_pool, find_medoid_with_sampling, RayonThreadPoolRef, VectorDataIterator,
        MAX_MEDOID_SAMPLE_SIZE,
    },
};
use tokio::task::JoinSet;
use tracing::{debug, info};

use crate::{
    build::builder::{
        core::{
            determine_build_strategy, DiskIndexBuilderCore, IndexBuildStrategy,
            MergedVamanaIndexWorkflow,
        },
        inmem_builder::{new_inmem_index_builder, InmemIndexBuilder},
        quantizer::BuildQuantizer,
        tokio::create_runtime,
    },
    storage::{
        quant::{PQGeneration, PQGenerationContext, QuantDataGenerator},
        DiskIndexWriter,
    },
    utils::instrumentation::{
        BuildMergedVamanaIndexCheckpoint, DiskIndexBuildCheckpoint, PerfLogger,
    },
    DiskIndexBuildParameters, QuantizationType,
};

/// Disk index builder that composes with DiskIndexBuilderCore.
pub struct DiskIndexBuilder<'a, Data, StorageProvider>
where
    Data: GraphDataType<VectorIdType = u32>,
    StorageProvider: StorageReadProvider + StorageWriteProvider,
{
    pub core: DiskIndexBuilderCore<'a, Data, StorageProvider>,
    /// Async-specific field: actual quantizers for async processing
    pub build_quantizer: BuildQuantizer,
}

impl<'a, Data, StorageProvider> Deref for DiskIndexBuilder<'a, Data, StorageProvider>
where
    Data: GraphDataType<VectorIdType = u32>,
    StorageProvider: StorageReadProvider + StorageWriteProvider,
{
    type Target = DiskIndexBuilderCore<'a, Data, StorageProvider>;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

impl<'a, Data, StorageProvider> DerefMut for DiskIndexBuilder<'a, Data, StorageProvider>
where
    Data: GraphDataType<VectorIdType = u32>,
    StorageProvider: StorageReadProvider + StorageWriteProvider,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.core
    }
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

        let build_quantizer = Self::train_build_quantizer(
            disk_build_param.build_quantization(),
            &index_writer.get_index_path_prefix(),
            &index_configuration,
            &pq_storage,
            storage_provider,
        )?;

        let core = DiskIndexBuilderCore {
            disk_build_param,
            index_configuration,
            index_writer,
            storage_provider,
            pq_storage,
            _phantom: std::marker::PhantomData,
        };

        Ok(Self {
            core,
            build_quantizer,
        })
    }

    fn train_build_quantizer(
        build_quantization_type: &QuantizationType,
        index_path_prefix: &str,
        index_configuration: &IndexConfiguration,
        pq_storage: &PQStorage,
        storage_provider: &StorageProvider,
    ) -> ANNResult<BuildQuantizer> {
        info!(
            "Training quantizer for {} quantized builds.",
            build_quantization_type.to_string()
        );

        BuildQuantizer::train::<Data, _>(
            build_quantization_type,
            index_path_prefix,
            index_configuration,
            pq_storage,
            storage_provider,
        )
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

        self.generate_compressed_data(pool.as_ref()).await?;
        logger.log_checkpoint(DiskIndexBuildCheckpoint::PqConstruction);

        self.build_inmem_index(pool.as_ref()).await?;
        logger.log_checkpoint(DiskIndexBuildCheckpoint::InmemIndexBuild);

        // Use physical file to pass the memory index to the disk writer
        self.create_disk_layout()?;
        logger.log_checkpoint(DiskIndexBuildCheckpoint::DiskLayout);

        Ok(())
    }

    async fn generate_compressed_data(&mut self, pool: RayonThreadPoolRef<'_>) -> ANNResult<()> {
        let num_points = self.index_configuration.max_points;
        let num_chunks = self.disk_build_param.search_pq_chunks();

        let storage_provider = self.core.storage_provider;

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
            IndexBuildStrategy::Merged => self.build_merged_vamana_index(pool).await,
            IndexBuildStrategy::OneShot => self.build_one_shot_vamana_index().await,
        }
    }

    async fn build_merged_vamana_index(&mut self, pool: RayonThreadPoolRef<'_>) -> ANNResult<()> {
        let mut logger = PerfLogger::new_disk_index_build_logger();
        let mut workflow = MergedVamanaIndexWorkflow::new(self, pool);

        // Partition data stage
        let num_parts = workflow.partition_data(self)?;
        logger.log_checkpoint(BuildMergedVamanaIndexCheckpoint::PartitionData);

        // build in-memory index for each partition
        for p in 0..num_parts {
            // build in-memory disk for current shard partition:{shard_base_file} and save to disk
            self.build_shard_index(&workflow.merged_index_prefix, p)
                .await?;
        }
        logger.log_checkpoint(BuildMergedVamanaIndexCheckpoint::BuildIndicesOnShards);

        workflow.merge_and_cleanup(self, num_parts)?;
        logger.log_checkpoint(BuildMergedVamanaIndexCheckpoint::MergeIndices);

        Ok(())
    }

    async fn build_shard_index(&self, merged_index_prefix: &str, shard_id: usize) -> ANNResult<()> {
        let shard_base_file =
            DiskIndexWriter::get_merged_index_subshard_data_file(merged_index_prefix, shard_id);

        let shard_ids_file =
            DiskIndexWriter::get_merged_index_subshard_id_map_file(merged_index_prefix, shard_id);
        self.retrieve_shard_data_from_ids::<Data::VectorDataType>(
            &self.index_writer.get_dataset_file(),
            &shard_ids_file,
            &shard_base_file,
        )?;
        info!("Generated data for shard {}", shard_id);

        let index_config = self.create_shard_index_config(&shard_base_file)?;
        let shard_index_file = DiskIndexWriter::get_merged_index_subshard_mem_index_file(
            merged_index_prefix,
            shard_id,
        );

        self.build_inmem_index_from_data(index_config, &shard_base_file, &shard_index_file)
            .await
    }

    async fn build_one_shot_vamana_index(&mut self) -> ANNResult<()> {
        self.build_inmem_index_from_data(
            self.index_configuration.clone(),
            &self.index_writer.get_dataset_file(),
            &self.index_writer.get_mem_index_file(),
        )
        .await
    }

    async fn build_inmem_index_from_data(
        &self,
        config: IndexConfiguration,
        data_path: &str,
        save_path: &str,
    ) -> ANNResult<()> {
        build_inmem_index::<Data::VectorDataType, _>(
            config,
            &self.build_quantizer,
            data_path,
            save_path,
            self.core.storage_provider,
        )
        .await
    }
}

async fn build_inmem_index<T, StorageProvider>(
    config: IndexConfiguration,
    quantizer: &BuildQuantizer,
    data_path: &str,
    save_path: &str,
    storage_provider: &StorageProvider,
) -> ANNResult<()>
where
    T: VectorRepr,
    StorageProvider: StorageReadProvider + StorageWriteProvider + 'static,
    <StorageProvider as StorageReadProvider>::Reader: std::marker::Send,
{
    // use either user-specified number of threads or default to available parallelism
    let num_tasks = NonZeroUsize::new(config.num_threads)
        .or_else(|| std::thread::available_parallelism().ok())
        .ok_or_else(|| ANNError::log_index_error("Failed to determine number of threads"))?;

    // Associated data will only be used in the write_disk_layout function which only requires the none-partitioned associated data stream.
    let dataset_iter = Arc::new(Mutex::new({
        let iter = VectorDataIterator::<_, T>::new(data_path, Option::None, storage_provider)?;
        iter.enumerate()
    }));

    let index_config = config.config.clone();
    let provider_parameters = DefaultProviderParameters {
        max_points: config.max_points,
        frozen_points: ONE,
        metric: config.dist_metric,
        dim: config.dim,
        max_degree: index_config.max_degree_u32().get(),
        prefetch_lookahead: config.prefetch_lookahead.map(|x| x.get()),
        prefetch_cache_line_level: config.prefetch_cache_line_level,
    };
    let index = new_inmem_index_builder::<T>(index_config, provider_parameters, quantizer)?;
    let medoid_id =
        set_start_point_to_medoid::<T, _>(&index, data_path, config.random_seed, storage_provider)?;
    let start_point = u32_try_from(medoid_id)?;

    run_build(&index, dataset_iter, num_tasks).await?;

    #[cfg(debug_assertions)]
    log_build_stats::<_>(&index).await?;

    run_final_prune(&index, num_tasks).await?;
    index
        .save_graph(
            storage_provider,
            &(start_point, DiskGraphOnly::new(save_path)),
        )
        .await?;

    Ok(())
}

#[cfg(debug_assertions)]
/// Log statistics about the build process
async fn log_build_stats<T: VectorRepr>(index: &Arc<dyn InmemIndexBuilder<T>>) -> ANNResult<()> {
    debug!(
        "Number of points reachable in the graph: {}",
        index.count_reachable_nodes().await?
    );

    let (full_vector, quant_vector) = index.counts_for_get_vector();
    let capacity = index.capacity();
    debug!(
        "Number of get vector calls per insert: {}",
        full_vector as f32 / capacity as f32
    );
    debug!(
        "Number of get quantized vector calls per insert: {}",
        quant_vector as f32 / capacity as f32
    );

    Ok(())
}

/// Convert a `usize` index into the `u32` internal id type, erroring if it does not fit.
///
/// The async index uses `u32` internal ids, so positions in the dataset must not exceed
/// `u32::MAX`.
fn u32_try_from(value: usize) -> ANNResult<u32> {
    u32::try_from(value)
        .map_err(|_| ANNError::log_index_error(format_args!("id {value} exceeds u32::MAX")))
}

fn set_start_point_to_medoid<T, StorageReader>(
    index: &Arc<dyn InmemIndexBuilder<T>>,
    path: &str,
    random_seed: Option<u64>,
    reader: &StorageReader,
) -> ANNResult<usize>
where
    T: VectorRepr,
    StorageReader: StorageReadProvider,
{
    let mut rng = diskann_providers::utils::create_rnd_from_optional_seed(random_seed);
    let (medoid, medoid_id) =
        find_medoid_with_sampling::<T, _>(path, reader, MAX_MEDOID_SAMPLE_SIZE, &mut rng)?;

    index.set_start_point(medoid.as_slice())?;

    debug!("Set start point to medoid ID: {}", medoid_id);

    Ok(medoid_id)
}

async fn run_build<T, I>(
    index: &Arc<dyn InmemIndexBuilder<T>>,
    iterator: Arc<Mutex<I>>,
    num_tasks: NonZeroUsize,
) -> ANNResult<()>
where
    T: VectorRepr,
    I: Iterator<Item = (usize, (Box<[T]>, ()))> + Send + 'static,
{
    let total_points = index.capacity();
    let partitions = async_tools::PartitionIter::new(total_points, num_tasks);

    let mut tasks = JoinSet::new();

    for partition in partitions {
        let index_clone = index.clone();
        let iterator_clone = iterator.clone();
        tasks.spawn(async move {
            for _ in partition {
                let vector_data = {
                    let mut guard = iterator_clone.lock().map_err(|_| {
                        ANNError::log_index_error("Poisoned mutex during construction")
                    })?;
                    guard.next()
                };

                match vector_data {
                    Some((i, (vector, _))) => {
                        let id = u32_try_from(i)?;
                        index_clone.insert_vector(id, vector.as_ref()).await?;
                    }
                    None => break,
                }
            }
            ANNResult::Ok(())
        });
    }

    // Wait for all tasks to complete.
    while let Some(res) = tasks.join_next().await {
        res.map_err(|_| ANNError::log_index_error("A spawned insert task failed"))??;
    }

    info!("Linked all points. Num points: #{}", total_points);
    Ok(())
}

async fn run_final_prune<T: VectorRepr>(
    index: &Arc<dyn InmemIndexBuilder<T>>,
    num_tasks: NonZeroUsize,
) -> ANNResult<()> {
    let partitions = async_tools::PartitionIter::new(index.total_points(), num_tasks);

    let mut tasks = JoinSet::new();

    for partition in partitions {
        let index_clone = index.clone();
        tasks.spawn(async move {
            let range = u32_try_from(partition.start)?..u32_try_from(partition.end)?;
            index_clone.final_prune(range).await
        });
    }

    // Wait for all final prune tasks to complete
    while let Some(res) = tasks.join_next().await {
        res.map_err(|_| ANNError::log_index_error("A spawned final prune task failed"))??;
    }

    Ok(())
}

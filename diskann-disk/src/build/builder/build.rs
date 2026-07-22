/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Async disk index builder implementation.
use std::{
    marker::PhantomData,
    num::NonZeroUsize,
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

#[cfg(feature = "pipnn")]
mod pipnn;

use crate::{
    build::builder::{
        core::{determine_build_strategy, IndexBuildStrategy, MergedVamanaIndexBuilder},
        inmem_builder::{new_inmem_index_builder, InmemIndexBuilder},
        quantizer::BuildQuantizer,
        tokio::create_runtime,
    },
    storage::{
        quant::{PQGeneration, PQGenerationContext, QuantDataGenerator},
        DiskIndexWriter,
    },
    utils::instrumentation::{DiskIndexBuildCheckpoint, PerfLogger},
    DiskIndexBuildParameters,
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
        #[cfg(feature = "pipnn")]
        let disk_build_param = {
            let mut disk_build_param = disk_build_param;
            if let Some(config) = disk_build_param.pipnn_config() {
                config.validate()?;
                let total_points = index_configuration
                    .max_points
                    .checked_add(index_configuration.num_frozen_pts.get())
                    .ok_or_else(|| {
                        ANNError::log_index_error("PiPNN point count overflows usize")
                    })?;
                let estimate = disk_build_param.use_vamana_if_pipnn_exceeds(
                    total_points,
                    index_configuration.dim,
                    std::mem::size_of::<Data::VectorDataType>(),
                    index_configuration.num_threads,
                )?;
                let selected = disk_build_param.build_algorithm();
                info!(
                    estimated_peak_bytes = estimate,
                    memory_limit_bytes = disk_build_param.build_memory_limit().in_bytes(),
                    algorithm = %selected,
                    "Selected graph build algorithm"
                );
            }
            disk_build_param
        };

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

        self.build_graph(pool.as_ref()).await?;
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

    async fn build_graph(&mut self, pool: RayonThreadPoolRef<'_>) -> ANNResult<()> {
        #[cfg(feature = "pipnn")]
        if let Some(config) = self.disk_build_param.pipnn_config() {
            let context = pipnn::prepare_context(self, config)?;
            return pipnn::build_graph(self, &context);
        }

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

pub(super) async fn build_inmem_index<T, StorageProvider>(
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

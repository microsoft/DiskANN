/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Shared disk pipeline and configuration-selected graph construction.
use std::{
    fmt,
    marker::PhantomData,
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

use crate::data_model::GraphDataType;
use diskann::{
    ANNResult,
    utils::{ONE, VectorRepr, async_tools},
};
use diskann_providers::storage::{StorageReadProvider, StorageWriteProvider};
use diskann_providers::{
    model::{
        IndexConfiguration, MAX_PQ_TRAINING_SET_SIZE, NUM_KMEANS_REPS_PQ, NUM_PQ_CENTROIDS,
        graph::provider::async_::inmem::DefaultProviderParameters,
    },
    storage::{DiskGraphOnly, PQStorage},
    utils::{
        MAX_MEDOID_SAMPLE_SIZE, RayonThreadPoolRef, VectorDataIterator, create_thread_pool,
        find_medoid_with_sampling,
    },
};
use tokio::task::JoinSet;
#[cfg(debug_assertions)]
use tracing::debug;
use tracing::info;

use crate::{
    DiskIndexBuildParameters,
    build::builder::{
        core::{IndexBuildStrategy, MergedVamanaIndexBuilder, determine_build_strategy},
        inmem_builder::{InmemIndexBuilder, StartPoint, new_inmem_index_builder},
        quantizer::BuildQuantizer,
        tokio::create_runtime,
    },
    build::configuration::{BuildParameters, DiskIndexBuildConfig},
    error::{ErrorKind, diskann_error},
    storage::{
        DiskIndexWriter,
        quant::{
            PQGeneration, PQGenerationContext, QuantDataGenerator, validate_data_generation_input,
        },
    },
    utils::instrumentation::{DiskIndexBuildCheckpoint, PerfLogger},
};

enum GraphBuild {
    Vamana {
        index: IndexConfiguration,
        parameters: DiskIndexBuildParameters,
        quantizer: BuildQuantizer,
    },
    #[cfg(feature = "pipnn")]
    Pipnn(diskann::graph::pipnn::Config),
}

/// Shared search-PQ, graph, disk-layout and cleanup pipeline.
pub struct DiskIndexBuilder<'a, Data, StorageProvider>
where
    Data: GraphDataType<VectorIdType = u32>,
    StorageProvider: StorageReadProvider + StorageWriteProvider,
{
    index_writer: DiskIndexWriter,
    pq_storage: PQStorage,
    storage_provider: &'a StorageProvider,
    parameters: BuildParameters,
    graph: GraphBuild,
    _phantom: PhantomData<Data>,
}

impl<Data, StorageProvider> fmt::Debug for DiskIndexBuilder<'_, Data, StorageProvider>
where
    Data: GraphDataType<VectorIdType = u32>,
    StorageProvider: StorageReadProvider + StorageWriteProvider,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DiskIndexBuilder")
            .field("parameters", &self.parameters)
            .finish_non_exhaustive()
    }
}

impl<'a, Data, StorageProvider> DiskIndexBuilder<'a, Data, StorageProvider>
where
    Data: GraphDataType<VectorIdType = u32>,
    Data::VectorDataType: VectorRepr,
    StorageProvider: StorageReadProvider + StorageWriteProvider + 'static,
    StorageProvider::Reader: Send,
{
    /// Creates a disk builder with one selected graph configuration.
    ///
    /// # Errors
    /// Returns Vamana graph-quantizer training or storage errors.
    pub fn new(
        storage_provider: &'a StorageProvider,
        config: DiskIndexBuildConfig,
        index_writer: DiskIndexWriter,
    ) -> ANNResult<Self> {
        let (parameters, graph) = match config {
            DiskIndexBuildConfig::Vamana { index, parameters } => {
                let common = BuildParameters::from((&index, &parameters));
                let quantizer = BuildQuantizer::train::<Data, _>(
                    parameters.build_quantization(),
                    &index_writer.get_index_path_prefix(),
                    &index,
                    &index_writer.get_dataset_file(),
                    storage_provider,
                )?;
                (
                    common,
                    GraphBuild::Vamana {
                        index,
                        parameters,
                        quantizer,
                    },
                )
            }
            #[cfg(feature = "pipnn")]
            DiskIndexBuildConfig::Pipnn {
                graph,
                metric,
                dim,
                num_points,
                num_threads,
                search_pq_chunks,
                compression_chunk_vector_count,
                seed,
            } => (
                BuildParameters {
                    metric,
                    dim,
                    num_points,
                    num_threads,
                    search_pq_chunks,
                    compression_chunk_vector_count,
                    seed,
                },
                GraphBuild::Pipnn(graph),
            ),
        };
        let pq_storage = PQStorage::new(
            &(index_writer.get_index_path_prefix() + "_pq_pivots.bin"),
            &(index_writer.get_index_path_prefix() + "_pq_compressed.bin"),
            Some(&index_writer.get_dataset_file()),
        );
        Ok(Self {
            index_writer,
            pq_storage,
            storage_provider,
            parameters,
            graph,
            _phantom: PhantomData,
        })
    }

    /// Builds search-PQ, the selected graph and the disk layout.
    ///
    /// # Errors
    /// Returns compression, graph-construction or storage errors.
    pub fn build(&mut self) -> ANNResult<()> {
        let mut logger = PerfLogger::new_disk_index_build_logger();
        let pool = create_thread_pool(self.parameters.num_threads)?;
        self.generate_compressed_data(pool.as_ref())?;
        logger.log_checkpoint(DiskIndexBuildCheckpoint::PqConstruction);

        match &self.graph {
            GraphBuild::Vamana {
                index,
                parameters,
                quantizer,
            } => {
                create_runtime(index.num_threads)?.block_on(build_vamana_graph::<Data, _>(
                    index,
                    parameters,
                    quantizer,
                    &self.index_writer,
                    self.storage_provider,
                    pool.as_ref(),
                ))?;
            }
            #[cfg(feature = "pipnn")]
            GraphBuild::Pipnn(config) => {
                super::pipnn::build_inmem_index::<Data::VectorDataType, _>(
                    &self.parameters,
                    config,
                    &self.index_writer.get_dataset_file(),
                    &self.index_writer.get_mem_index_file(),
                    self.storage_provider,
                    pool.as_ref(),
                )?
            }
        }
        logger.log_checkpoint(DiskIndexBuildCheckpoint::InmemIndexBuild);
        self.index_writer
            .create_disk_layout::<Data, StorageProvider>(self.storage_provider)?;
        self.index_writer
            .index_build_cleanup(self.storage_provider)?;
        logger.log_checkpoint(DiskIndexBuildCheckpoint::DiskLayout);
        Ok(())
    }

    fn generate_compressed_data(&mut self, pool: RayonThreadPoolRef<'_>) -> ANNResult<()> {
        let parameters = &self.parameters;
        let data_path = self.index_writer.get_dataset_file();
        let max_block_size = parameters.compression_chunk_vector_count;
        validate_data_generation_input(self.storage_provider, &data_path, max_block_size)?;
        let num_chunks = parameters.search_pq_chunks;
        info!(
            "Compressing data into {} bytes per vector for disk search",
            num_chunks.get()
        );

        let quantizer_context = PQGenerationContext {
            pq_storage: self.pq_storage.clone(),
            num_chunks: num_chunks.get(),
            max_kmeans_reps: NUM_KMEANS_REPS_PQ,
            num_centers: NUM_PQ_CENTROIDS,
            seed: parameters.seed,
            p_val: MAX_PQ_TRAINING_SET_SIZE / (parameters.num_points as f64),
            storage_provider: self.storage_provider,
            pool,
            dim: parameters.dim,
            metric: parameters.metric,
        };
        let generator = QuantDataGenerator::<
            Data::VectorDataType,
            PQGeneration<Data::VectorDataType, StorageProvider>,
        >::new(
            data_path,
            self.pq_storage.get_compressed_data_path().into(),
            &quantizer_context,
        )?;
        generator.generate_data(self.storage_provider, pool, max_block_size)
    }
}

async fn build_vamana_graph<Data, StorageProvider>(
    index: &IndexConfiguration,
    parameters: &DiskIndexBuildParameters,
    quantizer: &BuildQuantizer,
    writer: &DiskIndexWriter,
    storage: &StorageProvider,
    pool: RayonThreadPoolRef<'_>,
) -> ANNResult<()>
where
    Data: GraphDataType<VectorIdType = u32>,
    Data::VectorDataType: VectorRepr,
    StorageProvider: StorageReadProvider + StorageWriteProvider + 'static,
    StorageProvider::Reader: Send,
{
    match determine_build_strategy::<Data>(
        index,
        parameters.build_memory_limit().in_bytes() as f64,
        parameters.build_quantization(),
    ) {
        IndexBuildStrategy::Merged => {
            MergedVamanaIndexBuilder::<Data, _>::new(index, parameters, writer, quantizer, storage)
                .build(pool)
                .await
        }
        IndexBuildStrategy::OneShot => {
            build_inmem_index::<Data::VectorDataType, _>(
                index.clone(),
                quantizer,
                &writer.get_dataset_file(),
                &writer.get_mem_index_file(),
                storage,
            )
            .await
        }
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
        .ok_or_else(|| {
            diskann_error!(
                ErrorKind::IndexError,
                "Failed to determine number of threads"
            )
        })?;

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

    let start_point =
        start_point_from_medoid::<T, _>(data_path, config.random_seed, storage_provider)?;
    let start_point_id = start_point.id();
    let index =
        new_inmem_index_builder::<T>(index_config, provider_parameters, quantizer, start_point)?;

    run_build(&index, dataset_iter, num_tasks).await?;

    #[cfg(debug_assertions)]
    log_build_stats::<_>(&index).await?;

    run_final_prune(&index, num_tasks).await?;
    index
        .save_graph(
            storage_provider,
            &(start_point_id, DiskGraphOnly::new(save_path)),
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
        .map_err(|_| diskann_error!(ErrorKind::IndexError, "id {value} exceeds u32::MAX"))
}

pub(super) fn start_point_from_medoid<T, StorageReader>(
    path: &str,
    random_seed: Option<u64>,
    reader: &StorageReader,
) -> ANNResult<StartPoint<T>>
where
    T: VectorRepr,
    StorageReader: StorageReadProvider,
{
    let mut rng = diskann_providers::utils::create_rnd_from_optional_seed(random_seed);
    let (medoid, medoid_id) =
        find_medoid_with_sampling::<T, _>(path, reader, MAX_MEDOID_SAMPLE_SIZE, &mut rng)?;

    let medoid_id = u32::try_from(medoid_id)?;
    Ok(StartPoint::new(medoid, medoid_id))
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
                        diskann_error!(ErrorKind::IndexError, "Poisoned mutex during construction")
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
        res.map_err(|_| diskann_error!(ErrorKind::IndexError, "A spawned insert task failed"))??;
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
        res.map_err(|_| {
            diskann_error!(ErrorKind::IndexError, "A spawned final prune task failed")
        })??;
    }

    Ok(())
}

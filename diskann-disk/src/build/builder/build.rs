/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Shared disk pipeline and configuration-selected graph construction.
use std::{
    marker::PhantomData,
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

use crate::data_model::GraphDataType;
use diskann::{
    ANNResult,
    utils::{VectorRepr, async_tools},
};
use diskann_providers::storage::{StorageReadProvider, StorageWriteProvider};
use diskann_providers::{
    model::{MAX_PQ_TRAINING_SET_SIZE, NUM_KMEANS_REPS_PQ, NUM_PQ_CENTROIDS},
    storage::PQStorage,
    utils::{
        MAX_MEDOID_SAMPLE_SIZE, RayonThreadPoolRef, VectorDataIterator, create_thread_pool,
        find_medoid_with_sampling, load_metadata_from_file,
    },
};
use diskann_utils::io::Metadata;
use tokio::task::JoinSet;
#[cfg(debug_assertions)]
use tracing::debug;
use tracing::info;

use crate::{
    build::builder::{
        backend::{Backend, BuildElement},
        core::{IndexBuildStrategy, MergedVamanaIndexBuilder, determine_build_strategy},
        inmem_builder::{self, InmemIndexBuilder, StartPoint},
        tokio::create_runtime,
    },
    build::configuration::{DiskIndexBuildConfig, VamanaBuildConfig},
    error::{ErrorKind, diskann_error},
    storage::{
        DiskIndexWriter,
        quant::{
            PQGeneration, PQGenerationContext, QuantDataGenerator, validate_data_generation_input,
        },
    },
    utils::instrumentation::{DiskIndexBuildCheckpoint, PerfLogger},
};

mod sealed {
    use diskann::ANNResult;
    use diskann_providers::storage::{StorageReadProvider, StorageWriteProvider};
    use diskann_utils::io::Metadata;

    use super::DiskIndexBuilder;
    use crate::data_model::GraphDataType;

    pub trait GraphBuild<Data>: Sized
    where
        Data: GraphDataType<VectorIdType = u32>,
    {
        fn build_disk_index<StorageProvider>(
            &self,
            builder: &DiskIndexBuilder<'_, Data, StorageProvider, Self>,
            data_path: &str,
            metadata: Metadata,
        ) -> ANNResult<()>
        where
            StorageProvider: StorageReadProvider<Reader: Send> + StorageWriteProvider + 'static;
    }
}

/// Graph-construction configuration accepted by [`DiskIndexBuilder::build`].
///
/// Implemented for [`VamanaBuildConfig`] and, when the `pipnn` feature is
/// enabled, [`diskann::graph::pipnn::Config`].
pub trait GraphBuild<Data>: sealed::GraphBuild<Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
}

impl<Data, G> GraphBuild<Data> for G
where
    Data: GraphDataType<VectorIdType = u32>,
    G: sealed::GraphBuild<Data>,
{
}

/// Shared search-PQ, graph, disk-layout and cleanup pipeline.
pub struct DiskIndexBuilder<'a, Data, StorageProvider, GraphConfig = VamanaBuildConfig>
where
    Data: GraphDataType<VectorIdType = u32>,
    StorageProvider: StorageReadProvider + StorageWriteProvider,
{
    index_writer: DiskIndexWriter,
    pq_storage: PQStorage,
    storage_provider: &'a StorageProvider,
    config: DiskIndexBuildConfig<GraphConfig>,
    _phantom: PhantomData<Data>,
}

impl<'a, Data, StorageProvider, GraphConfig>
    DiskIndexBuilder<'a, Data, StorageProvider, GraphConfig>
where
    Data: GraphDataType<VectorIdType = u32>,
    StorageProvider: StorageReadProvider<Reader: Send> + StorageWriteProvider + 'static,
    GraphConfig: GraphBuild<Data>,
{
    /// Stores the configuration; all input processing happens in `build`.
    pub fn new(
        storage_provider: &'a StorageProvider,
        config: DiskIndexBuildConfig<GraphConfig>,
        index_writer: DiskIndexWriter,
    ) -> Self {
        let pq_storage = PQStorage::new(
            &(index_writer.get_index_path_prefix() + "_pq_pivots.bin"),
            &(index_writer.get_index_path_prefix() + "_pq_compressed.bin"),
            Some(&index_writer.get_dataset_file()),
        );
        Self {
            index_writer,
            pq_storage,
            storage_provider,
            config,
            _phantom: PhantomData,
        }
    }

    /// Builds search-PQ, the selected graph and the disk layout.
    ///
    /// # Errors
    /// Returns compression, graph-construction or storage errors.
    pub fn build(&mut self) -> ANNResult<()> {
        let data_path = self.index_writer.get_dataset_file();
        let metadata = load_metadata_from_file(self.storage_provider, &data_path)?;
        self.config
            .graph
            .build_disk_index(self, &data_path, metadata)
    }
}

impl<Data, StorageProvider, GraphConfig> DiskIndexBuilder<'_, Data, StorageProvider, GraphConfig>
where
    Data: GraphDataType<VectorIdType = u32>,
    StorageProvider: StorageReadProvider<Reader: Send> + StorageWriteProvider + 'static,
{
    fn build_with(
        &self,
        metadata: Metadata,
        build_graph: impl FnOnce(RayonThreadPoolRef<'_>) -> ANNResult<()>,
    ) -> ANNResult<()> {
        let mut logger = PerfLogger::new_disk_index_build_logger();
        let pool = create_thread_pool(self.config.common.num_threads)?;
        self.generate_compressed_data(metadata, pool.as_ref())?;
        logger.log_checkpoint(DiskIndexBuildCheckpoint::PqConstruction);
        build_graph(pool.as_ref())?;
        logger.log_checkpoint(DiskIndexBuildCheckpoint::InmemIndexBuild);
        self.index_writer
            .create_disk_layout::<Data, StorageProvider>(self.storage_provider)?;
        self.index_writer
            .index_build_cleanup(self.storage_provider)?;
        logger.log_checkpoint(DiskIndexBuildCheckpoint::DiskLayout);
        Ok(())
    }

    fn generate_compressed_data(
        &self,
        metadata: Metadata,
        pool: RayonThreadPoolRef<'_>,
    ) -> ANNResult<()> {
        let parameters = &self.config.common;
        let data_path = self.index_writer.get_dataset_file();
        let max_block_size = parameters.compression_chunk_vector_count;
        validate_data_generation_input(self.storage_provider, &data_path, max_block_size)?;
        let num_chunks = parameters.search_pq_chunks;
        let quantizer_context = PQGenerationContext {
            pq_storage: self.pq_storage.clone(),
            num_chunks: num_chunks.get(),
            max_kmeans_reps: NUM_KMEANS_REPS_PQ,
            num_centers: NUM_PQ_CENTROIDS,
            seed: parameters.seed,
            p_val: MAX_PQ_TRAINING_SET_SIZE / (metadata.npoints() as f64),
            storage_provider: self.storage_provider,
            pool,
            dim: metadata.ndims(),
            metric: parameters.metric,
        };
        info!(
            "Training search-PQ quantizer with {} chunks",
            num_chunks.get()
        );
        let generator = QuantDataGenerator::<
            Data::VectorDataType,
            PQGeneration<Data::VectorDataType, StorageProvider>,
        >::new(
            data_path,
            self.pq_storage.get_compressed_data_path().into(),
            &quantizer_context,
        )?;
        info!(
            "Compressing data into {} bytes per vector for disk search",
            num_chunks.get()
        );
        generator.generate_data(self.storage_provider, pool, max_block_size)
    }
}

impl<Data> sealed::GraphBuild<Data> for VamanaBuildConfig
where
    Data: GraphDataType<VectorIdType = u32, VectorDataType: BuildElement>,
{
    fn build_disk_index<StorageProvider>(
        &self,
        builder: &DiskIndexBuilder<'_, Data, StorageProvider, Self>,
        data_path: &str,
        metadata: Metadata,
    ) -> ANNResult<()>
    where
        StorageProvider: StorageReadProvider<Reader: Send> + StorageWriteProvider + 'static,
    {
        info!(
            "Training quantizer for {} quantized builds.",
            self.build_quantization.to_string()
        );
        let backend = Backend::<Data::VectorDataType>::new(
            &self.build_quantization,
            &builder.config.common,
            metadata.npoints(),
            data_path,
            &builder.index_writer.get_index_path_prefix(),
            builder.storage_provider,
        )?;
        let runtime = create_runtime(builder.config.common.num_threads)?;
        info!(
            "Starting index build: R={} L={} Indexing RAM budget={} T={}",
            self.graph.pruned_degree(),
            self.graph.l_build(),
            self.memory_budget.in_bytes(),
            builder.config.common.num_threads
        );
        builder.build_with(metadata, |pool| {
            runtime.block_on(async {
                match determine_build_strategy::<Data>(metadata, self) {
                    IndexBuildStrategy::Merged => {
                        MergedVamanaIndexBuilder::<Data, _>::new(
                            &builder.config.common,
                            &builder.index_writer,
                            &backend,
                            builder.storage_provider,
                        )
                        .build(self, metadata, pool)
                        .await
                    }
                    IndexBuildStrategy::OneShot => {
                        build_inmem_index::<Data::VectorDataType, _>(
                            self.graph.clone(),
                            inmem_builder::Config::new(
                                metadata.npoints(),
                                builder.config.common.num_threads,
                                self.prefetch_lookahead,
                            )?,
                            builder.config.common.seed,
                            &backend,
                            data_path,
                            &builder.index_writer.get_mem_index_file(),
                            builder.storage_provider,
                        )
                        .await
                    }
                }
            })
        })
    }
}

#[cfg(feature = "pipnn")]
impl<Data> sealed::GraphBuild<Data> for diskann::graph::pipnn::Config
where
    Data: GraphDataType<VectorIdType = u32, VectorDataType: super::pipnn::PipnnBuildElement>,
{
    fn build_disk_index<StorageProvider>(
        &self,
        builder: &DiskIndexBuilder<'_, Data, StorageProvider, Self>,
        data_path: &str,
        metadata: Metadata,
    ) -> ANNResult<()>
    where
        StorageProvider: StorageReadProvider<Reader: Send> + StorageWriteProvider + 'static,
    {
        info!(
            "Starting index build: algorithm=PiPNN R={} c_max={} leaf_k={} T={}",
            self.degree(),
            self.c_max(),
            self.leaf_k(),
            builder.config.common.num_threads
        );
        builder.build_with(metadata, |pool| {
            let start_id = start_point_from_medoid::<Data::VectorDataType, _>(
                data_path,
                builder.config.common.seed,
                builder.storage_provider,
            )?
            .id();
            super::pipnn::build_inmem_index::<Data::VectorDataType, _>(
                builder.config.common.metric,
                start_id,
                self,
                data_path,
                &builder.index_writer.get_mem_index_file(),
                builder.storage_provider,
                pool,
            )
        })
    }
}

pub(super) async fn build_inmem_index<T, StorageProvider>(
    graph_config: diskann::graph::Config,
    provider_config: inmem_builder::Config,
    seed: Option<u64>,
    backend: &Backend<T>,
    data_path: &str,
    save_path: &str,
    storage_provider: &StorageProvider,
) -> ANNResult<()>
where
    T: VectorRepr,
    StorageProvider: StorageReadProvider + StorageWriteProvider + 'static,
    <StorageProvider as StorageReadProvider>::Reader: std::marker::Send,
{
    let num_tasks = provider_config.num_threads;

    // Associated data will only be used in the write_disk_layout function which only requires the none-partitioned associated data stream.
    let dataset_iter = Arc::new(Mutex::new({
        let iter = VectorDataIterator::<_, T>::new(data_path, Option::None, storage_provider)?;
        iter.enumerate()
    }));

    let start_point = start_point_from_medoid::<T, _>(data_path, seed, storage_provider)?;

    let mut index = backend.builder(graph_config, provider_config, start_point)?;

    run_build(&index, dataset_iter, num_tasks).await?;

    #[cfg(debug_assertions)]
    log_build_stats::<_>(Arc::get_mut(&mut index).expect("there should be no clones remaining"));

    run_final_prune(&index, num_tasks).await?;

    let index = Arc::get_mut(&mut index).expect("there should be no clones remaining");
    index.save_graph_mut(&mut storage_provider.create_for_write(save_path)?)?;

    Ok(())
}

/// Log statistics about the build process
#[cfg(debug_assertions)]
fn log_build_stats<T: VectorRepr>(index: &mut dyn InmemIndexBuilder<T>) {
    // Hoisting out of the `debug!` macro forces evaluation of the reachable nodes regardless
    // of the log level.
    //
    // This is useful for the side-effects since `count_reachable_nodes` serves as an
    // extra check on the integrity of the graph.
    let reachable = index.count_reachable_nodes();
    debug!("Number of points reachable in the graph: {}", reachable);
}

/// Convert a `usize` index into the `u32` internal id type, erroring if it does not fit.
///
/// The async index uses `u32` internal ids, so positions in the dataset must not exceed
/// `u32::MAX`.
fn u32_try_from(value: usize) -> ANNResult<u32> {
    u32::try_from(value)
        .map_err(|_| diskann_error!(ErrorKind::IndexError, "id {value} exceeds u32::MAX"))
}

fn start_point_from_medoid<T, StorageReader>(
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
    let partitions = async_tools::PartitionIter::new(index.capacity(), num_tasks);

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

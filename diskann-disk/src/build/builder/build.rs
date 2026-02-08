/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Async disk index builder implementation.
use std::{
    marker::PhantomData,
    num::NonZeroUsize,
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex},
};

use diskann::{
    utils::{async_tools, vecid_from_usize, TryIntoVectorId, VectorRepr, ONE},
    ANNError, ANNErrorKind, ANNResult,
};
use diskann_inmem::DefaultProviderParameters;
use diskann_providers::storage::{StorageReadProvider, StorageWriteProvider};
use diskann_providers::{
    model::{
        graph::traits::{AdHoc, GraphDataType},
        IndexConfiguration, MAX_PQ_TRAINING_SET_SIZE, NUM_KMEANS_REPS_PQ, NUM_PQ_CENTROIDS,
    },
    storage::{AsyncIndexMetadata, DiskGraphOnly, PQStorage},
    utils::{
        create_thread_pool, find_medoid_with_sampling, load_bin, save_bin_u32, RayonThreadPool,
        VectorDataIterator, MAX_MEDOID_SAMPLE_SIZE,
    },
};
use tokio::task::JoinSet;
use tracing::{debug, info};

use crate::{
    build::{
        builder::{
            core::{
                determine_build_strategy, DiskIndexBuilderCore, IndexBuildStrategy,
                MergedVamanaIndexWorkflow,
            },
            inmem_builder::{load_inmem_index_builder, new_inmem_index_builder, InmemIndexBuilder},
            quantizer::BuildQuantizer,
            tokio::create_runtime,
        },
        chunking::{
            checkpoint::{
                CheckpointContext, CheckpointManager, CheckpointManagerExt,
                NaiveCheckpointRecordManager, OwnedCheckpointContext, Progress, WorkStage,
            },
            continuation::{process_while_resource_is_available_async, ChunkingConfig},
        },
    },
    storage::{
        quant::{GeneratorContext, PQGeneration, PQGenerationContext, QuantDataGenerator},
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
        Self::new_with_chunking_config(
            storage_provider,
            disk_build_param,
            index_configuration,
            index_writer,
            ChunkingConfig::default(),
            Box::<NaiveCheckpointRecordManager>::default(),
        )
    }

    /// Create a new async disk index builder with custom chunking configuration.
    pub fn new_with_chunking_config(
        storage_provider: &'a StorageProvider,
        disk_build_param: DiskIndexBuildParameters,
        index_configuration: IndexConfiguration,
        index_writer: DiskIndexWriter,
        chunking_config: ChunkingConfig,
        mut checkpoint_record_manager: Box<dyn CheckpointManager>,
    ) -> ANNResult<Self> {
        checkpoint_record_manager.execute_stage(
            WorkStage::Start,
            WorkStage::TrainBuildQuantizer,
            || Ok(()),
            || Ok(()),
        )?;

        let pq_storage = PQStorage::new(
            &(index_writer.get_index_path_prefix() + "_pq_pivots.bin"),
            &(index_writer.get_index_path_prefix() + "_pq_compressed.bin"),
            Some(&index_writer.get_dataset_file()),
        );

        let build_quantizer = Self::train_or_load_build_quantizer(
            disk_build_param.build_quantization(),
            &index_writer.get_index_path_prefix(),
            &index_configuration,
            &pq_storage,
            storage_provider,
            checkpoint_record_manager.as_mut(),
        )?;

        let core = DiskIndexBuilderCore {
            disk_build_param,
            index_configuration,
            index_writer,
            storage_provider,
            pq_storage,
            chunking_config,
            checkpoint_record_manager,
            _phantom: std::marker::PhantomData,
        };

        Ok(Self {
            core,
            build_quantizer,
        })
    }

    /// Train or load a quantizer for async builds, using checkpoint management.
    fn train_or_load_build_quantizer(
        build_quantization_type: &QuantizationType,
        index_path_prefix: &str,
        index_configuration: &IndexConfiguration,
        pq_storage: &PQStorage,
        storage_provider: &StorageProvider,
        checkpoint_record_manager: &mut dyn CheckpointManager,
    ) -> ANNResult<BuildQuantizer> {
        info!(
            "Training quantizer for {} quantized builds.",
            build_quantization_type.to_string()
        );

        checkpoint_record_manager.execute_stage(
            WorkStage::TrainBuildQuantizer,
            WorkStage::QuantizeFPV,
            || {
                BuildQuantizer::train::<Data, _>(
                    build_quantization_type,
                    index_path_prefix,
                    index_configuration,
                    pq_storage,
                    storage_provider,
                )
            },
            || {
                info!(
                "Skipping quantizer training, instead loading from already trained quantizer saved in the file system.",
                );
                BuildQuantizer::load(
                    build_quantization_type,
                    index_path_prefix,
                    storage_provider,
                )
            },
        )
    }

    pub fn build(&mut self) -> ANNResult<()> {
        let runtime = create_runtime(self.index_configuration.num_threads)?;
        runtime.block_on(async {
            match self.build_internal().await {
                Err(err) if err.kind() == ANNErrorKind::BuildInterrupted => {
                    info!(
                        "Index build was interrupted by continuation_checker, progress saved for resumption"
                    );
                    Ok(()) // Return success for controlled interruptions
                }
                result => result, // Pass through any other result (Ok or Err)
            }
        })
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

        self.generate_compressed_data(&pool).await?;
        logger.log_checkpoint(DiskIndexBuildCheckpoint::PqConstruction);

        self.build_inmem_index(&pool).await?;
        logger.log_checkpoint(DiskIndexBuildCheckpoint::InmemIndexBuild);

        // Use physical file to pass the memory index to the disk writer
        self.create_disk_layout()?;
        logger.log_checkpoint(DiskIndexBuildCheckpoint::DiskLayout);

        Ok(())
    }

    async fn generate_compressed_data(&mut self, pool: &RayonThreadPool) -> ANNResult<()> {
        let num_points = self.index_configuration.max_points;
        let num_chunks = self.disk_build_param.search_pq_chunks();

        let storage_provider = self.core.storage_provider;

        info!(
            "Compressing data into {} bytes per vector for disk search",
            num_chunks.get()
        );

        let mut checkpoint_context = OwnedCheckpointContext::new(
            self.checkpoint_record_manager.clone_box(),
            WorkStage::QuantizeFPV,
            WorkStage::InMemIndexBuild,
        );

        let offset = match checkpoint_context.get_resumption_point()? {
            Some(offset) => offset,
            None => {
                info!("Skip the DataCompression");
                return Ok(());
            }
        };

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

        let generator_context =
            GeneratorContext::new(offset, self.pq_storage.get_compressed_data_path().into());

        let generator = QuantDataGenerator::<
            Data::VectorDataType,
            PQGeneration<Data::VectorDataType, StorageProvider, &RayonThreadPool>,
        >::new(
            self.index_writer.get_dataset_file(),
            generator_context,
            &quantizer_context,
        )?;
        let progress = generator.generate_data(storage_provider, &pool, &self.chunking_config)?;

        checkpoint_context.update(progress.clone())?;
        if let Progress::Processed(progress_point) = progress {
            let message = format!(
                "[Stage:{:?}] Build interrupt at progress {}",
                checkpoint_context.current_stage(),
                progress_point
            );
            return Err(ANNError::log_build_interrupted(message));
        }

        Ok(())
    }

    async fn build_inmem_index(&mut self, pool: &RayonThreadPool) -> ANNResult<()> {
        match determine_build_strategy::<Data>(
            &self.index_configuration,
            self.disk_build_param.build_memory_limit().in_bytes() as f64,
            self.disk_build_param.build_quantization(),
        ) {
            IndexBuildStrategy::Merged => self.build_merged_vamana_index(pool).await,
            IndexBuildStrategy::OneShot => {
                self.build_one_shot_vamana_index_with_checkpoint_record()
                    .await
            }
        }
    }

    async fn build_merged_vamana_index(&mut self, pool: &RayonThreadPool) -> ANNResult<()> {
        let mut logger = PerfLogger::new_disk_index_build_logger();
        let mut workflow = MergedVamanaIndexWorkflow::new(self, pool);

        // Partition data stage
        let num_parts = workflow.partition_data(self)?;
        logger.log_checkpoint(BuildMergedVamanaIndexCheckpoint::PartitionData);

        // build in-memory index for each partition
        for p in 0..num_parts {
            let checkpoint_context = workflow.get_shard_context(self, p, num_parts);

            // build in-memory disk for current shard partition:{shard_base_file} and save to disk
            self.build_shard_index(&workflow.merged_index_prefix, p, checkpoint_context)
                .await?;
        }
        logger.log_checkpoint(BuildMergedVamanaIndexCheckpoint::BuildIndicesOnShards);

        workflow.merge_and_cleanup(self, num_parts)?;
        logger.log_checkpoint(BuildMergedVamanaIndexCheckpoint::MergeIndices);

        Ok(())
    }

    async fn build_shard_index(
        &self,
        merged_index_prefix: &str,
        shard_id: usize,
        checkpoint_context: CheckpointContext<'_>,
    ) -> ANNResult<()> {
        let stage = checkpoint_context.current_stage();
        let shard_base_file =
            DiskIndexWriter::get_merged_index_subshard_data_file(merged_index_prefix, shard_id);

        // Determine what action to take based on the checkpoint state
        let offset = match checkpoint_context.get_resumption_point()? {
            Some(offset) => offset,
            None => {
                info!(
                    "[Stage:{:?}] Skip build_shard_index for shard {} - no valid checkpoint exists",
                    stage, shard_id
                );
                return Ok(());
            }
        };

        // 1. If checkpoint is at 0, create the shard data from IDs
        if offset == 0 {
            let shard_ids_file = DiskIndexWriter::get_merged_index_subshard_id_map_file(
                merged_index_prefix,
                shard_id,
            );

            // based on id_maps, partition original data into {num_parts} shards and save them to disk temporarily
            self.retrieve_shard_data_from_ids::<Data::VectorDataType>(
                &self.index_writer.get_dataset_file(),
                &shard_ids_file,
                &shard_base_file,
            )?;
            info!("[Stage:{:?}] Generate data for shard {}", stage, shard_id);
        } else {
            info!(
                "[Stage:{:?}] Resume shard {} build with existing data",
                stage, shard_id
            );
        }

        // 2. build in-memory disk for current shard partition:{shard_base_file} and save to disk
        let index_config = self.create_shard_index_config(&shard_base_file)?;
        let shard_prefix =
            DiskIndexWriter::get_merged_index_subshard_prefix(merged_index_prefix, shard_id);
        let shard_index_file = DiskIndexWriter::get_merged_index_subshard_mem_index_file(
            merged_index_prefix,
            shard_id,
        );

        self.build_inmem_index_with_checkpoint(
            index_config,
            checkpoint_context.to_owned(),
            &shard_base_file,
            &shard_prefix,
            &shard_index_file,
        )
        .await
    }

    async fn build_one_shot_vamana_index_with_checkpoint_record(&mut self) -> ANNResult<()> {
        let checkpoint_context = OwnedCheckpointContext::new(
            self.checkpoint_record_manager.clone_box(),
            WorkStage::InMemIndexBuild,
            WorkStage::WriteDiskLayout,
        );

        self.build_inmem_index_with_checkpoint(
            self.index_configuration.clone(),
            checkpoint_context,
            &self.index_writer.get_dataset_file(),
            &self.index_writer.get_index_path_prefix(),
            &self.index_writer.get_mem_index_file(),
        )
        .await
    }

    async fn build_inmem_index_with_checkpoint(
        &self,
        config: IndexConfiguration,
        mut checkpoint_context: OwnedCheckpointContext,
        data_path: &str,
        index_path_prefix: &str,
        save_path: &str,
    ) -> ANNResult<()> {
        let stage = checkpoint_context.current_stage();
        // Check if we have a valid checkpoint for in-memory index building
        let offset = match checkpoint_context.get_resumption_point()? {
            Some(offset) => offset,
            None => {
                info!(
                    "[Stage:{:?}] Skip in-memory index build - no valid checkpoint exists",
                    stage
                );
                return Ok(());
            }
        };

        // Mark the checkpoint record as invalid. In case of a crash, the index build will start from scratch.
        checkpoint_context.mark_as_invalid()?;

        let progress = build_inmem_index::<Data::VectorDataType, _>(
            config,
            &self.build_quantizer,
            data_path,
            index_path_prefix,
            save_path,
            offset,
            &self.chunking_config,
            self.core.storage_provider,
        )
        .await?;

        checkpoint_context.update(progress.clone())?;

        match progress {
            Progress::Processed(processed) => {
                let message = format!(
                    "[Stage:{:?}] Build interrupt at progress {}",
                    stage, processed
                );
                Err(ANNError::log_build_interrupted(message))
            }
            Progress::Completed => Ok(()),
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn build_inmem_index<T, StorageProvider>(
    config: IndexConfiguration,
    quantizer: &BuildQuantizer,
    data_path: &str,
    index_path_prefix: &str,
    save_path: &str,
    offset: usize,
    chunking_config: &ChunkingConfig,
    storage_provider: &StorageProvider,
) -> ANNResult<Progress>
where
    T: VectorRepr,
    StorageProvider: StorageReadProvider + StorageWriteProvider + 'static,
    <StorageProvider as StorageReadProvider>::Reader: std::marker::Send,
{
    if offset >= config.max_points {
        return Err(ANNError::log_index_error(format!(
            "Offset {} exceeds max points {}",
            offset, config.max_points
        )));
    }

    // use either user-specified number of threads or default to available parallelism
    let num_tasks = NonZeroUsize::new(config.num_threads)
        .or_else(|| std::thread::available_parallelism().ok())
        .ok_or_else(|| ANNError::log_index_error("Failed to determine number of threads"))?;

    // Associated data will only be used in the write_disk_layout function which only requires the none-partitioned associated data stream.
    let dataset_iter = Arc::new(Mutex::new({
        let iter =
            VectorDataIterator::<_, AdHoc<T>>::new(data_path, Option::None, storage_provider)?;
        iter.enumerate().skip(offset)
    }));

    let index_store: IndexStore<'_, _, T> = if offset == 0 {
        IndexStore::new_index(
            config,
            quantizer,
            data_path,
            index_path_prefix,
            storage_provider,
        )
        .await?
    } else {
        IndexStore::from_checkpoint(
            config,
            quantizer,
            index_path_prefix,
            offset,
            storage_provider,
        )
        .await?
    };

    let index = &index_store.index;
    let progress =
        run_build_with_chunking(index, dataset_iter, num_tasks, offset, chunking_config).await?;

    #[cfg(debug_assertions)]
    log_build_stats::<_>(index).await?;

    match progress {
        Progress::Processed(processed) => {
            index_store.save_checkpoint(processed).await?;
        }

        Progress::Completed => {
            // If the progress is completed, we can run the final prune and save the index to disk.
            run_final_prune(index, num_tasks).await?;

            index_store.save_final_index(save_path).await?;
        }
    }

    Ok(progress)
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

async fn run_build_with_chunking<T, I>(
    index: &Arc<dyn InmemIndexBuilder<T>>,
    iterator: Arc<Mutex<I>>,
    num_tasks: NonZeroUsize,
    offset: usize,
    chunking_config: &ChunkingConfig,
) -> ANNResult<Progress>
where
    T: VectorRepr,
    I: Iterator<Item = (usize, (Box<[T]>, ()))> + Send + 'static,
{
    let total_points = index.capacity();
    let chunk_size = chunking_config.inmemory_build_chunk_vector_count;

    let chunks = (offset..total_points)
        .step_by(chunk_size) // Create an infinite iterator that steps by `chunk_size`.
        .take_while(move |start| start < &total_points) // Take elements while the start is less than the range.
        .map(move |start| (start, usize::min(start + chunk_size, total_points)));

    let progress = process_while_resource_is_available_async(
        |chunk| process_chunk(index, iterator.clone(), num_tasks, chunk.0, chunk.1),
        chunks,
        chunking_config.continuation_checker.clone_box(),
    )
    .await?
    .map(|num_chunks| num_chunks * chunk_size);

    match progress {
        Progress::Processed(num_points) => {
            info!(
                "Linked #{} points. Start #{}, end #{} ",
                num_points,
                offset,
                num_points + offset
            );
        }
        Progress::Completed => {
            info!("Linked all points. Num points: #{}", total_points);
        }
    }

    let progress = progress.map(|num_points| num_points + offset);

    Ok(progress)
}

async fn process_chunk<T, Iter>(
    index: &Arc<dyn InmemIndexBuilder<T>>,
    iterator: Arc<Mutex<Iter>>,
    num_tasks: NonZeroUsize,
    start: usize,
    end: usize,
) -> ANNResult<()>
where
    T: VectorRepr,
    Iter: Iterator<Item = (usize, (Box<[T]>, ()))> + Send + 'static,
{
    debug!("Processing chunk from #{} to #{}", start, end);

    let partitions = async_tools::PartitionIter::new(end - start, num_tasks);

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
                        let id = vecid_from_usize(i)?;
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

    debug!("Completed chunk #{} to #{}", start, end);
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
            let start_index = partition.start.try_into_vector_id()?;
            let end_index = partition.end.try_into_vector_id()?;

            let range = start_index..end_index;
            index_clone.final_prune(range).await
        });
    }

    // Wait for all final prune tasks to complete
    while let Some(res) = tasks.join_next().await {
        res.map_err(|_| ANNError::log_index_error("A spawned final prune task failed"))??;
    }

    Ok(())
}

/// Manages the persistence operations for in-memory index building
struct IndexStore<'a, S, T>
where
    S: StorageReadProvider + StorageWriteProvider,
{
    pub index: Arc<dyn InmemIndexBuilder<T>>,
    start_point: StartPoint,

    metadata: AsyncIndexMetadata,
    storage_provider: &'a S,
    _phantom: PhantomData<T>,
}

impl<'a, S, T> IndexStore<'a, S, T>
where
    S: StorageReadProvider + StorageWriteProvider + 'static,
    <S as StorageReadProvider>::Reader: std::marker::Send,
    T: VectorRepr,
{
    /// Create a new persistence manager with a fresh index
    pub async fn new_index(
        config: IndexConfiguration,
        build_quantizer: &BuildQuantizer,
        data_path: &'a str,
        index_path_prefix: &str,
        storage_provider: &'a S,
    ) -> ANNResult<IndexStore<'a, S, T>> {
        // Create new index
        let index_config = config.config.clone();

        let provider_parameters = DefaultProviderParameters {
            max_points: config.max_points,
            frozen_points: ONE,
            metric: config.dist_metric,
            dim: config.dim,
            // This is the true maximum degree.
            max_degree: index_config.max_degree_u32().get(),
            prefetch_lookahead: config.prefetch_lookahead.map(|x| x.get()),
            prefetch_cache_line_level: config.prefetch_cache_line_level,
        };

        let index =
            new_inmem_index_builder::<T>(index_config, provider_parameters, build_quantizer)?;

        let medoid_id = set_start_point_to_medoid::<T, _>(
            &index,
            data_path,
            config.random_seed,
            storage_provider,
        )?;
        let start_point = StartPoint::new(vecid_from_usize(medoid_id)?);

        Ok(Self {
            index,
            start_point,
            metadata: AsyncIndexMetadata::new(index_path_prefix),
            storage_provider,
            _phantom: PhantomData,
        })
    }

    /// Load an existing index from a checkpoint
    pub async fn from_checkpoint(
        config: IndexConfiguration,
        build_quantizer: &BuildQuantizer,
        index_path_prefix: &str,
        _offset: usize,
        storage_provider: &'a S,
    ) -> ANNResult<Self> {
        let metadata = AsyncIndexMetadata::new(index_path_prefix);

        // Load existing index from resumable context
        let index = load_inmem_index_builder::<T, _>(
            storage_provider,
            build_quantizer,
            config,
            index_path_prefix,
        )
        .await?;

        // Load existing start point
        let start_point =
            StartPoint::load(&metadata.additional_points_id_path(), storage_provider)?;

        Ok(Self {
            index,
            start_point,
            metadata,
            storage_provider,
            _phantom: PhantomData,
        })
    }

    /// Save intermediate state during build interruption
    pub async fn save_checkpoint(&self, _processed: usize) -> ANNResult<()> {
        self.index
            .save_index(self.storage_provider, &self.metadata)
            .await?;

        self.start_point.save(
            &self.metadata.additional_points_id_path(),
            self.storage_provider,
        )?;

        Ok(())
    }

    /// Save the finalized index to disk
    pub async fn save_final_index(&self, save_path: &str) -> ANNResult<()> {
        self.clean_temp_files()?;

        // Use physical file to contact with index writer
        self.index
            .save_graph(
                self.storage_provider,
                &(self.start_point.id(), DiskGraphOnly::new(save_path)),
            )
            .await?;

        Ok(())
    }

    /// Removes temporary files created during index building
    fn clean_temp_files(&self) -> ANNResult<()> {
        let files = [
            self.metadata.prefix().to_string(),
            self.metadata.data_path(),
            self.metadata.additional_points_id_path(),
        ];

        for file in files.iter() {
            if self.storage_provider.exists(file) {
                debug!("Deleting temporary file: {}", file);
                self.storage_provider.delete(file)?;
            }
        }
        Ok(())
    }
}

/// Manages persistence of start point IDs for resumable builds
struct StartPoint(u32);

impl StartPoint {
    fn new(id: u32) -> Self {
        Self(id)
    }

    fn load<StorageReader>(path: &str, reader: &StorageReader) -> ANNResult<Self>
    where
        StorageReader: StorageReadProvider,
    {
        if !reader.exists(path) {
            return Err(ANNError::log_file_not_found_error(format!(
                "Start point ID file {} does not exist",
                path
            )));
        }
        let (data, _, _) = load_bin::<u32, _>(&mut reader.open_reader(path)?, 0)?;

        let start_point_id = data.first().ok_or_else(|| {
            ANNError::log_invalid_file_format(format!("Start point ID file {} is empty", path))
        })?;

        debug!("Loaded start point ID {} from {}", *start_point_id, path);
        Ok(Self(*start_point_id))
    }

    fn save<StorageWriter>(&self, path: &str, storage_provider: &StorageWriter) -> ANNResult<()>
    where
        StorageWriter: StorageWriteProvider,
    {
        save_bin_u32(
            &mut storage_provider.create_for_write(path)?,
            std::slice::from_ref(&self.0),
            1,
            1,
            0,
        )?;
        debug!("Saved start point ID {} to {}", self.0, path);
        Ok(())
    }

    fn id(&self) -> u32 {
        self.0
    }
}

#[cfg(test)]
mod start_point_tests {
    use std::io::Write;

    use diskann_providers::storage::VirtualStorageProvider;
    use diskann_providers::utils::write_metadata;
    use vfs::MemoryFS;

    use super::*;

    #[test]
    fn test_start_point_creation() {
        let id = 42u32;
        let start_point = StartPoint::new(id);
        assert_eq!(start_point.id(), id);
    }

    #[test]
    fn test_start_point_save_and_load() {
        let file_path = "/start_point_test.bin";
        let fs = MemoryFS::new();
        let storage_provider = VirtualStorageProvider::new(fs);

        // Create and save a start point
        let id = 42u32;
        let start_point = StartPoint::new(id);
        start_point.save(file_path, &storage_provider).unwrap();

        // Load the start point and verify it matches the original
        let loaded_start_point = StartPoint::load(file_path, &storage_provider).unwrap();
        assert_eq!(loaded_start_point.id(), id);
    }

    #[test]
    fn test_start_point_load_nonexistent_file() {
        let storage_provider = VirtualStorageProvider::new(MemoryFS::new());
        let result = StartPoint::load("/nonexistent_file.bin", &storage_provider);
        assert_eq!(
            result.err().unwrap().kind(),
            ANNErrorKind::FileNotFoundError
        );
    }

    #[test]
    fn test_start_point_load_empty_file() {
        let file_path = "/empty_file.bin";
        let fs = MemoryFS::new();
        let storage_provider = VirtualStorageProvider::new(fs);

        // Create an empty file
        {
            let mut file = storage_provider.create_for_write(file_path).unwrap();
            file.write_all(&[]).unwrap();
        }

        let result = StartPoint::load(file_path, &storage_provider);
        assert_eq!(result.err().unwrap().kind(), ANNErrorKind::IOError);
    }

    #[test]
    fn test_start_point_load_invalid_data() {
        let file_path = "/invalid_data.bin";
        let fs = MemoryFS::new();
        let storage_provider = VirtualStorageProvider::new(fs);

        // Create a file with invalid data
        {
            let mut file = storage_provider.create_for_write(file_path).unwrap();
            let npts = 0;
            let dim = 1;
            write_metadata(&mut file, npts, dim).unwrap();
        }

        let result = StartPoint::load(file_path, &storage_provider);
        assert_eq!(
            result.err().unwrap().kind(),
            ANNErrorKind::InvalidFileFormatError
        );
    }
}

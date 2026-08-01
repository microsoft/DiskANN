/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    marker::PhantomData,
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

use diskann::{
    utils::{async_tools, VectorRepr, ONE},
    ANNResult,
};
use diskann_providers::{
    model::{graph::provider::async_::inmem::DefaultProviderParameters, IndexConfiguration},
    storage::{DiskGraphOnly, StorageReadProvider, StorageWriteProvider},
    utils::{find_medoid_with_sampling, VectorDataIterator, MAX_MEDOID_SAMPLE_SIZE},
};
use tokio::task::JoinSet;
use tracing::{debug, info};

use crate::{
    build::builder::quantizer::BuildQuantizer,
    error::{diskann_error, ErrorKind},
};

use super::index::VamanaBuildIndex;
/// Builds a complete Vamana graph from a dataset in one pass.
pub(in crate::build::builder) struct OneShotVamanaBuilder<'a, T, StorageProvider>
where
    T: VectorRepr,
    StorageProvider: StorageReadProvider + StorageWriteProvider,
{
    config: &'a IndexConfiguration,
    quantizer: &'a BuildQuantizer,
    data_path: String,
    save_path: String,
    storage_provider: &'a StorageProvider,
    _phantom: PhantomData<T>,
}

impl<'a, T, StorageProvider> OneShotVamanaBuilder<'a, T, StorageProvider>
where
    T: VectorRepr,
    StorageProvider: StorageReadProvider + StorageWriteProvider + 'static,
    <StorageProvider as StorageReadProvider>::Reader: Send,
{
    pub(in crate::build::builder) fn new(
        config: &'a IndexConfiguration,
        quantizer: &'a BuildQuantizer,
        data_path: String,
        save_path: String,
        storage_provider: &'a StorageProvider,
    ) -> Self {
        Self {
            config,
            quantizer,
            data_path,
            save_path,
            storage_provider,
            _phantom: PhantomData,
        }
    }

    pub(in crate::build::builder) async fn build(self) -> ANNResult<()> {
        let Self {
            config,
            quantizer,
            data_path,
            save_path,
            storage_provider,
            ..
        } = self;

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
            let iter = VectorDataIterator::<_, T>::new(&data_path, None, storage_provider)?;
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
        let index = VamanaBuildIndex::<T>::new(index_config, provider_parameters, quantizer)?;
        let medoid_id = Self::set_start_point_to_medoid(
            &index,
            &data_path,
            config.random_seed,
            storage_provider,
        )?;
        let start_point = Self::u32_try_from(medoid_id)?;

        Self::run_build(&index, dataset_iter, num_tasks).await?;

        #[cfg(debug_assertions)]
        Self::log_build_stats(&index).await?;

        Self::run_final_prune(&index, num_tasks).await?;
        index
            .save_graph(
                storage_provider,
                &(start_point, DiskGraphOnly::new(&save_path)),
            )
            .await?;

        Ok(())
    }

    /// Log statistics about the build process
    #[cfg(debug_assertions)]
    async fn log_build_stats(index: &VamanaBuildIndex<T>) -> ANNResult<()> {
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

    fn set_start_point_to_medoid(
        index: &VamanaBuildIndex<T>,
        path: &str,
        random_seed: Option<u64>,
        reader: &StorageProvider,
    ) -> ANNResult<usize> {
        let mut rng = diskann_providers::utils::create_rnd_from_optional_seed(random_seed);
        let (medoid, medoid_id) =
            find_medoid_with_sampling::<T, _>(path, reader, MAX_MEDOID_SAMPLE_SIZE, &mut rng)?;

        index.set_start_point(medoid.as_slice())?;

        debug!("Set start point to medoid ID: {}", medoid_id);

        Ok(medoid_id)
    }

    async fn run_build<I>(
        index: &VamanaBuildIndex<T>,
        iterator: Arc<Mutex<I>>,
        num_tasks: NonZeroUsize,
    ) -> ANNResult<()>
    where
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
                            diskann_error!(
                                ErrorKind::IndexError,
                                "Poisoned mutex during construction"
                            )
                        })?;
                        guard.next()
                    };

                    match vector_data {
                        Some((i, (vector, _))) => {
                            let id = Self::u32_try_from(i)?;
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
            res.map_err(|_| {
                diskann_error!(ErrorKind::IndexError, "A spawned insert task failed")
            })??;
        }

        info!("Linked all points. Num points: #{}", total_points);
        Ok(())
    }

    async fn run_final_prune(
        index: &VamanaBuildIndex<T>,
        num_tasks: NonZeroUsize,
    ) -> ANNResult<()> {
        let partitions = async_tools::PartitionIter::new(index.total_points(), num_tasks);

        let mut tasks = JoinSet::new();

        for partition in partitions {
            let index_clone = index.clone();
            tasks.spawn(async move {
                let range =
                    Self::u32_try_from(partition.start)?..Self::u32_try_from(partition.end)?;
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
}

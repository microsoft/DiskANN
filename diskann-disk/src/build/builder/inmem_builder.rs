/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    marker::PhantomData,
    num::NonZeroUsize,
    pin::Pin,
    sync::{Arc, Mutex},
};

use diskann::{
    graph::{
        glue::{InsertStrategy, PruneStrategy},
        Config, DiskANNIndex,
    },
    provider::DefaultContext,
    utils::{async_tools, VectorRepr, ONE},
    ANNError, ANNResult,
};
use diskann_providers::storage::{
    DynWriteProvider, StorageReadProvider, StorageWriteProvider, WriteProviderWrapper,
};
use diskann_providers::{
    index::diskann_async,
    model::graph::provider::async_::{
        common::{FullPrecision, NoDeletes, NoStore, Quantized, SetElementHelper, VectorStore},
        inmem::{
            DefaultProvider, DefaultProviderParameters, FullPrecisionProvider, SetStartPoints,
        },
    },
    model::IndexConfiguration,
    storage::{DiskGraphOnly, SaveWith},
    utils::{find_medoid_with_sampling, VectorDataIterator, MAX_MEDOID_SAMPLE_SIZE},
};
use diskann_utils::future::{AsyncFriendly, SendFuture};
use tokio::task::JoinSet;
use tracing::{debug, info};

use super::quantizer::BuildQuantizer;

/// Builder facade for in memory index construction and persistence.
///
/// Thread safety:
/// Implementors must be `Send` and `Sync`. Methods can be called from many tasks.
trait InmemIndexBuilder<T: Sized>: Send + Sync {
    /// Return the total capacity of the provider, **excluding** start points.
    fn capacity(&self) -> usize;

    /// Return the total capacity of the provider, **including** start points.
    fn total_points(&self) -> usize;

    /// Set a single start point to search.
    ///
    /// The slice must match the underlying vector type, else `WrongDataType` is returned.
    fn set_start_point(&self, start_point: &[T]) -> ANNResult<()>;

    /// Insert a vector with a `id`.
    ///
    /// The slice must match the underlying vector type.
    fn insert_vector<'a>(
        &'a self,
        id: u32,
        vector: &'a [T],
    ) -> Pin<Box<dyn SendFuture<ANNResult<()>> + 'a>>;

    /// Prune the built graph over `[range.start, range.end)`.
    fn final_prune(
        &self,
        range: core::ops::Range<u32>,
    ) -> Pin<Box<dyn SendFuture<ANNResult<()>> + '_>>;

    /// Persist only the graph file set.
    fn save_graph<'a>(
        &'a self,
        storage_provider: &'a dyn DynWriteProvider,
        start_point_and_path: &'a (u32, DiskGraphOnly),
    ) -> Pin<Box<dyn SendFuture<ANNResult<()>> + 'a>>;

    /// Return the number of vector reads for full_precision and quantized stores respectively.
    #[cfg(debug_assertions)]
    fn counts_for_get_vector(&self) -> (usize, usize);

    /// Count the number of nodes in the graph reachable from the given `start_points`.
    ///
    /// This function has a large memory footprint for large graphs and should not be called
    /// frequently. This is mainly for analysis and sanity tests.
    #[cfg(debug_assertions)]
    fn count_reachable_nodes(&self) -> Pin<Box<dyn SendFuture<ANNResult<usize>> + '_>>;
}

//////////////////////////////////
// FullPrecision Implementation //
//////////////////////////////////

impl<T> InmemIndexBuilder<T> for DiskANNIndex<FullPrecisionProvider<T>>
where
    T: VectorRepr,
{
    fn capacity(&self) -> usize {
        self.provider().capacity()
    }

    fn total_points(&self) -> usize {
        self.provider().total_points()
    }

    fn set_start_point(&self, start_point: &[T]) -> ANNResult<()> {
        self.provider()
            .set_start_points(std::iter::once(start_point))
    }

    fn insert_vector<'a>(
        &'a self,
        id: u32,
        vector: &'a [T],
    ) -> Pin<Box<dyn SendFuture<ANNResult<()>> + 'a>> {
        Box::pin(async move {
            self.insert(&FullPrecision, &DefaultContext, &id, vector)
                .await
        })
    }

    fn final_prune(
        &self,
        range: core::ops::Range<u32>,
    ) -> Pin<Box<dyn SendFuture<ANNResult<()>> + '_>> {
        Box::pin(async move {
            self.prune_range(&FullPrecision, &DefaultContext, range)
                .await
        })
    }

    fn save_graph<'a>(
        &'a self,
        storage_provider: &'a dyn DynWriteProvider,
        start_point_and_path: &'a (u32, DiskGraphOnly),
    ) -> Pin<Box<dyn SendFuture<ANNResult<()>> + 'a>> {
        Box::pin(async move {
            let wrapper = WriteProviderWrapper::new(storage_provider);
            self.save_with(&wrapper, start_point_and_path).await
        })
    }

    #[cfg(debug_assertions)]
    fn counts_for_get_vector(&self) -> (usize, usize) {
        self.provider().counts_for_get_vector()
    }

    #[cfg(debug_assertions)]
    fn count_reachable_nodes(&self) -> Pin<Box<dyn SendFuture<ANNResult<usize>> + '_>> {
        Box::pin(async move {
            let provider = self.provider();
            let start_points = provider.starting_points()?;
            let mut neighbor_accessor = provider.neighbors();
            self.count_reachable_nodes(&start_points, &mut neighbor_accessor)
                .await
        })
    }
}

//////////////////////////
// Quant Implementation //
//////////////////////////

struct QuantInMemBuilder<T, Q>
where
    Q: AsyncFriendly,
{
    index: DiskANNIndex<DefaultProvider<NoStore, Q>>,
    _vector_data_type: PhantomData<T>,
}

impl<T, Q> QuantInMemBuilder<T, Q>
where
    Q: AsyncFriendly,
{
    fn new(index: DiskANNIndex<DefaultProvider<NoStore, Q>>) -> Self {
        Self {
            index,
            _vector_data_type: PhantomData,
        }
    }

    fn index(&self) -> &DiskANNIndex<DefaultProvider<NoStore, Q>> {
        &self.index
    }
}

impl<T, Q> InmemIndexBuilder<T> for QuantInMemBuilder<T, Q>
where
    T: VectorRepr,
    Q: AsyncFriendly + VectorStore + SetElementHelper<T>,
    Quantized: for<'a> InsertStrategy<'a, DefaultProvider<NoStore, Q>, &'a [T]>
        + PruneStrategy<DefaultProvider<NoStore, Q>>,
    DefaultProvider<NoStore, Q>: SaveWith<(u32, u32, DiskGraphOnly), Error = ANNError>,
{
    fn capacity(&self) -> usize {
        self.index().provider().capacity()
    }

    fn total_points(&self) -> usize {
        self.index().provider().total_points()
    }

    fn set_start_point(&self, start_point: &[T]) -> ANNResult<()> {
        self.index()
            .provider()
            .set_start_points(std::iter::once(start_point))
    }

    fn insert_vector<'a>(
        &'a self,
        id: u32,
        vector: &'a [T],
    ) -> Pin<Box<dyn SendFuture<ANNResult<()>> + 'a>> {
        Box::pin(async move {
            self.index()
                .insert(&Quantized, &DefaultContext, &id, vector)
                .await
        })
    }

    fn final_prune(
        &self,
        range: core::ops::Range<u32>,
    ) -> Pin<Box<dyn SendFuture<ANNResult<()>> + '_>> {
        Box::pin(async move {
            self.index()
                .prune_range(&Quantized, &DefaultContext, range)
                .await
        })
    }

    fn save_graph<'a>(
        &'a self,
        storage_provider: &'a dyn DynWriteProvider,
        start_point_and_path: &'a (u32, DiskGraphOnly),
    ) -> Pin<Box<dyn SendFuture<ANNResult<()>> + 'a>> {
        Box::pin(async move {
            let wrapper = WriteProviderWrapper::new(storage_provider);
            self.index().save_with(&wrapper, start_point_and_path).await
        })
    }

    #[cfg(debug_assertions)]
    fn counts_for_get_vector(&self) -> (usize, usize) {
        self.index().provider().counts_for_get_vector()
    }

    #[cfg(debug_assertions)]
    fn count_reachable_nodes(&self) -> Pin<Box<dyn SendFuture<ANNResult<usize>> + '_>> {
        Box::pin(async move {
            let provider = self.index().provider();
            let start_points = provider.starting_points()?;
            let mut neighbor_accessor = provider.neighbors();
            self.index()
                .count_reachable_nodes(&start_points, &mut neighbor_accessor)
                .await
        })
    }
}

/// Create a new in-memory index builder for vectors of type `T`.
///
/// Chooses the builder implementation based on the given `BuildQuantizer`.
/// - `NoQuant` uses a plain index with no quantization.
/// - `Scalar1Bit` and `PQ` create quantized only indexes backed by `QuantInMemBuilder`.
///
/// # Parameters
/// * `config` – Index configuration.
/// * `build_quantizer` – Quantization strategy to apply.
///
/// # Returns
/// An `Arc<dyn InmemIndexBuilder>` wrapped in `ANNResult`.
///
/// # Errors
/// Returns an error if the underlying index creation fails.
fn new_inmem_index_builder<T>(
    config: Config,
    params: DefaultProviderParameters,
    build_quantizer: &BuildQuantizer,
) -> ANNResult<Arc<dyn InmemIndexBuilder<T>>>
where
    T: VectorRepr,
{
    match &build_quantizer {
        BuildQuantizer::NoQuant(_) => diskann_async::new_index::<T, _>(config, params, NoDeletes)
            .map(|index| index as Arc<dyn InmemIndexBuilder<T>>),
        BuildQuantizer::Scalar1Bit(q) => {
            let index = diskann_async::new_quant_only_index(config, params, q.clone(), NoDeletes)?;
            Ok(Arc::new(QuantInMemBuilder::<T, _>::new(index)))
        }
        BuildQuantizer::PQ(table) => {
            let index =
                diskann_async::new_quant_only_index(config, params, table.clone(), NoDeletes)?;
            Ok(Arc::new(QuantInMemBuilder::<T, _>::new(index)))
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
/// The in-memory index uses `u32` internal ids, so positions in the dataset must not exceed
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

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{marker::PhantomData, pin::Pin, sync::Arc};

use diskann::{
    graph::{
        glue::{InsertStrategy, PruneStrategy},
        Config, DiskANNIndex,
    },
    provider::DefaultContext,
    utils::VectorRepr,
    ANNError, ANNResult,
};
use diskann_providers::storage::{DynWriteProvider, StorageReadProvider, WriteProviderWrapper};
use diskann_providers::{
    index::diskann_async,
    model::{
        graph::provider::async_::{
            common::{FullPrecision, NoDeletes, NoStore, Quantized, SetElementHelper, VectorStore},
            inmem::{
                DefaultProvider, DefaultProviderParameters, DefaultQuant, FullPrecisionProvider,
                SQStore, SetStartPoints,
            },
        },
        IndexConfiguration,
    },
    storage::{
        index_storage::load_index, load_fp_index, AsyncIndexMetadata, DiskGraphOnly, SaveWith,
    },
};
use diskann_utils::future::{AsyncFriendly, SendFuture};

use super::quantizer::BuildQuantizer;

/// Builder facade for in memory index construction and persistence.
///
/// Thread safety:
/// Implementors must be `Send` and `Sync`. Methods can be called from many tasks.
pub(super) trait InmemIndexBuilder<T: Sized>: Send + Sync {
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

    /// Persist the full index layout and metadata.
    fn save_index<'a>(
        &'a self,
        storage_provider: &'a dyn DynWriteProvider,
        metadata: &'a AsyncIndexMetadata,
    ) -> Pin<Box<dyn SendFuture<ANNResult<()>> + 'a>>;

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
            self.insert(FullPrecision, &DefaultContext, &id, vector)
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

    fn save_index<'a>(
        &'a self,
        storage_provider: &'a dyn DynWriteProvider,
        metadata: &'a AsyncIndexMetadata,
    ) -> Pin<Box<dyn SendFuture<ANNResult<()>> + 'a>> {
        Box::pin(async move {
            let wrapper = WriteProviderWrapper::new(storage_provider);
            self.save_with(&wrapper, metadata).await
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

pub(super) struct QuantInMemBuilder<T, Q>
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
    pub fn new(index: DiskANNIndex<DefaultProvider<NoStore, Q>>) -> Self {
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
    Quantized: InsertStrategy<DefaultProvider<NoStore, Q>, [T]>
        + PruneStrategy<DefaultProvider<NoStore, Q>>,
    DefaultProvider<NoStore, Q>: SaveWith<(u32, AsyncIndexMetadata), Error = ANNError>,
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
                .insert(Quantized, &DefaultContext, &id, vector)
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

    fn save_index<'a>(
        &'a self,
        storage_provider: &'a dyn DynWriteProvider,
        metadata: &'a AsyncIndexMetadata,
    ) -> Pin<Box<dyn SendFuture<ANNResult<()>> + 'a>> {
        Box::pin(async move {
            let wrapper = WriteProviderWrapper::new(storage_provider);
            self.index().save_with(&wrapper, metadata).await
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
pub(super) fn new_inmem_index_builder<T>(
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

/// Loads an in memory index builder from storage based on the given `BuildQuantizer`.
///
/// Depending on the quantizer type:
/// - `NoQuant` loads a full precision index with `NoStore`.
/// - `Scalar1Bit` loads a quant only index with `SQStore<1>`.
/// - `PQ` loads a quant only index with `DefaultQuant`.
///
/// # Type Parameters
/// - `T`: Vector element type, must implement `VectorRepr`.
/// - `P`: Storage provider, must implement `StorageReadProvider`.
///
/// # Arguments
/// - `storage_provider`: Source to read index data.
/// - `build_quantizer`: Selects which index to load.
/// - `config`: Index configuration.
/// - `index_path_prefix`: Path prefix for index files.
///
/// # Returns
/// An `Arc<dyn InmemIndexBuilder>` ready for use, or an error if loading fails.
///
/// # Async
/// This function is async and must be awaited.
pub(super) async fn load_inmem_index_builder<T, P>(
    storage_provider: &P,
    build_quantizer: &BuildQuantizer,
    config: IndexConfiguration,
    index_path_prefix: &str,
) -> ANNResult<Arc<dyn InmemIndexBuilder<T>>>
where
    P: StorageReadProvider,
    T: VectorRepr,
{
    match build_quantizer {
        BuildQuantizer::NoQuant(_) => {
            load_fp_index::<T, _, NoStore>(storage_provider, index_path_prefix, config)
                .await
                .map(|index| Arc::new(index) as Arc<dyn InmemIndexBuilder<T>>)
        }
        BuildQuantizer::Scalar1Bit(_) => {
            let index =
                load_index::<_, NoStore, SQStore<1>>(storage_provider, index_path_prefix, config)
                    .await?;
            Ok(Arc::new(QuantInMemBuilder::<T, _>::new(index)))
        }
        BuildQuantizer::PQ(_) => {
            let index =
                load_index::<_, NoStore, DefaultQuant>(storage_provider, index_path_prefix, config)
                    .await?;
            Ok(Arc::new(QuantInMemBuilder::<T, _>::new(index)))
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_trait_definition() {
        // Test verifies the trait and types compile
        assert!(true);
    }
}

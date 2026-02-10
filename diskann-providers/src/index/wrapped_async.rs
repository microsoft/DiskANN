/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use futures_executor::block_on;
use std::{num::NonZeroUsize, sync::Arc};

use diskann::{
    ANNError, ANNResult,
    graph::{
        self, ConsolidateKind, InplaceDeleteMethod, SearchParams,
        glue::{
            self, AsElement, InplaceDeleteStrategy, InsertStrategy, PruneStrategy, SearchStrategy,
        },
        index::{DegreeStats, PartitionedNeighbors, SearchState, SearchStats},
        search_output_buffer,
    },
    neighbor::Neighbor,
    provider::{AsNeighbor, AsNeighborMut, DataProvider, Delete, SetElement},
    utils::{ONE, async_tools::VectorIdBoxSlice},
};

use crate::{
    model::IndexConfiguration,
    storage::{
        AsyncIndexMetadata, AsyncQuantLoadContext, LoadWith, StorageReadProvider,
        file_storage_provider::FileStorageProvider,
    },
};

pub struct DiskANNIndex<DP: DataProvider> {
    /// The underlying async DiskANNIndex.
    pub inner: Arc<graph::DiskANNIndex<DP>>,
    _runtime: Option<tokio::runtime::Runtime>,
    handle: tokio::runtime::Handle,
}

impl<DP> DiskANNIndex<DP>
where
    DP: DataProvider,
{
    /// Construct a synchronous `DiskANNIndex` with its own `tokio::runtime::Runtime`.
    ///
    /// A default configured multi-threaded runtime will be created and used behind the scenes. To use
    /// a specific Toktio runtime, use `DiskANNIndex::new_with_multi_thread_runtime()` or `DiskANNIndex::new_with_handle()`.
    pub fn new_with_multi_thread_runtime(config: graph::Config, data_provider: DP) -> Self {
        #[allow(clippy::expect_used)]
        let rt = tokio::runtime::Builder::new_multi_thread()
            .build()
            .expect("failed to create tokio runtime");

        let handle = rt.handle().clone();

        Self::new_internal(config, data_provider, Some(rt), handle, Some(ONE))
    }

    /// Construct a synchronous `DiskANNIndex` with its own `tokio::runtime::Runtime`.
    ///
    /// A default configured runtime that uses the curren thread will be created and used behind the scenes. To use
    /// a specific Toktio runtime, use `DiskANNIndex::new_with_multi_thread_runtime()` or `DiskANNIndex::new_with_handle()`.
    pub fn new_with_current_thread_runtime(config: graph::Config, data_provider: DP) -> Self {
        #[allow(clippy::expect_used)]
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("failed to create tokio runtime");

        let handle = rt.handle().clone();

        Self::new_internal(config, data_provider, Some(rt), handle, Some(ONE))
    }

    /// Construct a synchronous `DiskANNIndex` that uses a provided `tokio::runtime::Handle`.
    ///
    /// The `tokio::runtime::Runtime` is owned externally and we just keep a `Handle` to it.
    pub fn new_with_handle(
        config: graph::Config,
        data_provider: DP,
        handle: tokio::runtime::Handle,
        thread_hint: Option<NonZeroUsize>,
    ) -> Self {
        Self::new_internal(config, data_provider, None, handle, thread_hint)
    }

    fn new_internal(
        config: graph::Config,
        data_provider: DP,
        runtime: Option<tokio::runtime::Runtime>,
        handle: tokio::runtime::Handle,
        thread_hint: Option<NonZeroUsize>,
    ) -> Self {
        let inner = Arc::new(graph::DiskANNIndex::new(config, data_provider, thread_hint));

        Self {
            inner,
            _runtime: runtime,
            handle,
        }
    }
    pub fn insert<S, T>(
        &self,
        strategy: S,
        context: &DP::Context,
        id: &DP::ExternalId,
        vector: &T,
    ) -> ANNResult<()>
    where
        S: InsertStrategy<DP, T>,
        T: Sync + ?Sized,
        DP: SetElement<T>,
    {
        self.handle
            .block_on(self.inner.insert(strategy, context, id, vector))
    }

    pub fn multi_insert<S, T>(
        &self,
        strategy: S,
        context: &DP::Context,
        vector_id_pairs: Box<[VectorIdBoxSlice<DP::ExternalId, T>]>,
    ) -> ANNResult<()>
    where
        Self: 'static,
        T: Send + Sync + 'static,
        S: InsertStrategy<DP, [T]> + Clone + Send + Sync,
        DP: SetElement<[T]>,
        S::PruneStrategy: Clone,
        for<'a> glue::aliases::InsertPruneAccessor<'a, S, DP, [T]>: AsElement<&'a [T]>,
    {
        self.handle
            .block_on(self.inner.multi_insert(strategy, context, vector_id_pairs))
    }

    pub fn is_any_neighbor_deleted<NA>(
        &self,
        context: &DP::Context,
        accessor: &mut NA,
        vector_id: DP::InternalId,
    ) -> ANNResult<bool>
    where
        DP: Delete,
        NA: AsNeighbor<Id = DP::InternalId>,
    {
        self.handle.block_on(
            self.inner
                .is_any_neighbor_deleted(context, accessor, vector_id),
        )
    }

    pub fn drop_adj_list<NA>(&self, accessor: &mut NA, vector_id: DP::InternalId) -> ANNResult<()>
    where
        NA: AsNeighborMut<Id = DP::InternalId>,
    {
        self.handle
            .block_on(self.inner.drop_adj_list(accessor, vector_id))
    }

    #[allow(clippy::type_complexity)]
    pub fn get_undeleted_neighbors<NA>(
        &self,
        context: &DP::Context,
        accessor: &mut NA,
        vector_id: DP::InternalId,
    ) -> ANNResult<PartitionedNeighbors<DP::InternalId>>
    where
        DP: Delete,
        NA: AsNeighbor<Id = DP::InternalId>,
    {
        self.handle.block_on(
            self.inner
                .get_undeleted_neighbors(context, accessor, vector_id),
        )
    }

    pub fn inplace_delete<S>(
        &self,
        strategy: S,
        context: &DP::Context,
        id: &DP::ExternalId,
        num_to_replace: usize,
        inplace_delete_method: InplaceDeleteMethod,
    ) -> ANNResult<()>
    where
        S: InplaceDeleteStrategy<DP>
            + for<'a> SearchStrategy<DP, S::DeleteElement<'a>>
            + Sync
            + Clone,
        DP: Delete,
    {
        self.handle.block_on(self.inner.inplace_delete(
            strategy,
            context,
            id,
            num_to_replace,
            inplace_delete_method,
        ))
    }

    pub fn drop_deleted_neighbors<NA>(
        &self,
        context: &DP::Context,
        accessor: &mut NA,
        vector_id: DP::InternalId,
        only_orphans: bool,
    ) -> ANNResult<ConsolidateKind>
    where
        DP: Delete,
        NA: AsNeighborMut<Id = DP::InternalId>,
    {
        self.handle.block_on(self.inner.drop_deleted_neighbors(
            context,
            accessor,
            vector_id,
            only_orphans,
        ))
    }

    pub fn consolidate_vector<S>(
        &self,
        strategy: &S,
        context: &DP::Context,
        vector_id: DP::InternalId,
    ) -> ANNResult<ConsolidateKind>
    where
        DP: Delete,
        S: PruneStrategy<DP>,
    {
        self.handle
            .block_on(self.inner.consolidate_vector(strategy, context, vector_id))
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn search<S, T, O, OB>(
        &self,
        strategy: &S,
        context: &DP::Context,
        query: &T,
        search_params: &SearchParams,
        output: &mut OB,
    ) -> ANNResult<SearchStats>
    where
        T: Sync + ?Sized,
        S: SearchStrategy<DP, T, O>,
        O: Send,
        OB: search_output_buffer::SearchOutputBuffer<O> + Send,
    {
        self.handle.block_on(
            self.inner
                .search(strategy, context, query, search_params, output),
        )
    }

    #[allow(clippy::type_complexity)]
    pub fn start_paged_search<S, T>(
        &self,
        strategy: S,
        context: &DP::Context,
        query: &T,
        l_value: usize,
    ) -> ANNResult<SearchState<DP::InternalId, (S, S::QueryComputer)>>
    where
        S: SearchStrategy<DP, T>,
        T: Sync + ?Sized,
    {
        self.handle.block_on(
            self.inner
                .start_paged_search(strategy, context, query, l_value),
        )
    }

    #[allow(clippy::type_complexity)]
    pub fn start_paged_search_with_init_ids<S, T>(
        &self,
        strategy: S,
        context: &DP::Context,
        query: &T,
        l_value: usize,
        init_ids: Option<&[DP::InternalId]>,
    ) -> ANNResult<SearchState<DP::InternalId, (S, S::QueryComputer)>>
    where
        S: SearchStrategy<DP, T>,
        T: Sync + ?Sized,
    {
        self.handle.block_on(
            self.inner
                .start_paged_search_with_init_ids(strategy, context, query, l_value, init_ids),
        )
    }

    pub fn next_search_results<S, T>(
        &self,
        context: &DP::Context,
        search_state: &mut SearchState<DP::InternalId, (S, S::QueryComputer)>,
        k: usize,
        result_output: &mut [Neighbor<DP::InternalId>],
    ) -> ANNResult<usize>
    where
        S: SearchStrategy<DP, T>,
        T: Send + Sync + ?Sized,
    {
        self.handle.block_on(self.inner.next_search_results(
            context,
            search_state,
            k,
            result_output,
        ))
    }

    pub fn count_reachable_nodes<NA>(
        &self,
        start_points: &[DP::InternalId],
        accessor: &mut NA,
    ) -> ANNResult<usize>
    where
        NA: AsNeighbor<Id = DP::InternalId>,
    {
        self.handle
            .block_on(self.inner.count_reachable_nodes(start_points, accessor))
    }

    pub fn get_degree_stats<NA>(&self, accessor: &mut NA) -> ANNResult<DegreeStats>
    where
        for<'a> &'a DP: IntoIterator<Item = DP::InternalId, IntoIter: Send>,
        NA: AsNeighbor<Id = DP::InternalId>,
    {
        self.handle.block_on(self.inner.get_degree_stats(accessor))
    }
}

pub trait SyncLoadWith<DP>: Sized {
    fn load_with(path: &str, index_config: IndexConfiguration) -> ANNResult<Self>;
}

// Load static memory index from pre-built files synchronously
impl<DP> SyncLoadWith<DP> for DiskANNIndex<DP>
where
    DP: DataProvider<InternalId = u32> + LoadWith<AsyncQuantLoadContext, Error = ANNError>,
{
    fn load_with(path: &str, index_config: IndexConfiguration) -> ANNResult<DiskANNIndex<DP>> {
        let storage = FileStorageProvider;
        let data_provider = create_data_provider(&storage, path, &index_config);

        match data_provider {
            Ok(dp) => {
                let index =
                    DiskANNIndex::<DP>::new_with_current_thread_runtime(index_config.config, dp);
                Ok(index)
            }
            Err(e) => return Err(e),
        }
    }
}

pub fn create_data_provider<'a, P, DP>(
    provider: &P,
    path: &'a str,
    index_config: &'a IndexConfiguration,
) -> ANNResult<DP>
where
    P: StorageReadProvider,
    DP: DataProvider + LoadWith<AsyncQuantLoadContext, Error = ANNError>,
{
    let pq_context = AsyncQuantLoadContext {
        metadata: AsyncIndexMetadata::new(path),
        num_frozen_points: index_config.num_frozen_pts,
        metric: index_config.dist_metric,
        prefetch_lookahead: index_config.prefetch_lookahead.map(|x| x.get()),
        is_disk_index: false, // only support in-memory index loading here
        prefetch_cache_line_level: index_config.prefetch_cache_line_level,
    };

    let data_provider = DP::load_with(provider, &pq_context);

    block_on(data_provider)
}

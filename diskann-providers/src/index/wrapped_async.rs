/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{num::NonZeroUsize, sync::Arc};

use diskann::{
    ANNError, ANNResult,
    graph::{
        self, ConsolidateKind, InplaceDeleteMethod,
        glue::{
            self, AsElement, DefaultSearchStrategy, InplaceDeleteStrategy, InsertStrategy,
            PruneStrategy, SearchStrategy,
        },
        index::{DegreeStats, PartitionedNeighbors, SearchState, SearchStats},
        search::Knn,
        search_output_buffer,
    },
    neighbor::Neighbor,
    provider::{AsNeighbor, AsNeighborMut, DataProvider, Delete, SetElement},
    utils::{ONE, async_tools::VectorIdBoxSlice},
};

use crate::storage::{LoadWith, StorageReadProvider};

/// Synchronous wrapper around [`graph::DiskANNIndex`] that owns or borrows a tokio runtime.
pub struct DiskANNIndex<DP: DataProvider> {
    /// The underlying async DiskANNIndex.
    pub inner: Arc<graph::DiskANNIndex<DP>>,
    /// Keeps the runtime alive when `Self` owns it; `None` when using an external handle.
    _runtime: Option<tokio::runtime::Runtime>,
    handle: tokio::runtime::Handle,
}

/// Create a multi-threaded tokio runtime and return it together with its handle.
fn create_multi_thread_runtime() -> (tokio::runtime::Runtime, tokio::runtime::Handle) {
    #[allow(clippy::expect_used)]
    let rt = tokio::runtime::Builder::new_multi_thread()
        .build()
        .expect("failed to create tokio runtime");
    let handle = rt.handle().clone();
    (rt, handle)
}

/// Create a current-thread tokio runtime and return it together with its handle.
fn create_current_thread_runtime() -> (tokio::runtime::Runtime, tokio::runtime::Handle) {
    #[allow(clippy::expect_used)]
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("failed to create tokio runtime");
    let handle = rt.handle().clone();
    (rt, handle)
}

impl<DP> DiskANNIndex<DP>
where
    DP: DataProvider,
{
    /// Construct a synchronous `DiskANNIndex` with its own multi-threaded `tokio::runtime::Runtime`.
    ///
    /// A default multi-threaded runtime will be created and owned by `Self`. For a single-threaded
    /// runtime use [`new_with_current_thread_runtime`](Self::new_with_current_thread_runtime), or
    /// to supply an external runtime handle use [`new_with_handle`](Self::new_with_handle).
    pub fn new_with_multi_thread_runtime(config: graph::Config, data_provider: DP) -> Self {
        let (rt, handle) = create_multi_thread_runtime();
        Self::new_internal(config, data_provider, Some(rt), handle, Some(ONE))
    }

    /// Construct a synchronous `DiskANNIndex` with its own single-threaded `tokio::runtime::Runtime`.
    ///
    /// A default current-thread runtime will be created and owned by `Self`. For a multi-threaded
    /// runtime use [`new_with_multi_thread_runtime`](Self::new_with_multi_thread_runtime), or
    /// to supply an external runtime handle use [`new_with_handle`](Self::new_with_handle).
    pub fn new_with_current_thread_runtime(config: graph::Config, data_provider: DP) -> Self {
        let (rt, handle) = create_current_thread_runtime();
        Self::new_internal(config, data_provider, Some(rt), handle, Some(ONE))
    }

    /// Construct a synchronous `DiskANNIndex` that uses a provided `tokio::runtime::Handle`.
    ///
    /// The `tokio::runtime::Runtime` is owned externally and we just keep a `Handle` to it.
    /// `thread_hint` is forwarded to [`graph::DiskANNIndex::new`] to size internal thread pools;
    /// pass `None` to let it choose a default.
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

    /// Run an arbitrary async operation against the underlying
    /// [`graph::DiskANNIndex`] using this wrapper's tokio runtime.
    ///
    /// This is a catch-all escape hatch for async methods on the inner index
    /// that do not (yet) have a dedicated synchronous wrapper. The closure
    /// receives an `&Arc<graph::DiskANNIndex<DP>>` and should return a future.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let stats = index.run(|inner| inner.some_async_method(&ctx))?;
    /// ```
    pub fn run<F, Fut, R>(&self, f: F) -> R
    where
        F: FnOnce(&Arc<graph::DiskANNIndex<DP>>) -> Fut,
        Fut: core::future::Future<Output = R>,
    {
        self.handle.block_on(f(&self.inner))
    }

    /// Load a prebuilt index from storage with its own multi-threaded `tokio::runtime::Runtime`.
    ///
    /// This is the synchronous equivalent of
    /// [`LoadWith::load_with`](crate::storage::LoadWith::load_with).
    /// A default multi-threaded runtime is created and owned by `Self`.
    /// For a single-threaded runtime use [`load_with_current_thread_runtime`](Self::load_with_current_thread_runtime),
    /// or to supply an external runtime handle use [`load_with_handle`](Self::load_with_handle).
    pub fn load_with_multi_thread_runtime<T, P>(provider: &P, auxiliary: &T) -> ANNResult<Self>
    where
        graph::DiskANNIndex<DP>: LoadWith<T, Error = ANNError>,
        P: StorageReadProvider,
    {
        let (rt, handle) = create_multi_thread_runtime();
        let inner = handle.block_on(graph::DiskANNIndex::<DP>::load_with(provider, auxiliary))?;
        Ok(Self {
            inner: Arc::new(inner),
            _runtime: Some(rt),
            handle,
        })
    }

    /// Load a prebuilt index from storage with its own single-threaded `tokio::runtime::Runtime`.
    ///
    /// This is the synchronous equivalent of
    /// [`LoadWith::load_with`](crate::storage::LoadWith::load_with).
    /// A default current-thread runtime is created and owned by `Self`.
    /// For a multi-threaded runtime use [`load_with_multi_thread_runtime`](Self::load_with_multi_thread_runtime),
    /// or to supply an external runtime handle use [`load_with_handle`](Self::load_with_handle).
    pub fn load_with_current_thread_runtime<T, P>(provider: &P, auxiliary: &T) -> ANNResult<Self>
    where
        graph::DiskANNIndex<DP>: LoadWith<T, Error = ANNError>,
        P: StorageReadProvider,
    {
        let (rt, handle) = create_current_thread_runtime();
        let inner = handle.block_on(graph::DiskANNIndex::<DP>::load_with(provider, auxiliary))?;
        Ok(Self {
            inner: Arc::new(inner),
            _runtime: Some(rt),
            handle,
        })
    }

    /// Load a prebuilt index from storage using a provided `tokio::runtime::Handle`.
    ///
    /// This is the synchronous equivalent of
    /// [`LoadWith::load_with`](crate::storage::LoadWith::load_with).
    /// The `tokio::runtime::Runtime` is owned externally and we just keep a `Handle` to it.
    /// For an owned runtime use [`load_with_multi_thread_runtime`](Self::load_with_multi_thread_runtime)
    /// or [`load_with_current_thread_runtime`](Self::load_with_current_thread_runtime).
    pub fn load_with_handle<T, P>(
        provider: &P,
        auxiliary: &T,
        handle: tokio::runtime::Handle,
    ) -> ANNResult<Self>
    where
        graph::DiskANNIndex<DP>: LoadWith<T, Error = ANNError>,
        P: StorageReadProvider,
    {
        let inner = handle.block_on(graph::DiskANNIndex::<DP>::load_with(provider, auxiliary))?;
        Ok(Self {
            inner: Arc::new(inner),
            _runtime: None,
            handle,
        })
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

    pub fn search<S, T, O, OB>(
        &self,
        strategy: &S,
        context: &DP::Context,
        query: &T,
        search_params: &Knn,
        output: &mut OB,
    ) -> ANNResult<SearchStats>
    where
        T: Sync + ?Sized,
        S: DefaultSearchStrategy<DP, T, O>,
        O: Send,
        OB: search_output_buffer::SearchOutputBuffer<O> + Send,
    {
        let knn_search = *search_params;
        self.handle.block_on(
            self.inner
                .search(knn_search, strategy, context, query, output),
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use diskann::{
        graph::{self, search_output_buffer},
        provider::DefaultContext,
        utils::ONE,
    };
    use diskann_utils::test_data_root;
    use diskann_vector::distance::Metric;

    use super::DiskANNIndex;
    use crate::{
        index::diskann_async,
        model::{
            configuration::IndexConfiguration,
            graph::provider::async_::{
                common::{FullPrecision, TableBasedDeletes},
                inmem::{self, CreateFullPrecision, DefaultProvider},
            },
        },
        storage::{AsyncIndexMetadata, SaveWith, StorageReadProvider, VirtualStorageProvider},
        utils::create_rnd_from_seed_in_tests,
    };

    #[test]
    fn test_save_then_sync_load_round_trip() {
        // -- Build an index in async context and save it -----------------------
        let save_path = "/index";
        let file_path = "/sift/siftsmall_learn_256pts.fbin";

        let train_data = {
            let storage = VirtualStorageProvider::new_overlay(test_data_root());
            let mut reader = storage.open_reader(file_path).unwrap();
            diskann_utils::io::read_bin::<f32>(&mut reader).unwrap()
        };

        let pq_bytes = 8;
        let pq_table = diskann_async::train_pq(
            train_data.as_view(),
            pq_bytes,
            &mut create_rnd_from_seed_in_tests(0xe3c52ef001bc7ade),
            2,
        )
        .unwrap();

        let (build_config, parameters) = diskann_async::simplified_builder(
            20,
            32,
            Metric::L2,
            train_data.ncols(),
            train_data.nrows(),
            |_| {},
        )
        .unwrap();

        let fp_precursor =
            CreateFullPrecision::new(parameters.dim, parameters.prefetch_cache_line_level);
        let data_provider =
            DefaultProvider::new_empty(parameters, fp_precursor, pq_table, TableBasedDeletes)
                .unwrap();

        let index =
            DiskANNIndex::new_with_current_thread_runtime(build_config.clone(), data_provider);

        let storage = VirtualStorageProvider::new_memory();
        let ctx = DefaultContext;
        for (i, v) in train_data.row_iter().enumerate() {
            index.insert(FullPrecision, &ctx, &(i as u32), v).unwrap();
        }

        let save_metadata = AsyncIndexMetadata::new(save_path.to_string());
        let storage_ref = &storage;
        let metadata_ref = &save_metadata;
        index
            .run(|inner| {
                let inner = Arc::clone(inner);
                async move { inner.save_with(storage_ref, metadata_ref).await }
            })
            .unwrap();

        // -- Reload via the synchronous wrapped_async API ----------------------
        let load_config = IndexConfiguration::new(
            Metric::L2,
            train_data.ncols(),
            train_data.nrows(),
            ONE,
            1,
            build_config,
        );

        type TestProvider = inmem::FullPrecisionProvider<
            f32,
            crate::model::graph::provider::async_::FastMemoryQuantVectorProviderAsync,
            crate::model::graph::provider::async_::TableDeleteProviderAsync,
        >;

        let loaded: DiskANNIndex<TestProvider> =
            DiskANNIndex::load_with_current_thread_runtime(&storage, &(save_path, load_config))
                .unwrap();

        // -- Verify the loaded index is functional -----------------------------
        // A single search call is enough to confirm the sync wrapper loaded a
        // working index. Exhaustive search-correctness is tested elsewhere.
        let top_k = 5;
        let search_l = 20;
        let mut ids = vec![0u32; top_k];
        let mut distances = vec![0.0f32; top_k];
        let mut output = search_output_buffer::IdDistance::new(&mut ids, &mut distances);

        let query = train_data.row(0);
        let search_params = graph::search::Knn::new_default(top_k, search_l).unwrap();
        let stats = loaded
            .search(
                &FullPrecision,
                &DefaultContext,
                query,
                &search_params,
                &mut output,
            )
            .unwrap();

        assert_eq!(stats.result_count, top_k as u32);
        // The query is itself in the dataset, so the nearest neighbor must be at distance 0.
        assert_eq!(ids[0], 0);
        assert_eq!(distances[0], 0.0);
    }
}

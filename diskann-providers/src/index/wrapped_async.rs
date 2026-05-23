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
            Batch, DefaultSearchStrategy, InplaceDeleteStrategy, InsertStrategy,
            MultiInsertStrategy, PruneStrategy, SearchAccessor, SearchStrategy,
        },
        index::{DegreeStats, PartitionedNeighbors},
        search_output_buffer,
    },
    neighbor::Neighbor,
    provider::{AsNeighbor, AsNeighborMut, DataProvider, Delete, SetElement},
    utils::ONE,
};
use diskann_utils::Reborrow;

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

    pub fn insert<'a, S, T>(
        &'a self,
        strategy: &'a S,
        context: &'a DP::Context,
        id: &DP::ExternalId,
        vector: T,
    ) -> ANNResult<()>
    where
        S: InsertStrategy<'a, DP, T>,
        DP: SetElement<T>,
        T: Copy + Send + 'a,
    {
        self.handle
            .block_on(self.inner.insert(strategy, context, id, vector))
    }

    pub fn multi_insert<S, B>(
        &self,
        strategy: S,
        context: &DP::Context,
        vectors: Arc<B>,
        ids: Arc<[DP::ExternalId]>,
    ) -> ANNResult<()>
    where
        Self: 'static,
        S: MultiInsertStrategy<DP, B>,
        B: Batch,
        DP: for<'a> SetElement<B::Element<'a>>,
    {
        self.handle
            .block_on(self.inner.multi_insert(strategy, context, vectors, ids))
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
        S: InplaceDeleteStrategy<DP> + Sync + Clone,
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

    pub fn search<'a, S, T, O, OB, P>(
        &'a self,
        search_params: P,
        strategy: &'a S,
        context: &'a DP::Context,
        query: T,
        output: &mut OB,
    ) -> ANNResult<P::Output>
    where
        P: graph::search::Search<'a, DP, S, T>,
        S: DefaultSearchStrategy<'a, DP, T, O>,
        O: Send,
        OB: search_output_buffer::SearchOutputBuffer<O> + Send + ?Sized,
        T: 'a,
    {
        self.handle.block_on(
            self.inner
                .search(search_params, strategy, context, query, output),
        )
    }

    /// Begin a paged search over the index (synchronous wrapper).
    ///
    /// Returns a [`PagedSearch`] handle. See
    /// [`PagedSearch::next_page`] for retrieving results.
    pub fn paged_search<'a, S, T>(
        &'a self,
        strategy: &'a S,
        context: &'a DP::Context,
        query: T,
        l_value: usize,
    ) -> ANNResult<PagedSearch<'a, DP, S::SearchAccessor>>
    where
        S: SearchStrategy<'a, DP, T> + 'static,
        T: Copy + Send + 'a,
    {
        let inner = self
            .handle
            .block_on(self.inner.paged_search(strategy, context, query, l_value))?;
        Ok(PagedSearch {
            handle: self.handle.clone(),
            inner,
        })
    }

    /// Begin a paged search with explicit initial seed IDs (synchronous wrapper).
    pub fn paged_search_with_init_ids<'a, S, T>(
        &'a self,
        strategy: &'a S,
        context: &'a DP::Context,
        query: T,
        l_value: usize,
        init_ids: Option<&'a [DP::InternalId]>,
    ) -> ANNResult<PagedSearch<'a, DP, S::SearchAccessor>>
    where
        S: SearchStrategy<'a, DP, T> + 'static,
        T: Copy + Send + 'a,
    {
        let inner = self.handle.block_on(
            self.inner
                .paged_search_with_init_ids(strategy, context, query, l_value, init_ids),
        )?;
        Ok(PagedSearch {
            handle: self.handle.clone(),
            inner,
        })
    }

    /// Begin a synchronous paged search over the index.
    ///
    /// This will construct a [`noawait::PagedSearch`] and initialize search with the
    /// providers start points. Pages can be retrieved with [`noawait::PagedSearch::next`].
    ///
    /// **Caution**: This method should only be used if is known that all functions reachable
    /// via the implementation of [`SearchStrategy`] are known to be synchronous and never
    /// truly await. This allows [`noawait::PagedSearch`] to be much more efficient.
    pub fn paged_search_no_await<S, T>(
        &self,
        strategy: S,
        context: DP::Context,
        query: T,
        l_value: usize,
    ) -> ANNResult<noawait::PagedSearch<DP::InternalId>>
    where
        T: for<'a> Reborrow<'a, Target: Copy + Send> + 'static,
        S: for<'a> SearchStrategy<'a, DP, <T as Reborrow<'a>>::Target> + 'static,
    {
        noawait::PagedSearch::new(self.inner.clone(), strategy, context, query, l_value)
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

    pub fn get_degree_stats<NA, Itr>(&self, accessor: &mut NA, itr: Itr) -> ANNResult<DegreeStats>
    where
        Itr: IntoIterator<Item = DP::InternalId, IntoIter: Send> + Send,
        NA: AsNeighbor<Id = DP::InternalId>,
    {
        self.handle
            .block_on(self.inner.get_degree_stats(accessor, itr))
    }
}

/// Synchronous wrapper around [`graph::search::PagedSearch`] that owns a tokio runtime handle.
///
/// Created by [`DiskANNIndex::paged_search`]. Each call to [`next_page`](Self::next_page)
/// blocks the current thread to drive the underlying async search forward.
pub struct PagedSearch<'a, DP, A>
where
    DP: DataProvider,
    A: SearchAccessor<Id = DP::InternalId> + 'a,
{
    handle: tokio::runtime::Handle,
    inner: graph::search::PagedSearch<'a, DP, A>,
}

impl<'a, DP, A> PagedSearch<'a, DP, A>
where
    DP: DataProvider,
    A: SearchAccessor<Id = DP::InternalId> + 'a,
{
    /// Returns the next page of at most `k` nearest-neighbor results.
    ///
    /// Blocks the current thread. Returns an empty `Vec` when the search is exhausted.
    pub fn next_page(&mut self, k: usize) -> ANNResult<Vec<Neighbor<DP::InternalId>>> {
        self.handle.block_on(self.inner.next_page(k))
    }
}

pub mod noawait {
    //! Implementations of a synchronous wrapper around [`diskann::graph::DiskANNIndex`] that
    //! assume the [`Accessor`] and associated implementations never truly `await` and are
    //! in fact synchronous.
    //!
    //! With this assumption, we can perform lighter-weight communication with the index
    //! by assuming that each `poll` returns ready.
    //!
    //! **Do not use this if your index ever actually await**: Doing so will lead to deadlock!

    use super::*;

    use std::{
        cell::RefCell,
        pin::Pin,
        rc::Rc,
        task::{Context, Poll, Waker},
    };

    use diskann::{ANNErrorKind, utils::VectorId};
    use diskann_utils::Reborrow;
    use thiserror::Error;

    type Input = Rc<RefCell<Option<usize>>>;
    type Output<I> = Rc<RefCell<Option<Vec<Neighbor<I>>>>>;

    fn channel<I>() -> (Input, Output<I>)
    where
        I: VectorId,
    {
        let input = Rc::new(RefCell::new(None));
        let output = Rc::new(RefCell::new(None));
        (input, output)
    }

    fn step<I>(fut: Pin<&mut dyn Future<Output = I>>) -> Option<I> {
        let mut cx = Context::from_waker(Waker::noop());
        match fut.poll(&mut cx) {
            Poll::Ready(v) => Some(v),
            Poll::Pending => None,
        }
    }

    /// A synchronous wrapper for [`graph::search::PagedSearch`]
    ///
    /// See: [`super::DiskANNIndex::paged_search_no_await`].
    pub struct PagedSearch<I: VectorId> {
        // The `input` is wrapped in an `Option` so we can fuse `searcher` if it exits
        // with an error. Polling a completed future risk panicking.
        //
        // We construct `searcher` to pull its next-page size from this input.
        input: Option<Input>,

        // Output yielded from polling `searcher`.
        output: Output<I>,

        // We shut down the future by running `Drop`. Thus, the only way it can actually
        // finish is if it returns with an error.
        searcher: Pin<Box<dyn Future<Output = ANNError>>>,
    }

    impl<I> PagedSearch<I>
    where
        I: VectorId,
    {
        /// Construct a new [`PagedSearch`].
        ///
        /// This works by creating a small async task using [`graph::search::PagedSearch`]
        /// internally. The requested k-nearest neighors are sent using a `Rc<RefCell<_>>`
        /// channel and the actual neighbors are retrieved from a similar data structure.
        ///
        /// Under the assumption that the implementation of [`graph::search::PagedSearch`]'s
        /// implementations are fully synchronous, we can directly poll this task instead
        /// of going through a runtime since we (theoretically) control the only suspension
        /// point.
        ///
        /// Doing so allows stepping the task state machine to be done with a single function
        /// call to `Future::poll`.
        ///
        /// Obviously, if the "noawait" assumption is broken, then the inner async job may
        /// yield before our control point, but we can detect this situation since no output
        /// will be generated on the output channel.
        ///
        /// We rely on `Drop` to clean up the paged search resources.
        pub(super) fn new<DP, S, T>(
            index: Arc<diskann::graph::DiskANNIndex<DP>>,
            strategy: S,
            context: DP::Context,
            query: T,
            l_value: usize,
        ) -> ANNResult<Self>
        where
            DP: DataProvider<InternalId = I>,
            T: for<'a> Reborrow<'a, Target: Copy + Send> + 'static,
            S: for<'a> SearchStrategy<'a, DP, <T as Reborrow<'a>>::Target> + 'static,
        {
            // Prepare the input and output channels used to communicate with the search task.
            let (input, output) = channel::<I>();
            let input_clone = input.clone();
            let output_clone = output.clone();

            // Create the search task.
            let mut searcher: Pin<Box<dyn Future<Output = ANNError>>> = Box::pin(async move {
                // The assumption of `noawait` is that this call will always resolve to
                // `Poll::Ready`.
                let mut state = match index
                    .paged_search(&strategy, &context, query.reborrow(), l_value)
                    .await
                {
                    Ok(state) => state,
                    Err(err) => return err,
                };

                loop {
                    // This is the await point that pauses the future.
                    //
                    // Under the "noawait" assumption, this should be the only point where
                    // this future ever yields `Pending` and is where we expect the future
                    // to stop every time we poll it.
                    futures_util::pending!();

                    // We control the invocation of poll and should always ensure that
                    // input is available.
                    let k_value = match input_clone.take() {
                        Some(value) => value,
                        None => return InternalInvariantViolated::MissingInput.into(),
                    };

                    // Step paged search and propagate any errors.
                    let page = match state.next_page(k_value).await {
                        Ok(page) => page,
                        Err(err) => return err,
                    };

                    // Send output to the caller.
                    output_clone.replace(Some(page));
                }
            });

            // Drive the inner future one step to initialize paged search.
            if let Some(err) = step(searcher.as_mut()) {
                return Err(err);
            }

            let this = Self {
                input: Some(input),
                output,
                searcher,
            };
            Ok(this)
        }

        /// Retrieve the next results from paged search, returning any errors.
        ///
        /// If [`next`](Self::next) previously returned with an error, it will continue
        /// to do so.
        pub fn next(&mut self, k: usize) -> ANNResult<Vec<Neighbor<I>>> {
            // Prepare input. We use the presence of the input channel to decide whether
            // or not it is safe to poll search task.
            match self.input.as_ref() {
                Some(input) => input.replace(Some(k)),
                None => {
                    return Err(ANNError::message(
                        ANNErrorKind::Opaque,
                        "paged searcher errored and is no longer runnable",
                    ));
                }
            };

            // Progress the future.
            //
            // The only reason to return return `Some` is if the inner future aborts with
            // an error. Here, we fuse the searcher to prevent panics on re-enters and
            // forward the error.
            if let Some(result) = step(self.searcher.as_mut()) {
                self.input = None;
                return Err(result);
            }

            // Profit!
            match self.output.take() {
                Some(v) => Ok(v),
                None => Err(InternalInvariantViolated::MissingOutput.into()),
            }
        }
    }

    #[derive(Debug, Clone, Copy, Error)]
    enum InternalInvariantViolated {
        #[error("INTERNAL: input channel was not configured")]
        MissingInput,
        #[error("noawait contract violated: future suspended before expected yield point")]
        MissingOutput,
    }

    impl From<InternalInvariantViolated> for ANNError {
        #[track_caller]
        #[cold]
        fn from(err: InternalInvariantViolated) -> Self {
            Self::new(ANNErrorKind::Opaque, err)
        }
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
            crate::utils::create_thread_pool(2).unwrap().as_ref(),
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
            index.insert(&FullPrecision, &ctx, &(i as u32), v).unwrap();
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
        let kind = graph::search::Knn::new_default(top_k, search_l).unwrap();
        let stats = loaded
            .search(kind, &FullPrecision, &DefaultContext, query, &mut output)
            .unwrap();

        assert_eq!(stats.result_count, top_k as u32);
        // The query is itself in the dataset, so the nearest neighbor must be at distance 0.
        assert_eq!(ids[0], 0);
        assert_eq!(distances[0], 0.0);
    }

    fn wrapped_test_provider() -> DiskANNIndex<graph::test::provider::Provider> {
        let provider =
            graph::test::provider::Provider::grid(graph::test::synthetic::Grid::One, 100).unwrap();

        DiskANNIndex::new_with_current_thread_runtime(
            graph::config::Builder::new(
                provider.max_degree(),
                diskann::graph::config::MaxDegree::same(),
                100,
                (Metric::L2).into(),
            )
            .build()
            .unwrap(),
            provider,
        )
    }

    // Test the `noawait` paged searcher.
    //
    // This relies on the test-provider being no-await.
    #[test]
    fn test_paged_search_noawait() {
        let index = wrapped_test_provider();

        for page_size in [1, 5, 9, 12] {
            let mut paged = index
                .paged_search_no_await::<_, Vec<f32>>(
                    graph::test::provider::Strategy::new(),
                    graph::test::provider::Context::new(),
                    vec![0.0],
                    10.max(page_size),
                )
                .unwrap();

            let mut i = 0u32;
            loop {
                let v = paged.next(page_size).unwrap();
                assert!(
                    v.len() <= page_size,
                    "candidates returned ({}) exceeded page size ({})",
                    v.len(),
                    page_size,
                );

                if v.is_empty() {
                    break;
                }

                for neighbor in v {
                    assert_ne!(
                        neighbor.id,
                        u32::MAX,
                        "paged search should not return start point",
                    );
                    assert_eq!(
                        neighbor.id, i,
                        "monotonicity should at least hold for the 1d grid"
                    );
                    assert_eq!(
                        neighbor.distance,
                        (i as f32) * (i as f32),
                        "distance was computed incorrectly!",
                    );
                    i += 1;
                }
            }

            // Search is exhausted - make sure that subsequent searches yield empty vectors.
            let exhausted = paged.next(5).unwrap();
            assert!(
                exhausted.is_empty(),
                "expected an empty vector when exhausted - instead got {:?}",
                exhausted
            );
        }
    }

    // Verify that the searcher is properly fused when it returns with an error.
    #[test]
    fn test_paged_search_noawait_fuse() {
        let index = wrapped_test_provider();

        // To do this test, we request more neighbors than the search-L, which triggers
        // an inner error.
        let search_l = 10;
        let bigger_than_search_l = 20;

        let mut paged = index
            .paged_search_no_await::<_, Vec<f32>>(
                graph::test::provider::Strategy::new(),
                graph::test::provider::Context::new(),
                vec![0.0],
                search_l,
            )
            .unwrap();

        let expected = "search_param_l";
        let err = paged.next(bigger_than_search_l).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains(expected),
            "expected error message to contain \"{}\" - instead got\n\n{}",
            expected,
            msg,
        );

        // Now that we've yielded an error - the next time we request pages should also error.
        let err = paged.next(10).unwrap_err();
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("paged searcher errored"),
            "unexpected error message:\n\n{}",
            err_msg
        );
    }
}

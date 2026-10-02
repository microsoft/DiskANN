/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::Arc;

use diskann::{
    ANNResult,
    graph::{
        self, InplaceDeleteMethod,
        glue::{DefaultSearchStrategy, InplaceDeleteStrategy, InsertStrategy},
        search_output_buffer,
    },
    provider::{DataProvider, Delete, SetElement},
    utils::ONE,
};

/// Synchronous wrapper around [`graph::DiskANNIndex`] for Garnet.
pub(crate) struct DiskANNIndex<DP: DataProvider> {
    pub(crate) inner: Arc<graph::DiskANNIndex<DP>>,
    _runtime: Option<tokio::runtime::Runtime>,
    handle: tokio::runtime::Handle,
}

fn create_current_thread_runtime() -> (tokio::runtime::Runtime, tokio::runtime::Handle) {
    #[expect(clippy::expect_used)]
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("failed to create tokio runtime");
    let handle = rt.handle().clone();
    (rt, handle)
}

impl<DP: DataProvider> DiskANNIndex<DP> {
    pub(crate) fn new_with_current_thread_runtime(
        config: graph::Config,
        data_provider: DP,
    ) -> Self {
        let (rt, handle) = create_current_thread_runtime();
        let inner = Arc::new(graph::DiskANNIndex::new(config, data_provider, Some(ONE)));
        Self {
            inner,
            _runtime: Some(rt),
            handle,
        }
    }

    pub(crate) fn insert<'a, S, T>(
        &'a self,
        strategy: &'a S,
        context: &'a DP::Context,
        id: &DP::ExternalId,
        vector: T,
    ) -> ANNResult<()>
    where
        S: InsertStrategy<'a, DP, T>,
        DP: SetElement<T>,
        T: Copy + Send,
    {
        self.handle
            .block_on(self.inner.insert(strategy, context, id, vector))
    }

    pub(crate) fn inplace_delete<S>(
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

    pub(crate) fn search<'a, S, T, O, OB, P>(
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
    {
        self.handle.block_on(
            self.inner
                .search(search_params, strategy, context, query, output),
        )
    }
}

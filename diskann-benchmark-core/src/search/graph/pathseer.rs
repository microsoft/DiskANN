/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::Arc;

use diskann::{
    ANNResult,
    graph::{self, ext::labeled, glue},
    provider,
};
use diskann_utils::{
    future::AsyncFriendly,
    views::rowmajor::{self, Matrix},
};

use crate::search::{self, Search, graph::KnnParams, graph::Strategy};

/// A built-in helper for benchmarking filtered K-nearest neighbors search
/// using the PathSeer search method.
///
/// This is intended to be used in conjunction with [`search::search`] or [`search::search_all`]
/// and provides some basic additional metrics for the latter. Result aggregation for
/// [`search::search_all`] is provided by the [`search::graph::knn::Aggregator`] type (same
/// aggregator as [`search::graph::knn::KNN`]).
///
/// The provided implementation of [`Search`] accepts [`KnnParams`]
/// and returns [`search::graph::knn::Metrics`] as additional output.
#[derive(Debug)]
pub struct PathSeer<DP, T, S>
where
    DP: provider::DataProvider,
{
    index: Arc<graph::DiskANNIndex<DP>>,
    queries: Arc<rowmajor::Owned<T>>,
    strategy: Strategy<S>,
    labels: Arc<[Arc<dyn labeled::QueryLabelProvider<DP::InternalId>>]>,
    method_id: graph::search::PathSeerMethod,
}

impl<DP, T, S> PathSeer<DP, T, S>
where
    DP: provider::DataProvider,
{
    /// Construct a new [`PathSeer`] searcher.
    ///
    /// If `strategy` is one of the container variants of [`Strategy`], its length
    /// must match the number of rows in `queries`. If this is the case, then the
    /// strategies will have a querywise correspondence (see [`search::SearchResults`])
    /// with the query matrix.
    ///
    /// Additionally, the length of `labels` must match the number of rows in `queries`
    /// and will be used in querywise correspondence with `queries`.
    ///
    /// # Errors
    ///
    /// Returns an error under the following conditions.
    ///
    /// 1. The number of elements in `strategy` is not compatible with the number of rows in
    ///    `queries`.
    ///
    /// 2. The number of label providers in `labels` is not equal to the number of rows in
    ///    `queries`.
    pub fn new(
        index: Arc<graph::DiskANNIndex<DP>>,
        queries: Arc<rowmajor::Owned<T>>,
        strategy: Strategy<S>,
        labels: Arc<[Arc<dyn labeled::QueryLabelProvider<DP::InternalId>>]>,
        method_id: graph::search::PathSeerMethod,
    ) -> anyhow::Result<Arc<Self>> {
        strategy.length_compatible(queries.nrows())?;

        if labels.len() != queries.nrows() {
            Err(anyhow::anyhow!(
                "Number of label providers ({}) must be equal to the number of queries ({})",
                labels.len(),
                queries.nrows()
            ))
        } else {
            Ok(Arc::new(Self {
                index,
                queries,
                strategy,
                labels,
                method_id,
            }))
        }
    }
}

impl<DP, T, S> Search for PathSeer<DP, T, S>
where
    DP: provider::PathSeerGraph<Context: Default, ExternalId: search::Id>,
    S: for<'a> glue::DefaultSearchStrategy<
            'a,
            DP,
            &'a [T],
            DP::ExternalId,
            SearchAccessor: glue::SearchAccessor,
        > + Clone
        + AsyncFriendly,
    T: AsyncFriendly + Clone,
{
    type Id = DP::ExternalId;
    type Parameters = KnnParams;
    type Output = super::knn::Metrics;

    fn num_queries(&self) -> usize {
        self.queries.nrows()
    }

    fn id_count(&self, parameters: &Self::Parameters) -> search::IdCount {
        search::IdCount::Fixed(parameters.k_value())
    }

    async fn search<O>(
        &self,
        parameters: &Self::Parameters,
        buffer: &mut O,
        index: usize,
    ) -> ANNResult<Self::Output>
    where
        O: graph::SearchOutputBuffer<DP::ExternalId> + Send,
    {
        let context = DP::Context::default();
        let knn = parameters.knn;
        let pathseer_search = graph::search::PathSeer::new(knn).with_method(self.method_id);
        let strategy =
            labeled::Filtered::new(self.strategy.get(index)?.clone(), &*self.labels[index]);
        let stats = self
            .index
            .search(
                pathseer_search,
                &strategy,
                &context,
                self.queries.row(index),
                buffer,
            )
            .await?;

        Ok(super::knn::Metrics {
            comparisons: stats.cmps,
            hops: stats.hops,
        })
    }
}

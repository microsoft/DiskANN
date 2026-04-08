/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::Arc;

use diskann::{
    ANNResult,
    graph::{self, glue},
    provider,
};
use diskann_utils::{future::AsyncFriendly, views::Matrix};

use crate::search::{self, Search, graph::Strategy};

/// A built-in helper for benchmarking filtered K-nearest neighbors search
/// using the two-queue search method.
///
/// This is intended to be used in conjunction with [`search::search`] or [`search::search_all`]
/// and provides some basic additional metrics for the latter. Result aggregation for
/// [`search::search_all`] is provided by the [`search::graph::knn::Aggregator`] type (same
/// aggregator as [`search::graph::knn::KNN`]).
///
/// The provided implementation of [`Search`] accepts [`graph::search::Knn`]
/// and returns [`search::graph::knn::Metrics`] as additional output.
#[derive(Debug)]
pub struct TwoQueue<DP, T, S>
where
    DP: provider::DataProvider,
{
    index: Arc<graph::DiskANNIndex<DP>>,
    queries: Arc<Matrix<T>>,
    strategy: Strategy<S>,
    labels: Arc<[Arc<dyn graph::index::QueryLabelProvider<DP::InternalId>>]>,
    max_candidates: usize,
}

impl<DP, T, S> TwoQueue<DP, T, S>
where
    DP: provider::DataProvider,
{
    /// Construct a new [`TwoQueue`] searcher.
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
        queries: Arc<Matrix<T>>,
        strategy: Strategy<S>,
        labels: Arc<[Arc<dyn graph::index::QueryLabelProvider<DP::InternalId>>]>,
        max_candidates: usize,
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
                max_candidates,
            }))
        }
    }
}

impl<DP, T, S> Search for TwoQueue<DP, T, S>
where
    DP: provider::DataProvider<Context: Default, ExternalId: search::Id>,
    S: for<'a> glue::DefaultSearchStrategy<DP, &'a [T], DP::ExternalId> + Clone + AsyncFriendly,
    T: AsyncFriendly + Clone,
{
    type Id = DP::ExternalId;
    type Parameters = graph::search::Knn;
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
        let two_queue_search =
            graph::search::TwoQueueSearch::new(*parameters, &*self.labels[index], self.max_candidates);
        let result = self
            .index
            .search(
                two_queue_search,
                self.strategy.get(index)?,
                &context,
                self.queries.row(index),
                buffer,
            )
            .await?;

        Ok(super::knn::Metrics {
            comparisons: result.stats.cmps,
            hops: result.stats.hops,
        })
    }
}

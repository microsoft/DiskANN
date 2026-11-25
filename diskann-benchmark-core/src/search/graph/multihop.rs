/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{num::NonZeroUsize, sync::Arc};

use diskann::{
    ANNResult,
    error::IntoANNResult,
    graph::{self, compat, ext::labeled, glue},
};
use diskann_utils::{
    future::AsyncFriendly,
    views::rowmajor::{self, Matrix},
};

use crate::search::{self, Search, graph::KnnParams, graph::Strategy};

/// A built-in helper for benchmarking filtered K-nearest neighbors search
/// using the multi-hop search method.
///
/// This is intended to be used in conjunction with [`search::search`] or [`search::search_all`]
/// and provides some basic additional metrics for the latter. Result aggregation for
/// [`search::search_all`] is provided by the [`search::graph::knn::Aggregator`] type (same
/// aggregator as [`search::graph::knn::KNN`]).
///
/// The provided implementation of [`Search`] accepts [`KnnParams`]
/// and returns [`search::graph::knn::Metrics`] as additional output.
#[derive(Debug)]
pub struct MultiHop<DP, T, S>
where
    DP: compat::DataProvider,
{
    index: Arc<graph::compat::DiskANNIndex<DP>>,
    queries: Arc<rowmajor::Owned<T>>,
    strategy: Strategy<S>,
    labels: Arc<[Arc<dyn labeled::QueryLabelProvider<DP::InternalId>>]>,
}

impl<DP, T, S> MultiHop<DP, T, S>
where
    DP: compat::DataProvider,
{
    /// Construct a new [`MultiHop`] searcher.
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
        index: Arc<graph::compat::DiskANNIndex<DP>>,
        queries: Arc<rowmajor::Owned<T>>,
        strategy: Strategy<S>,
        labels: Arc<[Arc<dyn labeled::QueryLabelProvider<DP::InternalId>>]>,
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
            }))
        }
    }
}

impl<DP, T, S> Search for MultiHop<DP, T, S>
where
    DP: compat::DataProvider<Context: Default, ExternalId: search::Id>,
    S: for<'a> compat::SearchStrategy<
            'a,
            DP,
            &'a [T],
            SearchAccessor: glue::SearchAccessor + glue::DefaultPostProcessor<DP::ExternalId>,
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
        let max_two_hop_candidates =
            NonZeroUsize::new((self.index.config().max_degree().get() / 2).max(1));
        let multihop_search = graph::search::MultihopFilterSearch::new(knn, max_two_hop_candidates);
        let accessor = self
            .strategy
            .get(index)?
            .search_accessor(self.index.provider(), &context, self.queries.row(index))
            .into_ann_result()?;
        let mut accessor = labeled::FilteredAccessor::new(accessor, &*self.labels[index]);
        let processor = glue::DefaultPostProcessor::default_post_processor(&accessor);
        let stats = graph::FilteredSearch::filtered_search(
            multihop_search,
            &mut accessor,
            processor,
            buffer,
        )
        .await?;

        Ok(super::knn::Metrics {
            comparisons: stats.cmps,
            hops: stats.hops,
        })
    }
}

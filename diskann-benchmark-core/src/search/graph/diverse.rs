/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A built-in helper for benchmarking diversity-aware K-nearest neighbors search.

use std::sync::Arc;

use diskann::{
    ANNResult,
    graph::{self, glue},
    neighbor::AttributeValueProvider,
    provider,
};
use diskann_utils::{future::AsyncFriendly, views::Matrix};

use crate::search::{self, graph::Strategy, graph::knn::Metrics};

/// A built-in helper for benchmarking diversity-aware search via
/// [`graph::search::Diverse`].
///
/// This mirrors [`super::KNN`] but runs each query through a
/// [`graph::search::Diverse`] wrapper constructed from the shared
/// [`AttributeValueProvider`] and diversity parameters stored on this struct. The
/// [`Search::Parameters`] remain the base [`graph::search::Knn`] parameters so that the
/// same benchmark driving code (search list sweeps, recall computation, aggregation) can be
/// reused unchanged.
///
/// # Type Parameters
///
/// - `DP`: The data provider type.
/// - `T`: The query element type.
/// - `S`: The search strategy type.
/// - `P`: The attribute value provider used to derive diversity attributes.
#[derive(Debug)]
pub struct DiverseKNN<DP, T, S, P>
where
    DP: provider::DataProvider,
    P: AttributeValueProvider,
{
    index: Arc<graph::DiskANNIndex<DP>>,
    queries: Arc<Matrix<T>>,
    strategy: Strategy<S>,
    attribute_provider: Arc<P>,
    diverse_attribute_id: usize,
    diverse_results_k: usize,
}

impl<DP, T, S, P> DiverseKNN<DP, T, S, P>
where
    DP: provider::DataProvider,
    P: AttributeValueProvider,
{
    /// Construct a new [`DiverseKNN`] searcher.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of elements in `strategy` is not compatible with
    /// the number of rows in `queries`.
    pub fn new(
        index: Arc<graph::DiskANNIndex<DP>>,
        queries: Arc<Matrix<T>>,
        strategy: Strategy<S>,
        attribute_provider: Arc<P>,
        diverse_attribute_id: usize,
        diverse_results_k: usize,
    ) -> anyhow::Result<Arc<Self>> {
        strategy.length_compatible(queries.nrows())?;

        Ok(Arc::new(Self {
            index,
            queries,
            strategy,
            attribute_provider,
            diverse_attribute_id,
            diverse_results_k,
        }))
    }

    /// Access the index.
    pub fn index(&self) -> &Arc<graph::DiskANNIndex<DP>> {
        &self.index
    }
}

impl<DP, T, S, P> search::Search for DiverseKNN<DP, T, S, P>
where
    DP: provider::DataProvider<Context: Default, ExternalId: search::Id>,
    S: for<'a> glue::DefaultSearchStrategy<
            'a,
            DP,
            &'a [T],
            DP::ExternalId,
            SearchAccessor: glue::SearchAccessor,
        > + Clone
        + AsyncFriendly,
    P: AttributeValueProvider<Id = DP::InternalId> + AsyncFriendly,
    graph::search::Diverse<P>:
        for<'a> graph::Search<'a, DP, S, &'a [T], Output = graph::index::SearchStats>,
    T: AsyncFriendly + Clone,
{
    type Id = DP::ExternalId;
    type Parameters = graph::search::Knn;
    type Output = Metrics;

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
        let strategy = self.strategy.get(index)?;

        let diverse_params = graph::DiverseSearchParams::new(
            self.diverse_attribute_id,
            self.diverse_results_k,
            self.attribute_provider.clone(),
        );
        let diverse_search = graph::search::Diverse::new(*parameters, diverse_params);

        let stats = self
            .index
            .search(
                diverse_search,
                strategy,
                &context,
                self.queries.row(index),
                buffer,
            )
            .await?;

        Ok(Metrics {
            comparisons: stats.cmps,
            hops: stats.hops,
        })
    }
}

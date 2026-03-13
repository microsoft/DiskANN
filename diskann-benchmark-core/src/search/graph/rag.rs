/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A built-in helper for benchmarking RAG (Retrieval-Augmented Generation) search.
//!
//! This mirrors [`super::KNN`] but uses [`graph::search::RagSearch`] as the search
//! parameters, applying diversity-maximizing reranking in post-processing.

use std::sync::Arc;

use diskann::{
    ANNResult,
    graph::{self, glue},
    provider,
};
use diskann_benchmark_runner::utils::{MicroSeconds, percentiles};
use diskann_utils::{future::AsyncFriendly, views::Matrix};

use crate::{
    search::{self, Search, graph::Strategy},
    utils,
};

/// A built-in helper for benchmarking the RAG search method
/// [`graph::DiskANNIndex::search`] with [`graph::search::RagSearch`].
///
/// This is identical to [`super::KNN`] in structure but uses [`graph::search::RagSearch`]
/// as the search parameters, which applies diversity-maximizing reranking via greedy
/// orthogonalization in post-processing.
///
/// The provided implementation of [`Search`] accepts [`graph::search::RagSearch`]
/// and returns [`super::knn::Metrics`] as additional output.
#[derive(Debug)]
pub struct RAG<DP, T, S>
where
    DP: provider::DataProvider,
{
    index: Arc<graph::DiskANNIndex<DP>>,
    queries: Arc<Matrix<T>>,
    strategy: Strategy<S>,
}

impl<DP, T, S> RAG<DP, T, S>
where
    DP: provider::DataProvider,
{
    /// Construct a new [`RAG`] searcher.
    ///
    /// If `strategy` is one of the container variants of [`Strategy`], its length
    /// must match the number of rows in `queries`.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of elements in `strategy` is not compatible with
    /// the number of rows in `queries`.
    pub fn new(
        index: Arc<graph::DiskANNIndex<DP>>,
        queries: Arc<Matrix<T>>,
        strategy: Strategy<S>,
    ) -> anyhow::Result<Arc<Self>> {
        strategy.length_compatible(queries.nrows())?;

        Ok(Arc::new(Self {
            index,
            queries,
            strategy,
        }))
    }
}

impl<DP, T, S> Search for RAG<DP, T, S>
where
    DP: provider::DataProvider<Context: Default, ExternalId: search::Id>,
    S: glue::SearchStrategy<DP, [T], DP::ExternalId>
        + glue::PostProcess<graph::search::RagSearchParams, DP, [T], DP::ExternalId>
        + Clone
        + AsyncFriendly,
    T: AsyncFriendly + Clone,
{
    type Id = DP::ExternalId;
    type Parameters = graph::search::RagSearch;
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
        let rag_search = parameters.clone();
        let stats = self
            .index
            .search(
                rag_search,
                self.strategy.get(index)?,
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

/// An [`search::Aggregate`]d summary of multiple [`RAG`] search runs.
///
/// This reuses [`super::knn::Summary`] since the output format is identical —
/// the only difference is the search parameters type.
pub struct Aggregator<'a, I> {
    groundtruth: &'a dyn crate::recall::Rows<I>,
    recall_k: usize,
    recall_n: usize,
}

impl<'a, I> Aggregator<'a, I> {
    /// Construct a new [`Aggregator`] using `groundtruth` for recall computation.
    pub fn new(
        groundtruth: &'a dyn crate::recall::Rows<I>,
        recall_k: usize,
        recall_n: usize,
    ) -> Self {
        Self {
            groundtruth,
            recall_k,
            recall_n,
        }
    }
}

impl<I> search::Aggregate<graph::search::RagSearch, I, super::knn::Metrics> for Aggregator<'_, I>
where
    I: crate::recall::RecallCompatible,
{
    type Output = super::knn::Summary;

    fn aggregate(
        &mut self,
        run: search::Run<graph::search::RagSearch>,
        mut results: Vec<search::SearchResults<I, super::knn::Metrics>>,
    ) -> anyhow::Result<super::knn::Summary> {
        // Compute the recall using just the first result.
        let recall = match results.first() {
            Some(first) => crate::recall::knn(
                self.groundtruth,
                None,
                first.ids().as_rows(),
                self.recall_k,
                self.recall_n,
                true,
            )?,
            None => anyhow::bail!("Results must be non-empty"),
        };

        let mut mean_latencies = Vec::with_capacity(results.len());
        let mut p90_latencies = Vec::with_capacity(results.len());
        let mut p99_latencies = Vec::with_capacity(results.len());

        results.iter_mut().for_each(|r| {
            match percentiles::compute_percentiles(r.latencies_mut()) {
                Ok(values) => {
                    let percentiles::Percentiles { mean, p90, p99, .. } = values;
                    mean_latencies.push(mean);
                    p90_latencies.push(p90);
                    p99_latencies.push(p99);
                }
                Err(_) => {
                    let zero = MicroSeconds::new(0);
                    mean_latencies.push(0.0);
                    p90_latencies.push(zero);
                    p99_latencies.push(zero);
                }
            }
        });

        // Extract the inner Knn parameters so we can produce a knn::Summary.
        let knn_params = *run.parameters().knn();

        Ok(super::knn::Summary {
            setup: run.setup().clone(),
            parameters: knn_params,
            end_to_end_latencies: results.iter().map(|r| r.end_to_end_latency()).collect(),
            recall,
            mean_latencies,
            p90_latencies,
            p99_latencies,
            mean_cmps: utils::average_all(
                results
                    .iter()
                    .flat_map(|r| r.output().iter().map(|o| o.comparisons)),
            ),
            mean_hops: utils::average_all(
                results
                    .iter()
                    .flat_map(|r| r.output().iter().map(|o| o.hops)),
            ),
        })
    }
}

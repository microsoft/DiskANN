/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{num::NonZeroUsize, sync::Arc};

use diskann::{
    ANNResult,
    graph::{self, ext::labeled, glue},
    provider,
};
use diskann_benchmark_runner::utils::{MicroSeconds, percentiles};
use diskann_utils::{future::AsyncFriendly, views::Matrix};

use crate::{
    recall,
    search::{self, Search, graph::Strategy},
};

/// A built-in helper for benchmarking the filtered range search method
/// [`graph::DiskANNIndex::search`] with [`graph::search::FilteredRange`].
///
/// This is intended to be used in conjunction with [`search::search`] or
/// [`search::search_all`] and provides some basic additional metrics for the
/// latter. Result aggregation for [`search::search_all`] is provided by the
/// [`Aggregator`] type.
///
/// The provided implementation of [`Search`] accepts
/// [`graph::search::FilteredRange`] and returns [`Metrics`] as additional output.
#[derive(Debug)]
pub struct FilteredRange<DP, T, S>
where
    DP: provider::DataProvider,
{
    index: Arc<graph::DiskANNIndex<DP>>,
    queries: Arc<Matrix<T>>,
    strategy: Strategy<S>,
    labels: Arc<[Arc<dyn labeled::QueryLabelProvider<DP::InternalId>>]>,
}

impl<DP, T, S> FilteredRange<DP, T, S>
where
    DP: provider::DataProvider,
{
    /// Construct a new [`FilteredRange`] searcher.
    ///
    /// If `strategy` is one of the container variants of [`Strategy`], its length
    /// must match the number of rows in `queries`. If this is the case, then the
    /// strategies will have a querywise correspondence (see [`search::SearchResults`])
    /// with the query matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of elements in `strategy` is not compatible with
    /// the number of rows in `queries`.
    pub fn new(
        index: Arc<graph::DiskANNIndex<DP>>,
        queries: Arc<Matrix<T>>,
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

/// Placeholder for filtered range search metrics.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct Metrics {}

impl<DP, T, S> Search for FilteredRange<DP, T, S>
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
    T: AsyncFriendly + Clone,
{
    type Id = DP::ExternalId;
    type Parameters = graph::search::FilteredRange;
    type Output = Metrics;

    fn num_queries(&self) -> usize {
        self.queries.nrows()
    }

    fn id_count(&self, parameters: &Self::Parameters) -> search::IdCount {
        search::IdCount::Dynamic(NonZeroUsize::new(parameters.starting_l()))
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
        let filtered_range_search = *parameters;
        let strategy =
            labeled::Filtered::new(self.strategy.get(index)?.clone(), &*self.labels[index]);
        let _ = self
            .index
            .search(
                filtered_range_search,
                &strategy,
                &context,
                self.queries.row(index),
                buffer,
            )
            .await?;

        Ok(Metrics {})
    }
}

/// An [`search::Aggregate`]d summary of multiple [`FilteredRange`] search runs returned by
/// the provided [`Aggregator`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Summary {
    pub setup: search::Setup,
    pub parameters: graph::search::FilteredRange,
    pub end_to_end_latencies: Vec<MicroSeconds>,
    pub mean_latencies: Vec<f64>,
    pub p90_latencies: Vec<MicroSeconds>,
    pub p99_latencies: Vec<MicroSeconds>,
    pub average_precision: recall::AveragePrecisionMetrics,
}

/// A [`search::Aggregate`] for collecting the results of multiple [`FilteredRange`] search
/// runs.
pub struct Aggregator<'a, I> {
    groundtruth: &'a dyn crate::recall::Rows<I>,
}

impl<'a, I> Aggregator<'a, I> {
    pub fn new(groundtruth: &'a dyn crate::recall::Rows<I>) -> Self {
        Self { groundtruth }
    }
}

impl<I> search::Aggregate<graph::search::FilteredRange, I, Metrics> for Aggregator<'_, I>
where
    I: crate::recall::RecallCompatible,
{
    type Output = Summary;

    #[inline(never)]
    fn aggregate(
        &mut self,
        run: search::Run<graph::search::FilteredRange>,
        mut results: Vec<search::SearchResults<I, Metrics>>,
    ) -> anyhow::Result<Summary> {
        let average_precision = match results.first() {
            Some(first) => {
                crate::recall::average_precision(first.ids().as_rows(), self.groundtruth)?
            }
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

        Ok(Summary {
            setup: run.setup().clone(),
            parameters: *run.parameters(),
            end_to_end_latencies: results.iter().map(|r| r.end_to_end_latency()).collect(),
            mean_latencies,
            p90_latencies,
            p99_latencies,
            average_precision,
        })
    }
}

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
use diskann_benchmark_runner::utils::{MicroSeconds, percentiles};
use diskann_providers::model::graph::provider::async_::DeterminantDiversitySearchParams;
use diskann_utils::{future::AsyncFriendly, views::Matrix};

use crate::{
    recall,
    search::{self, Search, graph::Strategy},
    utils,
};

#[derive(Debug, Clone, Copy)]
pub struct Parameters {
    pub inner: graph::search::Knn,
    pub processor: DeterminantDiversitySearchParams,
}

#[derive(Debug)]
pub struct DeterminantDiversity<DP, T, S>
where
    DP: provider::DataProvider,
{
    index: Arc<graph::DiskANNIndex<DP>>,
    queries: Arc<Matrix<T>>,
    strategy: Strategy<S>,
}

impl<DP, T, S> DeterminantDiversity<DP, T, S>
where
    DP: provider::DataProvider,
{
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

impl<DP, T, S> Search for DeterminantDiversity<DP, T, S>
where
    DP: provider::DataProvider<Context: Default, ExternalId: search::Id>,
    S: glue::DefaultSearchStrategy<DP, [T], DP::ExternalId> + Clone + AsyncFriendly,
    DeterminantDiversitySearchParams:
        for<'a> glue::SearchPostProcess<S::SearchAccessor<'a>, [T], DP::ExternalId> + Send + Sync,
    T: AsyncFriendly + Clone,
{
    type Id = DP::ExternalId;
    type Parameters = Parameters;
    type Output = super::knn::Metrics;

    fn num_queries(&self) -> usize {
        self.queries.nrows()
    }

    fn id_count(&self, parameters: &Self::Parameters) -> search::IdCount {
        search::IdCount::Fixed(parameters.inner.k_value())
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
        let stats = self
            .index
            .search_with(
                parameters.inner,
                self.strategy.get(index)?,
                parameters.processor,
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

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Summary {
    pub setup: search::Setup,
    pub parameters: Parameters,
    pub end_to_end_latencies: Vec<MicroSeconds>,
    pub mean_latencies: Vec<f64>,
    pub p90_latencies: Vec<MicroSeconds>,
    pub p99_latencies: Vec<MicroSeconds>,
    pub recall: recall::RecallMetrics,
    pub mean_cmps: f64,
    pub mean_hops: f64,
}

pub struct Aggregator<'a, I> {
    groundtruth: &'a dyn crate::recall::Rows<I>,
    recall_k: usize,
    recall_n: usize,
}

impl<'a, I> Aggregator<'a, I> {
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

impl<I> search::Aggregate<Parameters, I, super::knn::Metrics> for Aggregator<'_, I>
where
    I: crate::recall::RecallCompatible,
{
    type Output = Summary;

    fn aggregate(
        &mut self,
        run: search::Run<Parameters>,
        mut results: Vec<search::SearchResults<I, super::knn::Metrics>>,
    ) -> anyhow::Result<Summary> {
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

        Ok(Summary {
            setup: run.setup().clone(),
            parameters: *run.parameters(),
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

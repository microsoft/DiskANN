/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A built-in helper for benchmarking K-nearest neighbors.

use std::{num::NonZeroUsize, sync::Arc};

use diskann::{
    ANNResult,
    graph::{self, glue},
    provider,
};
use diskann_benchmark_runner::utils::{MicroSeconds, percentiles};
use diskann_utils::{future::AsyncFriendly, views::Matrix};

use crate::{
    recall,
    search::{self, Search, graph::Strategy},
    utils,
};

/// A built-in helper for benchmarking the K-nearest neighbors method
/// [`graph::DiskANNIndex::search`].
///
/// This is intended to be used in conjunction with [`search::search`] or
/// [`search::search_all`] and provides some basic additional metrics for
/// the latter. Result aggregation for [`search::search_all`] is provided
/// by the [`Aggregator`] type.
///
/// The provided implementation of [`Search`] accepts [`graph::SearchParams`]
/// and returns [`Metrics`] as additional output.
#[derive(Debug)]
pub struct KNN<DP, T, S>
where
    DP: provider::DataProvider,
{
    index: Arc<graph::DiskANNIndex<DP>>,
    queries: Arc<Matrix<T>>,
    strategy: Strategy<S>,
}

impl<DP, T, S> KNN<DP, T, S>
where
    DP: provider::DataProvider,
{
    /// Construct a new [`KNN`] searcher.
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
    ) -> anyhow::Result<Arc<Self>> {
        strategy.length_compatible(queries.nrows())?;

        Ok(Arc::new(Self {
            index,
            queries,
            strategy,
        }))
    }
}

/// Additional metrics collected during [`KNN`] search.
///
/// # Note
///
/// This struct is marked as non-exhaustive to allow for future additions.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct Metrics {
    /// The number of distance comparisons performed during search.
    pub comparisons: u32,
    /// The number of candidates expanded during search.
    pub hops: u32,
}

impl<DP, T, S> Search for KNN<DP, T, S>
where
    DP: provider::DataProvider<Context: Default, ExternalId: search::Id>,
    S: glue::SearchStrategy<DP, [T], DP::ExternalId> + Clone + AsyncFriendly,
    T: AsyncFriendly + Clone,
{
    type Id = DP::ExternalId;
    type Parameters = graph::SearchParams;
    type Output = Metrics;

    fn num_queries(&self) -> usize {
        self.queries.nrows()
    }

    fn id_count(&self, parameters: &Self::Parameters) -> search::IdCount {
        search::IdCount::Fixed(NonZeroUsize::new(parameters.k_value).unwrap_or(diskann::utils::ONE))
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
            .search(
                self.strategy.get(index)?,
                &context,
                self.queries.row(index),
                parameters,
                buffer,
            )
            .await?;

        Ok(Metrics {
            comparisons: stats.cmps,
            hops: stats.hops,
        })
    }
}

/// An [`search::Aggregate`]d summary of multiple [`KNN`] search runs
/// returned by the provided [`Aggregator`].
///
/// This struct is marked as non-exhaustive to allow for future additions.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Summary {
    /// The [`search::Setup`] used for the batch of runs.
    pub setup: search::Setup,

    /// The [`Search::Parameters`] used for the batch of runs.
    pub parameters: graph::SearchParams,

    /// The end-to-end latency for each repetition in the batch.
    pub end_to_end_latencies: Vec<MicroSeconds>,

    /// The average latency for individual queries.
    ///
    /// This contains one entry per repetition in the batch.
    pub mean_latencies: Vec<f64>,

    /// The 90th percentile latency for individual queries.
    ///
    /// This contains one entry per repetition in the batch.
    pub p90_latencies: Vec<MicroSeconds>,

    /// The 99th percentile latency for individual queries.
    ///
    /// This contains one entry per repetition in the batch.
    pub p99_latencies: Vec<MicroSeconds>,

    /// The recall metrics for search.
    ///
    /// This implementation assumes that search is deterministic and only
    /// uses the first repetition's results to compute recall.
    pub recall: recall::RecallMetrics,

    /// The average number of distance comparisons per query.
    pub mean_cmps: f64,

    /// The average number of neighbor hops per query.
    pub mean_hops: f64,
}

/// A [`search::Aggregate`] for collecting the results of multiple [`KNN`] search runs.
///
/// In addition to collecting latencies and other metrics, this aggregator computes
/// recall using a provided groundtruth.
///
/// The aggregated results are available as a [`Summary`].
pub struct Aggregator<'a, I> {
    groundtruth: &'a dyn crate::recall::Rows<I>,
    recall_k: usize,
    recall_n: usize,
}

impl<'a, I> Aggregator<'a, I> {
    /// Construct a new [`Aggregator`] using `groundtruth` for recall computation.
    ///
    /// Recall will be computed as `recall_k`-NN recall over the top `recall_n` neighbors.
    ///
    /// This implementation allows fewer than `recall_n` neighbors to be returned
    /// per query without error.
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

impl<I> search::Aggregate<graph::SearchParams, I, Metrics> for Aggregator<'_, I>
where
    I: crate::recall::RecallCompatible,
{
    type Output = Summary;

    fn aggregate(
        &mut self,
        run: search::Run<graph::SearchParams>,
        mut results: Vec<search::SearchResults<I, Metrics>>,
    ) -> anyhow::Result<Summary> {
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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use diskann::graph::test::provider;

    #[test]
    fn test_knn() {
        let nearest_neighbors = 5;

        let index = search::graph::test_grid_provider();

        let mut queries = Matrix::new(0.0f32, 5, index.provider().dim());
        queries.row_mut(0).copy_from_slice(&[0.0, 0.0, 0.0, 0.0]);
        queries.row_mut(1).copy_from_slice(&[4.0, 0.0, 0.0, 0.0]);
        queries.row_mut(2).copy_from_slice(&[0.0, 4.0, 0.0, 0.0]);
        queries.row_mut(3).copy_from_slice(&[0.0, 0.0, 4.0, 0.0]);
        queries.row_mut(4).copy_from_slice(&[0.0, 0.0, 0.0, 4.0]);

        let queries = Arc::new(queries);

        let knn = KNN::new(
            index,
            queries.clone(),
            Strategy::broadcast(provider::Strategy::new()),
        )
        .unwrap();

        // Test the standard search interface.
        let rt = crate::tokio::runtime(2).unwrap();
        let results = search::search(
            knn.clone(),
            graph::SearchParams::new(nearest_neighbors, 10, None).unwrap(),
            NonZeroUsize::new(2).unwrap(),
            &rt,
        )
        .unwrap();

        assert_eq!(results.len(), queries.nrows());
        let rows = results.ids().as_rows();
        assert_eq!(*rows.row(0).first().unwrap(), 0);

        for r in 0..rows.nrows() {
            assert_eq!(rows.row(r).len(), nearest_neighbors);
        }

        const TWO: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let setup = search::Setup {
            threads: TWO,
            tasks: TWO,
            reps: TWO,
        };

        // Try the aggregated strategy.
        let parameters = [
            search::Run::new(
                graph::SearchParams::new(nearest_neighbors, 10, None).unwrap(),
                setup.clone(),
            ),
            search::Run::new(
                graph::SearchParams::new(nearest_neighbors, 15, None).unwrap(),
                setup.clone(),
            ),
        ];

        let recall_k = nearest_neighbors;
        let recall_n = nearest_neighbors;

        let all =
            search::search_all(knn, parameters, Aggregator::new(rows, recall_k, recall_n)).unwrap();

        assert_eq!(all.len(), 2);
        for summary in all {
            assert_eq!(summary.setup, setup);
            assert_eq!(summary.end_to_end_latencies.len(), TWO.get());
            assert_eq!(summary.mean_latencies.len(), TWO.get());
            assert_eq!(summary.p90_latencies.len(), TWO.get());
            assert_eq!(summary.p99_latencies.len(), TWO.get());

            assert_ne!(summary.mean_cmps, 0.0);
            assert_ne!(summary.mean_hops, 0.0);

            let recall = summary.recall;
            assert_eq!(recall.recall_k, recall_k);
            assert_eq!(recall.recall_n, recall_n);
            assert_eq!(recall.num_queries, queries.nrows());
            assert_eq!(recall.average, 1.0, "we used a search as the groundtruth");
        }
    }

    #[test]
    fn test_knn_error() {
        let index = search::graph::test_grid_provider();

        let queries = Arc::new(Matrix::new(0.0f32, 1, index.provider().dim()));
        let strategy = provider::Strategy::new();

        let err = KNN::new(
            index,
            queries.clone(),
            Strategy::collection([strategy, strategy]),
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("2 strategies were provided when 1 was expected"),
            "failed with {msg}"
        );
    }
}

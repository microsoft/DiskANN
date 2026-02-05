/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

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
};

/// A built-in helper for benchmarking the range search method
/// [`graph::DiskANNIndex::range_search`].
///
/// This is intended to be used in conjunction with [`search::search`] or
/// [`search::search_all`] and provides some basic additional metrics for
/// the latter. Result aggregation for [`search::search_all`] is provided
/// by the [`Aggregator`] type.
///
/// The provided implementation of [`Search`] accepts
/// [`graph::RangeSearchParams`] and returns [`Metrics`] as additional output.
#[derive(Debug)]
pub struct Range<DP, T, S>
where
    DP: provider::DataProvider,
{
    index: Arc<graph::DiskANNIndex<DP>>,
    queries: Arc<Matrix<T>>,
    strategy: Strategy<S>,
}

impl<DP, T, S> Range<DP, T, S>
where
    DP: provider::DataProvider,
{
    /// Construct a new [`Range`] searcher.
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

/// Placeholder for range search metrics.
///
/// This struct currently does not contain any fields, but may be augmented
/// in the future. The use of `#[non_exhaustive]` prevents breaking changes.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct Metrics {}

impl<DP, T, S> Search for Range<DP, T, S>
where
    DP: provider::DataProvider<Context: Default, ExternalId: search::Id>,
    S: glue::SearchStrategy<DP, [T], DP::ExternalId> + Clone + AsyncFriendly,
    T: AsyncFriendly + Clone,
{
    type Id = DP::ExternalId;
    type Parameters = graph::RangeSearchParams;
    type Output = Metrics;

    fn num_queries(&self) -> usize {
        self.queries.nrows()
    }

    fn id_count(&self, parameters: &Self::Parameters) -> search::IdCount {
        search::IdCount::Dynamic(NonZeroUsize::new(parameters.starting_l_value))
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
        let (_, ids, distances) = self
            .index
            .range_search(
                self.strategy.get(index)?,
                &context,
                self.queries.row(index),
                parameters,
            )
            .await?;
        buffer.extend(std::iter::zip(ids.into_iter(), distances.into_iter()));

        Ok(Metrics {})
    }
}

/// An [`search::Aggregate`]d summary of multiple [`Range`] search runs
/// returned by the provided [`Aggregator`].
///
/// This struct is marked as non-exhaustive to allow for future additions.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Summary {
    /// The [`search::Setup`] used for the batch of runs.
    pub setup: search::Setup,

    /// The [`graph::RangeSearchParams`] used for the batch of runs.
    pub parameters: graph::RangeSearchParams,

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

    /// The average precision metrics for the batch of runs.
    ///
    /// This implementation assumes that search is deterministic and only
    /// uses the first repetition's results to compute the average precision.
    pub average_precision: recall::AveragePrecisionMetrics,
}

/// A [`search::Aggregate`] for collecting the results of multiple [`Range`] search runs.
///
/// In addition to collecting latencies and other metrics, this aggregator computes
/// [`crate::recall::AveragePrecisionMetrics`] using a provided groundtruth.
///
/// The aggregated results are available as a [`Summary`].
pub struct Aggregator<'a, I> {
    groundtruth: &'a dyn crate::recall::Rows<I>,
}

impl<'a, I> Aggregator<'a, I> {
    /// Construct a new [`Aggregator`] using `groundtruth` for average precision computation.
    pub fn new(groundtruth: &'a dyn crate::recall::Rows<I>) -> Self {
        Self { groundtruth }
    }
}

impl<I> search::Aggregate<graph::RangeSearchParams, I, Metrics> for Aggregator<'_, I>
where
    I: crate::recall::RecallCompatible,
{
    type Output = Summary;

    #[inline(never)]
    fn aggregate(
        &mut self,
        run: search::Run<graph::RangeSearchParams>,
        mut results: Vec<search::SearchResults<I, Metrics>>,
    ) -> anyhow::Result<Summary> {
        // Compute the recall using just the first result.
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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use diskann::graph::test::provider;

    #[test]
    fn test_range() {
        let index = search::graph::test_grid_provider();

        let mut queries = Matrix::new(0.0f32, 5, index.provider().dim());
        queries.row_mut(0).copy_from_slice(&[0.0, 0.0, 0.0, 0.0]);
        queries.row_mut(1).copy_from_slice(&[4.0, 0.0, 0.0, 0.0]);
        queries.row_mut(2).copy_from_slice(&[0.0, 4.0, 0.0, 0.0]);
        queries.row_mut(3).copy_from_slice(&[0.0, 0.0, 4.0, 0.0]);
        queries.row_mut(4).copy_from_slice(&[0.0, 0.0, 0.0, 4.0]);

        let queries = Arc::new(queries);

        let range = Range::new(
            index,
            queries.clone(),
            Strategy::broadcast(provider::Strategy::new()),
        )
        .unwrap();

        // Test the standard search interface.
        let rt = crate::tokio::runtime(2).unwrap();
        let results = search::search(
            range.clone(),
            graph::RangeSearchParams::new(None, 10, None, 2.0, None, 0.8, 1.2).unwrap(),
            NonZeroUsize::new(2).unwrap(),
            &rt,
        )
        .unwrap();

        assert_eq!(results.len(), queries.nrows());
        let rows = results.ids().as_rows();
        assert_eq!(*rows.row(0).first().unwrap(), 0);
        const TWO: NonZeroUsize = NonZeroUsize::new(2).unwrap();
        let setup = search::Setup {
            threads: TWO,
            tasks: TWO,
            reps: TWO,
        };

        // Try the aggregated strategy.
        let parameters = [
            search::Run::new(
                graph::RangeSearchParams::new(None, 10, None, 2.0, None, 0.8, 1.2).unwrap(),
                setup.clone(),
            ),
            search::Run::new(
                graph::RangeSearchParams::new(None, 15, None, 2.0, None, 0.8, 1.2).unwrap(),
                setup.clone(),
            ),
        ];

        let all = search::search_all(range, parameters, Aggregator::new(rows)).unwrap();

        assert_eq!(all.len(), 2);
        for summary in all {
            assert_eq!(summary.setup, setup);
            assert_eq!(summary.end_to_end_latencies.len(), TWO.get());
            assert_eq!(summary.mean_latencies.len(), TWO.get());
            assert_eq!(summary.p90_latencies.len(), TWO.get());
            assert_eq!(summary.p99_latencies.len(), TWO.get());

            let ap = summary.average_precision;
            assert_eq!(ap.num_queries, queries.nrows());
            assert_eq!(
                ap.average_precision, 1.0,
                "we used a search as the groundtruth"
            );
        }
    }

    #[test]
    fn test_range_error() {
        let index = search::graph::test_grid_provider();

        let queries = Arc::new(Matrix::new(0.0f32, 2, index.provider().dim()));
        let strategy = provider::Strategy::new();

        let err = Range::new(index, queries.clone(), Strategy::collection([strategy])).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("1 strategy was provided when 2 were expected"),
            "failed with {msg}"
        );
    }
}

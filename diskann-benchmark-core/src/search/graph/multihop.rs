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
use diskann_utils::{future::AsyncFriendly, views::Matrix};

use crate::search::{self, Search, graph::Strategy};

/// A built-in helper for benchmarking filtered K-nearest neighbors search
/// using the [multi-hop](graph::DiskANNIndex::multihop_search) search method.
///
/// This is intended to be used in conjunction with [`search::search`] or [`search::search_all`]
/// and provides some basic additional metrics for the latter. Result aggregation for
/// [`search::search_all`] is provided by the [`search::graph::knn::Aggregator`] type (same
/// aggregator as [`search::graph::KNN`]).
///
/// The provided implementation of [`Search`] accepts [`graph::SearchParams`]
/// and returns [`search::graph::knn::Metrics`] as additional output.
#[derive(Debug)]
pub struct MultiHop<DP, T, S>
where
    DP: provider::DataProvider,
{
    index: Arc<graph::DiskANNIndex<DP>>,
    queries: Arc<Matrix<T>>,
    strategy: Strategy<S>,
    labels: Arc<[Arc<dyn graph::index::QueryLabelProvider<DP::InternalId>>]>,
}

impl<DP, T, S> MultiHop<DP, T, S>
where
    DP: provider::DataProvider,
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
        index: Arc<graph::DiskANNIndex<DP>>,
        queries: Arc<Matrix<T>>,
        strategy: Strategy<S>,
        labels: Arc<[Arc<dyn graph::index::QueryLabelProvider<DP::InternalId>>]>,
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
    DP: provider::DataProvider<Context: Default, ExternalId: search::Id>,
    S: glue::SearchStrategy<DP, [T], DP::ExternalId> + Clone + AsyncFriendly,
    T: AsyncFriendly + Clone,
{
    type Id = DP::ExternalId;
    type Parameters = graph::SearchParams;
    type Output = super::knn::Metrics;

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
            .multihop_search(
                self.strategy.get(index)?,
                &context,
                self.queries.row(index),
                parameters,
                buffer,
                &*self.labels[index],
            )
            .await?;

        Ok(super::knn::Metrics {
            comparisons: stats.cmps,
            hops: stats.hops,
        })
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use diskann::graph::{index::QueryLabelProvider, test::provider};

    // A simple [`QueryLabelProvider`] that rejects odd indices.
    #[derive(Debug)]
    struct NoOdds;

    impl graph::index::QueryLabelProvider<u32> for NoOdds {
        fn is_match(&self, id: u32) -> bool {
            id.is_multiple_of(2)
        }
    }

    #[test]
    fn test_multihop() {
        let nearest_neighbors = 5;

        let index = search::graph::test_grid_provider();

        let mut queries = Matrix::new(0.0f32, 5, index.provider().dim());
        queries.row_mut(0).copy_from_slice(&[0.0, 0.0, 0.0, 0.0]);
        queries.row_mut(1).copy_from_slice(&[4.0, 0.0, 0.0, 0.0]);
        queries.row_mut(2).copy_from_slice(&[0.0, 4.0, 0.0, 0.0]);
        queries.row_mut(3).copy_from_slice(&[0.0, 0.0, 4.0, 0.0]);
        queries.row_mut(4).copy_from_slice(&[0.0, 0.0, 0.0, 4.0]);

        let queries = Arc::new(queries);

        let multihop = MultiHop::new(
            index,
            queries.clone(),
            Strategy::broadcast(provider::Strategy::new()),
            (0..queries.nrows())
                .map(|_| -> Arc<dyn QueryLabelProvider<_>> { Arc::new(NoOdds {}) })
                .collect(),
        )
        .unwrap();

        // Test the standard search interface.
        let rt = crate::tokio::runtime(2).unwrap();
        let results = search::search(
            multihop.clone(),
            graph::SearchParams::new(nearest_neighbors, 10, None).unwrap(),
            NonZeroUsize::new(2).unwrap(),
            &rt,
        )
        .unwrap();

        assert_eq!(results.len(), queries.nrows());
        let rows = results.ids().as_rows();
        assert_eq!(*rows.row(0).first().unwrap(), 0);

        // Check that only even IDs are returned.
        for r in 0..rows.nrows() {
            assert_eq!(rows.row(r).len(), nearest_neighbors);
            for &id in rows.row(r) {
                assert_eq!(id % 2, 0, "Found odd ID {} in row {}", id, r);
            }
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

        let all = search::search_all(
            multihop,
            parameters,
            search::graph::knn::Aggregator::new(rows, recall_k, recall_n),
        )
        .unwrap();

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
    fn test_multihop_error() {
        let index = search::graph::test_grid_provider();
        let queries = Arc::new(Matrix::new(0.0f32, 2, index.provider().dim()));

        let labels: Arc<[_]> = (0..queries.nrows() + 1)
            .map(|_| -> Arc<dyn QueryLabelProvider<_>> { Arc::new(NoOdds {}) })
            .collect();

        let strategy = provider::Strategy::new();

        // Error for a mismatch between strategies and queries.
        let err = MultiHop::new(
            index.clone(),
            queries.clone(),
            Strategy::collection([strategy]),
            labels.clone(),
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("1 strategy was provided when 2 were expected"),
            "failed with {msg}"
        );

        // Error for a mismatch between label providers and queries.
        let err = MultiHop::new(
            index,
            queries.clone(),
            Strategy::broadcast(strategy),
            labels.clone(),
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains(
                "Number of label providers (3) must be equal to the number of queries (2)"
            ),
            "failed with {msg}"
        );
    }
}

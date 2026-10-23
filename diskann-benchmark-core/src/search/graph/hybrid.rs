/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Benchmark adapter for the library's count-routed filtered search.

use std::{num::NonZeroU64, sync::Arc};

use diskann::{
    ANNResult,
    graph::{self, ext::labeled, glue},
    provider,
};
use diskann_utils::{future::AsyncFriendly, views::Matrix};

use crate::search::{self, Search, graph::KnnParams, graph::Strategy};

/// Benchmark multiple queries with library-level hybrid filtered search.
#[derive(Debug)]
pub struct Hybrid<DP, T, S, L>
where
    DP: provider::DataProvider,
    L: labeled::CandidateLabelProvider<DP::InternalId>,
{
    index: Arc<graph::DiskANNIndex<DP>>,
    queries: Arc<Matrix<T>>,
    strategy: Strategy<S>,
    labels: Arc<[Arc<L>]>,
    threshold: NonZeroU64,
}

impl<DP, T, S, L> Hybrid<DP, T, S, L>
where
    DP: provider::DataProvider,
    L: labeled::CandidateLabelProvider<DP::InternalId> + 'static,
{
    /// Create a querywise hybrid benchmark.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible query counts, missing label counts, or a zero
    /// match-count threshold.
    pub fn new(
        index: Arc<graph::DiskANNIndex<DP>>,
        queries: Arc<Matrix<T>>,
        strategy: Strategy<S>,
        labels: Arc<[Arc<L>]>,
        threshold: u64,
    ) -> anyhow::Result<Arc<Self>> {
        strategy.length_compatible(queries.nrows())?;
        anyhow::ensure!(
            queries.nrows() > 0,
            "hybrid search requires at least one query"
        );
        anyhow::ensure!(
            labels.len() == queries.nrows(),
            "hybrid query and filter row counts differ"
        );
        anyhow::ensure!(
            labels
                .iter()
                .all(|label| label.match_upper_bound().is_some()),
            "hybrid search requires a counted label index"
        );
        let threshold = NonZeroU64::new(threshold)
            .ok_or_else(|| anyhow::anyhow!("hybrid threshold is zero"))?;
        Ok(Arc::new(Self {
            index,
            queries,
            strategy,
            labels,
            threshold,
        }))
    }
}

impl<DP, T, S, L> Search for Hybrid<DP, T, S, L>
where
    DP: provider::DataProvider<Context: Default, ExternalId: search::Id>,
    S: for<'a> glue::DefaultSearchStrategy<
            'a,
            DP,
            &'a [T],
            DP::ExternalId,
            SearchAccessor: glue::SearchAccessor + glue::RandomAccessQueryDistance,
        > + Clone
        + AsyncFriendly,
    T: AsyncFriendly + Clone,
    L: labeled::CandidateLabelProvider<DP::InternalId> + 'static,
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
        let filter = &*self.labels[index];
        let hybrid = graph::search::HybridFilterSearch::new(
            parameters.knn,
            parameters.k_value().get(),
            self.threshold.get(),
            filter,
        )?;
        let strategy = labeled::Filtered::new(self.strategy.get(index)?.clone(), filter);
        let stats = self
            .index
            .search(hybrid, &strategy, &context, self.queries.row(index), buffer)
            .await?;

        let mut metrics = super::knn::Metrics::new(stats.stats.cmps, stats.stats.hops);
        metrics.hybrid_timings = stats.timings;
        Ok(metrics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskann::graph::{config, test};
    use std::num::NonZeroUsize;

    #[derive(Debug)]
    struct StubFilter {
        bound: u64,
        ids: Vec<u32>,
    }

    impl labeled::QueryLabelProvider<u32> for StubFilter {
        fn is_match(&self, id: u32) -> bool {
            self.ids.contains(&id)
        }
    }

    impl labeled::CandidateLabelProvider<u32> for StubFilter {
        fn match_upper_bound(&self) -> Option<u64> {
            Some(self.bound)
        }

        fn visit_candidates(&self, mut visit: impl FnMut(u32)) {
            for &id in &self.ids {
                visit(id);
            }
        }
    }

    #[test]
    fn benchmark_adapter_dispatches_library_search_for_each_query() {
        let provider = test::provider::Provider::grid(test::synthetic::Grid::One, 4).unwrap();
        let config = config::Builder::new(
            provider.max_degree(),
            config::MaxDegree::same(),
            10,
            provider.distance_metric().into(),
        )
        .build()
        .unwrap();
        let index = Arc::new(graph::DiskANNIndex::new(config, provider, None));
        let queries = Arc::new(Matrix::new(0.0f32, 3, 1));
        let filters: Arc<[Arc<StubFilter>]> = vec![
            Arc::new(StubFilter {
                bound: 2,
                ids: vec![1, 2],
            }),
            Arc::new(StubFilter {
                bound: 4,
                ids: vec![0, 1, 2, 3],
            }),
            Arc::new(StubFilter {
                bound: 0,
                ids: Vec::new(),
            }),
        ]
        .into();
        let hybrid = Hybrid::new(
            index,
            queries,
            Strategy::broadcast(test::provider::Strategy::new()),
            filters,
            3,
        )
        .unwrap();
        let rt = crate::tokio::runtime(1).unwrap();
        let results = search::search(
            hybrid,
            KnnParams::new(2, 10).unwrap(),
            NonZeroUsize::new(1).unwrap(),
            &rt,
        )
        .unwrap();
        let ids = results.ids().as_rows();
        assert_eq!(ids.row(0), &[1, 2]);
        assert_eq!(ids.row(1), &[0, 1]);
        assert!(ids.row(2).is_empty());
        assert_eq!(results.output()[0].comparisons, 2);
        assert_eq!(results.output()[0].hops, 0);
        assert_eq!(
            results.output()[0]
                .hybrid_timings
                .map(|timings| timings.candidate_count),
            Some(2)
        );
        assert!(results.output()[1].hops > 0);
        assert!(results.output()[1].hybrid_timings.is_none());
        assert_eq!(results.output()[2].comparisons, 0);
        assert_eq!(
            results.output()[2].hybrid_timings,
            Some(graph::search::HybridPhaseTimings::default())
        );
    }
}

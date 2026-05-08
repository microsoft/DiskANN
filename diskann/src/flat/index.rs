/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! [`FlatIndex`] — the index wrapper for a [`DataProvider`](crate::provider::DataProvider)
//! over which we do flat search.
use std::num::NonZeroUsize;

use diskann_utils::future::SendFuture;

use crate::{
    ANNResult,
    error::{ErrorExt, IntoANNResult},
    flat::{DistancesUnordered, SearchStrategy},
    graph::{SearchOutputBuffer, glue::SearchPostProcess},
    neighbor::{Neighbor, NeighborPriorityQueue},
    provider::{BuildQueryComputer, DataProvider},
};

/// Statistics collected during a flat search.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SearchStats {
    /// The total number of distance computations performed during the scan.
    pub cmps: u32,

    /// The total number of results written to the output buffer.
    pub result_count: u32,
}

/// A `'static` thin wrapper around a [`DataProvider`] used for flat search.
///
/// The provider is owned by the index. The index is constructed once at process startup and
/// shared across requests; per-query state lives in the [`crate::flat::OnElementsUnordered`]
/// implementation that the [`SearchStrategy`] produces.
#[derive(Debug)]
pub struct FlatIndex<P: DataProvider> {
    /// The backing provider.
    provider: P,
}

impl<P: DataProvider> FlatIndex<P> {
    /// Construct a new [`FlatIndex`] around `provider`.
    pub fn new(provider: P) -> Self {
        Self { provider }
    }

    /// Borrow the underlying provider.
    pub fn provider(&self) -> &P {
        &self.provider
    }

    /// Brute-force k-nearest-neighbor flat search.
    ///
    /// Streams every element produced by the strategy's visitor through the query
    /// computer, keeps the best `k` candidates in a [`NeighborPriorityQueue`], and hands
    /// the survivors to the post-processor.
    ///
    /// # Arguments
    /// - `k`: number of nearest neighbors to return.
    /// - `strategy`: produces the per-query iterator and the query computer. See [`SearchStrategy`].
    /// - `processor`: post-processes the survivor candidates into the output type.
    /// - `context`: per-request context threaded through to the provider.
    /// - `query`: the query.
    /// - `output`: caller-owned output buffer.
    pub fn knn_search<S, T, O, OB, PP>(
        &self,
        k: NonZeroUsize,
        strategy: &S,
        processor: &PP,
        context: &P::Context,
        query: T,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<SearchStats>>
    where
        S: SearchStrategy<P, T>,
        T: Copy + Send + Sync,
        O: Send,
        OB: SearchOutputBuffer<O> + Send + ?Sized,
        PP: for<'a> SearchPostProcess<S::Visitor<'a>, T, O> + Send + Sync,
    {
        async move {
            let mut visitor = strategy
                .create_visitor(&self.provider, context)
                .into_ann_result()?;

            let computer = visitor.build_query_computer(query).into_ann_result()?;

            let k = k.get();
            let mut queue = NeighborPriorityQueue::new(k);
            let mut cmps: u32 = 0;

            visitor
                .distances_unordered(&computer, |id, dist| {
                    cmps += 1;
                    queue.insert(Neighbor::new(id, dist));
                })
                .await
                .escalate("flat scan must complete to produce correct k-NN results")?;

            let result_count = processor
                .post_process(&mut visitor, query, &computer, queue.iter().take(k), output)
                .await
                .into_ann_result()? as u32;

            Ok(SearchStats { cmps, result_count })
        }
    }
}

/////////////
// Tests ///
/////////////

#[cfg(test)]
mod tests {
    use crate::flat::{
        FlatIndex,
        test::{
            harness::KnnOracleRun,
            provider::{self as flat_provider, Strategy},
        },
    };
    use crate::graph::test::synthetic::Grid;

    fn fixture(grid: Grid, size: usize) -> (FlatIndex<flat_provider::Provider>, usize) {
        let provider = flat_provider::Provider::grid(grid, size);
        let len = provider.len();
        (FlatIndex::new(provider), len)
    }

    /// `knn_search` returns a `Send` future, and a shared `&FlatIndex` can serve
    /// many concurrent searches on a multi-threaded runtime, each producing the
    /// correct top-k independently.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn knn_search() {
        use std::sync::Arc;

        let (index, len) = fixture(Grid::Two, 4);
        let index = Arc::new(index);

        // Mix of corner, axis-aligned, and off-grid queries; k spans 1..=len.
        let cases: &[(&[f32], usize)] = &[
            (&[-1.0, -1.0], 1),
            (&[1.0, 1.0], len),
            (&[-1.0, 1.0], len / 2),
            (&[1.0, -1.0], len - 1),
            (&[0.0, 0.0], 3),
            (&[3.0, 3.0], len),
            (&[-2.0, 0.5], 2),
            (&[0.5, -0.5], len),
        ];

        let mut set = tokio::task::JoinSet::new();
        for (query, k) in cases {
            let index = Arc::clone(&index);
            let query: Vec<f32> = query.to_vec();
            let k = *k;
            set.spawn(async move {
                let outcome = KnnOracleRun::run(&index, &Strategy::new(), &query, k)
                    .await
                    .expect("knn_search failed");
                (query, k, outcome)
            });
        }

        while let Some(joined) = set.join_next().await {
            let (query, k, outcome) = joined.expect("task panicked");
            assert_eq!(
                outcome.top_k, outcome.ground_truth,
                "query = {query:?}, k = {k}: top-k must match brute force",
            );
            assert_eq!(outcome.stats.cmps as usize, len);
            assert_eq!(outcome.stats.result_count as usize, k.min(len));
        }
    }

    /// A transient error from the visitor's scan must escalate up through `knn_search`.
    #[test]
    fn transient_scan_error() {
        let (index, _len) = fixture(Grid::Two, 3);

        // The flat scan must touch every id, so any transient id is guaranteed to be
        // hit.
        for transient_ids in [&[0u32][..], &[3][..], &[1, 2, 5][..]] {
            let err = KnnOracleRun::run_sync(
                &index,
                &Strategy::with_transient(transient_ids.iter().copied()),
                &[1.0, 0.0],
                4,
            )
            .expect_err("transient error during full scan must escalate");

            let msg = format!("{err}");
            assert!(
                transient_ids
                    .iter()
                    .any(|id| msg.contains(&format!("id {id}"))),
                "transients = {transient_ids:?}: expected error to name one of the \
                 transient ids, got: {msg}",
            );
        }
    }
}

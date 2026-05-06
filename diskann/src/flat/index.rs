/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! [`FlatIndex`] — the index wrapper for an on which we do flat search.
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
    /// Streams every element produced by the strategy's iterator through the query
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

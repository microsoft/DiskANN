/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! [`FlatIndex`] — the index wrapper for an on which we do flat search.

use std::marker::PhantomData;
use std::num::NonZeroUsize;

use diskann_utils::future::SendFuture;
use diskann_vector::PreprocessedDistanceFunction;

use crate::{
    ANNResult,
    error::IntoANNResult,
    flat::{FlatIterator, FlatPostProcess, FlatSearchStrategy},
    graph::{SearchOutputBuffer, index::SearchStats},
    neighbor::{Neighbor, NeighborPriorityQueue},
    provider::DataProvider,
};

/// A `'static` thin wrapper around a [`DataProvider`] used for flat search.
///
/// The provider is owned by the index. The index is constructed once at process startup and
/// shared across requests; per-query state lives in the [`crate::flat::FlatIterator`] that
/// the [`crate::flat::FlatSearchStrategy`] produces.
#[derive(Debug)]
pub struct FlatIndex<P: DataProvider> {
    /// The backing provider.
    pub provider: P,
    _marker: PhantomData<fn() -> P>,
}

impl<P: DataProvider> FlatIndex<P> {
    /// Construct a new [`FlatIndex`] around `provider`.
    pub fn new(provider: P) -> Self {
        Self {
            provider,
            _marker: PhantomData,
        }
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
    /// - `strategy`: produces the per-query iterator and the query computer. See [`FlatSearchStrategy`]
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
        query: &T,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<SearchStats>>
    where
        S: FlatSearchStrategy<P, T>,
        T: ?Sized + Sync,
        O: Send,
        OB: SearchOutputBuffer<O> + Send + ?Sized,
        PP: for<'a> FlatPostProcess<S::Iter<'a>, T, O> + Send + Sync,
    {
        async move {
            let mut iter = strategy
                .create_iter(&self.provider, context)
                .into_ann_result()?;
            let computer = strategy.build_query_computer(query).into_ann_result()?;

            let k = k.get();
            let mut queue = NeighborPriorityQueue::new(k);
            let mut cmps: u32 = 0;

            iter.on_elements_unordered(|id, element| {
                let dist = computer.evaluate_similarity(element);
                cmps += 1;
                queue.insert(Neighbor::new(id, dist));
            })
            .await
            .into_ann_result()?;

            let result_count = processor
                .post_process(&mut iter, query, queue.iter().take(k), output)
                .await
                .into_ann_result()? as u32;

            Ok(SearchStats {
                cmps,
                hops: 0,
                result_count,
                range_search_second_round: false,
            })
        }
    }
}

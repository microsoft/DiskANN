/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! IVF index wrapper.

use std::num::NonZeroUsize;

use diskann_utils::future::SendFuture;

use crate::{
    ANNResult,
    error::{ANNError, ANNErrorKind, ErrorExt, IntoANNResult},
    graph::SearchOutputBuffer,
    ivf::{InsertAccessor, InsertStrategy, ListAccessor, SearchAccessor, SearchStrategy},
    neighbor::{Neighbor, NeighborPriorityQueue, NeighborPriorityQueueIdType},
    provider::{DataProvider, Guard, SetElement},
};

/// Statistics collected during an IVF search.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SearchStats {
    /// Distance computations performed while scanning lists.
    pub cmps: u32,

    /// Results written to the output buffer.
    pub result_count: u32,
}

/// IVF index wrapper over a [`DataProvider`].
#[derive(Debug)]
pub struct IvfIndex<P: DataProvider> {
    provider: P,
}

impl<P: DataProvider> IvfIndex<P> {
    /// Construct a new index around `provider`.
    pub fn new(provider: P) -> Self {
        Self { provider }
    }

    /// Borrow the underlying provider.
    pub fn provider(&self) -> &P {
        &self.provider
    }

    /// Run IVF k-nearest-neighbor search.
    pub fn knn_search<'a, S, T, OB>(
        &'a self,
        k: NonZeroUsize,
        nprobe: usize,
        strategy: &'a S,
        context: &'a P::Context,
        query: T,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<SearchStats>>
    where
        S: SearchStrategy<'a, P, T>,
        S::ListId: Eq,
        P::InternalId: NeighborPriorityQueueIdType,
        T: Copy + Send,
        OB: SearchOutputBuffer<P::InternalId> + Send + ?Sized,
    {
        async move {
            let mut list_accessor = strategy
                .list_accessor(&self.provider, context, query)
                .into_ann_result()?;

            let mut lists: Vec<Neighbor<S::ListId>> = Vec::with_capacity(nprobe);
            list_accessor
                .select_lists(nprobe, &mut lists)
                .await
                .escalate("IVF coarse list selection must complete")?;

            let mut search_accessor = strategy
                .search_accessor(&self.provider, context, query)
                .into_ann_result()?;

            let k = k.get();
            let mut queue = NeighborPriorityQueue::new(k);
            let mut cmps: u32 = 0;

            search_accessor
                .scan_lists(lists.iter().map(|n| n.id), |id, dist| {
                    cmps += 1;
                    queue.insert(Neighbor::new(id, dist));
                })
                .await
                .escalate("IVF list scan must complete to produce correct k-NN results")?;

            let result_count =
                output.extend(queue.iter().take(k).map(|n| (n.id, n.distance))) as u32;

            Ok(SearchStats { cmps, result_count })
        }
    }

    /// Insert a vector under external id `id`.
    pub fn insert<'a, S, T>(
        &'a self,
        strategy: &'a S,
        context: &'a P::Context,
        id: &P::ExternalId,
        vector: T,
    ) -> impl SendFuture<ANNResult<()>>
    where
        S: InsertStrategy<'a, P, T>,
        S::ListId: Eq,
        P: SetElement<T>,
        T: Copy + Send,
    {
        async move {
            let guard = self
                .provider
                .set_element(context, id, vector)
                .await
                .escalate("IVF insert requires a successful `set_element`")?;

            let internal_id = guard.id();

            let mut list_accessor = strategy
                .list_accessor(&self.provider, context, vector)
                .into_ann_result()?;

            let mut lists: Vec<Neighbor<S::ListId>> = Vec::with_capacity(1);

            list_accessor
                .select_lists(1, &mut lists)
                .await
                .escalate("IVF insert must select a target list")?;

            let list = lists.first().map(|n| n.id).ok_or_else(|| {
                ANNError::message(
                    ANNErrorKind::IndexError,
                    "IVF insert: list selection returned no candidate list",
                )
            })?;

            let mut insert_accessor = strategy
                .insert_accessor(&self.provider, context)
                .into_ann_result()?;

            insert_accessor
                .append(list, internal_id, vector)
                .await
                .escalate("IVF insert must append the vector to its assigned list")?;

            guard.complete().await;

            Ok(())
        }
    }
}

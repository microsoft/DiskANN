/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! IVF policy and orchestration, independent of physical list storage.

use std::{collections::BTreeSet, marker::PhantomData, num::NonZeroUsize};

use diskann_utils::future::SendFuture;
use thiserror::Error;

use super::traits::{BuildAccessor, Nearest, SearchAccessor, SearchStrategy};
use crate::{
    ANNResult,
    error::IntoANNResult,
    graph::SearchOutputBuffer,
    ivf::{
        region::AsMatRef,
        split::{self, KMeansSplit, Plan, Split, policies},
        traits::BuildStrategy,
    },
    neighbor::NeighborPriorityQueue,
    provider::ExecutionContext,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReassignCount {
    None,
    Count(NonZeroUsize),
}

/// Parameters for incremental k-means splitting.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Dimension of every canonical point and centroid supplied by the provider.
    pub dim: NonZeroUsize,
    /// Split a touched list when its pending membership exceeds this size.
    pub split_threshold: NonZeroUsize,
    /// Number of children `k` that each split parent is replaced with.
    pub split_k: NonZeroUsize,
    /// Nearest-list query count per split parent, before filtering split parents.
    ///
    /// - [`ReassignCount::None`] corresponds to no neighborhood gathering.
    /// - [`ReassignCount::Count`] coresponds to a non-zero neighborhood size to be gathered.
    pub reassign_count: ReassignCount,
    /// Positive number of Lloyd steps per parent.
    pub split_iterations: NonZeroUsize,
    pub seed: u64,
}

#[derive(Debug, Error)]
#[error("split_threshold must be at least 2, got {0}")]
pub struct ConfigError(pub usize);

/// Work successfully published by one insertion.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct InsertStats {
    pub inserted: usize,
    pub splits: usize,
}

/// An IVF index with serialized mutation through the underlying provider.
#[derive(Debug)]
pub struct IVFIndex<C: ExecutionContext> {
    config: Config,
    phantom: PhantomData<C>,
}

impl<C: ExecutionContext> IVFIndex<C> {
    /// # Errors
    ///
    /// Returns [`ConfigError`] when the split threshold is less than two.
    pub fn new(config: Config) -> Result<Self, ConfigError> {
        Ok(Self {
            config,
            phantom: PhantomData::default(),
        })
    }

    pub fn insert_batch<'a, S, B>(
        &'a mut self,
        strategy: &'a S,
        context: &'a C,
        points: B,
    ) -> impl SendFuture<ANNResult<InsertStats>>
    where
        S: BuildStrategy<'a, B, C>,
        S::Accessor: for<'b> BuildAccessor<B, VectorsRef<'b>: AsMatRef<'b, f32>>,
        B: Send + Sync,
    {
        let config = self.config;

        async move {
            let mut accessor = strategy.build_accessor(context).into_ann_result()?;

            let mut parents = BTreeSet::new();

            let inserted = accessor
                .route(points, |list, size| {
                    if size.get() > config.split_threshold.get() {
                        parents.insert(list);
                    }
                })
                .await?;

            if !parents.is_empty() {
                let mut plan = Plan::new(parents.iter().copied());

                if let ReassignCount::Count(n) = config.reassign_count {
                    let router = accessor.nearest();

                    for &parent in &parents {
                        router
                            .find(
                                parent,
                                n,
                                |id| !parents.contains(&id),
                                |id, _| plan.link(parent, id),
                            )
                            .await?;
                    }
                }

                let lists = plan.sources().map(|(_, list)| list);
                let (region, mut update) = accessor.fill(lists).await?;

                KMeansSplit::new(
                    config.split_k,
                    config.split_iterations,
                    config.dim,
                    policies::NearestCentroid,
                    policies::FirstPoints,
                )
                .split(&region, &mut update, &plan)?;
            };

            accessor.finish().await?;

            Ok(InsertStats {
                inserted: inserted.get(),
                splits: parents.len(),
            })
        }
    }

    /// Append the best `k` candidates from the provider-selected lists to `output`.
    ///
    /// # Errors
    ///
    /// Accessor construction or scan failures leave `output` unchanged.
    pub fn search<'a, S, T, OB>(
        &'a self,
        strategy: &'a S,
        context: &'a C,
        query: T,
        k: NonZeroUsize,
        nprobe: NonZeroUsize,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<usize>>
    where
        S: SearchStrategy<'a, T, C>,
        T: Copy + Send + Sync,
        OB: SearchOutputBuffer<S::Id> + Send + ?Sized,
    {
        async move {
            let accessor = strategy.search_accessor(context, query).into_ann_result()?;
            let mut queue = NeighborPriorityQueue::new(k.get());
            accessor.scan(nprobe, &mut queue).await?;

            Ok(output.extend(queue.iter()))
        }
    }
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! IVF policy and orchestration, independent of physical list storage.

use std::{
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
    num::NonZeroUsize,
};

use diskann_utils::future::SendFuture;
use thiserror::Error;

use super::traits::{BuildAccessor, Nearest, SearchAccessor, SearchStrategy};
use crate::{
    ANNResult,
    error::IntoANNResult,
    graph::SearchOutputBuffer,
    ivf::{
        region::{AsMatRef, Update},
        split::{KMeansSplit, Linked, Links, Reassign, Split, policies},
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
    /// Nearest-list query count per child of a split, excluding its siblings.
    ///
    /// - [`ReassignCount::None`] skips reassignment.
    /// - [`ReassignCount::Count`] lets each child exchange points with that many lists.
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
                let parents: Vec<_> = parents.iter().copied().collect();

                // Split: replace every parent with fitted children.
                let mut children = Vec::new();
                {
                    let (region, mut update) = accessor.fill(parents.iter().copied()).await?;

                    KMeansSplit::new(
                        config.split_k,
                        config.split_iterations,
                        config.dim,
                        policies::NearestCentroid,
                        policies::FirstPoints,
                    )
                    .split(
                        &region,
                        &mut update,
                        &parents,
                        |parent, child| children.push((parent, child)),
                    )?;

                    update.finish().await?;
                }

                // Reassign: each child exchanges points with its nearest lists, other than
                // its siblings.
                if let ReassignCount::Count(n) = config.reassign_count {
                    // Each child's parent, to recognize siblings.
                    let parent_of: BTreeMap<_, _> = children
                        .iter()
                        .map(|&(parent, child)| (child, parent))
                        .collect();

                    let mut links = Links::new(children.iter().map(|&(_, child)| child));
                    
                    {
                        let router = accessor.nearest();

                        for (parent, child) in children {
                            // Skip the child itself and its siblings
                            router
                                .find(
                                    child,
                                    n,
                                    |id| parent_of.get(&id) != Some(&parent),
                                    |id, _| links.link(child, id),
                                )
                                .await?;
                        }
                    }

                    let lists = links.lists().iter().copied();
                    let (region, mut update) = accessor.fill(lists).await?;

                    Linked::new(policies::NearestCentroid).reassign(
                        &region,
                        &mut update,
                        &links,
                    )?;

                    update.finish().await?;
                }
            };

            // Commit the updates registered from the split and re-assign operations.
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

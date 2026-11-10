/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! [`IVFIndex`] and its configuration.

use std::num::NonZeroUsize;

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Debug,
    hash::Hash,
};

use diskann_quantization::Parallelism;
use diskann_utils::future::SendFuture;
use thiserror::Error;

use super::traits::SearchAccessor;
use crate::{
    ANNResult,
    graph::SearchOutputBuffer,
    neighbor::{NeighborPriorityQueue, NeighborPriorityQueueIdType},
};

use super::{
    region::{AsMatRef, Region},
    split::{KMeansSplit, Linked, Links, Reassign, Split, policies},
    traits::BuildAccessor,
};

/// Number of nearby lists each child of a split considers doing re-assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReassignCount {
    /// Skip reassignment
    None,
    /// Link each child to `Count` nearest lists, other than its siblings.
    Count(NonZeroUsize),
}

/// Configuration of an [`IVFIndex`].
#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Dimension of the vectors and centroids that providers fill.
    pub dim: NonZeroUsize,
    /// Split a list when routing leaves it with more than this many members.
    pub split_threshold: NonZeroUsize,
    /// Number of children that replace each split list.
    pub split_k: NonZeroUsize,
    /// Number of nearby lists each child considers for re-assignment after a split.
    pub reassign_count: ReassignCount,
    /// Number of Lloyd iterations that fit the children of each split list.
    pub split_iterations: NonZeroUsize,
}

/// Error returned by [`IVFIndex::new`] for an invalid [`Config`].
#[derive(Debug, Error)]
pub enum ConfigError {
    /// [`Config::split_k`] is 1, so splitting would not shrink a list.
    #[error("split_k must be at least 2, got {0}")]
    TooFewChildren(usize),
    /// [`Config::split_k`] exceeds the smallest size of a split list.
    #[error("split_k ({split_k}) must be at most split_threshold + 1 ({split_threshold} + 1)")]
    TooManyChildren {
        split_k: usize,
        split_threshold: usize,
    },
}

/// Work successfully published by one insertion.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct InsertStats {
    /// Number of points inserted.
    pub inserted: usize,
    /// Number of lists split.
    pub splits: usize,
}

/// An incremental IVF index.
///
/// The index holds only its [`Config`]. Inserting needs the `kmeans` feature.
///
/// Providers shared by several indexes must still keep build operations apart;
/// see [`build_accessor`](super::traits::BuildStrategy::build_accessor).
#[derive(Debug)]
pub struct IVFIndex {
    config: Config,
}

impl IVFIndex {
    /// Construct an index with `config`.
    ///
    /// # Errors
    ///
    /// Fails if [`Config::split_k`] is less than 2, or greater than
    /// [`Config::split_threshold`] plus one.
    pub fn new(config: Config) -> Result<Self, ConfigError> {
        let (split_k, split_threshold) = (config.split_k.get(), config.split_threshold.get());

        if split_k < 2 {
            return Err(ConfigError::TooFewChildren(split_k));
        }
        if split_k - 1 > split_threshold {
            return Err(ConfigError::TooManyChildren {
                split_k,
                split_threshold,
            });
        }

        Ok(Self { config })
    }

    /// Return the configuration of the index.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Insert a batch of points as one build operation, splitting every list that routing
    /// leaves with more than [`Config::split_threshold`] members and then re-assigning
    /// points if [`ReassignCount`] in `config` is not zero.
    ///
    /// # Errors
    ///
    /// Fails if the batch is empty, or if a provider call or split fails. Any failure
    /// abandons the operation: the accessor is dropped without being finished.
    #[cfg(feature = "ivf")]
    pub fn insert_batch<'a, A, B>(
        &'a mut self,
        mut accessor: A,
        points: B,
        parallelism: Parallelism,
    ) -> impl SendFuture<ANNResult<InsertStats>>
    where
        A: for<'b> BuildAccessor<B, VectorsRef<'b>: AsMatRef<'b, f32>>,
        B: Send + Sync,
    {
        let config = self.config;

        async move {
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
                    let mut region = accessor.fill(parents.iter().copied()).await?;

                    KMeansSplit::new(
                        config.split_k,
                        config.split_iterations,
                        config.dim,
                        policies::NearestCentroid,
                        policies::FirstPoints,
                    )
                    .split(
                        &mut region,
                        &parents,
                        parallelism,
                        |parent, child| children.push((parent, child)),
                    )?;

                    region.finish().await?;
                }

                // Reassign: each child potentially exchanges points with its nearest lists, other than
                // its siblings.
                if let ReassignCount::Count(n) = config.reassign_count {
                    // Each child's parent, to recognize siblings.
                    let parent_of: BTreeMap<_, _> = children
                        .iter()
                        .map(|&(parent, child)| (child, parent))
                        .collect();

                    let mut links = Links::new(children.iter().map(|&(_, child)| child));

                    for (parent, child) in children {
                        accessor
                            .find(
                                child,
                                n,
                                // Skip the child itself and its siblings.
                                |id| parent_of.get(&id) != Some(&parent),
                                |id, _| links.link(child, id),
                            )
                            .await?;
                    }

                    let lists = links.lists().iter().copied();
                    let mut region = accessor.fill(lists).await?;

                    Linked::new(policies::NearestCentroid).reassign(
                        &mut region,
                        &links,
                        parallelism,
                    )?;

                    region.finish().await?;
                }
            };

            // Publish the routed points and the changes of both stages.
            accessor.finish().await?;

            Ok(InsertStats {
                inserted: inserted.get(),
                splits: parents.len(),
            })
        }
    }

    /// Search the `nprobe` lists nearest `query`, and append the best `k` members found to
    /// `output`, nearest first. Return the number of results appended.
    ///
    /// # Errors
    ///
    /// Fails if the accessor cannot be created or the scan fails, leaving `output`
    /// unchanged.
    pub fn search<'a, A, OB>(
        &'a self,
        accessor: A,
        k: NonZeroUsize,
        nprobe: NonZeroUsize,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<usize>>
    where
        A: SearchAccessor,
        A::Id: NeighborPriorityQueueIdType,
        OB: SearchOutputBuffer<A::Id> + Send + ?Sized,
    {
        async move {
            let mut queue = NeighborPriorityQueue::new(k.get());
            accessor.scan(nprobe, &mut queue).await?;

            Ok(output.extend(queue.iter()))
        }
    }
}

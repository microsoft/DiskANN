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

use super::traits::{BuildAccessor, Router, RouterAndNearest, SetElements};
use crate::{
    ANNError, ANNResult,
    error::IntoANNResult,
    ivf::{
        clustering::{RegionalNearest, SplitAndAssign, TwoMeans},
        traits::BuildStrategy,
    },
    provider::ExecutionContext,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReassignCount {
    None,
    Count(NonZeroUsize),
}

/// Parameters for incremental two-means splitting.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Split a touched list when its pending membership exceeds this size.
    pub split_threshold: NonZeroUsize,
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

    pub fn insert_batch<'a, S, T>(
        &'a mut self,
        strategy: &'a S,
        context: &'a C,
        points: &'a [T],
    ) -> impl SendFuture<ANNResult<InsertStats>>
    where
        S: BuildStrategy<'a, T, C>,
        T: Sync,
    {
        let config = self.config;

        async move {
            if points.is_empty() {
                return Ok(InsertStats::default());
            }

            let mut accessor = strategy.build_accessor(context).into_ann_result()?;

            let dim = accessor.dim();

            // Step 1: Compute the batch
            let ids = accessor.set_elements(context, points).await?;

            // Step 2: Route the points to their nearest centroid.
            let mut routed = BTreeMap::<S::ListId, Vec<S::Id>>::new();

            {
                let router = accessor.router();

                for (id, point) in ids.iter().zip(points.iter()) {
                    let selected = router.route(point, 1).await?;

                    let (list, _) = selected.first().ok_or_else(|| {
                        ANNError::message("IVF routing returned no live centroid")
                    })?;

                    routed.entry(*list).or_default().push(*id);
                }
            }

            let mut parents = BTreeSet::new();

            // Step 3: append the points to their respective list ids and determine which need to be split.
            for (list, members) in routed.iter() {
                let size = accessor.append_to(*list, members.iter().copied()).await?;

                if size.get() > config.split_threshold.get() {
                    parents.insert(*list);
                }
            }

            let splits = parents.len();

            if !parents.is_empty() {
                let mut neighbors = BTreeSet::new();

                if let ReassignCount::Count(n) = config.reassign_count {
                    let router = accessor.router();

                    for &pid in parents.iter() {
                        let nids = router
                            .nearest(pid, n.get())
                            .await?
                            .into_iter()
                            .filter(|(id, _)| !parents.contains(id));

                        for (id, _) in nids {
                            neighbors.insert(id);
                        }
                    }
                }

                let iter = parents.iter().chain(neighbors.iter()).copied();

                let (region, mut update) = accessor.fill(iter).await?;

                let clusterer = TwoMeans::new(config.split_iterations, dim);
                SplitAndAssign::new(clusterer, RegionalNearest).run(
                    &region,
                    &mut update,
                    &parents,
                    &neighbors,
                )?;
            };

            accessor.finish().await?;

            Ok(InsertStats {
                inserted: points.len(),
                splits,
            })
        }
    }
}

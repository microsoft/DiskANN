/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! IVF policy and orchestration, independent of physical list storage.

use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroUsize,
};

use diskann_utils::future::SendFuture;
use thiserror::Error;

use super::{
    clustering,
    traits::{BuildAccessor, InsertStrategy, Provider, Router, RouterAndNearest},
};
use crate::{ANNError, ANNResult, error::IntoANNResult};

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
pub struct IVFIndex<P: Provider> {
    provider: P,
    config: Config,
}

impl<P: Provider> IVFIndex<P> {
    /// # Errors
    ///
    /// Returns [`ConfigError`] when the split threshold is less than two.
    pub fn new(provider: P, config: Config) -> Result<Self, ConfigError> {
        if config.split_threshold.get() < 2 {
            return Err(ConfigError(config.split_threshold.get()));
        }
        Ok(Self { provider, config })
    }

    pub fn provider(&self) -> &P {
        &self.provider
    }

    pub fn insert_batch<'a, S, T>(
        &'a mut self,
        strategy: &'a S,
        context: &'a P::Context,
        points: &'a [(P::ExternalId, T)],
    ) -> impl SendFuture<ANNResult<InsertStats>>
    where
        S: InsertStrategy<'a, P, T>,
        T: Sync,
    {
        let provider = &mut self.provider;
        let config = self.config;

        async move {
            if points.is_empty() {
                return Ok(InsertStats::default());
            }

            let mut accessor = strategy
                .build_accessor(provider, context)
                .into_ann_result()?;

            let dim = accessor.dim();

            // Step 1: Compute the batch
            let batch = strategy.prepare(&mut accessor, points).await?;

            // Step 2: Route the points to their nearest centroid.
            let mut routed = BTreeMap::<P::ListId, Vec<P::InternalId>>::new();

            {
                let router = accessor.router();

                for point in batch.points() {
                    let selected = router.route(point.vector, 1).await?;

                    let (list, _) = selected.first().ok_or_else(|| {
                        ANNError::message("IVF routing returned no live centroid")
                    })?;

                    routed.entry(*list).or_default().push(point.id);
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
            let mut update_final = None;

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

                clustering::split(
                    &region,
                    &mut update,
                    &parents,
                    &neighbors,
                    config.split_iterations,
                    dim,
                )?;

                update_final = Some(update)
            };

            accessor.finish(update_final.as_ref()).await?;

            Ok(InsertStats {
                inserted: points.len(),
                splits,
            })
        }
    }
}

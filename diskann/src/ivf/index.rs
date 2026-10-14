/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! IVF policy and orchestration, independent of physical list storage.

use std::collections::BTreeMap;

use diskann_utils::future::SendFuture;
use rand::{SeedableRng, rngs::StdRng};
use thiserror::Error;

use super::{
    clustering,
    traits::{BuildAccessor, InsertStrategy, Provider, Router},
};
use crate::{ANNError, ANNResult, error::IntoANNResult};

/// Parameters for incremental two-means splitting.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Split a touched list when its pending membership exceeds this size.
    pub split_threshold: usize,
    /// Lloyd steps per parent. Zero still performs one step.
    pub two_means_iterations: usize,
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
    rng: StdRng,
}

impl<P: Provider> IVFIndex<P> {
    /// # Errors
    ///
    /// Returns [`ConfigError`] when the split threshold is less than two.
    pub fn new(provider: P, config: Config) -> Result<Self, ConfigError> {
        if config.split_threshold < 2 {
            return Err(ConfigError(config.split_threshold));
        }
        Ok(Self {
            provider,
            rng: StdRng::seed_from_u64(config.seed),
            config,
        })
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
        let rng = &mut self.rng;

        async move {
            if points.is_empty() {
                return Ok(InsertStats::default());
            }
            let mut accessor = strategy
                .build_accessor(provider, context)
                .into_ann_result()?;

            // Step 1: Compute the batch
            let batch = strategy.prepare(&mut accessor, points).await?;

            // Step 2: Route the points to their nearest centroid.
            let mut routed = BTreeMap::<P::ListId, Vec<P::InternalId>>::new();

            {
                let router = accessor.router();

                for point in batch.points() {
                    let selected = router.select(point.vector, 1).await?;
                    let list = selected
                        .first()
                        .ok_or_else(|| ANNError::message("IVF routing returned no live centroid"))?
                        .0;

                    routed.entry(list).or_default().push(point.id);
                }
            }

            let mut parents = Vec::with_capacity(routed.len());

            // append the points to their respective list ids and determine which need to be split.
            for (list, members) in routed {
                let size = accessor.append_to(list, &members).await?;
                if size > config.split_threshold {
                    parents.push(list);
                }
            }

            let splits = parents.len();
            let update = if !parents.is_empty() {
                let dim = accessor.dim();
                let (region, mut update) = accessor.fill(&parents).await?;
                clustering::split(
                    &region,
                    &mut update,
                    &parents,
                    dim,
                    config.two_means_iterations,
                    rng,
                )?;
                Some(update)
            } else {
                None
            };

            accessor.finish(update.as_ref()).await?;

            Ok(InsertStats {
                inserted: points.len(),
                splits,
            })
        }
    }
}

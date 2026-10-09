/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! IVF policy and orchestration, independent of physical list storage.

use std::collections::BTreeMap;

use diskann_utils::{
    future::SendFuture,
    views::rowmajor::{self, Matrix},
};
use rand::{SeedableRng, rngs::StdRng};
use thiserror::Error;

use super::{
    clustering,
    traits::{
        BuildAccessor, BuildStrategy, CoarseAccessor, InsertStrategy, ListAccessor, Provider,
    },
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

    /// Install initial empty lists and their centroids in one build operation.
    ///
    /// # Errors
    ///
    /// Fails for an already initialized index, or a provider error.
    /// Failed publication is provider-defined.
    pub fn initialize<'a, S>(
        &'a mut self,
        strategy: &'a S,
        context: &'a P::Context,
        centroids: rowmajor::Ref<'_, f32>,
    ) -> impl SendFuture<ANNResult<()>>
    where
        S: BuildStrategy<'a, P>,
    {
        async move {
            if centroids.nrows() == 0 || centroids.ncols() == 0 {
                return Err(ANNError::message(
                    "IVF initialization needs nonempty centroids",
                ));
            }
            let mut accessor = strategy
                .build_accessor(&mut self.provider, context)
                .into_ann_result()?;

            if !accessor.coarse().is_empty() {
                return Err(ANNError::message("the IVF index is already initialized"));
            }

            for centroid in centroids.rows() {
                // TO DO: Need to probably introduce an initialization strategy
                accessor.lists().create_list(centroid, Vec::new()).await?;
            }
            accessor.finish().await
        }
    }

    /// Route and provisionally append a batch, split oversized lists, then publish.
    ///
    /// All split parents form one working set. Their members choose among every
    /// fitted child, never among retiring parent centroids. Other appends keep their
    /// routes. Point coordinates are borrowed throughout fitting and assignment.
    ///
    /// # Errors
    ///
    /// Fails before publication if the index is uninitialized or preparation,
    /// routing, filling, fitting, or list mutation fails. State after a failed
    /// `finish` is provider-defined. An empty batch is a no-op.
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
            if accessor.coarse().is_empty() {
                return Err(ANNError::message("the IVF index is not initialized"));
            }
            let batch = strategy.prepare(&mut accessor, points).await?;

            let mut routed = BTreeMap::<P::ListId, Vec<P::InternalId>>::new();

            for point in batch.points() {
                let selected = accessor.coarse().select(point.vector, 1).await?;
                let list = selected
                    .first()
                    .ok_or_else(|| ANNError::message("IVF routing returned no live centroid"))?
                    .id;
                routed.entry(list).or_default().push(point.id);
            }

            // Stable list order also makes seeded fitting independent of hash order.
            let mut parents = Vec::with_capacity(routed.len());
            for (list, members) in routed {
                let mut lists = accessor.lists();
                lists.append_members(list, members).await?;
                if lists.list_size(list).await? > config.split_threshold {
                    parents.push(list);
                }
            }
            let splits = parents.len();
            if !parents.is_empty() {
                let children = {
                    let dim = accessor.dim();
                    let view = accessor.fill(&parents).await?;
                    clustering::split(&view, dim, config.two_means_iterations, rng)?
                };
                for (centroid, members) in children.centroids.rows().zip(children.members) {
                    accessor.lists().create_list(centroid, members).await?;
                }
                for parent in parents {
                    accessor.lists().retire_list(parent).await?;
                }
            }

            accessor.finish().await?;

            Ok(InsertStats {
                inserted: points.len(),
                splits,
            })
        }
    }
}

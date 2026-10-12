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
    partition::Partitioning,
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
    /// TO DO: needs 
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
                accessor.lists().create_list(centroid, Vec::new()).await?;
            }
            accessor.finish().await
        }
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
                    .0;

                routed.entry(list).or_default().push(point.id);
            }

            let mut parents = Vec::with_capacity(routed.len());

            for (list, members) in routed {
                let size = accessor.lists().append_members(list, members).await?;
                if size > config.split_threshold {
                    parents.push(list);
                }
            }

            let splits = parents.len();
            if !parents.is_empty() {
                let replacement = {
                    let dim = accessor.dim();
                    let view = accessor.fill(&parents).await?;
                    clustering::split(&view, &parents, dim, config.two_means_iterations, rng)?
                };
                Self::apply_partitioning(&mut accessor.lists(), &parents, replacement).await?;
            }

            accessor.finish().await?;

            Ok(InsertStats {
                inserted: points.len(),
                splits,
            })
        }
    }

    /// Apply destination memberships before retiring the replaced lists.
    pub(super) async fn apply_partitioning<A>(
    accessor: &mut A,
    retired: &[A::ListId],
    partitioning: Partitioning<A::InternalId, A::ListId>,
) -> ANNResult<()>
where
    A: ListAccessor,
{
    for (centroid, partition) in partitioning.centroids.rows().zip(partitioning.partitions) {
        match partition.id {
            Some(id) => accessor.set_members(id, partition.members).await?,
            None => {
                accessor.create_list(centroid, partition.members).await?;
            }
        }
    }
    for &id in retired {
        accessor.retire_list(id).await?;
    }
    Ok(())
}
}




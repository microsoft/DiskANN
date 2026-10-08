/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Dynamic IVF index orchestration.
//!
//! The index drives one [`InsertAccessor`] per mutation: it plans a
//! [`Deltas`](super::update::Deltas) against the accessor's view and hands it to
//! [`InsertAccessor::update`]. It never touches point, centroid, or list storage directly.

use diskann_utils::{
    future::SendFuture,
    views::rowmajor::{self, Matrix, MatrixMut},
};
use rand::{SeedableRng, rngs::StdRng};
use thiserror::Error;

use crate::{
    ANNError, ANNResult,
    error::{ErrorExt, IntoANNResult},
    ivf::{
        dynamic::{Centroids, InsertAccessor, MaintenanceStrategy, Provider, Stage},
        split::plan_split_update,
        update::{self, Deltas},
    },
};

/// Parameters for the online split policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    /// Split a list once an insert batch would grow it beyond this many points.
    pub split_threshold: usize,
    /// Nearby lists reassigned together with each split parent.
    pub reassign_neighbors: usize,
    /// Lloyd iterations used to fit split children.
    pub two_means_iterations: usize,
    /// Seed for split-child initialization.
    pub seed: u64,
}

/// Invalid [`DynamicIvfConfig`] values.
#[derive(Debug, Clone, Copy, Error, PartialEq, Eq)]
pub enum ConfigError {
    #[error("split_threshold must be at least 2, got {0}")]
    SplitThreshold(usize),
    #[error("reassign_neighbors must be at least 1")]
    ReassignNeighbors,
}

impl From<ConfigError> for ANNError {
    #[track_caller]
    fn from(err: ConfigError) -> Self {
        ANNError::new(err)
    }
}

/// Work performed by one [`DynamicIvfIndex::insert_batch`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InsertStats {
    /// Points added to the index.
    pub inserted: usize,
    /// Lists split into two children.
    pub splits: usize,
    /// Previously indexed points moved by regional reassignment.
    pub reassigned: usize,
}

/// An incrementally maintained IVF index.
///
/// Mutations take `&mut self` and lend the provider exclusively to one
/// [`MaintenanceAccessor`], so they cannot overlap searches through the same value.
#[derive(Debug)]
pub struct IVFIndex<P: Provider> {
    provider: P,
    config: Config,
    rng: StdRng,
}

impl<P: Provider> IVFIndex<P> {
    /// Construct an index over `provider`.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if `config` is invalid.
    pub fn new(provider: P, config: Config) -> Result<Self, ConfigError> {
        if config.split_threshold < 2 {
            return Err(ConfigError::SplitThreshold(config.split_threshold));
        }
        if config.reassign_neighbors == 0 {
            return Err(ConfigError::ReassignNeighbors);
        }
        Ok(Self {
            provider,
            rng: StdRng::seed_from_u64(config.seed),
            config,
        })
    }

    /// Borrow the underlying provider.
    pub fn provider(&self) -> &P {
        &self.provider
    }

    /// Install the initial centroids.
    ///
    /// # Errors
    ///
    /// Fails if `centroids` is empty, non-finite, or of the wrong dimension, the index
    /// already has centroids, or the maintenance accessor fails.
    pub fn initialize<'a, S, T>(
        &'a mut self,
        strategy: &'a S,
        context: &'a P::Context,
        centroids: rowmajor::Ref<'_, f32>,
    ) -> impl SendFuture<ANNResult<()>>
    where
        S: MaintenanceStrategy<'a, P, T>,
        T: Sync,
    {
        let provider = &mut self.provider;

        async move {
            let count = centroids.nrows();
            if count == 0 {
                return Err(ANNError::message("Cannot initialize with zero centroids."));
            }

            let mut accessor = strategy
                .maintenance_accessor(provider, context)
                .into_ann_result()?;

            let mut installs = Vec::with_capacity(centroids.nrows());

            for c in centroids.rows() {
                let id = accessor.stage_centroid(c).await?;
                installs.push(update::Delta::CentroidDelta {
                    id,
                    delta: update::CentroidDelta::Install,
                });
            }

            accessor
                .update(Deltas::<P::InternalId, _>::new(installs))
                .await
                .escalate("initialize must apply the initial centroids")?;

            Ok(())
        }
    }

    /// Insert a batch, splitting every list the batch pushes past `split_threshold`.
    ///
    /// All split parents form one region. Their points are assigned across all fitted
    /// children, never back to retiring parents. Other staged points keep their routes.
    /// The batch and its splits are then applied as one
    /// [`Deltas`](super::update::Deltas).
    ///
    /// # Errors
    ///
    /// Fails if the index is not initialized, the provider returns inconsistent
    /// data, or any accessor call fails. State after an `update` failure is
    /// provider-defined.
    pub fn insert_batch<'a, S, T>(
        &'a mut self,
        strategy: &'a S,
        context: &'a P::Context,
        points: &'a [(P::ExternalId, T)],
    ) -> impl SendFuture<ANNResult<InsertStats>>
    where
        S: MaintenanceStrategy<'a, P, T>,
        T: Sync + Copy,
    {
        let provider = &mut self.provider;
        let config = self.config;
        let rng = &mut self.rng;

        async move {
            if points.is_empty() {
                return Ok(InsertStats::default());
            }

            let mut accessor = strategy
                .maintenance_accessor(provider, context)
                .into_ann_result()?;

            // stage the input points
            let mut vectors = rowmajor::Owned::from_element(points.len(), accessor.dim(), f32::NAN);
            let mut ids = Vec::with_capacity(points.len());
            let mut routed = hashbrown::HashMap::<P::ListId, Vec<usize>>::new();

            for (position, ((id, v), out)) in points.iter().zip(vectors.rows_mut()).enumerate() {
                //stage the input points first.
                let id = accessor.stage_point(id, *v, out).await.into_ann_result()?;
                ids.push(id);

                // route the points to their nearest centroid.
                let parent = accessor
                    .centroids()
                    .select(out, 1)
                    .escalate("insert must route every staged point")?
                    .first()
                    .map(|s| s.id)
                    .ok_or_else(|| ANNError::message("Didn't return list"))?;

                routed.entry(parent).or_default().push(position);
            }

            // filter the routed lists for ones that need splitting.
            let mut parents = Vec::new();
            for (&list, positions) in &routed {
                let len = accessor.list_size(list).into_ann_result()?;

                if len + positions.len() > config.split_threshold {
                    parents.push(list);
                }
            }

            // Fit every split's children before assigning points and building the update.
            let update = plan_split_update::<P, _, T>(
                &mut accessor,
                config.two_means_iterations,
                rng,
                &ids,
                vectors.as_view(),
                &routed,
                &parents,
            )
            .await?;

            accessor
                .update(update)
                .await
                .escalate("insert must apply its update")?;

            Ok(InsertStats::default())
        }
    }
}

#[cfg(test)]
mod tests {}

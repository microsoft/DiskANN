/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Dynamic IVF index orchestration.
//!
//! The index drives one [`InsertAccessor`] per mutation: it plans a
//! [`Deltas`](super::update::Deltas) against the accessor's view and hands it to
//! [`InsertAccessor::update`]. Gathering and update construction use only the accessor;
//! splitting and assignment operate on an in-memory region.

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
        traits::{Centroids, InsertAccessor, MaintenanceStrategy, Provider, Reader, Stage},
        online::index_error,
        split::{Assign, NearestCentroid, Region, SourceList, Split, TwoMeansSplit},
        update::{self, CentroidDelta, Delta, Deltas, MoveTo},
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

/// A staged point and its borrowed canonical vector, ready for regional gathering.
pub(super) struct StagedPoint<'a, I> {
    pub(super) id: I,
    pub(super) vector: &'a [f32],
}

/// Staged points routed to one existing list, in insertion order.
pub(super) struct ListInsertion<'a, I, L> {
    pub(super) list: L,
    pub(super) staged: Vec<StagedPoint<'a, I>>,
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

            let mut vectors = rowmajor::Owned::from_element(points.len(), accessor.dim(), f32::NAN);
            let mut routed =
                hashbrown::HashMap::<P::ListId, Vec<StagedPoint<'_, P::InternalId>>>::new();

            for ((id, v), out) in points.iter().zip(vectors.rows_mut()) {
                let id = accessor.stage_point(id, *v, out).await.into_ann_result()?;

                let parent = accessor
                    .centroids()
                    .select(out, 1)
                    .escalate("insert must route every staged point")?
                    .first()
                    .map(|s| s.id)
                    .ok_or_else(|| ANNError::message("Didn't return list"))?;

                routed
                    .entry(parent)
                    .or_default()
                    .push(StagedPoint { id, vector: out });
            }

            let mut split_inputs = Vec::with_capacity(routed.len());
            let mut deltas = Vec::with_capacity(routed.len());
            for (list, staged) in routed {
                let len = accessor.list_size(list).into_ann_result()?;

                if len + staged.len() > config.split_threshold {
                    split_inputs.push(ListInsertion { list, staged });
                } else {
                    deltas.push(update::Delta::PointAppends {
                        to: list,
                        ids: staged.into_iter().map(|point| point.id).collect(),
                    });
                }
            }

            if !split_inputs.is_empty() {
                let mut region = Self::gather_region::<_, T>(&mut accessor, &split_inputs).await?;
                let parents: Vec<_> = split_inputs.iter().map(|input| input.list).collect();
                TwoMeansSplit {
                    iterations: config.two_means_iterations,
                }
                .split(&mut region, &parents, rng)?;
                let assignments = NearestCentroid.assign(&region)?;
                deltas.extend(
                    Self::build_deltas::<_, T>(&mut accessor, &region, &assignments).await?,
                );
            }

            accessor
                .update(Deltas::new(deltas))
                .await
                .escalate("insert must apply its update")?;

            Ok(InsertStats::default())
        }
    }

    /// Gather committed members and staged points into their final region rows.
    pub(super) async fn gather_region<A, T>(
        accessor: &mut A,
        inputs: &[ListInsertion<'_, P::InternalId, P::ListId>],
    ) -> ANNResult<Region<P::InternalId, P::ListId>>
    where
        A: InsertAccessor<P, T>,
        T: Sync,
    {
        let mut lists = Vec::with_capacity(inputs.len());
        let mut rows = 0;
        for input in inputs {
            let id = input.list;
            let member_count = accessor.get_members(id).into_ann_result()?.len();
            let end = rows + member_count + input.staged.len();
            lists.push(SourceList {
                id,
                rows: rows..end,
                member_count,
            });
            rows = end;
        }

        let dim = accessor.dim();
        let mut point_ids = Vec::with_capacity(rows);
        let mut points = rowmajor::Owned::try_from_element(rows, dim, 0.0)?;
        let mut centroids = rowmajor::Owned::try_from_element(inputs.len(), dim, 0.0)?;
        let mut centroid_ids = Vec::with_capacity(inputs.len());

        for (index, (list, input)) in lists.iter().zip(inputs).enumerate() {
            point_ids.extend_from_slice(accessor.get_members(list.id).into_ann_result()?);

            let member_end = list.rows.start + list.member_count;
            accessor
                .reader()
                .read_into(
                    list.id,
                    points
                        .subview_mut(list.rows.start..member_end)
                        .ok_or_else(|| {
                            index_error("member rows lie outside the region's point matrix")
                        })?,
                )
                .await
                .escalate("split must read the requested list's vectors")?;
            let mut tail = points
                .subview_mut(member_end..list.rows.end)
                .ok_or_else(|| index_error("staged rows lie outside the region's point matrix"))?;
            for (point, out) in input.staged.iter().zip(tail.rows_mut()) {
                point_ids.push(point.id);
                out.copy_from_slice(point.vector);
            }

            let catalog = accessor.centroids();
            let centroid = catalog
                .centroid(list.id)
                .ok_or_else(|| index_error(format!("centroid {} is unavailable", list.id)))?;
            centroids.row_mut(index).copy_from_slice(&centroid);
            centroid_ids.push(Some(list.id));
        }

        Ok(Region {
            lists,
            point_ids,
            points,
            centroid_ids,
            centroids,
        })
    }

    /// Stage proposed centroids and translate aligned assignments into partition deltas.
    pub(super) async fn build_deltas<A, T>(
        accessor: &mut A,
        region: &Region<P::InternalId, P::ListId>,
        assignments: &[usize],
    ) -> ANNResult<Vec<Delta<P::InternalId, P::ListId>>>
    where
        A: InsertAccessor<P, T>,
        T: Sync,
    {
        let mut deltas = Vec::with_capacity(2 * region.centroid_ids.len() + region.lists.len());
        let mut destinations = Vec::with_capacity(region.centroid_ids.len());
        for (id, centroid) in region.centroid_ids.iter().zip(region.centroids.rows()) {
            let id = match id {
                Some(id) => *id,
                None => {
                    let id = accessor.stage_centroid(centroid).await?;
                    deltas.push(Delta::CentroidDelta {
                        id,
                        delta: CentroidDelta::Install,
                    });
                    id
                }
            };
            destinations.push(id);
        }

        let mut append_counts = vec![0; destinations.len()];

        for (index, list) in region.lists.iter().enumerate() {
            let retired = !region.centroid_ids.contains(&Some(list.id));
            let mut moves = Vec::with_capacity(list.member_count);
            for (&id, &to) in region
                .member_ids(index)
                .iter()
                .zip(&assignments[list.rows.start..list.rows.start + list.member_count])
            {
                let to = destinations[to];
                if to != list.id {
                    moves.push(MoveTo::new(id, to));
                }
            }
            if retired {
                deltas.push(Delta::CentroidDelta {
                    id: list.id,
                    delta: CentroidDelta::Retire {
                        moves: moves.into_boxed_slice(),
                    },
                });
            } else if !moves.is_empty() {
                deltas.push(Delta::PointMoves {
                    from: list.id,
                    moves: moves.into_boxed_slice(),
                });
            }
            for &to in &assignments[list.rows.start + list.member_count..list.rows.end] {
                append_counts[to] += 1;
            }
        }

        let mut appends: Vec<_> = append_counts.into_iter().map(Vec::with_capacity).collect();
        for (index, list) in region.lists.iter().enumerate() {
            for (&id, &to) in region
                .staged_ids(index)
                .iter()
                .zip(&assignments[list.rows.start + list.member_count..list.rows.end])
            {
                appends[to].push(id);
            }
        }
        for (to, ids) in destinations.into_iter().zip(appends) {
            if !ids.is_empty() {
                deltas.push(Delta::PointAppends {
                    to,
                    ids: ids.into_boxed_slice(),
                });
            }
        }
        Ok(deltas)
    }
}

#[cfg(test)]
mod tests {}

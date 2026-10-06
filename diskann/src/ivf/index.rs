/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Dynamic IVF index orchestration.
//!
//! The index drives one [`InsertAccessor`] per mutation: it plans a [`Deltas`] against
//! the accessor's view and hands it to [`InsertAccessor::update`]. It never touches
//! point, centroid, or list storage directly.

use diskann_utils::{
    future::SendFuture,
    views::{Matrix, MatrixView},
};
use rand::{SeedableRng, rngs::StdRng};
use thiserror::Error;

use crate::{
    ANNError, ANNErrorKind, ANNResult,
    error::{ErrorExt, IntoANNResult},
    ivf::{
        dynamic::{
            Centroids, InsertAccessor, MaintenanceStrategy, Provider, Reader, StageElements,
        },
        online::{TwoMeans, index_error, two_means},
        update::{CentroidDelta, Delta, Deltas, MoveTo},
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
        ANNError::new(ANNErrorKind::IndexConfigError, err)
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

    // /// Install the initial centroids.
    // ///
    // /// # Errors
    // ///
    // /// Fails if `centroids` is empty, non-finite, or of the wrong dimension, the index
    // /// already has centroids, or the maintenance accessor fails.
    // pub fn initialize<'a, S>(
    //     &'a mut self,
    //     strategy: &'a S,
    //     context: &'a P::Context,
    //     centroids: Matrix<f32>,
    // ) -> impl SendFuture<ANNResult<()>>
    // where
    //     S: MaintenanceStrategy<'a, P>,
    // {
    //     let provider = &mut self.provider;

    //     async move {
    //         let count = centroids.nrows();
    //         if count == 0 {
    //             return Err(index_error("initialize requires at least one centroid"));
    //         }
    //         if centroids.as_slice().iter().any(|x| !x.is_finite()) {
    //             return Err(index_error("initial centroids must be finite"));
    //         }

    //         let mut accessor = strategy
    //             .maintenance_accessor(provider, context)
    //             .into_ann_result()?;

    //         if !accessor.centroids().is_empty() {
    //             return Err(index_error("dynamic IVF index is already initialized"));
    //         }
    //         let dim = accessor.dim();
    //         if dim == 0 || centroids.ncols() != dim {
    //             return Err(index_error(format!(
    //                 "initial centroids have dimension {}, expected {dim}",
    //                 centroids.ncols()
    //             )));
    //         }

    //         let list_ids = accessor
    //             .reserve_list_ids(count)
    //             .await
    //             .escalate("initialize must reserve list ids")?;
    //         check_reserved(&accessor.centroids(), &list_ids, count)?;

    //         let installs = list_ids
    //             .into_iter()
    //             .zip(centroids.row_iter())
    //             .map(|(id, centroid)| CentroidDelta::Install {
    //                 id,
    //                 centroid: centroid.into(),
    //             })
    //             .collect();

    //         accessor
    //             .apply(InsertionUpdate::new(installs, Vec::new()))
    //             .await
    //             .escalate("initialize must apply the initial centroids")
    //     }
    // }

    /// Insert a batch, splitting every list the batch pushes past `split_threshold`.
    ///
    /// Routing and split planning run against the accessor's view before any change.
    /// The batch, and any splits it triggers, are then applied as one
    /// [`InsertionUpdate`].
    ///
    /// # Errors
    ///
    /// Fails if the index is not initialized, the provider returns inconsistent
    /// data, or any accessor call fails. State after an `apply` failure is
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
            let mut vectors = Matrix::<f32>::new(f32::NAN, points.len(), accessor.dim());
            let mut ids = Vec::with_capacity(points.len());
            let mut routed = hashbrown::HashMap::<P::ListId, Vec<usize>>::new();

            for (position, ((id, v), out)) in points.iter().zip(vectors.row_iter_mut()).enumerate()
            {
                //stage the input points first.
                let id = accessor
                    .stage_point(id, *v, out)
                    .await
                    .escalate("Unable to stage point")?;
                ids.push(id);

                // route the points to their nearest centroid
                let parent = accessor
                    .centroids()
                    .select(out, 1)
                    .escalate("Unable to select")?
                    .first()
                    .map(|s| s.id)
                    .ok_or_else(|| {
                        ANNError::message(ANNErrorKind::IndexError, "Didn't return list")
                    })?;

                routed.entry(parent).or_default().push(position);
            }

            // filter the routed lists for ones that need splitting.
            let mut parents = Vec::new();
            for (&list, positions) in &routed {
                let len = accessor.list_size(list).escalate("Unable to get len")?;

                if len + positions.len() > config.split_threshold {
                    parents.push(list);
                }
            }
            parents.sort_unstable();

            // for the ones that need splitting - split and re-assign necessary points in neighbors
            let update = Self::split::<_, T>(
                &mut accessor,
                &config,
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

    /// Build the update for one insert batch: split every list in `parents` and append
    /// every staged point to the list that ends up holding it.
    ///
    /// Batch position `i` is the staged point `ids[i]`, whose full-precision vector is
    /// row `i` of `vectors`. `routed` maps every routed list to the batch positions
    /// routed to it, and `parents` names the subset of those lists that overflow.
    ///
    /// # Errors
    ///
    /// Fails if an accessor call fails or a split list holds fewer than two points.
    async fn split<A, T>(
        accessor: &mut A,
        config: &Config,
        rng: &mut StdRng,
        ids: &[P::InternalId],
        vectors: MatrixView<'_, f32>,
        routed: &hashbrown::HashMap<P::ListId, Vec<usize>>,
        parents: &[P::ListId],
    ) -> ANNResult<Deltas<P::InternalId, P::ListId>>
    where
        A: InsertAccessor<P, T>,
        T: Sync,
    {
        let mut deltas = Vec::new();

        // Lists that keep their centroid take their staged points as they are.
        for (&list, positions) in routed {
            if parents.contains(&list) {
                continue;
            }
            deltas.push(Delta::PointAppends {
                to: list,
                ids: positions.iter().map(|&position| ids[position]).collect(),
            });
        }

        // Overflowing lists are replaced by two children, which take their staged
        // points instead.
        for &parent in parents {
            let staged: &[usize] = routed.get(&parent).map_or(&[], Vec::as_slice);
            deltas.extend(
                Self::split_list::<_, T>(accessor, config, rng, ids, vectors, parent, staged)
                    .await?,
            );
        }

        Ok(Deltas::new(deltas))
    }

    /// Plan one list's split: fit two children over every point the list would hold,
    /// then place each of those points on the nearer child and retire the parent.
    ///
    /// `staged` holds the batch positions routed to `parent`.
    ///
    /// # Errors
    ///
    /// Fails if an accessor call fails or the list holds fewer than two points.
    async fn split_list<A, T>(
        accessor: &mut A,
        config: &Config,
        rng: &mut StdRng,
        ids: &[P::InternalId],
        vectors: MatrixView<'_, f32>,
        parent: P::ListId,
        staged: &[usize],
    ) -> ANNResult<Vec<Delta<P::InternalId, P::ListId>>>
    where
        A: InsertAccessor<P, T>,
        T: Sync,
    {
        // let reader = accessor.reader();

        // let num_members = accessor.list_size(parent)?;

        // let mut points = Matrix::new(f32::NAN, num_members + staged,  reader.dim());

        // reader.read_into(parent, points.as_mut_view()).await?;
        let dim = accessor.dim();
        let members = accessor
            .get_members(parent)
            .escalate("split must read the parent's members")?
            .to_vec();

        // Everything the parent would hold: its current members, then its staged points.
        let mut points = Matrix::new(f32::NAN, members.len() + staged.len(), dim);
        {
            let mut member_vectors = Matrix::new(f32::NAN, members.len(), dim);
            accessor
                .reader()
                .read_into(parent, member_vectors.as_mut_view())
                .await
                .escalate("split must read the parent's vectors")?;

            let sources = member_vectors
                .row_iter()
                .chain(staged.iter().map(|&position| vectors.row(position)));
            for (row, source) in points.row_iter_mut().zip(sources) {
                row.copy_from_slice(source);
            }
        }

        let TwoMeans {
            centroids,
            children,
        } = two_means(points.as_view(), config.two_means_iterations, rng)?;

        let mut deltas = Vec::new();
        let mut child_ids = Vec::new();

        // Install both children before anything is placed into them.
        for centroid in centroids.row_iter() {
            let id = accessor.stage_centroid(centroid).await?;
            child_ids.push(id);

            deltas.push(Delta::CentroidDelta {
                id,
                delta: CentroidDelta::Install {
                    centroid: centroid.into(),
                },
            });
        }

        // Retire the parent, moving every member it held onto that member's child.
        let moves = members
            .iter()
            .zip(&children)
            .map(|(&id, &child)| MoveTo::new(id, child_ids[child]))
            .collect();

        deltas.push(Delta::CentroidDelta {
            id: parent,
            delta: CentroidDelta::Retire { moves },
        });

        // Append the staged points routed here onto their child.
        for (child, &list) in child_ids.iter().enumerate() {
            let appended: Box<[_]> = staged
                .iter()
                .zip(&children[members.len()..])
                .filter(|&(_, &group)| group == child)
                .map(|(&position, _)| ids[position])
                .collect();
            if !appended.is_empty() {
                deltas.push(Delta::PointAppends {
                    to: list,
                    ids: appended,
                });
            }
        }

        Ok(deltas)
    }
}

#[cfg(test)]
mod tests {}

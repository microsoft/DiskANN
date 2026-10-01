/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Dynamic IVF index orchestration.
//!
//! The index drives one [`MaintenanceAccessor`] per mutation: it plans the update
//! against the accessor's view and hands the result to [`Apply`]. It never touches
//! point, centroid, or list storage directly.

use diskann_utils::{future::SendFuture, views::Matrix};
use rand::{SeedableRng, rngs::StdRng};
use thiserror::Error;

use crate::{
    ANNError, ANNErrorKind, ANNResult,
    error::{ErrorExt, IntoANNResult},
    ivf::{
        dynamic::{Apply, CentroidIndex, ListProvider, MaintenanceAccessor, MaintenanceStrategy},
        online::{SplitPlan, StagedBatch, check_reserved, index_error},
        update::{Appends, Bootstrap, CentroidBlock, SplitInsert},
    },
};

/// Parameters for the online split policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynamicIvfConfig {
    /// Split a list once an insert batch would grow it beyond this many points.
    pub split_threshold: usize,
    /// Hard cap on live lists; `None` allows unbounded growth.
    pub max_clusters: Option<usize>,
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
pub struct DynamicIvfIndex<P: ListProvider> {
    provider: P,
    config: DynamicIvfConfig,
    rng: StdRng,
}

impl<P: ListProvider> DynamicIvfIndex<P> {
    /// Construct an index over `provider`.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if `config` is invalid.
    pub fn new(provider: P, config: DynamicIvfConfig) -> Result<Self, ConfigError> {
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
        centroids: Matrix<f32>,
    ) -> impl SendFuture<ANNResult<()>>
    where
        S: MaintenanceStrategy<'a, P, T>,
        S::MaintenanceAccessor: Apply<Bootstrap<P::ListId>>,
        T: Send,
    {
        let provider = &mut self.provider;

        async move {
            let count = centroids.nrows();
            if count == 0 {
                return Err(index_error("initialize requires at least one centroid"));
            }
            if centroids.as_slice().iter().any(|x| !x.is_finite()) {
                return Err(index_error("initial centroids must be finite"));
            }

            let mut accessor = strategy
                .maintenance_accessor(provider, context)
                .into_ann_result()?;

            if !accessor.centroids().is_empty() {
                return Err(index_error("dynamic IVF index is already initialized"));
            }
            let dim = accessor.dim();
            if dim == 0 || centroids.ncols() != dim {
                return Err(index_error(format!(
                    "initial centroids have dimension {}, expected {dim}",
                    centroids.ncols()
                )));
            }

            let list_ids = accessor
                .reserve_list_ids(count)
                .await
                .escalate("initialize must reserve list ids")?;
            check_reserved(accessor.centroids(), &list_ids, count)?;

            accessor
                .apply(Bootstrap::new(CentroidBlock::new(list_ids, centroids)))
                .await
                .escalate("initialize must apply the initial centroids")
        }
    }

    /// Insert a batch, splitting every list the batch pushes past `split_threshold`.
    ///
    /// Routing and split planning run against the accessor's view before any change.
    /// The batch is then applied as one [`Appends`], or as one [`SplitInsert`] when
    /// lists split.
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
        S::MaintenanceAccessor:
            Apply<Appends<P::InternalId, P::ListId>> + Apply<SplitInsert<P::InternalId, P::ListId>>,
        T: Copy + Send + Sync,
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

            if accessor.centroids().is_empty() {
                return Err(index_error(
                    "insert requires an initialized dynamic IVF index",
                ));
            }

            let batch = StagedBatch::stage(&mut accessor, points).await?;
            let parents = Self::select_parents(&config, &mut accessor, &batch).await?;

            if parents.is_empty() {
                let update = batch.into_appends();
                let inserted = update.len();
                accessor
                    .apply(update)
                    .await
                    .escalate("insert must apply routed points")?;
                return Ok(InsertStats {
                    inserted,
                    splits: 0,
                    reassigned: 0,
                });
            }

            let update = SplitPlan::gather(&mut accessor, &config, batch, parents)
                .await?
                .solve(config.two_means_iterations, rng)?;
            let stats = InsertStats {
                inserted: update.appends().len(),
                splits: update.splits().len(),
                reassigned: update.splits().num_evacuated() + update.reassignments().len(),
            };
            accessor
                .apply(update)
                .await
                .escalate("insert must apply the split update")?;
            Ok(stats)
        }
    }

    /// Choose the routed lists to split.
    ///
    /// Admits every list the batch pushes past `split_threshold`, largest projected
    /// size first, while the live-list count stays within `max_clusters`. Parents are
    /// returned in ascending list order.
    ///
    /// Takes the configuration rather than `&self` because the provider is lent to
    /// `accessor` for the whole mutation.
    ///
    /// # Errors
    ///
    /// Fails if the accessor fails.
    async fn select_parents<A, T>(
        config: &DynamicIvfConfig,
        accessor: &mut A,
        batch: &StagedBatch<A::Id, A::ListId>,
    ) -> ANNResult<Vec<A::ListId>>
    where
        A: MaintenanceAccessor<T>,
        T: Send,
    {
        let mut overflowing = Vec::new();
        for (list, incoming) in batch.routed() {
            let len = accessor
                .list_metadata(list)
                .await
                .escalate("insert must read routed list sizes")?
                .len;
            let projected = len + incoming;
            if projected > config.split_threshold {
                overflowing.push((projected, list));
            }
        }

        // Each split adds one live list.
        let budget = config.max_clusters.map_or(usize::MAX, |max| {
            max.saturating_sub(accessor.centroids().len())
        });
        overflowing.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        overflowing.truncate(budget);

        let mut parents: Vec<_> = overflowing.into_iter().map(|(_, list)| list).collect();
        parents.sort_unstable();
        Ok(parents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ivf::test::provider::{Faults, Provider, Strategy},
        provider::DefaultContext,
    };

    fn config(split_threshold: usize, reassign_neighbors: usize) -> DynamicIvfConfig {
        DynamicIvfConfig {
            split_threshold,
            max_clusters: None,
            reassign_neighbors,
            two_means_iterations: 10,
            seed: 7,
        }
    }

    /// Pair each vector with a fresh external id, counting up from `first`.
    fn batch<const D: usize>(first: u32, vectors: &[[f32; D]]) -> Vec<(u32, &[f32])> {
        (first..)
            .zip(vectors.iter().map(|vector| &vector[..]))
            .collect()
    }

    fn matrix<const D: usize>(rows: &[[f32; D]]) -> Matrix<f32> {
        let data: Box<[f32]> = rows.iter().flatten().copied().collect();
        Matrix::try_from(data, rows.len(), D).unwrap()
    }

    async fn initialized(
        provider: Provider,
        config: DynamicIvfConfig,
        centroids: Matrix<f32>,
    ) -> DynamicIvfIndex<Provider> {
        let mut index = DynamicIvfIndex::new(provider, config).unwrap();
        index
            .initialize::<_, &[f32]>(&Strategy, &DefaultContext, centroids)
            .await
            .unwrap();
        index
    }

    #[test]
    fn new_rejects_invalid_config() {
        let new = |config| DynamicIvfIndex::new(Provider::new(1), config).err();
        assert_eq!(new(config(1, 1)), Some(ConfigError::SplitThreshold(1)));
        assert_eq!(new(config(4, 0)), Some(ConfigError::ReassignNeighbors));
    }

    #[tokio::test]
    async fn initialize_rejects_bad_centroids() {
        let mut index = DynamicIvfIndex::new(Provider::new(2), config(10, 1)).unwrap();
        let wrong_dim = index
            .initialize::<_, &[f32]>(&Strategy, &DefaultContext, matrix(&[[0.0, 0.0, 0.0]]))
            .await
            .unwrap_err();
        assert!(wrong_dim.to_string().contains("dimension"), "{wrong_dim}");
        let non_finite = index
            .initialize::<_, &[f32]>(&Strategy, &DefaultContext, matrix(&[[f32::NAN, 0.0]]))
            .await
            .unwrap_err();
        assert!(non_finite.to_string().contains("finite"), "{non_finite}");

        index
            .initialize::<_, &[f32]>(&Strategy, &DefaultContext, matrix(&[[0.0, 0.0]]))
            .await
            .unwrap();
        let again = index
            .initialize::<_, &[f32]>(&Strategy, &DefaultContext, matrix(&[[1.0, 1.0]]))
            .await
            .unwrap_err();
        assert!(again.to_string().contains("already initialized"), "{again}");
    }

    #[tokio::test]
    async fn insert_requires_initialization() {
        let mut index = DynamicIvfIndex::new(Provider::new(1), config(4, 1)).unwrap();
        let empty = index
            .insert_batch(&Strategy, &DefaultContext, &batch::<1>(0, &[]))
            .await
            .unwrap();
        assert_eq!(empty, InsertStats::default());
        let err = index
            .insert_batch(&Strategy, &DefaultContext, &batch(0, &[[1.0]]))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("initialized"), "{err}");
    }

    #[tokio::test]
    async fn insert_without_overflow_appends_to_nearest_lists() {
        let centroids = matrix(&[[0.0, 0.0], [10.0, 10.0]]);
        let mut index = initialized(Provider::new(2), config(10, 1), centroids).await;
        let vectors = [[1.0, 1.0], [9.0, 9.0], [2.0, 0.0]];
        let stats = index
            .insert_batch(&Strategy, &DefaultContext, &batch(0, &vectors))
            .await
            .unwrap();
        assert_eq!(
            stats,
            InsertStats {
                inserted: 3,
                splits: 0,
                reassigned: 0
            }
        );

        let provider = index.provider();
        provider.check();
        assert_eq!(
            provider.member_vectors(0),
            vec![vec![1.0, 1.0], vec![2.0, 0.0]]
        );
        assert_eq!(provider.member_vectors(1), vec![vec![9.0, 9.0]]);
    }

    #[tokio::test]
    async fn max_clusters_caps_splits() {
        let mut capped = config(2, 1);
        capped.max_clusters = Some(2);
        let mut index = initialized(Provider::new(1), capped, matrix(&[[0.0], [10.0]])).await;
        let stats = index
            .insert_batch(
                &Strategy,
                &DefaultContext,
                &batch(0, &[[1.0], [2.0], [3.0]]),
            )
            .await
            .unwrap();
        assert_eq!(stats.splits, 0);
        assert_eq!(index.provider().lists(), vec![0, 1]);
        index.provider().check();
    }

    /// Lists A (0), B (20), and C (40) in one dimension. Inserting 0.5, 8, and 9
    /// overflows A, which splits into children near 1/6 and 8.5, with B and C as
    /// its neighbors.
    #[tokio::test]
    async fn split_places_parent_points_and_moves_neighbor_points_only_toward_children() {
        let provider = Provider::from_lists(
            1,
            &[
                (&[0.0], &[&[-1.0], &[1.0]]),
                (&[20.0], &[&[12.0], &[25.0], &[33.0]]),
                (&[40.0], &[&[41.0]]),
            ],
        );
        let mut index = DynamicIvfIndex::new(provider, config(4, 2)).unwrap();
        let stats = index
            .insert_batch(
                &Strategy,
                &DefaultContext,
                &batch(100, &[[0.5], [8.0], [9.0]]),
            )
            .await
            .unwrap();
        // A's members -1 and 1 are evacuated, and 12 moves out of B.
        assert_eq!(
            stats,
            InsertStats {
                inserted: 3,
                splits: 1,
                reassigned: 3
            }
        );

        let provider = index.provider();
        provider.check();
        assert_eq!(provider.lists(), vec![1, 2, 3, 4]);
        let low = provider.list_of(&[0.5]).unwrap();
        let high = provider.list_of(&[8.0]).unwrap();
        assert!((provider.centroid(low).unwrap()[0] - 1.0 / 6.0).abs() < 1e-6);
        assert_eq!(provider.centroid(high).unwrap(), &[8.5]);
        assert_eq!(
            provider.member_vectors(low),
            vec![vec![-1.0], vec![0.5], vec![1.0]]
        );
        assert_eq!(
            provider.member_vectors(high),
            vec![vec![8.0], vec![9.0], vec![12.0]]
        );
        // 33 is closer to C than to B, but neighbor points only move toward children.
        assert_eq!(provider.member_vectors(1), vec![vec![25.0], vec![33.0]]);
        assert_eq!(provider.member_vectors(2), vec![vec![41.0]]);
    }

    #[tokio::test]
    async fn repeated_batches_keep_the_partition_valid() {
        let centroids = matrix(&[[0.0, 0.0], [10.0, 10.0]]);
        let mut index = initialized(Provider::new(2), config(8, 2), centroids).await;
        let vectors: Vec<[f32; 2]> = (0..300)
            .map(|i| {
                let t = i as f32;
                [(t * 0.37).sin() * 20.0, (t * 0.11).cos() * 20.0]
            })
            .collect();

        // Insert in batches of 25, checking the partition after each batch.
        let mut stats = InsertStats::default();
        for (first, chunk) in (0..).step_by(25).zip(vectors.chunks(25)) {
            let batch = index
                .insert_batch(&Strategy, &DefaultContext, &batch(first, chunk))
                .await
                .unwrap();
            index.provider().check();
            stats.inserted += batch.inserted;
            stats.splits += batch.splits;
            stats.reassigned += batch.reassigned;
        }

        let provider = index.provider();
        assert_eq!(stats.inserted, 300);
        assert_eq!(provider.len(), 300);
        assert!(stats.splits > 1 && stats.reassigned > 0, "{stats:?}");
        assert_eq!(provider.lists().len(), 2 + stats.splits);
        assert_eq!(provider.num_retired(), stats.splits);
    }

    #[tokio::test]
    async fn inconsistent_provider_responses_fail_before_any_change() {
        let fault = |set: fn(&mut Faults)| {
            let mut faults = Faults::default();
            set(&mut faults);
            faults
        };
        let cases = [
            (
                fault(|f| f.skip_vector = true),
                "no finite canonical vector",
            ),
            (fault(|f| f.nan_vector = true), "no finite canonical vector"),
            (fault(|f| f.repeat_stage = true), "staging returned point"),
            (fault(|f| f.reuse_lists = true), "reserved live list id"),
        ];
        for (faults, expected) in cases {
            // Inserting 0.5 and 8 overflows list 0, so every case reaches the split path.
            let provider =
                Provider::from_lists(1, &[(&[0.0], &[&[-1.0], &[1.0]]), (&[20.0], &[&[12.0]])])
                    .with_faults(faults);
            let before = provider.clone();
            let mut index = DynamicIvfIndex::new(provider, config(3, 1)).unwrap();
            let err = index
                .insert_batch(&Strategy, &DefaultContext, &batch(100, &[[0.5], [8.0]]))
                .await
                .unwrap_err();
            assert!(err.to_string().contains(expected), "{faults:?}: {err}");
            assert!(index.provider().same_contents(&before), "{faults:?}");
        }
    }
}

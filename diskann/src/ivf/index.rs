/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Dynamic IVF index orchestration.
//!
//! The index owns split policy and drives one [`MaintenanceAccessor`] per mutation;
//! it never touches point, centroid, or list storage directly.

use std::{
    collections::{BTreeMap, BTreeSet, btree_map::Entry},
    fmt::{Debug, Display},
};

use diskann_utils::{future::SendFuture, views::MatrixView};
use rand::{Rng, SeedableRng, rngs::StdRng};
use thiserror::Error;

use crate::{
    ANNError, ANNErrorKind, ANNResult,
    error::{ErrorExt, IntoANNResult},
    ivf::dynamic::{
        CentroidDelta, CentroidIndex, CentroidRecord, MaintenanceAccessor, MaintenanceStrategy,
        PartitionUpdate, PointMove,
    },
    provider::DataProvider,
    utils::VectorId,
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
/// Mutations take `&mut self`, so they cannot overlap searches through the same value.
#[derive(Debug)]
pub struct DynamicIvfIndex<P: DataProvider> {
    provider: P,
    config: DynamicIvfConfig,
    rng: StdRng,
}

impl<P: DataProvider> DynamicIvfIndex<P> {
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

    /// Borrow the split policy.
    pub fn config(&self) -> &DynamicIvfConfig {
        &self.config
    }

    /// Install the initial centroids.
    ///
    /// # Errors
    ///
    /// Fails if `centroids` is empty, the index already has centroids, or the
    /// maintenance accessor fails.
    pub fn initialize<'a, S, T>(
        &'a mut self,
        strategy: &'a S,
        context: &'a P::Context,
        centroids: MatrixView<'a, f32>,
    ) -> impl SendFuture<ANNResult<()>>
    where
        S: MaintenanceStrategy<'a, P, T>,
        T: Send,
    {
        let provider = &self.provider;
        async move {
            let count = centroids.nrows();
            if count == 0 {
                return Err(index_error("initialize requires at least one centroid"));
            }

            let mut accessor = strategy
                .maintenance_accessor(provider, context)
                .into_ann_result()?;

            if !accessor.centroids().is_empty() {
                return Err(index_error("dynamic IVF index is already initialized"));
            }

            let list_ids = accessor
                .reserve_list_ids(count)
                .await
                .escalate("initialize must reserve list ids")?;
            check_reserved(count, list_ids.len())?;

            let centroids = CentroidDelta {
                insert: list_ids
                    .into_iter()
                    .zip(centroids.row_iter())
                    .map(|(id, vector)| CentroidRecord {
                        id,
                        vector: vector.into(),
                    })
                    .collect(),
                retire: Vec::new(),
            };

            accessor
                .apply(PartitionUpdate {
                    centroids,
                    ..Default::default()
                })
                .await
                .escalate("initialize must apply the initial centroids")
        }
    }

    /// Insert a batch, splitting every list the batch pushes past `split_threshold`.
    ///
    /// Routing and split planning run against the accessor's view before any change;
    /// the whole batch is then handed to [`MaintenanceAccessor::apply`] as one update.
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
        T: Copy + Send + Sync,
    {
        let provider = &self.provider;
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

            let (new_ids, mut vectors) = stage_points(&mut accessor, points).await?;

            let mut routed: BTreeMap<_, Vec<_>> = BTreeMap::new();

            for &id in &new_ids {
                let plan = accessor
                    .centroids()
                    .select(vectors.get(id)?, 1)
                    .await
                    .escalate("insert must route every point")?;
                let list = plan
                    .selected()
                    .first()
                    .map(|selected| selected.id)
                    .ok_or_else(|| index_error("centroid selection returned no list"))?;
                routed.entry(list).or_default().push(id);
            }

            let parents = select_split_parents(&mut accessor, &config, &routed).await?;

            if parents.is_empty() {
                let update = PartitionUpdate {
                    point_moves: routed_moves(&routed, &BTreeMap::new()),
                    ..Default::default()
                };
                accessor
                    .apply(update)
                    .await
                    .escalate("insert must apply routed points")?;

                return Ok(InsertStats {
                    inserted: new_ids.len(),
                    ..Default::default()
                });
            }

            let plan = SplitPlan::prepare(&mut accessor, &config, &parents, &mut vectors).await?;
            let children =
                plan.fit_children(&routed, &vectors, config.two_means_iterations, rng)?;

            let child_ids = accessor
                .reserve_list_ids(children.len())
                .await
                .escalate("split must reserve child list ids")?;
            check_reserved(children.len(), child_ids.len())?;

            let placements = plan.reassign(&routed, &vectors, &child_ids, &children)?;

            let mut point_moves = routed_moves(&routed, &placements);
            let mut reassigned = 0;
            for (&from, ids) in &plan.members {
                for &id in ids {
                    if let Some(&to) = placements.get(&id)
                        && to != from
                    {
                        reassigned += 1;
                        point_moves.push(PointMove {
                            id,
                            from: Some(from),
                            to: Some(to),
                        });
                    }
                }
            }

            let splits = parents.len();
            let update = PartitionUpdate {
                centroids: CentroidDelta {
                    insert: child_ids
                        .iter()
                        .zip(children)
                        .map(|(&id, vector)| CentroidRecord { id, vector })
                        .collect(),
                    retire: parents,
                },
                point_moves,
                ..Default::default()
            };

            accessor
                .apply(update)
                .await
                .escalate("insert must apply the split update")?;

            Ok(InsertStats {
                inserted: new_ids.len(),
                splits,
                reassigned,
            })
        }
    }
}

///////////////////
// Insert phases //
///////////////////

/// Canonical vectors gathered for one mutation, all of dimension `dim`.
struct Vectors<Id> {
    dim: usize,
    rows: BTreeMap<Id, Box<[f32]>>,
}

impl<Id: VectorId> Vectors<Id> {
    fn get(&self, id: Id) -> ANNResult<&[f32]> {
        self.rows
            .get(&id)
            .map(|row| &**row)
            .ok_or_else(|| index_error(format!("no canonical vector for point {id}")))
    }

    fn check_dim(&self, len: usize) -> ANNResult<()> {
        if len == self.dim {
            Ok(())
        } else {
            Err(index_error(format!(
                "vector dimension {len} does not match batch dimension {}",
                self.dim
            )))
        }
    }

    fn check_dims(&self) -> ANNResult<()> {
        self.rows
            .values()
            .try_for_each(|row| self.check_dim(row.len()))
    }
}

/// Stage every point and materialize its canonical vector.
async fn stage_points<A, T>(
    accessor: &mut A,
    points: &[(A::ExternalId, T)],
) -> ANNResult<(Vec<A::Id>, Vectors<A::Id>)>
where
    A: MaintenanceAccessor<T>,
    T: Copy + Send + Sync,
{
    let mut ids = Vec::with_capacity(points.len());
    for (external, element) in points {
        ids.push(
            accessor
                .stage_insert(external, *element)
                .await
                .escalate("insert must stage every point")?,
        );
    }

    let mut rows: BTreeMap<A::Id, Box<[f32]>> = BTreeMap::new();
    accessor
        .read_staged_vectors(ids.iter().copied(), |id, vector| {
            rows.insert(id, Box::from(vector));
        })
        .await
        .escalate("insert must read staged canonical vectors")?;

    let dim = rows.values().next().map_or(0, |row| row.len());
    if dim == 0 {
        return Err(index_error("staged canonical vectors must be non-empty"));
    }
    let vectors = Vectors { dim, rows };
    vectors.check_dims()?;
    Ok((ids, vectors))
}

/// Route each staged point to its nearest live list.
async fn route<A, T>(
    accessor: &mut A,
    ids: &[A::Id],
    vectors: &Vectors<A::Id>,
) -> ANNResult<BTreeMap<A::ListId, Vec<A::Id>>>
where
    A: MaintenanceAccessor<T>,
    T: Send,
{
    let centroids = accessor.centroids();
    let mut routed: BTreeMap<A::ListId, Vec<A::Id>> = BTreeMap::new();

    for &id in ids {
        let plan = centroids
            .select(vectors.get(id)?, 1)
            .await
            .escalate("insert must route every point")?;
        let list = plan
            .selected()
            .first()
            .map(|selected| selected.id)
            .ok_or_else(|| index_error("centroid selection returned no list"))?;
        routed.entry(list).or_default().push(id);
    }

    Ok(routed)
}

/// Return routed lists whose projected size exceeds the threshold, largest first
/// under the cluster cap, in ascending id order.
async fn select_split_parents<A, T>(
    accessor: &mut A,
    config: &DynamicIvfConfig,
    routed: &BTreeMap<A::ListId, Vec<A::Id>>,
) -> ANNResult<Vec<A::ListId>>
where
    A: MaintenanceAccessor<T>,
    T: Send,
{
    let mut sizes = BTreeMap::new();
    accessor
        .list_metadata(routed.keys().copied(), |metadata| {
            sizes.insert(metadata.id, metadata.len);
        })
        .await
        .escalate("insert must read routed list sizes")?;

    let mut overflowing = Vec::new();
    for (&list, incoming) in routed {
        let current = sizes
            .get(&list)
            .copied()
            .ok_or_else(|| index_error(format!("no metadata for list {list}")))?;
        let projected = current + incoming.len();
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

/// One split parent and the surviving lists reassigned alongside it.
struct Region<L> {
    parent: L,
    neighbors: Vec<L>,
}

/// Everything read from the accessor to split a set of parents.
struct SplitPlan<Id, L> {
    regions: Vec<Region<L>>,
    neighbor_centroids: BTreeMap<L, Box<[f32]>>,
    /// Current members of every parent and neighbor list.
    members: BTreeMap<L, Vec<Id>>,
}

impl<Id: VectorId, L: VectorId> SplitPlan<Id, L> {
    async fn prepare<A, T>(
        accessor: &mut A,
        config: &DynamicIvfConfig,
        parents: &[L],
        vectors: &mut Vectors<Id>,
    ) -> ANNResult<Self>
    where
        A: MaintenanceAccessor<T, Id = Id, ListId = L>,
        T: Send,
    {
        let parent_set: BTreeSet<L> = parents.iter().copied().collect();
        let centroids = accessor.centroids();
        let mut regions = Vec::with_capacity(parents.len());
        let mut neighbor_centroids = BTreeMap::new();

        for &parent in parents {
            let anchor = centroids
                .centroid(parent)
                .ok_or_else(|| index_error(format!("split parent {parent} is not live")))?;
            let selected = centroids
                .select(anchor, config.reassign_neighbors + 1)
                .await
                .escalate("split must find neighbor lists")?;

            // Other parents are retired by this update, so they are never candidates.
            let neighbors: Vec<L> = selected
                .into_selected()
                .into_iter()
                .map(|selected| selected.id)
                .filter(|id| !parent_set.contains(id))
                .take(config.reassign_neighbors)
                .collect();

            for &neighbor in &neighbors {
                if let Entry::Vacant(entry) = neighbor_centroids.entry(neighbor) {
                    let vector = centroids.centroid(neighbor).ok_or_else(|| {
                        index_error(format!("neighbor list {neighbor} is not live"))
                    })?;
                    vectors.check_dim(vector.len())?;
                    entry.insert(Box::from(vector));
                }
            }
            regions.push(Region { parent, neighbors });
        }

        let lists: BTreeSet<L> = regions
            .iter()
            .flat_map(|region| {
                std::iter::once(region.parent).chain(region.neighbors.iter().copied())
            })
            .collect();

        let mut members: BTreeMap<L, Vec<Id>> = BTreeMap::new();
        accessor
            .read_members(lists.iter().copied(), |list, ids| {
                members.entry(list).or_default().extend_from_slice(ids)
            })
            .await
            .escalate("split must read region members")?;

        accessor
            .read_vectors(members.values().flatten().copied(), |id, vector| {
                vectors.rows.insert(id, Box::from(vector));
            })
            .await
            .escalate("split must read region canonical vectors")?;
        vectors.check_dims()?;

        Ok(Self {
            regions,
            neighbor_centroids,
            members,
        })
    }

    /// Existing members plus incoming points of `list`.
    fn points_of<'s>(
        &'s self,
        list: L,
        routed: &'s BTreeMap<L, Vec<Id>>,
    ) -> impl Iterator<Item = Id> + 's {
        self.members
            .get(&list)
            .into_iter()
            .chain(routed.get(&list))
            .flatten()
            .copied()
    }

    /// Fit two children per parent with one joint k-means over every parent's points.
    fn fit_children(
        &self,
        routed: &BTreeMap<L, Vec<Id>>,
        vectors: &Vectors<Id>,
        iterations: usize,
        rng: &mut StdRng,
    ) -> ANNResult<Vec<Box<[f32]>>> {
        let mut points = Vec::new();
        let mut centers = Vec::with_capacity(2 * self.regions.len());
        for region in &self.regions {
            let start = points.len();
            for id in self.points_of(region.parent, routed) {
                points.push(vectors.get(id)?);
            }

            let span = points.len() - start;
            if span < 2 {
                return Err(index_error(format!(
                    "split parent {} has fewer than two points",
                    region.parent
                )));
            }
            let a = rng.random_range(0..span);
            let b = (a + rng.random_range(1..span)) % span;
            centers.push(points[start + a].to_vec());
            centers.push(points[start + b].to_vec());
        }

        lloyd(&points, &mut centers, iterations.max(1));
        Ok(centers.into_iter().map(Vec::into_boxed_slice).collect())
    }

    /// Place every region point on its nearest candidate across all regions containing it.
    fn reassign(
        &self,
        routed: &BTreeMap<L, Vec<Id>>,
        vectors: &Vectors<Id>,
        child_ids: &[L],
        children: &[Box<[f32]>],
    ) -> ANNResult<BTreeMap<Id, L>> {
        let mut best: BTreeMap<Id, (L, f32)> = BTreeMap::new();
        for ((region, ids), vecs) in self
            .regions
            .iter()
            .zip(child_ids.chunks_exact(2))
            .zip(children.chunks_exact(2))
        {
            let mut candidates: Vec<(L, &[f32])> = Vec::with_capacity(region.neighbors.len() + 2);
            for &neighbor in &region.neighbors {
                let vector = self.neighbor_centroids.get(&neighbor).ok_or_else(|| {
                    index_error(format!("missing centroid for neighbor list {neighbor}"))
                })?;
                candidates.push((neighbor, vector));
            }
            candidates.extend(ids.iter().copied().zip(vecs.iter().map(|v| &**v)));

            let lists = std::iter::once(region.parent).chain(region.neighbors.iter().copied());
            for list in lists {
                for id in self.points_of(list, routed) {
                    let (to, distance) = nearest(vectors.get(id)?, &candidates)
                        .ok_or_else(|| index_error("split region has no candidate lists"))?;
                    match best.entry(id) {
                        Entry::Vacant(entry) => {
                            entry.insert((to, distance));
                        }
                        Entry::Occupied(mut entry) => {
                            if distance < entry.get().1 {
                                entry.insert((to, distance));
                            }
                        }
                    }
                }
            }
        }
        Ok(best.into_iter().map(|(id, (list, _))| (id, list)).collect())
    }
}

/// Moves for newly inserted points: their split placement, else their route.
fn routed_moves<Id: VectorId, L: VectorId>(
    routed: &BTreeMap<L, Vec<Id>>,
    placements: &BTreeMap<Id, L>,
) -> Vec<PointMove<Id, L>> {
    routed
        .iter()
        .flat_map(|(&route, ids)| {
            ids.iter().map(move |&id| PointMove {
                id,
                from: None,
                to: Some(placements.get(&id).copied().unwrap_or(route)),
            })
        })
        .collect()
}

/////////////
// Helpers //
/////////////

// Clustering and reassignment use squared L2 regardless of the search metric.
fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn nearest<L: Copy>(point: &[f32], candidates: &[(L, &[f32])]) -> Option<(L, f32)> {
    candidates
        .iter()
        .map(|&(list, centroid)| (list, squared_l2(point, centroid)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
}

fn lloyd(points: &[&[f32]], centers: &mut [Vec<f32>], iterations: usize) {
    let dim = centers.first().map_or(0, Vec::len);
    if dim == 0 {
        return;
    }
    let mut sums = vec![0.0f32; centers.len() * dim];
    let mut counts = vec![0usize; centers.len()];
    for _ in 0..iterations {
        sums.fill(0.0);
        counts.fill(0);
        for &point in points {
            let Some(c) = centers
                .iter()
                .map(|center| squared_l2(point, center))
                .enumerate()
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(c, _)| c)
            else {
                continue;
            };
            counts[c] += 1;
            for (sum, x) in sums[c * dim..(c + 1) * dim].iter_mut().zip(point) {
                *sum += x;
            }
        }
        // Empty clusters keep their previous center.
        for ((center, sum), &count) in centers.iter_mut().zip(sums.chunks_exact(dim)).zip(&counts) {
            if count > 0 {
                let scale = 1.0 / count as f32;
                for (x, s) in center.iter_mut().zip(sum) {
                    *x = s * scale;
                }
            }
        }
    }
}

#[track_caller]
fn index_error<D>(message: D) -> ANNError
where
    D: Display + Debug + Send + Sync + 'static,
{
    ANNError::message(ANNErrorKind::IndexError, message)
}

#[track_caller]
fn check_reserved(expected: usize, actual: usize) -> ANNResult<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(index_error(format!(
            "accessor reserved {actual} ids, expected {expected}"
        )))
    }
}

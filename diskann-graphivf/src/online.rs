/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Online graph-IVF clustering with split, dissolve, and reassignment.
//!
//! [`OnlineClusterer`] builds the IVF partition incrementally instead of in a
//! single batch Lloyd pass. Points are routed to their nearest centroid via a
//! mutable centroid graph; when a cluster grows past a threshold it is split,
//! and the points of the split cluster together with the points of its
//! graph-neighboring clusters are reassigned among the new and neighboring
//! centroids.
//!
//! There is one insert path, [`insert_batch`](OnlineClusterer::insert_batch):
//! validate and route the batch, select overflows from projected sizes, prepare
//! every split against unchanged state, then commit the inserts and reassign
//! each split region. A batch large enough to be worth the dispatch routes
//! across the thread pool; reassignment is always a GEMM.
//!
//! Points can also be removed with
//! [`delete_batch`](OnlineClusterer::delete_batch): select underflows from
//! projected post-delete sizes and prepare survivor candidates first, then drop
//! the deleted points and retire the selected clusters. Retiring *dissolves* a
//! cluster — it leaves the centroid graph and its remaining members are
//! scattered onto preselected survivors by the same GEMM the split path uses —
//! so a split is `+1` live cluster and a merge is `-1`. Splits are insert-driven
//! and merges are delete-driven, and neither triggers the other, so the two
//! cannot cascade.
//!
//! Ordinary fallible work is completed before mutation. Structural changes go
//! through a private registry that owns both the centroid table and graph, and
//! point moves go through a partition that owns both inverted lists and reverse
//! assignments. A commit can still fail after an irreversible graph operation;
//! that poisons the clusterer and all later mutation, search, and flush requests
//! are rejected.
//!
//! The whole IVF mapping (inverted lists and the point→centroid assignment) is
//! kept in memory and mutated in place; [`OnlineClusterer::flush`] serializes it
//! once, at the end, into the same on-disk format a batch build produces, so the
//! result loads and searches through the unchanged
//! [`GraphIvfIndex`](crate::GraphIvfIndex) path.
//!
//! Points are preloaded as an `f32` matrix and "streamed" by feeding their row
//! indices to [`insert_batch`](OnlineClusterer::insert_batch); this keeps the
//! experiment free of disk I/O while still exercising the incremental build
//! logic. For a normalizing metric (cosine) the caller must pre-normalize the
//! points.

use std::path::Path;
use std::time::Instant;

use diskann_disk::utils::compute_closest_centers;
use diskann_providers::utils::{create_thread_pool, ParallelIteratorInPool, RayonThreadPool};
use diskann_utils::views::{Matrix, MatrixView};
use diskann_vector::distance::Metric as VectorMetric;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use tokio::runtime::Runtime;

use crate::{
    centroids::{self, AdjacencyCensus},
    cluster::{self, sq_l2},
    index::{with_suffix, CENTROIDS_SUFFIX, GRAPH_SUFFIX, LISTS_SUFFIX, META_SUFFIX},
    params::{EmptyClusterPolicy, OnlineCentroidRouting, OnlineParams},
    storage::{self, Layout},
    GraphIvfError, Result,
};

use diskann::{utils::VectorRepr, ANNError};

mod search;
mod seed;
mod state;
mod telemetry;

use state::{CentroidRegistry, DetachedPoint, IvfPartition};

pub use search::{CentroidRecall, OnlineSearchStats, OnlineSearcher};
pub use seed::SeedStrategy;
pub use telemetry::{BuildTelemetry, MergeEvent, SplitEvent};

/// Sentinel in the partition's reverse map for a point that is not currently
/// in the index — either never inserted, deleted, or temporarily detached
/// during reassignment. Shared with the list writer, which skips rows carrying
/// it.
const UNASSIGNED: u32 = storage::NOT_INDEXED;

/// Points routed per parallel work unit in [`OnlineClusterer::insert_batch`].
const ROUTE_CHUNK: usize = 256;

/// Maximum points gathered into one contiguous tile for a GEMM reassignment
/// call. Reassigning a whole split region at once would need `|P| * dim` floats
/// (gigabytes for a large `reassign_neighbors`), so points are streamed through
/// a tile that bounds the scratch to `REASSIGN_TILE * dim` floats.
const REASSIGN_TILE: usize = 4096;

/// Route one point to its nearest live centroid via the centroid graph.
///
/// The centroid graph is mutated in place as clusters split and dissolve:
/// retired slots are recycled, and the in-edges repaired around a departing
/// centroid can leave a region thinly connected. A narrow beam can then
/// occasionally exhaust its frontier without reaching a live centroid, so the
/// search is retried with the wider beam before giving up and, as a last
/// resort, falls back to a brute-force scan over the live centroids.
/// Successful narrow-beam routes are unchanged.
///
/// `beams` is `None` under exact routing, where there is no graph to walk and
/// the scan below is the primary path rather than a fallback.
fn route_one(
    centroids: &CentroidRegistry,
    runtime: &Runtime,
    point: &[f32],
    beams: Option<(usize, usize)>,
) -> Result<u32> {
    let mut ids = [0u32; 1];
    let mut dist = [0.0f32; 1];
    if let Some((base_l, wide_l)) = beams {
        for l in [base_l, wide_l] {
            if centroids.search(runtime, point, l, &mut ids, &mut dist)? > 0 {
                return Ok(ids[0]);
            }
        }
    }
    if centroids.exact_search(point, &mut ids, &mut dist)? > 0 {
        return Ok(ids[0]);
    }
    Err(GraphIvfError::invalid(
        "no live centroid available for assignment",
    ))
}

#[derive(Default)]
struct NeighborScratch {
    ids: Vec<u32>,
    distances: Vec<f32>,
}

#[derive(Default)]
struct ReassignScratch {
    centroid_vectors: Vec<f32>,
    point_tile: Vec<f32>,
    best: Vec<u32>,
}

#[derive(Default)]
struct MaintenanceScratch {
    points: Vec<DetachedPoint>,
    candidates: Vec<u32>,
    neighbors: NeighborScratch,
    reassign: ReassignScratch,
}

struct SplitParentPlan {
    id: u32,
    members: Vec<u32>,
    neighbors: Vec<u32>,
    children: [Box<[f32]>; 2],
    two_means_us: u64,
}

struct SplitPlan {
    parents: Vec<SplitParentPlan>,
    rng_after: StdRng,
    started: Instant,
}

struct MergeVictimPlan {
    id: u32,
    members: Vec<u32>,
    candidates: Vec<u32>,
    search_us: u64,
}

struct MergePlan {
    victims: Vec<MergeVictimPlan>,
    started: Instant,
}

/// An incremental graph-IVF clusterer with insert-driven splits, delete-driven
/// dissolves, and live in-memory search.
pub struct OnlineClusterer {
    /// The full corpus, preloaded; row `pid` is point `pid`.
    points: Matrix<f32>,
    dim: usize,
    params: OnlineParams,

    /// Mirrored vector table and mutable navigation graph.
    centroids: CentroidRegistry,
    /// The point↔centroid mapping (inverted lists plus the reverse map).
    partition: IvfPartition,

    /// Current-thread runtime driving the graph search/insert/delete calls.
    runtime: Runtime,
    /// Thread pool for routing and split k-means.
    pool: RayonThreadPool,
    rng: StdRng,

    /// Build telemetry (routing, splits, deletes, merges, and latencies).
    telemetry: BuildTelemetry,

    /// Reusable buffers for neighborhood search and point reassignment.
    scratch: MaintenanceScratch,

    /// Set before the first in-place step of a mutation and cleared only after
    /// the complete operation succeeds. Graph retirement cannot be rolled
    /// back, so a failed commit permanently disables further use.
    poisoned: Option<String>,
}

impl OnlineClusterer {
    /// Create a clusterer over `points`, obtaining the initial centroids from
    /// `seed` (see [`SeedStrategy`]). This is the ergonomic entry point;
    /// experiments typically pass [`SeedStrategy::Warmup`].
    ///
    /// # Errors
    ///
    /// Returns an error if `num_threads` is zero, the seed strategy fails, or
    /// the resulting centroids violate the invariants checked by [`new`].
    ///
    /// [`new`]: Self::new
    pub fn with_seed(
        points: Matrix<f32>,
        seed: SeedStrategy,
        params: OnlineParams,
    ) -> Result<Self> {
        if params.num_threads == 0 {
            return Err(GraphIvfError::invalid("num_threads must be non-zero"));
        }
        let initial = seed.resolve(points.as_view(), &params)?;
        Self::new(points, initial, params)
    }

    /// Create a clusterer over `points`, seeded with an explicit `initial`
    /// centroid matrix (one row each). Streaming begins with zero points
    /// assigned.
    ///
    /// This is the low-level primitive; [`with_seed`] wraps it with pluggable
    /// centroid bootstrapping.
    ///
    /// # Errors
    ///
    /// Returns an error if the shapes are inconsistent, `centroid_capacity` is
    /// smaller than the initial centroid count, or `split_threshold < 2`.
    ///
    /// [`with_seed`]: Self::with_seed
    pub fn new(points: Matrix<f32>, initial: Matrix<f32>, params: OnlineParams) -> Result<Self> {
        let dim = points.ncols();
        let num_points = points.nrows();
        let initial_k = initial.nrows();

        if dim == 0 || num_points == 0 {
            return Err(GraphIvfError::invalid("empty corpus"));
        }
        if initial.ncols() != dim {
            return Err(GraphIvfError::invalid(format!(
                "initial centroid dim ({}) does not match point dim ({dim})",
                initial.ncols()
            )));
        }
        if initial_k == 0 {
            return Err(GraphIvfError::invalid("need at least one initial centroid"));
        }
        if params.centroid_capacity < initial_k {
            return Err(GraphIvfError::invalid(format!(
                "centroid_capacity ({}) is smaller than the initial centroid count ({initial_k})",
                params.centroid_capacity
            )));
        }
        if params.split_threshold < 2 {
            return Err(GraphIvfError::invalid("split_threshold must be >= 2"));
        }
        if params.reassign_neighbors < 1 {
            return Err(GraphIvfError::invalid("reassign_neighbors must be >= 1"));
        }
        params.routing.validate()?;
        // Merging needs a hysteresis gap below the split threshold. Without
        // one, a dissolve spills onto a neighbor, overflowing it; the split
        // that follows produces two half-size children, either of which may
        // land back under the merge line.
        if params.merges_enabled() && 2 * params.merge_threshold > params.split_threshold {
            return Err(GraphIvfError::invalid(format!(
                "merge_threshold ({}) leaves no hysteresis below split_threshold ({}); \
                 require 2 * merge_threshold <= split_threshold",
                params.merge_threshold, params.split_threshold
            )));
        }
        if params.num_threads == 0 {
            return Err(GraphIvfError::invalid("num_threads must be non-zero"));
        }

        // `centroid_capacity` is the total id budget (live + retired); size the
        // graph and the id-indexed vectors to exactly that.
        let capacity = params.centroid_capacity.max(initial_k);

        // Clustering and centroid navigation always run in squared-L2, matching
        // a batch build, whatever metric candidate scoring uses.
        let graph = if let OnlineCentroidRouting::Graph { graph, .. } = params.routing {
            let init_mat = Matrix::try_from(
                initial.as_slice().to_vec().into_boxed_slice(),
                initial_k,
                dim,
            )
            .map_err(|_| GraphIvfError::invalid("initial centroid matrix shape mismatch"))?;
            Some(centroids::build_mutable(
                init_mat,
                &graph,
                params.num_threads,
                capacity,
                VectorMetric::L2,
            )?)
        } else {
            None
        };
        let centroids = CentroidRegistry::new(&initial, capacity, graph);
        let partition = IvfPartition::new(capacity, num_points);

        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .map_err(ANNError::from)?;
        let pool = create_thread_pool(params.num_threads)?;

        Ok(Self {
            points,
            dim,
            params,
            centroids,
            partition,
            runtime,
            pool,
            rng: StdRng::seed_from_u64(params.seed),
            telemetry: BuildTelemetry::default(),
            scratch: MaintenanceScratch::default(),
            poisoned: None,
        })
    }

    fn ensure_healthy(&self) -> Result<()> {
        if let Some(reason) = &self.poisoned {
            return Err(GraphIvfError::poisoned(reason.clone()));
        }
        Ok(())
    }

    fn begin_commit(&mut self) {
        debug_assert!(self.poisoned.is_none());
        self.poisoned = Some("mutation did not complete".to_owned());
    }

    fn finish_commit(&mut self, result: &Result<()>) {
        match result {
            Ok(()) => self.poisoned = None,
            Err(error) => self.poisoned = Some(error.to_string()),
        }
    }

    /// Whether a previous mutation failed after changing in-memory state.
    ///
    /// A poisoned clusterer remains available for diagnostics through
    /// [`telemetry`](Self::telemetry), but insert, delete, search, and flush
    /// operations are rejected.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.is_some()
    }

    /// Number of live clusters.
    pub fn num_clusters(&self) -> usize {
        self.centroids.live_count()
    }

    /// Read-only access to the build telemetry accumulated so far.
    pub fn telemetry(&self) -> &BuildTelemetry {
        &self.telemetry
    }

    /// Current size of every live cluster (points assigned to it), in no
    /// particular order. Useful for inspecting the final size distribution.
    pub fn cluster_sizes(&self) -> Vec<usize> {
        self.centroids
            .live_ids()
            .map(|cid| self.partition.list_len(cid))
            .collect()
    }

    /// Census the out-edges of the live centroid graph, or `None` under
    /// [`CentroidSearch::Exact`](crate::CentroidSearch::Exact), which maintains
    /// no graph.
    ///
    /// Diagnostic for graph health under churn: reports how much of the
    /// graph's out-degree still points at a readable slot. Reads adjacency
    /// lists and slot status only, so it costs `O(num_clusters * degree)`
    /// status checks and no distance computations.
    ///
    /// # Errors
    ///
    /// Returns an error if the clusterer is poisoned or the graph read fails.
    pub fn centroid_adjacency_census(&self) -> Result<Option<AdjacencyCensus>> {
        self.ensure_healthy()?;
        self.centroids.adjacency_census(&self.runtime)
    }

    /// Clustering residual: the sum of squared distances from every assigned
    /// point to its centroid. Lower is a tighter clustering.
    pub fn residual(&self) -> f64 {
        let mut sum = 0.0f64;
        for (cid, cv) in self.centroids.iter_live() {
            for &pid in self.partition.members(cid) {
                sum += sq_l2(self.points.row(pid as usize), cv) as f64;
            }
        }
        sum
    }

    /// Insert a batch of points, routing each to its nearest centroid and then
    /// splitting whichever clusters that pushed past `split_threshold`.
    ///
    /// A batch is processed in phases rather than point by point:
    ///
    /// 1. **Route.** Splits are deferred to the end of the batch, so routing is
    ///    read-only with respect to the centroid graph and a batch large enough
    ///    to be worth the dispatch runs across the whole thread pool.
    /// 2. **Split each overflow.** Every routed-to cluster that now overflows is
    ///    bisected on its own by a local 2-means over its members, including the
    ///    points this batch routed to it. Parents are independent of one
    ///    another, so a cluster splits identically however many others
    ///    overflowed in the same batch.
    /// 3. **Reassign per split region.** Each split parent's neighborhood is
    ///    then reassigned in turn, as a GEMM.
    ///
    /// Routes are computed against the pre-batch partition, so a point that
    /// lands in a cluster which then splits is routed slightly stale; phase 3
    /// re-examines exactly that cluster's region.
    ///
    /// # Errors
    ///
    /// Returns an error if any point id is out of range or is already present
    /// in the index, or if routing, clustering, or graph mutation fails. Input
    /// validation and routing happen before mutation. An error after mutation
    /// begins poisons the clusterer, and all later mutations, searches, and
    /// flushes are rejected.
    pub fn insert_batch(&mut self, pids: &[u32]) -> Result<()> {
        self.ensure_healthy()?;
        if pids.is_empty() {
            return Ok(());
        }
        let num_points = self.points.nrows();
        for &pid in pids {
            if pid as usize >= num_points {
                return Err(GraphIvfError::invalid("point id out of range"));
            }
            if self.partition.assignment(pid) != UNASSIGNED {
                return Err(GraphIvfError::invalid(format!(
                    "point {pid} is already present; delete it before re-inserting"
                )));
            }
        }
        // A pid named twice in one batch would be appended to its list twice.
        // Catching it here, before anything is mutated, keeps a rejected insert
        // a no-op.
        let mut sorted = pids.to_vec();
        sorted.sort_unstable();
        if let Some(dup) = sorted.windows(2).find(|w| w[0] == w[1]) {
            return Err(GraphIvfError::invalid(format!(
                "point {} appears more than once in the same insert batch",
                dup[0]
            )));
        }

        // 1. Route the whole batch in parallel against the frozen graph.
        let route_start = Instant::now();
        let mut routes = vec![0u32; pids.len()];
        self.route_batch(pids, &mut routes)?;
        self.telemetry.routing_us += route_start.elapsed().as_micros() as u64;

        // Complete every fallible planning step against the old partition.
        // Parent plans explicitly include points from this batch that will be
        // attached during commit.
        let parents = self.select_split_parents(&routes);
        let split = if parents.is_empty() {
            None
        } else {
            let parent_set: std::collections::HashSet<u32> = parents.iter().copied().collect();
            let mut incoming = std::collections::HashMap::<u32, Vec<u32>>::new();
            for (&pid, &cid) in pids.iter().zip(&routes) {
                if parent_set.contains(&cid) {
                    incoming.entry(cid).or_default().push(pid);
                }
            }
            Some(self.prepare_split(&parents, &incoming)?)
        };

        self.begin_commit();
        let result = self.commit_insert(pids, &routes, split);
        if result.is_err() {
            self.reattach_pending_points();
        }
        self.finish_commit(&result);
        result
    }

    /// Apply routes already validated and computed by [`insert_batch`].
    fn commit_insert(
        &mut self,
        pids: &[u32],
        routes: &[u32],
        split: Option<SplitPlan>,
    ) -> Result<()> {
        // 1. Append every point to its routed cluster. Both failure modes —
        //    already present, and named twice — were ruled out before commit.
        for (&pid, &cid) in pids.iter().zip(routes.iter()) {
            self.partition.attach_new(pid, cid);
        }
        self.telemetry.total_inserts += pids.len() as u64;

        match split {
            Some(plan) => self.commit_split(plan),
            None => Ok(()),
        }
    }

    /// Select overflowing routed-to clusters using their projected post-insert
    /// sizes, then apply id-budget and live-cluster admission limits.
    fn select_split_parents(&self, routes: &[u32]) -> Vec<u32> {
        let mut sorted = routes.to_vec();
        sorted.sort_unstable();
        let mut parents = Vec::new();
        let mut start = 0;
        while start < sorted.len() {
            let cid = sorted[start];
            let end = sorted[start..]
                .iter()
                .position(|&candidate| candidate != cid)
                .map_or(sorted.len(), |offset| start + offset);
            if self.partition.list_len(cid) + end - start > self.params.split_threshold {
                parents.push(cid);
            }
            start = end;
        }

        let mut admitted = parents.len().min(self.centroids.alloc_budget() / 2);
        if let Some(max_clusters) = self.params.max_clusters {
            admitted = admitted.min(max_clusters.saturating_sub(self.centroids.live_count()));
        }
        if admitted < parents.len() {
            let incoming_count = |cid: u32| {
                sorted.partition_point(|&route| route <= cid)
                    - sorted.partition_point(|&route| route < cid)
            };
            parents.sort_unstable_by(|&a, &b| {
                let size_a = self.partition.list_len(a) + incoming_count(a);
                let size_b = self.partition.list_len(b) + incoming_count(b);
                size_b.cmp(&size_a).then(a.cmp(&b))
            });
            parents.truncate(admitted);
            parents.sort_unstable();
        }
        parents
    }

    /// Delete a batch of points, then merge whichever clusters that emptied
    /// past [`merge_threshold`](OnlineParams::merge_threshold).
    ///
    /// The batch first groups points by their current cluster and selects
    /// underflows from projected post-delete sizes. It then snapshots each
    /// admitted victim's remaining members and finds survivor candidates while
    /// the graph and partition are unchanged. Only after that preparation
    /// succeeds are the delete filters, centroid retirements, and point
    /// placements committed.
    ///
    /// Deletion is structurally cheap here in a way it is not for a graph
    /// index: nothing points *at* an indexed point — the centroid graph spans
    /// centroids, not points — so removing one cannot disconnect anything, and
    /// there is no tombstone to keep or consolidation pass to run. A deleted
    /// point's id becomes free and may be inserted again later.
    ///
    /// Deleting never splits. A survivor that absorbs a dissolved cluster's
    /// members can land above `split_threshold`, but it is left for the next
    /// insert routed to it: keeping splits insert-driven and merges
    /// delete-driven makes split/merge cascades structurally impossible rather
    /// than merely bounded.
    ///
    /// # Errors
    ///
    /// Returns an error if any point id is out of range, if any point is not
    /// currently present in the index, or if the merge pass fails. A failure
    /// after list removal begins poisons the clusterer.
    pub fn delete_batch(&mut self, pids: &[u32]) -> Result<()> {
        self.ensure_healthy()?;
        if pids.is_empty() {
            return Ok(());
        }
        let num_points = self.points.nrows();

        // 1. Pair every point with the cluster it currently sits in. Sorting by
        //    (cluster, point) groups the batch by cluster and leaves each group
        //    itself sorted, which is what `remove_sorted` wants; deduplicating
        //    makes a repeated id within one batch harmless.
        let mut by_cluster = Vec::with_capacity(pids.len());
        for &pid in pids {
            if pid as usize >= num_points {
                return Err(GraphIvfError::invalid("point id out of range"));
            }
            let cid = self.partition.assignment(pid);
            if cid == UNASSIGNED {
                return Err(GraphIvfError::invalid(format!(
                    "point {pid} is not present in the index"
                )));
            }
            by_cluster.push((cid, pid));
        }
        by_cluster.sort_unstable();
        by_cluster.dedup();

        // Select merge victims from projected post-delete sizes, and complete
        // their fallible candidate searches before removing a single point.
        let merge = if self.params.merges_enabled() {
            let mut victims = Vec::new();
            let mut start = 0;
            while start < by_cluster.len() {
                let cid = by_cluster[start].0;
                let end = by_cluster[start..]
                    .iter()
                    .position(|&(candidate, _)| candidate != cid)
                    .map_or(by_cluster.len(), |offset| start + offset);
                if self.partition.list_len(cid) - (end - start) < self.params.merge_threshold {
                    victims.push(cid);
                }
                start = end;
            }

            let floor = self.params.effective_min_clusters();
            let admitted = victims
                .len()
                .min(self.centroids.live_count().saturating_sub(floor));
            victims.sort_unstable_by_key(|&cid| {
                let deleted = by_cluster
                    .iter()
                    .filter(|&&(candidate, _)| candidate == cid)
                    .count();
                (self.partition.list_len(cid) - deleted, cid)
            });
            victims.truncate(admitted);

            if victims.is_empty() {
                None
            } else {
                let deleted: std::collections::HashSet<u32> =
                    by_cluster.iter().map(|&(_, pid)| pid).collect();
                Some(self.prepare_merge(&victims, &deleted)?)
            }
        } else {
            None
        };

        self.begin_commit();
        let result = self.commit_delete(&by_cluster, merge);
        if result.is_err() {
            self.reattach_pending_points();
        }
        self.finish_commit(&result);
        result
    }

    /// Remove a validated, sorted `(cluster, point)` batch and maintain any
    /// underfull clusters.
    fn commit_delete(&mut self, by_cluster: &[(u32, u32)], merge: Option<MergePlan>) -> Result<()> {
        let delete_start = Instant::now();

        // 1. Filter each touched cluster's list in a single pass.
        let mut group: Vec<u32> = Vec::new();
        let mut deleted = 0usize;
        let mut start = 0usize;
        while start < by_cluster.len() {
            let cid = by_cluster[start].0;
            let end = by_cluster[start..]
                .iter()
                .position(|&(c, _)| c != cid)
                .map_or(by_cluster.len(), |off| start + off);
            group.clear();
            group.extend(by_cluster[start..end].iter().map(|&(_, pid)| pid));
            self.partition.remove_sorted(cid, &group);
            deleted += group.len();
            start = end;
        }
        self.telemetry.total_deletes += deleted as u64;
        self.telemetry.delete_us += delete_start.elapsed().as_micros() as u64;

        match merge {
            Some(plan) => self.commit_merge(plan),
            None => Ok(()),
        }
    }

    /// The `want` live centroids nearest `c`'s own centroid vector, in
    /// ascending distance and excluding `c` itself.
    ///
    /// Used by [`prepare_split`](Self::prepare_split) to define its working region:
    /// one cluster plus its centroid-space neighborhood, searched *before* any
    /// retirement so it sees the pre-mutation graph. The result is therefore
    /// only a candidate list — the caller re-checks liveness after publishing
    /// its own structural change.
    ///
    /// `k = want + 1` reserves the slot `c` itself takes at distance zero.
    fn region_neighbors(&mut self, c: u32, want: usize) -> Result<Vec<u32>> {
        let query = self
            .centroids
            .get(c)
            .expect("region is anchored on a live centroid")
            .to_vec();
        let mut neighbors = self.nearest_live_centroids(&query, want + 1)?;
        neighbors.retain(|&x| x != c);
        neighbors.truncate(want);
        Ok(neighbors)
    }

    /// The `want` nearest live centroids to `query`, in ascending distance.
    ///
    /// Anchored on a vector rather than a centroid id so callers can search
    /// around copied centroid vectors without extending a table borrow across
    /// mutable scratch access.
    fn nearest_live_centroids(&mut self, query: &[f32], want: usize) -> Result<Vec<u32>> {
        let search_k = want.max(1);
        // Ignored under exact routing, which scans every live centroid.
        let search_l = self
            .params
            .routing
            .neighbor_beam(search_k)
            .unwrap_or(search_k);

        let scratch = &mut self.scratch.neighbors;
        scratch.ids.clear();
        scratch.ids.resize(search_k, 0);
        scratch.distances.clear();
        scratch.distances.resize(search_k, 0.0);

        let found = self.centroids.search(
            &self.runtime,
            query,
            search_l,
            &mut scratch.ids,
            &mut scratch.distances,
        )?;
        let mut out = scratch.ids[..found].to_vec();

        out.retain(|&x| self.centroids.is_live(x));
        out.truncate(want);
        Ok(out)
    }

    /// Route every point of `pids` to its nearest live centroid, writing the
    /// centroid ids into `out`.
    ///
    /// The graph is not mutated during a batch, so the searches are independent.
    /// A batch large enough to fill more than one chunk is spread across the
    /// thread pool, each worker driving its own current-thread runtime; a
    /// smaller one is not worth the dispatch and the runtimes it would build, so
    /// it runs on the clusterer's own runtime.
    fn route_batch(&self, pids: &[u32], out: &mut [u32]) -> Result<()> {
        debug_assert_eq!(pids.len(), out.len());
        let centroids = &self.centroids;
        let points = &self.points;
        let beams = self.params.routing.route_beams();

        if pids.len() <= ROUTE_CHUNK {
            for (pid, slot) in pids.iter().zip(out.iter_mut()) {
                *slot = route_one(centroids, &self.runtime, points.row(*pid as usize), beams)?;
            }
            return Ok(());
        }

        out.par_chunks_mut(ROUTE_CHUNK)
            .enumerate()
            .try_for_each_in_pool(self.pool.as_ref(), |(ci, chunk)| -> Result<()> {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .build()
                    .map_err(ANNError::from)?;
                for (j, slot) in chunk.iter_mut().enumerate() {
                    let point = points.row(pids[ci * ROUTE_CHUNK + j] as usize);
                    *slot = route_one(centroids, &runtime, point, beams)?;
                }
                Ok(())
            })
    }

    /// Prepare every parent's split without changing centroid or partition
    /// state.
    ///
    /// Each parent is bisected on its own by a local 2-means over its members,
    /// exactly as a batch of one would be. Parents that overflow in the same
    /// batch do not see one another, so a split is planned identically however
    /// many others fired alongside it.
    ///
    /// `parents` must be sorted, live, hold at least two members each, and fit
    /// the id budget and cluster cap — [`insert_batch`](Self::insert_batch)
    /// guarantees all four.
    fn prepare_split(
        &mut self,
        parents: &[u32],
        incoming: &std::collections::HashMap<u32, Vec<u32>>,
    ) -> Result<SplitPlan> {
        let started = Instant::now();

        // The `reassign_neighbors` live centroids nearest each parent, found by
        // searching the parent's own centroid vector. The searches run before the
        // parents are retired and the children inserted, so they see the
        // pre-split centroids; the parent itself (distance 0) is dropped, and its
        // own two children join the candidate set once they exist. `k = s + 1`
        // reserves the slot the parent takes.
        let s = self.params.reassign_neighbors;
        let mut rng_after = self.rng.clone();
        let mut parent_plans = Vec::with_capacity(parents.len());
        for &c in parents {
            let mut members = self.partition.members(c).to_vec();
            if let Some(inserted) = incoming.get(&c) {
                members.extend_from_slice(inserted);
            }
            let neighbors = self.region_neighbors(c, s)?;

            let kmeans_start = Instant::now();
            let children = self.two_means(&members, &mut rng_after)?;
            parent_plans.push(SplitParentPlan {
                id: c,
                members,
                neighbors,
                children,
                two_means_us: kmeans_start.elapsed().as_micros() as u64,
            });
        }

        Ok(SplitPlan {
            parents: parent_plans,
            rng_after,
            started,
        })
    }

    /// Bisect `members` into two centroids, seeded with two distinct members
    /// drawn from `rng`.
    fn two_means(&self, members: &[u32], rng: &mut StdRng) -> Result<[Box<[f32]>; 2]> {
        let dim = self.dim;
        let m = members.len();
        debug_assert!(m >= 2);

        let mut buf = vec![0.0f32; m * dim];
        for (i, &pid) in members.iter().enumerate() {
            buf[i * dim..(i + 1) * dim].copy_from_slice(self.points.row(pid as usize));
        }
        let data = Matrix::try_from(buf.into_boxed_slice(), m, dim)
            .map_err(|_| GraphIvfError::invalid("split sub-matrix shape mismatch"))?;

        let a = rng.random_range(0..m);
        let mut b = rng.random_range(0..m);
        if b == a {
            b = (a + 1) % m;
        }
        let mut seed = vec![0.0f32; 2 * dim];
        seed[..dim].copy_from_slice(data.row(a));
        seed[dim..].copy_from_slice(data.row(b));
        let mut children = Matrix::try_from(seed.into_boxed_slice(), 2, dim)
            .map_err(|_| GraphIvfError::invalid("split seed shape mismatch"))?;

        let mut assigner = cluster::ExactAssigner::default();
        cluster::lloyd(
            data.as_view(),
            &mut children,
            &mut assigner,
            self.params.two_means_iters.max(1),
            EmptyClusterPolicy::PreserveOld,
            self.params.normalize_centroids,
            &self.pool,
        )?;

        Ok([
            children.row(0).to_vec().into_boxed_slice(),
            children.row(1).to_vec().into_boxed_slice(),
        ])
    }

    /// Publish a prepared split and reassign each affected region.
    ///
    /// The caller has already marked the clusterer poisoned. All preparation
    /// that can run against the old graph and partition is complete, but graph
    /// publication and GEMM assignment remain fallible and irreversible.
    fn commit_split(&mut self, plan: SplitPlan) -> Result<()> {
        let parent_ids: Vec<u32> = plan.parents.iter().map(|parent| parent.id).collect();
        let children: Vec<Box<[f32]>> = plan
            .parents
            .iter()
            .flat_map(|parent| parent.children.clone())
            .collect();
        let child_ids = self
            .centroids
            .apply_split(&self.runtime, &parent_ids, children)?;
        let live_after = self.centroids.live_count();
        let mut events = Vec::with_capacity(plan.parents.len());
        let mut total_reassigned = 0u64;

        // Liveness no longer moves — `apply_split` retired every parent and
        // created every child — so the batch's whole working set is known
        // before the first region runs, and regions that share a neighbor
        // collapse onto one entry.
        let mut updated = std::collections::HashSet::new();
        for (parent, born) in plan.parents.iter().zip(child_ids.chunks_exact(2)) {
            updated.extend(
                parent
                    .neighbors
                    .iter()
                    .copied()
                    .filter(|&c| self.centroids.is_live(c)),
            );
            updated.extend(born.iter().copied());
        }
        let clusters_updated = updated.len();

        // Reassign each split region in turn. The candidates are the parent's
        // own two children plus the neighbors picked before the mutation, less
        // any neighbor that was itself a parent of this batch and has since
        // been retired — that region is covered by its own turn.
        for (parent, born) in plan.parents.iter().zip(child_ids.chunks_exact(2)) {
            let cluster_size = parent.members.len();

            self.scratch.candidates.clear();
            self.scratch.candidates.extend(
                parent
                    .neighbors
                    .iter()
                    .copied()
                    .filter(|&c| self.centroids.is_live(c)),
            );
            let num_neighbors = self.scratch.candidates.len();
            self.scratch.candidates.extend_from_slice(born);

            // Candidate points: the parent's own members plus everything its
            // surviving neighbors hold. The children's lists are still empty —
            // no other region can have placed a point on them — so the k-means
            // assignment is not applied separately; reassignment places every
            // member against the neighbors too, which strictly refines it.
            self.scratch.points.clear();
            self.partition
                .detach_members_into(parent.id, &mut self.scratch.points);
            debug_assert!(self.scratch.points[..cluster_size]
                .iter()
                .map(|point| point.id)
                .eq(parent.members.iter().copied()));
            for i in 0..num_neighbors {
                let cid = self.scratch.candidates[i];
                self.partition
                    .detach_members_into(cid, &mut self.scratch.points);
            }

            let reassign_start = Instant::now();
            let num_reassigned = match self.reassign_scratch_points() {
                Ok(count) => count,
                Err(error) => {
                    self.telemetry.total_splits += events.len() as u64;
                    self.telemetry.total_reassigned += total_reassigned;
                    self.telemetry.splits.extend(events);
                    self.telemetry.split_us += plan.started.elapsed().as_micros() as u64;
                    return Err(error);
                }
            };
            let reassign_us = reassign_start.elapsed().as_micros() as u64;

            total_reassigned += num_reassigned as u64;
            events.push(SplitEvent {
                insert_index: self.telemetry.total_inserts,
                cluster: parent.id,
                cluster_size,
                num_neighbors,
                num_reassigned,
                live_after,
                clusters_updated,
                two_means_us: parent.two_means_us,
                reassign_us,
                total_us: parent.two_means_us + reassign_us,
            });
        }

        self.rng = plan.rng_after;
        self.telemetry.total_splits += events.len() as u64;
        self.telemetry.total_reassigned += total_reassigned;
        self.telemetry.splits.extend(events);
        self.telemetry.split_us += plan.started.elapsed().as_micros() as u64;
        Ok(())
    }

    /// Snapshot every merge victim before any structural mutation.
    ///
    /// The counterpart of a split, but deliberately not its mirror. A split
    /// has to re-examine the neighbors' points because
    /// it *fits* new centroids, and a point that was nearest its own centroid
    /// may now be nearer a child. A merge fits nothing, and removing a centroid
    /// cannot change a surviving point's nearest-among-survivors — so the
    /// neighbors' members are provably already where they belong, and only the
    /// victim's own points are re-placed. That is the whole operator: no
    /// k-means and no new centroid.
    ///
    /// Because nothing is fitted, a merge consumes no centroid id. Deletes are
    /// therefore free against the
    /// [`centroid_capacity`](OnlineParams::centroid_capacity) budget, which
    /// only splits draw down.
    ///
    /// Candidate search runs before retirement so graph navigation remains
    /// fallible but non-mutating. Every victim in the batch is explicitly
    /// excluded from every result, including the exact fallback, so the saved
    /// candidates remain live after the whole batch is retired. Placement then
    /// runs only after all victims have left the graph and table.
    ///
    /// `victims` must be live, distinct, and leave at least one cluster
    /// standing; [`delete_batch`](Self::delete_batch) guarantees all of these.
    fn prepare_merge(
        &self,
        victims: &[u32],
        deleted: &std::collections::HashSet<u32>,
    ) -> Result<MergePlan> {
        let started = Instant::now();
        let victim_set: std::collections::HashSet<u32> = victims.iter().copied().collect();
        let survivor_count = self.centroids.live_count().saturating_sub(victims.len());
        let want = self.params.reassign_neighbors.min(survivor_count);
        // Ignored under exact routing, which scans every live centroid.
        let floor = want.saturating_add(victims.len()).max(1);
        let search_l = self.params.routing.neighbor_beam(floor).unwrap_or(floor);
        let victims = victims
            .iter()
            .map(|&id| {
                let anchor = self.centroids.get(id).ok_or_else(|| {
                    GraphIvfError::invalid(format!("merge victim {id} is not live"))
                })?;
                let search_start = Instant::now();
                let mut ids = vec![0u32; want.saturating_add(victims.len()).max(1)];
                let mut distances = vec![0.0f32; ids.len()];
                let found = self.centroids.search(
                    &self.runtime,
                    anchor,
                    search_l,
                    &mut ids,
                    &mut distances,
                )?;
                ids.truncate(found);
                ids.retain(|candidate| {
                    self.centroids.is_live(*candidate) && !victim_set.contains(candidate)
                });
                ids.truncate(want);
                if ids.len() < want {
                    let mut exact: Vec<(u32, f32)> = self
                        .centroids
                        .iter_live()
                        .filter(|(candidate, _)| !victim_set.contains(candidate))
                        .map(|(candidate, vector)| (candidate, sq_l2(anchor, vector)))
                        .collect();
                    exact.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                    ids = exact
                        .into_iter()
                        .take(want)
                        .map(|(candidate, _)| candidate)
                        .collect();
                }
                if ids.is_empty() {
                    return Err(GraphIvfError::invalid(format!(
                        "retiring cluster {id} found no surviving centroid"
                    )));
                }
                Ok(MergeVictimPlan {
                    id,
                    members: self
                        .partition
                        .members(id)
                        .iter()
                        .copied()
                        .filter(|pid| !deleted.contains(pid))
                        .collect(),
                    candidates: ids,
                    search_us: search_start.elapsed().as_micros() as u64,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(MergePlan { victims, started })
    }

    /// Retire and scatter a prepared merge batch.
    fn commit_merge(&mut self, plan: MergePlan) -> Result<()> {
        let victim_ids: Vec<u32> = plan.victims.iter().map(|victim| victim.id).collect();

        // Retire the whole batch up front, so no later search can offer a
        // centroid that is itself on its way out.
        self.centroids.retire_all(&self.runtime, &victim_ids)?;
        let live_after = self.centroids.live_count();

        // Scatter each victim's members over the survivors nearest its anchor.
        let op_index = self.telemetry.total_inserts + self.telemetry.total_deletes;
        let mut events = Vec::with_capacity(plan.victims.len());
        let mut total_reassigned = 0u64;
        for victim in &plan.victims {
            self.scratch.candidates.clear();
            self.scratch
                .candidates
                .extend_from_slice(&victim.candidates);
            self.scratch.points.clear();
            self.partition
                .detach_members_into(victim.id, &mut self.scratch.points);
            debug_assert!(self
                .scratch
                .points
                .iter()
                .map(|point| point.id)
                .eq(victim.members.iter().copied()));

            let reassign_start = Instant::now();
            let num_reassigned = match self.reassign_scratch_points() {
                Ok(count) => count,
                Err(error) => {
                    self.telemetry.total_merges += events.len() as u64;
                    self.telemetry.total_merge_reassigned += total_reassigned;
                    self.telemetry.merges.extend(events);
                    self.telemetry.merge_us += plan.started.elapsed().as_micros() as u64;
                    return Err(error);
                }
            };
            let reassign_us = reassign_start.elapsed().as_micros() as u64;

            total_reassigned += num_reassigned as u64;
            events.push(MergeEvent {
                op_index,
                victim: victim.id,
                victim_size: victim.members.len(),
                num_neighbors: self.scratch.candidates.len(),
                num_reassigned,
                live_after,
                search_us: victim.search_us,
                reassign_us,
                total_us: victim.search_us + reassign_us,
            });
        }

        self.telemetry.total_merges += events.len() as u64;
        self.telemetry.total_merge_reassigned += total_reassigned;
        self.telemetry.merges.extend(events);
        self.telemetry.merge_us += plan.started.elapsed().as_micros() as u64;
        Ok(())
    }

    /// Reassign all currently detached scratch points to the nearest scratch
    /// candidate, returning how many actually changed cluster.
    ///
    /// Distances are computed as `‖p‖² - 2p·c + ‖c‖²` over a `tile x |cands|`
    /// GEMM instead of a scalar loop per point, which is what makes a large
    /// `reassign_neighbors` affordable. Each point must already be detached;
    /// candidate lists that are not part of the working region remain intact.
    fn reassign_scratch_points(&mut self) -> Result<usize> {
        let MaintenanceScratch {
            points,
            candidates,
            reassign,
            ..
        } = &mut self.scratch;
        if candidates.is_empty() || points.is_empty() {
            return Ok(0);
        }
        let dim = self.dim;
        let nc = candidates.len();

        reassign.centroid_vectors.clear();
        reassign.centroid_vectors.resize(nc * dim, 0.0);
        for (i, &c) in candidates.iter().enumerate() {
            reassign.centroid_vectors[i * dim..(i + 1) * dim]
                .copy_from_slice(self.centroids.get(c).expect("candidate is live"));
        }
        let rows = REASSIGN_TILE.min(points.len());
        reassign.point_tile.clear();
        reassign.point_tile.resize(rows * dim, 0.0);
        reassign.best.clear();
        reassign.best.resize(rows, 0);

        let mut num_reassigned = 0usize;
        for chunk in points.chunks(rows) {
            let n = chunk.len();
            for (i, point) in chunk.iter().enumerate() {
                reassign.point_tile[i * dim..(i + 1) * dim]
                    .copy_from_slice(self.points.row(point.id as usize));
            }
            compute_closest_centers(
                &reassign.point_tile[..n * dim],
                n,
                dim,
                &reassign.centroid_vectors,
                nc,
                1,
                &mut reassign.best[..n],
                None,
                None,
                self.pool.as_ref(),
            )?;
            for (i, point) in chunk.iter().copied().enumerate() {
                let cid = candidates[reassign.best[i] as usize];
                if self.partition.attach_detached(point, cid) {
                    num_reassigned += 1;
                }
            }
        }

        Ok(num_reassigned)
    }

    /// Restore any scratch points that remain detached after a failed
    /// reassignment. This keeps the partition internally inspectable even
    /// though the enclosing clusterer remains poisoned because its centroid
    /// graph may have been partially changed.
    fn reattach_pending_points(&mut self) {
        for point in self.scratch.points.iter().copied() {
            if self.partition.assignment(point.id) != UNASSIGNED {
                continue;
            }
            if self.centroids.is_live(point.previous) {
                self.partition.attach_detached(point, point.previous);
            } else if let Some(&fallback) = self
                .scratch
                .candidates
                .iter()
                .find(|&&candidate| self.centroids.is_live(candidate))
            {
                self.partition.attach_detached(point, fallback);
            }
        }
    }

    /// Serialize the in-memory IVF mapping to `prefix` in the batch on-disk
    /// format (`.graphivf_centroids.fbin`, `.graphivf_lists`, `.graphivf_meta`,
    /// `.graphivf_graph`), densely remapping live centroid ids to
    /// `0..num_clusters`.
    ///
    /// The live centroid graph is written out under that same remapping, so a
    /// load replays it instead of rebuilding. Edges into centroids retired by
    /// earlier deletions name nothing that is being written and are dropped, so
    /// a heavily churned graph is saved sparser than it ran; a clusterer routing
    /// exactly has no graph and writes none.
    ///
    /// Clustering runs on `f32`, but the inverted lists are written from
    /// `stored` — the corpus in its on-disk element type `T` (e.g.
    /// [`MinMaxElement<8>`](diskann_providers::common::MinMaxElement)), copied
    /// verbatim. `stored` must have one row per corpus point, in the same order
    /// as the clustering points (row `pid` is point `pid`). To store `f32`
    /// lists, pass the clustering points themselves (`T = f32`). The centroid
    /// graph is always written as `f32`.
    ///
    /// The result loads through
    /// [`GraphIvfIndex::<T>::load`](crate::GraphIvfIndex::load).
    ///
    /// # Errors
    ///
    /// Returns an error if the clusterer is poisoned or `stored`'s row count
    /// does not match the corpus. Points that were never inserted or have been
    /// deleted are omitted from the serialized index.
    pub fn flush<T: VectorRepr>(&self, prefix: &Path, stored: MatrixView<'_, T>) -> Result<()> {
        self.ensure_healthy()?;
        let num_points = self.points.nrows();
        if stored.nrows() != num_points {
            return Err(GraphIvfError::invalid(format!(
                "stored corpus has {} rows but clustering corpus has {num_points}",
                stored.nrows()
            )));
        }

        // Dense remap of live centroid ids to a contiguous 0..k range.
        let (remap, centroids_mat) = self.centroids.densify()?;
        let k = centroids_mat.nrows();

        // Dense per-point assignments. A point that is not currently in the
        // index — never inserted, or deleted — is marked `UNASSIGNED` and
        // skipped by the writer. With deletes there is no way to tell those two
        // apart, and "the index holds whatever is live" is the only reading
        // that stays true; the ids of the points that *are* written are still
        // their original corpus rows, so groundtruth keeps lining up.
        let mut dense = vec![UNASSIGNED; num_points];
        let mut live_points = 0u64;
        for (pid, slot) in dense.iter_mut().enumerate() {
            let c = self.partition.assignment(pid as u32);
            if c != UNASSIGNED {
                *slot = remap[c as usize];
                live_points += 1;
            }
        }

        // Write centroids (always f32).
        let centroids_path = with_suffix(prefix, CENTROIDS_SUFFIX);
        storage::write_centroids(&centroids_path, centroids_mat.as_view())?;

        // Persist the live centroid graph under the same dense numbering, so a
        // load replays it instead of rebuilding.
        let snapshot = if k > 0 {
            self.centroids
                .snapshot_graph(&self.runtime, centroids_mat.row(0))?
        } else {
            None
        };
        if let Some(snapshot) = &snapshot {
            storage::write_graph(&with_suffix(prefix, GRAPH_SUFFIX), snapshot)?;
        }

        // Write inverted lists from the stored representation and the metadata.
        let stored_dim = stored.ncols();
        let lists_path = with_suffix(prefix, LISTS_SUFFIX);
        let (counts, offsets) = storage::write_lists_stored::<T>(&lists_path, stored, &dense, k)?;
        let layout = Layout {
            dim: stored_dim,
            metric: self.params.metric,
            element_size: std::mem::size_of::<T>(),
            num_points: live_points,
            graph: self.params.routing.stored_graph_params(),
            has_graph: snapshot.is_some(),
            counts,
            offsets,
        };
        storage::write_metadata(&with_suffix(prefix, META_SUFFIX), &layout)?;
        Ok(())
    }

    /// Open a query handle over the index in its *current* state.
    ///
    /// The handle borrows the clusterer, so no insert or delete can run while
    /// one is alive; conversely several handles can be opened at once and
    /// driven in parallel. Open one per search phase, not one per query — each
    /// carries a tokio runtime and reusable scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if the clusterer was poisoned by a failed mutation or
    /// if the query handle's runtime cannot be created.
    pub fn searcher(&self) -> Result<OnlineSearcher<'_>> {
        OnlineSearcher::new(self)
    }
}

#[cfg(test)]
mod tests;

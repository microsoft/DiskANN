/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Online (incremental) graph-IVF clustering with split-and-reassign.
//!
//! [`OnlineClusterer`] builds the IVF partition one point at a time instead of
//! in a single batch Lloyd pass. Points are streamed in; each is routed to its
//! nearest centroid via a mutable centroid graph. When a cluster grows past a
//! threshold it is split into two by a local 2-means, and the points of the
//! split cluster together with the points of its graph-neighboring clusters are
//! reassigned among the new and neighboring centroids.
//!
//! The whole IVF mapping (inverted lists and the point→centroid assignment) is
//! kept in memory and mutated in place; [`OnlineClusterer::flush`] serializes it
//! once, at the end, into the same on-disk format a batch build produces, so the
//! result loads and searches through the unchanged
//! [`GraphIvfIndex`](crate::GraphIvfIndex) path.
//!
//! Points are preloaded as an `f32` matrix and "streamed" by feeding their row
//! indices to [`OnlineClusterer::insert`]; this keeps the experiment free of
//! disk I/O while still exercising the incremental build logic. For a
//! normalizing metric (cosine) the caller must pre-normalize the points.

use std::path::Path;
use std::time::Instant;

use diskann_providers::utils::{create_thread_pool, RayonThreadPool};
use diskann_utils::views::{Matrix, MatrixView};
use diskann_vector::distance::Metric as VectorMetric;
use rand::{rngs::StdRng, Rng, SeedableRng};
use tokio::runtime::Runtime;

use crate::{
    centroids::{self, MutableCentroidGraph},
    cluster::{self, sq_l2},
    index::{with_suffix, CENTROIDS_SUFFIX, LISTS_SUFFIX, META_SUFFIX},
    params::{EmptyClusterPolicy, OnlineParams},
    storage::{self, Layout},
    GraphIvfError, Result,
};

use diskann::{utils::VectorRepr, ANNError};

/// Sentinel in [`OnlineClusterer::assignments`] for a point that has not been
/// inserted yet.
const UNASSIGNED: u32 = u32::MAX;

/// How the initial centroid set an [`OnlineClusterer`] starts from is produced.
///
/// This is the extensible seam for bootstrapping the clusterer. Experiments
/// rarely start from an empty partition; the common case is [`Warmup`], a light
/// k-means over a prefix of the corpus. [`Explicit`] passes an already-computed
/// centroid matrix through unchanged. New strategies (e.g. k-means++ seeding, a
/// sampled-not-prefixed warmup, or loading centroids from disk) can be added as
/// further variants without changing the constructor surface.
///
/// [`Warmup`]: SeedStrategy::Warmup
/// [`Explicit`]: SeedStrategy::Explicit
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SeedStrategy {
    /// Use a precomputed centroid matrix as-is (one row per centroid).
    Explicit(Matrix<f32>),
    /// Bootstrap by running a lightweight exact k-means over the first
    /// `warmup_points` points of the corpus, yielding `num_centroids`
    /// centroids refined for `iters` Lloyd iterations.
    ///
    /// `iters == 0` skips refinement and uses the sampled points directly.
    /// `warmup_points` is clamped to `[num_centroids, corpus_len]`.
    Warmup {
        /// Number of initial centroids to produce.
        num_centroids: usize,
        /// Number of leading corpus points to cluster.
        warmup_points: usize,
        /// Lloyd iterations for the warmup clustering.
        iters: usize,
    },
}

impl SeedStrategy {
    /// Resolve this strategy into a concrete centroid matrix over `points`.
    fn resolve(self, points: MatrixView<'_, f32>, params: &OnlineParams) -> Result<Matrix<f32>> {
        match self {
            SeedStrategy::Explicit(centroids) => Ok(centroids),
            SeedStrategy::Warmup {
                num_centroids,
                warmup_points,
                iters,
            } => warmup_kmeans(points, num_centroids, warmup_points, iters, params),
        }
    }
}

/// Run a lightweight exact k-means over the first `warmup_points` corpus points
/// to bootstrap `num_centroids` initial centroids (see
/// [`SeedStrategy::Warmup`]).
fn warmup_kmeans(
    points: MatrixView<'_, f32>,
    num_centroids: usize,
    warmup_points: usize,
    iters: usize,
    params: &OnlineParams,
) -> Result<Matrix<f32>> {
    let dim = points.ncols();
    let n = points.nrows();

    if num_centroids == 0 {
        return Err(GraphIvfError::invalid("num_centroids must be non-zero"));
    }
    if num_centroids > n {
        return Err(GraphIvfError::invalid(format!(
            "warmup num_centroids ({num_centroids}) exceeds corpus size ({n})"
        )));
    }
    // Cluster at least `num_centroids` and at most the whole corpus.
    let warmup_n = warmup_points.clamp(num_centroids, n);

    // The warmup window: the first `warmup_n` corpus points.
    let mut window = vec![0.0f32; warmup_n * dim];
    for (dst, r) in window.chunks_mut(dim).zip(0..warmup_n) {
        dst.copy_from_slice(points.row(r));
    }
    let window = Matrix::try_from(window.into_boxed_slice(), warmup_n, dim)
        .map_err(|_| GraphIvfError::invalid("warmup window shape mismatch"))?;

    // Forgy initialization: sample `num_centroids` distinct window rows.
    let mut rng = StdRng::seed_from_u64(params.seed);
    let idx = rand::seq::index::sample(&mut rng, warmup_n, num_centroids).into_vec();
    let mut cbuf = vec![0.0f32; num_centroids * dim];
    for (dst, &r) in cbuf.chunks_mut(dim).zip(idx.iter()) {
        dst.copy_from_slice(window.row(r));
    }
    let mut centroids = Matrix::try_from(cbuf.into_boxed_slice(), num_centroids, dim)
        .map_err(|_| GraphIvfError::invalid("warmup centroid shape mismatch"))?;

    // Refine with exact Lloyd's (cheap on a small prefix). `iters == 0` leaves
    // the sampled centers untouched.
    if iters > 0 {
        let pool = create_thread_pool(params.num_threads)?;
        let mut assigner = cluster::ExactAssigner::default();
        cluster::lloyd(
            window.as_view(),
            &mut centroids,
            &mut assigner,
            iters,
            EmptyClusterPolicy::PreserveOld,
            params.normalize_centroids,
            &pool,
        )?;
    }
    Ok(centroids)
}

/// One cluster-split event, recorded as it happens during an online build.
///
/// Splits are the only structural events in the build (routing an insert never
/// changes the partition unless it triggers a split), so the ordered list of
/// [`SplitEvent`]s is a complete timeline of how the centroid count grew and how
/// much reassignment work each split cost — enough to reconstruct, for any point
/// in the stream, the live cluster count, cumulative reassignments, and split
/// latency.
#[derive(Debug, Clone, Copy)]
pub struct SplitEvent {
    /// Number of inserts completed (inclusive) when this split fired. Serves as
    /// the build-progress timestamp, in `[1, corpus_len]`.
    pub insert_index: u64,
    /// The centroid id that was split (and retired).
    pub cluster: u32,
    /// Size of the split cluster at split time (the overflow that triggered it).
    pub cluster_size: usize,
    /// Number of live graph-neighbor clusters drawn into the reassignment.
    pub num_neighbors: usize,
    /// Points that actually changed cluster in this split's reassignment pass.
    /// Every member of the split (retired) cluster is counted (each moves to a
    /// child centroid); a pooled neighbor point is counted only if it landed on
    /// a different centroid than it held before.
    pub num_reassigned: usize,
    /// Live centroid count immediately after the split (net `+1`).
    pub live_after: usize,
    /// Wall-clock of the local 2-means, in microseconds.
    pub two_means_us: u64,
    /// Wall-clock of the reassignment pass, in microseconds.
    pub reassign_us: u64,
    /// Wall-clock of the whole split (2-means + graph mutation + reassign), in
    /// microseconds.
    pub total_us: u64,
}

/// Telemetry accumulated over an online build.
///
/// Always collected (the overhead is a few timers per insert). The per-split
/// timeline in [`splits`](Self::splits) is the primary artifact for analyzing
/// how splits and reassignments distribute across the build;
/// [`write_csv`](Self::write_csv) dumps it for offline analysis.
#[derive(Debug, Clone, Default)]
pub struct BuildTelemetry {
    /// Total points inserted.
    pub total_inserts: u64,
    /// Total splits performed.
    pub total_splits: u64,
    /// Total points that actually changed cluster, summed across all splits (a
    /// point moved by two different splits counts twice). Points re-examined by
    /// a split but routed back to their existing centroid are not counted.
    pub total_reassigned: u64,
    /// Cumulative time routing inserts through the centroid graph, microseconds.
    pub routing_us: u64,
    /// Cumulative time in split handling (2-means + graph mutation + reassign),
    /// microseconds.
    pub split_us: u64,
    /// Ordered per-split records (see [`SplitEvent`]).
    pub splits: Vec<SplitEvent>,
}

impl BuildTelemetry {
    /// Write the per-split timeline to `path` as CSV (one row per split, with a
    /// header). Columns match the fields of [`SplitEvent`].
    ///
    /// # Errors
    ///
    /// Returns any I/O error from creating or writing the file.
    pub fn write_csv(&self, path: &Path) -> std::io::Result<()> {
        use std::fmt::Write as _;
        let mut out = String::with_capacity(64 + self.splits.len() * 48);
        out.push_str(
            "insert_index,cluster,cluster_size,num_neighbors,num_reassigned,\
             live_after,two_means_us,reassign_us,total_us\n",
        );
        for e in &self.splits {
            let _ = writeln!(
                out,
                "{},{},{},{},{},{},{},{},{}",
                e.insert_index,
                e.cluster,
                e.cluster_size,
                e.num_neighbors,
                e.num_reassigned,
                e.live_after,
                e.two_means_us,
                e.reassign_us,
                e.total_us,
            );
        }
        std::fs::write(path, out)
    }
}

/// Id-indexed store of centroid vectors with soft deletion.
///
/// A centroid id is permanent: [`alloc`](Self::alloc) hands out the next id and
/// never reuses a retired one, so the table is sized to the whole id budget up
/// front. A `None` slot is a retired (split) centroid; a `Some` slot is live.
struct CentroidTable {
    dim: usize,
    /// Centroid vectors indexed by id; `None` marks a retired centroid.
    vecs: Vec<Option<Box<[f32]>>>,
    /// Number of live (non-retired) centroids.
    live_count: usize,
    /// Next unused centroid id.
    next_id: u32,
}

impl CentroidTable {
    /// Create a table with `capacity` id slots, seeding ids `0..initial.nrows()`
    /// from `initial` and leaving the remaining slots free for [`alloc`].
    ///
    /// [`alloc`]: Self::alloc
    fn new(initial: &Matrix<f32>, capacity: usize) -> Self {
        let initial_k = initial.nrows();
        let mut vecs: Vec<Option<Box<[f32]>>> = Vec::with_capacity(capacity);
        for i in 0..initial_k {
            vecs.push(Some(initial.row(i).to_vec().into_boxed_slice()));
        }
        for _ in initial_k..capacity {
            vecs.push(None);
        }
        Self {
            dim: initial.ncols(),
            vecs,
            live_count: initial_k,
            next_id: initial_k as u32,
        }
    }

    /// Total number of id slots (live + retired + free).
    #[cfg(test)]
    fn capacity(&self) -> usize {
        self.vecs.len()
    }

    /// Number of live (non-retired) centroids.
    fn live_count(&self) -> usize {
        self.live_count
    }

    /// Whether `id` is a live centroid.
    fn is_live(&self, id: u32) -> bool {
        self.vecs.get(id as usize).is_some_and(Option::is_some)
    }

    /// The vector of centroid `id`, or `None` if retired or out of range.
    fn get(&self, id: u32) -> Option<&[f32]> {
        self.vecs.get(id as usize).and_then(|s| s.as_deref())
    }

    /// Whether the id budget can accommodate `n` more [`alloc`](Self::alloc)s.
    fn can_alloc(&self, n: usize) -> bool {
        self.next_id as usize + n <= self.vecs.len()
    }

    /// Allocate the next id, storing `vec` as its centroid.
    ///
    /// # Errors
    ///
    /// Returns an error if the id budget (capacity) is exhausted.
    fn alloc(&mut self, vec: Box<[f32]>) -> Result<u32> {
        let id = self.next_id;
        if (id as usize) >= self.vecs.len() {
            return Err(GraphIvfError::invalid(
                "centroid capacity exceeded; increase centroid_capacity",
            ));
        }
        self.vecs[id as usize] = Some(vec);
        self.next_id += 1;
        self.live_count += 1;
        Ok(id)
    }

    /// Retire centroid `id` (soft delete). Its slot stays occupied but empty; a
    /// no-op if `id` is already retired.
    fn retire(&mut self, id: u32) {
        if self.vecs[id as usize].take().is_some() {
            self.live_count -= 1;
        }
    }

    /// Iterate over live centroids as `(id, vector)` pairs, in ascending id
    /// order.
    fn iter_live(&self) -> impl Iterator<Item = (u32, &[f32])> {
        self.vecs
            .iter()
            .enumerate()
            .filter_map(|(id, slot)| slot.as_deref().map(|v| (id as u32, v)))
    }

    /// Ids of the live centroids, in ascending order.
    fn live_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.vecs
            .iter()
            .enumerate()
            .filter_map(|(id, slot)| slot.as_ref().map(|_| id as u32))
    }

    /// Brute-force nearest live centroid to `point` by squared-L2, or `None` if
    /// no live centroid exists.
    fn nearest(&self, point: &[f32]) -> Option<u32> {
        let mut best = None;
        let mut best_d = f32::INFINITY;
        for (id, v) in self.iter_live() {
            let d = sq_l2(point, v);
            if d < best_d {
                best_d = d;
                best = Some(id);
            }
        }
        best
    }

    /// Nearest candidate centroid to `point` by squared-L2; retired candidates
    /// are skipped. `cands` must be non-empty; if every candidate is retired the
    /// first is returned.
    fn nearest_among(&self, point: &[f32], cands: &[u32]) -> u32 {
        let mut best = cands[0];
        let mut best_d = f32::INFINITY;
        for &cand in cands {
            if let Some(v) = self.get(cand) {
                let d = sq_l2(point, v);
                if d < best_d {
                    best_d = d;
                    best = cand;
                }
            }
        }
        best
    }

    /// Densely pack the live centroids into a contiguous `k x dim` matrix and
    /// return `(remap, matrix)`, where `remap[old_id]` is the new dense index of
    /// a live centroid and [`UNASSIGNED`] for a retired id.
    fn densify(&self) -> Result<(Vec<u32>, Matrix<f32>)> {
        let live: Vec<usize> = (0..self.vecs.len())
            .filter(|&c| self.vecs[c].is_some())
            .collect();
        let k = live.len();
        let mut remap = vec![UNASSIGNED; self.vecs.len()];
        let mut cbuf = vec![0.0f32; k * self.dim];
        for (new, &old) in live.iter().enumerate() {
            remap[old] = new as u32;
            cbuf[new * self.dim..(new + 1) * self.dim]
                .copy_from_slice(self.vecs[old].as_ref().expect("live"));
        }
        let mat = Matrix::try_from(cbuf.into_boxed_slice(), k, self.dim)
            .map_err(|_| GraphIvfError::invalid("centroid matrix shape mismatch"))?;
        Ok((remap, mat))
    }
}

/// The IVF point↔centroid mapping: the id-indexed inverted lists and the
/// reverse per-point assignment, kept consistent through [`assign`](Self::assign).
struct IvfPartition {
    /// `lists[c]` holds the ids of points currently assigned to centroid `c`;
    /// empty for retired ids.
    lists: Vec<Vec<u32>>,
    /// `assignments[pid]` is the centroid id of point `pid`, or [`UNASSIGNED`]
    /// before it is inserted.
    assignments: Vec<u32>,
}

impl IvfPartition {
    /// Create a partition with `capacity` empty inverted lists over `num_points`
    /// initially unassigned points.
    fn new(capacity: usize, num_points: usize) -> Self {
        Self {
            lists: (0..capacity).map(|_| Vec::new()).collect(),
            assignments: vec![UNASSIGNED; num_points],
        }
    }

    /// Assign point `pid` to centroid `cid`, appending it to the inverted list
    /// and updating the reverse map. Returns the previous assignment
    /// ([`UNASSIGNED`] if the point was not yet assigned).
    fn assign(&mut self, pid: u32, cid: u32) -> u32 {
        let prev = self.assignments[pid as usize];
        self.lists[cid as usize].push(pid);
        self.assignments[pid as usize] = cid;
        prev
    }

    /// The current assignment of point `pid`.
    fn assignment(&self, pid: u32) -> u32 {
        self.assignments[pid as usize]
    }

    /// The members (point ids) currently in centroid `cid`'s inverted list.
    fn members(&self, cid: u32) -> &[u32] {
        &self.lists[cid as usize]
    }

    /// Number of points in centroid `cid`'s inverted list.
    fn list_len(&self, cid: u32) -> usize {
        self.lists[cid as usize].len()
    }

    /// Remove and return centroid `cid`'s inverted list, leaving it empty. The
    /// reverse assignments of those points are left untouched — the caller is
    /// expected to reassign every returned point.
    fn take_members(&mut self, cid: u32) -> Vec<u32> {
        std::mem::take(&mut self.lists[cid as usize])
    }

    /// Restore centroid `cid`'s inverted list, rolling back a prior
    /// [`take_members`](Self::take_members).
    fn restore_members(&mut self, cid: u32, members: Vec<u32>) {
        self.lists[cid as usize] = members;
    }
}

/// An incremental graph-IVF clusterer driven by point insertion with
/// split-and-reassign cluster maintenance.
pub struct OnlineClusterer {
    /// The full corpus, preloaded; row `pid` is point `pid`.
    points: Matrix<f32>,
    dim: usize,
    params: OnlineParams,

    /// Id-indexed centroid store with soft deletion (retirement on split).
    table: CentroidTable,
    /// The point↔centroid mapping (inverted lists plus the reverse map).
    partition: IvfPartition,

    /// Mutable centroid graph (L2 navigation) used to route inserts.
    graph: MutableCentroidGraph,
    /// Current-thread runtime driving the graph search/insert/delete calls.
    runtime: Runtime,
    /// Thread pool for the internal 2-means.
    pool: RayonThreadPool,
    rng: StdRng,

    /// Build telemetry (splits, reassignments, latencies).
    telemetry: BuildTelemetry,

    /// Scratch reused by split reassignment: the candidate-point buffer.
    scratch_pool: Vec<u32>,
    /// Scratch reused by split reassignment: the candidate-centroid buffer.
    scratch_cands: Vec<u32>,
    /// Scratch reused for the nearest-centroid search that selects reassignment
    /// candidate clusters: the centroid-id buffer.
    scratch_neighbors: Vec<u32>,
    /// Scratch reused for the nearest-centroid search: the distance buffer.
    scratch_dist: Vec<f32>,
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
        if params.num_threads == 0 {
            return Err(GraphIvfError::invalid("num_threads must be non-zero"));
        }

        // `centroid_capacity` is the total id budget (live + retired); size the
        // graph and the id-indexed vectors to exactly that.
        let capacity = params.centroid_capacity.max(initial_k);

        let init_mat = Matrix::try_from(
            initial.as_slice().to_vec().into_boxed_slice(),
            initial_k,
            dim,
        )
        .map_err(|_| GraphIvfError::invalid("initial centroid matrix shape mismatch"))?;
        let graph = centroids::build_mutable(
            init_mat,
            &params.graph,
            params.num_threads,
            capacity,
            VectorMetric::L2,
        )?;

        let table = CentroidTable::new(&initial, capacity);
        let partition = IvfPartition::new(capacity, num_points);

        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .map_err(ANNError::from)?;
        let pool = create_thread_pool(params.num_threads)?;

        Ok(Self {
            points,
            dim,
            params,
            table,
            partition,
            graph,
            runtime,
            pool,
            rng: StdRng::seed_from_u64(params.seed),
            telemetry: BuildTelemetry::default(),
            scratch_pool: Vec::new(),
            scratch_cands: Vec::new(),
            scratch_neighbors: Vec::new(),
            scratch_dist: Vec::new(),
        })
    }

    /// Number of live clusters.
    pub fn num_clusters(&self) -> usize {
        self.table.live_count()
    }

    /// Read-only access to the build telemetry accumulated so far (splits,
    /// reassignments, latencies).
    pub fn telemetry(&self) -> &BuildTelemetry {
        &self.telemetry
    }

    /// Current size of every live cluster (points assigned to it), in no
    /// particular order. Useful for inspecting the final size distribution.
    pub fn cluster_sizes(&self) -> Vec<usize> {
        self.table
            .live_ids()
            .map(|cid| self.partition.list_len(cid))
            .collect()
    }

    /// Clustering residual: the sum of squared distances from every assigned
    /// point to its centroid. Lower is a tighter clustering.
    pub fn residual(&self) -> f64 {
        let mut sum = 0.0f64;
        for (cid, cv) in self.table.iter_live() {
            for &pid in self.partition.members(cid) {
                sum += sq_l2(self.points.row(pid as usize), cv) as f64;
            }
        }
        sum
    }

    /// Insert point `pid`: route it to its nearest centroid, then split that
    /// cluster (with neighborhood reassignment) if it exceeds the threshold.
    pub fn insert(&mut self, pid: u32) -> Result<()> {
        let p = pid as usize;
        if p >= self.points.nrows() {
            return Err(GraphIvfError::invalid("point id out of range"));
        }

        let route_start = Instant::now();
        let c = self.assign_nearest(pid)?;
        self.telemetry.routing_us += route_start.elapsed().as_micros() as u64;

        self.partition.assign(pid, c);
        self.telemetry.total_inserts += 1;

        // Split when the cluster overflows, unless an explicit live-cluster cap
        // is set and reached, or the id budget cannot fit two more children.
        let under_cap = self
            .params
            .max_clusters
            .is_none_or(|k| self.table.live_count() < k);
        let id_budget_ok = self.table.can_alloc(2);
        if self.partition.list_len(c) > self.params.split_threshold && under_cap && id_budget_ok {
            self.split(c)?;
        }
        Ok(())
    }

    /// Route point `pid` to its nearest live centroid via the centroid graph.
    ///
    /// The mutable centroid graph accumulates soft-deleted (tombstoned) slots as
    /// clusters split — near the target cluster count roughly half the graph can
    /// be tombstones. A narrow beam can then occasionally exhaust its frontier on
    /// tombstoned nodes and return no live centroid, so we widen the search list
    /// before giving up and, as a last resort, fall back to a brute-force scan
    /// over the live centroids. Successful narrow-beam routes are unchanged.
    fn assign_nearest(&self, pid: u32) -> Result<u32> {
        let mut ids = [0u32; 1];
        let mut dist = [0.0f32; 1];
        let base_l = self.params.assign_l.max(1);
        let wide_l = base_l.saturating_mul(8).max(512);
        for l in [base_l, wide_l] {
            let n = centroids::search_mut(
                &self.graph,
                &self.runtime,
                self.points.row(pid as usize),
                l,
                &mut ids,
                &mut dist,
            )?;
            if n > 0 {
                return Ok(ids[0]);
            }
        }
        self.table
            .nearest(self.points.row(pid as usize))
            .ok_or_else(|| GraphIvfError::invalid("no live centroid available for assignment"))
    }

    /// Split cluster `c` into two child centroids via a local 2-means, then
    /// reassign the points of `c` and of its `reassign_neighbors` nearest
    /// centroid clusters among the two children and those neighboring centroids.
    fn split(&mut self, c: u32) -> Result<()> {
        let split_start = Instant::now();

        // Take C's members out; if too small to split, restore and bail.
        let members = self.partition.take_members(c);
        if members.len() < 2 {
            self.partition.restore_members(c, members);
            return Ok(());
        }
        let cluster_size = members.len();

        // 1. Two child centroids from a local 2-means over C.
        let two_means_start = Instant::now();
        let two = self.two_means(&members)?;
        let two_means_us = two_means_start.elapsed().as_micros() as u64;
        let child1: Box<[f32]> = two.row(0).to_vec().into_boxed_slice();
        let child2: Box<[f32]> = two.row(1).to_vec().into_boxed_slice();

        // 2. Candidate clusters: the `s` live centroids nearest to `c`, found by
        //    searching `c`'s own centroid vector in the centroid graph. The
        //    search runs before `c` is deleted and the children inserted, so it
        //    sees the pre-split centroids; `c` itself (distance 0) is dropped
        //    from the results below. `k = s + 1` reserves a slot for `c`.
        let c_vec: Box<[f32]> = self
            .table
            .get(c)
            .expect("splitting a live centroid")
            .to_vec()
            .into_boxed_slice();
        let s = self.params.reassign_neighbors;
        let k = s + 1;
        let l = self.params.reassign_l.max(k);
        let mut neighbors = std::mem::take(&mut self.scratch_neighbors);
        let mut dist = std::mem::take(&mut self.scratch_dist);
        neighbors.clear();
        neighbors.resize(k, 0);
        dist.clear();
        dist.resize(k, 0.0);
        let found = centroids::search_mut(
            &self.graph,
            &self.runtime,
            &c_vec,
            l,
            &mut neighbors,
            &mut dist,
        )?;
        neighbors.truncate(found);
        neighbors.retain(|&x| x != c && self.table.is_live(x));
        neighbors.truncate(s);
        let num_neighbors = neighbors.len();

        // 3. Allocate the two child ids and retire the parent.
        let id1 = self.table.alloc(child1)?;
        let id2 = self.table.alloc(child2)?;
        self.table.retire(c);

        // 4. Mutate the graph: delete c, insert the two children.
        centroids::delete_centroid(&self.graph, &self.runtime, c)?;
        {
            let v1 = self.table.get(id1).expect("just set");
            centroids::insert_centroid(&self.graph, &self.runtime, id1, v1)?;
        }
        {
            let v2 = self.table.get(id2).expect("just set");
            centroids::insert_centroid(&self.graph, &self.runtime, id2, v2)?;
        }

        // 5. Candidate centroids = neighbors ∪ {id1, id2}.
        let mut cands = std::mem::take(&mut self.scratch_cands);
        cands.clear();
        cands.extend_from_slice(&neighbors);
        cands.push(id1);
        cands.push(id2);

        // 6. Candidate points = C ∪ (points of every neighbor cluster).
        let mut pool = std::mem::take(&mut self.scratch_pool);
        pool.clear();
        pool.extend_from_slice(&members);
        for &nc in &neighbors {
            let taken = self.partition.take_members(nc);
            pool.extend_from_slice(&taken);
        }
        // c's list is already empty (members were taken); keep it empty.

        // 7. Reassign every pooled point to its nearest candidate centroid and
        //    rebuild the affected inverted lists. Only points that actually land
        //    in a different cluster than before count as reassigned; a point
        //    routed back to the same centroid it already held does not. (Every
        //    member of the retired cluster `c` necessarily moves, since `c` is
        //    not among the candidates.)
        let reassign_start = Instant::now();
        let mut num_reassigned = 0usize;
        for &pid in &pool {
            let best = self
                .table
                .nearest_among(self.points.row(pid as usize), &cands);
            let prev = self.partition.assign(pid, best);
            if best != prev {
                num_reassigned += 1;
            }
        }
        let reassign_us = reassign_start.elapsed().as_micros() as u64;

        // Return the scratch buffers for reuse.
        self.scratch_pool = pool;
        self.scratch_cands = cands;
        self.scratch_neighbors = neighbors;
        self.scratch_dist = dist;

        // Record the split in the build telemetry.
        let total_us = split_start.elapsed().as_micros() as u64;
        self.telemetry.total_splits += 1;
        self.telemetry.total_reassigned += num_reassigned as u64;
        self.telemetry.split_us += total_us;
        self.telemetry.splits.push(SplitEvent {
            insert_index: self.telemetry.total_inserts,
            cluster: c,
            cluster_size,
            num_neighbors,
            num_reassigned,
            live_after: self.table.live_count(),
            two_means_us,
            reassign_us,
            total_us,
        });
        Ok(())
    }

    /// Run a 2-means over the given member points, returning the two child
    /// centroids as a `2 x dim` matrix.
    fn two_means(&mut self, members: &[u32]) -> Result<Matrix<f32>> {
        let dim = self.dim;
        let m = members.len();
        debug_assert!(m >= 2);

        let mut buf = vec![0.0f32; m * dim];
        for (i, &pid) in members.iter().enumerate() {
            buf[i * dim..(i + 1) * dim].copy_from_slice(self.points.row(pid as usize));
        }
        let sub = Matrix::try_from(buf.into_boxed_slice(), m, dim)
            .map_err(|_| GraphIvfError::invalid("split sub-matrix shape mismatch"))?;

        // Seed with two distinct member points.
        let a = self.rng.random_range(0..m);
        let mut b = self.rng.random_range(0..m);
        if b == a {
            b = (a + 1) % m;
        }
        let mut seed = vec![0.0f32; 2 * dim];
        seed[0..dim].copy_from_slice(sub.row(a));
        seed[dim..2 * dim].copy_from_slice(sub.row(b));
        let mut centroids = Matrix::try_from(seed.into_boxed_slice(), 2, dim)
            .map_err(|_| GraphIvfError::invalid("split seed shape mismatch"))?;

        let mut assigner = cluster::ExactAssigner::default();
        cluster::lloyd(
            sub.as_view(),
            &mut centroids,
            &mut assigner,
            self.params.two_means_iters.max(1),
            EmptyClusterPolicy::PreserveOld,
            self.params.normalize_centroids,
            &self.pool,
        )?;
        Ok(centroids)
    }

    /// Serialize the in-memory IVF mapping to `prefix` in the batch on-disk
    /// format (`.graphivf_centroids.fbin`, `.graphivf_lists`, `.graphivf_meta`),
    /// densely remapping live centroid ids to `0..num_clusters`.
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
    /// Returns an error if `stored`'s row count does not match the corpus, or if
    /// any corpus point has not been inserted yet.
    pub fn flush<T: VectorRepr>(&self, prefix: &Path, stored: MatrixView<'_, T>) -> Result<()> {
        let num_points = self.points.nrows();
        if stored.nrows() != num_points {
            return Err(GraphIvfError::invalid(format!(
                "stored corpus has {} rows but clustering corpus has {num_points}",
                stored.nrows()
            )));
        }

        // Dense remap of live centroid ids to a contiguous 0..k range.
        let (remap, centroids_mat) = self.table.densify()?;
        let k = centroids_mat.nrows();

        // Dense per-point assignments.
        let mut dense = vec![0u32; num_points];
        for (pid, slot) in dense.iter_mut().enumerate() {
            let c = self.partition.assignment(pid as u32);
            if c == UNASSIGNED {
                return Err(GraphIvfError::invalid(
                    "cannot flush: some points have not been inserted",
                ));
            }
            *slot = remap[c as usize];
        }

        // Write centroids (always f32).
        let centroids_path = with_suffix(prefix, CENTROIDS_SUFFIX);
        storage::write_centroids(&centroids_path, centroids_mat.as_view())?;

        // Write inverted lists from the stored representation and the metadata.
        let stored_dim = stored.ncols();
        let lists_path = with_suffix(prefix, LISTS_SUFFIX);
        let (counts, offsets) = storage::write_lists_stored::<T>(&lists_path, stored, &dense, k)?;
        let layout = Layout {
            dim: stored_dim,
            metric: self.params.metric,
            element_size: std::mem::size_of::<T>(),
            num_points: num_points as u64,
            graph: self.params.graph,
            counts,
            offsets,
        };
        storage::write_metadata(&with_suffix(prefix, META_SUFFIX), &layout)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraphIvfIndex, GraphParams, Metric, SearchParams};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    fn mat(data: Vec<f32>, nrows: usize, ncols: usize) -> Matrix<f32> {
        Matrix::try_from(data.into_boxed_slice(), nrows, ncols).unwrap()
    }

    fn params(target: usize, threshold: usize) -> OnlineParams {
        OnlineParams {
            max_clusters: Some(target),
            centroid_capacity: target.saturating_mul(2).max(1),
            split_threshold: threshold,
            assign_l: 32,
            reassign_neighbors: 8,
            reassign_l: 32,
            two_means_iters: 10,
            graph: GraphParams::default(),
            metric: Metric::L2,
            normalize_centroids: false,
            num_threads: 2,
            seed: 0,
        }
    }

    /// Two well-separated Gaussian-ish blobs in 2D.
    fn two_blobs(per: usize, seed: u64) -> (Matrix<f32>, usize) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut v = Vec::new();
        for _ in 0..per {
            v.push(rng.random_range(-1.0..1.0));
            v.push(rng.random_range(-1.0..1.0));
        }
        for _ in 0..per {
            v.push(20.0 + rng.random_range(-1.0..1.0));
            v.push(20.0 + rng.random_range(-1.0..1.0));
        }
        let n = per * 2;
        (mat(v, n, 2), n)
    }

    /// Brute-force squared-L2.
    fn sqd(a: &[f32], b: &[f32]) -> f64 {
        a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum()
    }

    /// Optimal residual for a fixed centroid set: every point to its globally
    /// nearest centroid. The online (local) assignment can only be >= this.
    fn optimal_residual(points: &Matrix<f32>, centroids: &[Box<[f32]>]) -> f64 {
        let mut sum = 0.0;
        for p in 0..points.nrows() {
            let row = points.row(p);
            let best = centroids
                .iter()
                .map(|c| sqd(row, c))
                .fold(f64::INFINITY, f64::min);
            sum += best;
        }
        sum
    }

    fn live_centroids(c: &OnlineClusterer) -> Vec<Box<[f32]>> {
        c.table
            .iter_live()
            .map(|(_, v)| v.to_vec().into_boxed_slice())
            .collect()
    }

    // ----- centroid-graph mutable ops -----

    #[test]
    fn mutable_graph_insert_delete_search() {
        // Four centroids at distinct corners; capacity leaves room to insert.
        let cents = mat(vec![0.0, 0.0, 10.0, 0.0, 0.0, 10.0, 10.0, 10.0], 4, 2);
        let graph =
            centroids::build_mutable(cents, &GraphParams::default(), 2, 8, VectorMetric::L2)
                .unwrap();
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();

        let mut ids = [0u32; 1];
        let mut dist = [0.0f32; 1];

        // Query near centroid 3 (10,10) -> returns 3.
        centroids::search_mut(&graph, &rt, &[9.5, 9.5], 8, &mut ids, &mut dist).unwrap();
        assert_eq!(ids[0], 3);

        // Delete centroid 3; the same query now returns a different live one.
        centroids::delete_centroid(&graph, &rt, 3).unwrap();
        centroids::search_mut(&graph, &rt, &[9.5, 9.5], 8, &mut ids, &mut dist).unwrap();
        assert_ne!(ids[0], 3);

        // Insert a new centroid (id 4) right at the query; it wins.
        centroids::insert_centroid(&graph, &rt, 4, &[9.5, 9.5]).unwrap();
        centroids::search_mut(&graph, &rt, &[9.5, 9.5], 8, &mut ids, &mut dist).unwrap();
        assert_eq!(ids[0], 4);
    }

    // ----- clusterer invariants -----

    /// Every inserted point is accounted for exactly once in a live cluster.
    fn assert_invariants(c: &OnlineClusterer, inserted: usize) {
        // live_count matches the centroid table.
        let live = c.table.iter_live().count();
        assert_eq!(live, c.table.live_count());
        if let Some(k) = c.params.max_clusters {
            assert!(c.table.live_count() <= k);
        }
        assert!(c.table.live_count() <= c.table.capacity());

        // Sum of live list lengths == inserted count; retired ids hold nothing.
        let mut total = 0usize;
        for cid in 0..c.table.capacity() as u32 {
            if c.table.is_live(cid) {
                total += c.partition.list_len(cid);
            } else {
                assert!(
                    c.partition.members(cid).is_empty(),
                    "retired centroid has points"
                );
            }
        }
        assert_eq!(total, inserted);

        // Every assigned point points to a live centroid.
        for pid in 0..inserted {
            let a = c.partition.assignment(pid as u32);
            assert_ne!(a, UNASSIGNED);
            assert!(c.table.is_live(a));
        }
    }

    #[test]
    fn no_split_matches_nearest_centroid() {
        // High threshold => no splits; pure online assignment with fixed
        // centroids. Residual must equal the optimal for those centroids.
        let (points, n) = two_blobs(40, 1);
        let initial = mat(vec![0.0, 0.0, 20.0, 20.0], 2, 2);
        let mut c = OnlineClusterer::new(points.clone(), initial, params(2, 10_000)).unwrap();
        for pid in 0..n as u32 {
            c.insert(pid).unwrap();
        }
        assert_invariants(&c, n);
        assert_eq!(c.num_clusters(), 2);

        let opt = optimal_residual(&points, &live_centroids(&c));
        // Graph routing is approximate, but for two far-apart blobs it is exact.
        assert!(
            (c.residual() - opt).abs() < 1e-3,
            "res={} opt={}",
            c.residual(),
            opt
        );
    }

    #[test]
    fn split_creates_cluster_and_tightens() {
        // Start with ONE centroid; a low threshold forces a split of the single
        // overfull cluster into the two blobs. Points are streamed in shuffled
        // order so both blobs are represented by the time the split fires.
        let (points, n) = two_blobs(60, 2);
        let initial = mat(vec![10.0, 10.0], 1, 2);
        let mut c = OnlineClusterer::new(points.clone(), initial, params(2, 30)).unwrap();

        let mut order: Vec<u32> = (0..n as u32).collect();
        let mut rng = StdRng::seed_from_u64(99);
        for i in (1..order.len()).rev() {
            order.swap(i, rng.random_range(0..=i));
        }
        for &pid in &order {
            c.insert(pid).unwrap();
        }
        assert_invariants(&c, n);
        assert_eq!(
            c.num_clusters(),
            2,
            "the overfull cluster should have split"
        );

        // With two centroids at the blob centers the residual is far below the
        // single-centroid residual, and never below the optimal-for-2.
        let opt2 = optimal_residual(&points, &live_centroids(&c));
        assert!(c.residual() >= opt2 - 1e-3);
        // Sanity: two tight blobs => small residual per point.
        assert!(
            c.residual() / (n as f64) < 5.0,
            "residual too large: {}",
            c.residual()
        );
    }

    #[test]
    fn many_splits_preserve_invariants_and_bound_residual() {
        // Random data, several initial centroids, many splits.
        let mut rng = StdRng::seed_from_u64(7);
        let (nn, dim) = (600usize, 8usize);
        let mut v = vec![0.0f32; nn * dim];
        for x in v.iter_mut() {
            *x = rng.random_range(-1.0..1.0);
        }
        let points = mat(v, nn, dim);

        // 4 initial centroids drawn from the data.
        let mut ib = vec![0.0f32; 4 * dim];
        for i in 0..4 {
            let src = rng.random_range(0..nn);
            ib[i * dim..(i + 1) * dim].copy_from_slice(points.row(src));
        }
        let initial = mat(ib, 4, dim);

        let mut c = OnlineClusterer::new(points.clone(), initial, params(16, 40)).unwrap();
        for pid in 0..nn as u32 {
            c.insert(pid).unwrap();
        }
        assert_invariants(&c, nn);
        assert!(c.num_clusters() > 4, "expected some splits to occur");
        assert!(c.num_clusters() <= 16);

        // Online (local) residual is never below the optimal assignment for the
        // same centroid set.
        let opt = optimal_residual(&points, &live_centroids(&c));
        assert!(
            c.residual() >= opt - 1e-3,
            "res={} opt={}",
            c.residual(),
            opt
        );
    }

    #[test]
    fn uncapped_splits_until_threshold_equilibrium() {
        // `max_clusters: None` removes the live-cluster ceiling: splitting is
        // driven purely by the threshold and continues for every point, so the
        // count grows well past any small fixed target and mean cluster size
        // settles near the split threshold.
        let mut rng = StdRng::seed_from_u64(11);
        let (nn, dim) = (800usize, 6usize);
        let mut v = vec![0.0f32; nn * dim];
        for x in v.iter_mut() {
            *x = rng.random_range(-1.0..1.0);
        }
        let points = mat(v, nn, dim);
        let initial = mat(points.row(0).to_vec(), 1, dim);

        let mut p = params(1, 20);
        p.max_clusters = None; // uncapped: threshold-driven only
        p.centroid_capacity = 4 * nn; // generous id budget, never binds

        let mut c = OnlineClusterer::new(points, initial, p).unwrap();
        for pid in 0..nn as u32 {
            c.insert(pid).unwrap();
        }
        assert_invariants(&c, nn);

        // Far more than the single seed centroid; roughly `~ 2 * nn / threshold`.
        assert!(c.num_clusters() > 10, "got {}", c.num_clusters());
        let mean = nn as f64 / c.num_clusters() as f64;
        assert!(mean <= 21.0, "mean cluster size {mean} exceeds threshold");
    }

    #[test]
    fn telemetry_records_splits_and_reassignments() {
        // A split-heavy run records one telemetry event per split, with a
        // monotonic insert-index timeline and sane counters.
        let (points, n) = two_blobs(60, 21);
        let initial = mat(vec![10.0, 10.0], 1, 2);
        let mut c = OnlineClusterer::new(points, initial, params(8, 25)).unwrap();

        let mut order: Vec<u32> = (0..n as u32).collect();
        let mut rng = StdRng::seed_from_u64(5);
        for i in (1..order.len()).rev() {
            order.swap(i, rng.random_range(0..=i));
        }
        for &pid in &order {
            c.insert(pid).unwrap();
        }

        let t = c.telemetry();
        assert_eq!(t.total_inserts, n as u64);
        assert!(t.total_splits >= 1, "expected at least one split");
        assert_eq!(t.splits.len() as u64, t.total_splits);

        // Per-split records are consistent and ordered in build time.
        let mut prev = 0u64;
        let mut reassigned_sum = 0u64;
        for e in &t.splits {
            assert!(
                e.insert_index >= prev,
                "insert_index must be non-decreasing"
            );
            assert!(e.insert_index >= 1 && e.insert_index <= n as u64);
            prev = e.insert_index;
            assert!(e.cluster_size >= 2);
            assert!(e.num_reassigned >= e.cluster_size); // all of C always moves
            reassigned_sum += e.num_reassigned as u64;
        }
        assert_eq!(reassigned_sum, t.total_reassigned);
        assert_eq!(
            t.splits.last().unwrap().live_after,
            c.num_clusters(),
            "last split's live_after should match the final cluster count"
        );

        // Cluster sizes cover every live cluster and sum to the corpus.
        let sizes = c.cluster_sizes();
        assert_eq!(sizes.len(), c.num_clusters());
        assert_eq!(sizes.iter().sum::<usize>(), n);

        // CSV export writes a header plus one row per split.
        let dir = std::env::temp_dir().join(format!("graphivf_tel_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let csv = dir.join("splits.csv");
        t.write_csv(&csv).unwrap();
        let text = std::fs::read_to_string(&csv).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[0].starts_with("insert_index,cluster,cluster_size"));
        assert_eq!(lines.len(), 1 + t.splits.len());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn flush_roundtrips_through_load_and_search() {
        let (points, n) = two_blobs(50, 3);
        let initial = mat(vec![10.0, 10.0], 1, 2);
        let mut c = OnlineClusterer::new(points.clone(), initial, params(2, 25)).unwrap();
        for pid in 0..n as u32 {
            c.insert(pid).unwrap();
        }
        assert_eq!(c.num_clusters(), 2);

        let dir = std::env::temp_dir().join(format!("graphivf_online_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let prefix = dir.join("idx");
        c.flush(&prefix, c.points.as_view()).unwrap();

        let index = GraphIvfIndex::<f32>::load(&prefix, 2).unwrap();
        assert_eq!(index.num_clusters(), 2);
        let mut searcher = index.searcher().unwrap();

        // A query in blob 0 should retrieve blob-0 points (small distances).
        let sp = SearchParams {
            nlist: 2,
            centroid_search_l: 8,
        };
        let results = searcher.search(&[0.0f32, 0.0], 5, &sp).unwrap();
        assert!(!results.is_empty());
        // Nearest neighbor is within blob 0 (distance well under the blob gap).
        assert!(
            results[0].1 < 25.0,
            "nn distance {} too large",
            results[0].1
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_bad_params() {
        let (points, _) = two_blobs(10, 4);
        // centroid_capacity < initial (params maps target -> capacity = 2*target,
        // so target 1 gives capacity 2 < the 3 initial centroids).
        let initial = mat(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0], 3, 2);
        assert!(OnlineClusterer::new(points.clone(), initial, params(1, 10)).is_err());
        // threshold < 2
        let initial = mat(vec![0.0, 0.0], 1, 2);
        assert!(OnlineClusterer::new(points, initial, params(4, 1)).is_err());
    }

    // ----- seeding -----

    #[test]
    fn warmup_seed_bootstraps_centroids() {
        // Warmup k-means over a prefix of two well-separated blobs recovers a
        // sensible starting partition, and streaming continues from it.
        let (points, n) = two_blobs(80, 11);
        let seed = SeedStrategy::Warmup {
            num_centroids: 4,
            warmup_points: 60,
            iters: 15,
        };
        let mut c = OnlineClusterer::with_seed(points.clone(), seed, params(8, 10_000)).unwrap();
        // The clusterer starts with exactly the requested centroids.
        assert_eq!(c.num_clusters(), 4);

        for pid in 0..n as u32 {
            c.insert(pid).unwrap();
        }
        assert_invariants(&c, n);

        // Warmed-up centroids sit inside the blobs, so the residual matches the
        // optimal assignment for that centroid set (no split here).
        let opt = optimal_residual(&points, &live_centroids(&c));
        assert!(
            c.residual() >= opt - 1e-3,
            "res={} opt={}",
            c.residual(),
            opt
        );
    }

    #[test]
    fn warmup_zero_iters_uses_sampled_points() {
        // With iters == 0 the sampled prefix points are used verbatim as
        // centroids (no refinement), and every centroid is a real corpus point.
        let (points, _) = two_blobs(40, 12);
        let seed = SeedStrategy::Warmup {
            num_centroids: 3,
            warmup_points: 20,
            iters: 0,
        };
        let c = OnlineClusterer::with_seed(points.clone(), seed, params(8, 10_000)).unwrap();
        assert_eq!(c.num_clusters(), 3);
        for (_, cv) in c.table.iter_live() {
            let is_corpus_point = (0..points.nrows()).any(|r| points.row(r) == cv);
            assert!(is_corpus_point, "unrefined centroid must be a corpus point");
        }
    }

    #[test]
    fn explicit_seed_matches_new() {
        // SeedStrategy::Explicit is a pass-through equivalent to `new`.
        let (points, _) = two_blobs(10, 13);
        let initial = mat(vec![0.0, 0.0, 20.0, 20.0], 2, 2);
        let c =
            OnlineClusterer::with_seed(points, SeedStrategy::Explicit(initial), params(4, 10_000))
                .unwrap();
        assert_eq!(c.num_clusters(), 2);
    }

    #[test]
    fn warmup_rejects_bad_config() {
        let (points, _) = two_blobs(10, 14); // 20 points
                                             // more centroids than points
        let seed = SeedStrategy::Warmup {
            num_centroids: 100,
            warmup_points: 10,
            iters: 5,
        };
        assert!(OnlineClusterer::with_seed(points.clone(), seed, params(200, 10)).is_err());
        // zero centroids
        let seed = SeedStrategy::Warmup {
            num_centroids: 0,
            warmup_points: 10,
            iters: 5,
        };
        assert!(OnlineClusterer::with_seed(points, seed, params(8, 10)).is_err());
    }
}

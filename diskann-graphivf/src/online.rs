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
use diskann_utils::{
    io::write_bin,
    views::{Matrix, MatrixView},
};
use diskann_vector::distance::Metric as VectorMetric;
use rand::{rngs::StdRng, Rng, SeedableRng};
use tokio::runtime::Runtime;

use crate::{
    centroids::{self, MutableCentroidGraph},
    cluster::{self, sq_l2},
    index::{with_suffix, CENTROIDS_SUFFIX, LISTS_SUFFIX, META_SUFFIX},
    params::{EmptyClusterPolicy, GraphParams, Metric},
    storage::{self, Layout},
    GraphIvfError, Result,
};

use diskann::{utils::VectorRepr, ANNError};

/// Sentinel in [`OnlineClusterer::assignments`] for a point that has not been
/// inserted yet.
const UNASSIGNED: u32 = u32::MAX;

/// Parameters for an [`OnlineClusterer`].
#[derive(Debug, Clone, Copy)]
pub struct OnlineParams {
    /// Target number of live clusters. Splitting stops once this many clusters
    /// exist; also sets the centroid-graph capacity (`2 * target_clusters`, to
    /// cover every id soft-deletion leaves behind).
    pub target_clusters: usize,
    /// A cluster is split once it holds strictly more than this many points.
    /// Must be `>= 2`.
    pub split_threshold: usize,
    /// Centroid-graph search-list size used to route each inserted point.
    pub assign_l: usize,
    /// Number of 2-means iterations used to split a cluster.
    pub two_means_iters: usize,
    /// Centroid-graph construction parameters.
    pub graph: GraphParams,
    /// Metric recorded in the flushed index metadata. Clustering and graph
    /// navigation always use squared-L2 (as in a batch build); this only
    /// controls how the *loaded* index scores at search time.
    pub metric: Metric,
    /// L2-normalize the two child centroids after a split (for unit-normalized
    /// corpora).
    pub normalize_centroids: bool,
    /// Run a final refinement pass before flushing: replace every live centroid
    /// with the mean of the points currently assigned to it (L2-normalized when
    /// [`normalize_centroids`](Self::normalize_centroids) is set). This is
    /// applied by [`OnlineClusterer::refine_centroids`], not automatically.
    pub refine_centroids: bool,
    /// Worker threads for the internal 2-means and graph construction.
    pub num_threads: usize,
    /// RNG seed for split seeding (reproducibility).
    pub seed: u64,
}

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
    /// Total points reassigned: the split cluster plus every neighbor cluster.
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
    /// Total point-reassignments summed across all splits (a point moved by two
    /// different splits counts twice).
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

/// An incremental graph-IVF clusterer driven by point insertion with
/// split-and-reassign cluster maintenance.
pub struct OnlineClusterer {
    /// The full corpus, preloaded; row `pid` is point `pid`.
    points: Matrix<f32>,
    dim: usize,
    params: OnlineParams,

    /// Centroid vectors indexed by centroid id; `None` marks a soft-deleted
    /// (split) centroid whose id is retired.
    centroid_vecs: Vec<Option<Box<[f32]>>>,
    /// Inverted lists: `lists[c]` holds the ids of points currently assigned to
    /// centroid `c`. Empty for retired ids.
    lists: Vec<Vec<u32>>,
    /// Reverse map: `assignments[pid]` is the centroid id of point `pid`, or
    /// [`UNASSIGNED`] before it is inserted.
    assignments: Vec<u32>,
    /// Number of live (non-retired) centroids.
    live_count: usize,
    /// Next unused centroid id.
    next_id: u32,

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
    /// Scratch reused for reading centroid graph neighbors.
    scratch_neighbors: Vec<u32>,
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
    /// Returns an error if the shapes are inconsistent, `target_clusters` is
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
        if params.target_clusters < initial_k {
            return Err(GraphIvfError::invalid(format!(
                "target_clusters ({}) is smaller than the initial centroid count ({initial_k})",
                params.target_clusters
            )));
        }
        if params.split_threshold < 2 {
            return Err(GraphIvfError::invalid("split_threshold must be >= 2"));
        }
        if params.num_threads == 0 {
            return Err(GraphIvfError::invalid("num_threads must be non-zero"));
        }

        // Every split retires one id and allocates two, so the number of ids
        // ever used is bounded by `2 * target_clusters`; size the graph and the
        // id-indexed vectors to that.
        let capacity = params.target_clusters.saturating_mul(2).max(initial_k);

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

        let mut centroid_vecs: Vec<Option<Box<[f32]>>> = Vec::with_capacity(capacity);
        let mut lists: Vec<Vec<u32>> = Vec::with_capacity(capacity);
        for i in 0..initial_k {
            centroid_vecs.push(Some(initial.row(i).to_vec().into_boxed_slice()));
            lists.push(Vec::new());
        }
        for _ in initial_k..capacity {
            centroid_vecs.push(None);
            lists.push(Vec::new());
        }

        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .map_err(ANNError::from)?;
        let pool = create_thread_pool(params.num_threads)?;

        Ok(Self {
            points,
            dim,
            params,
            centroid_vecs,
            lists,
            assignments: vec![UNASSIGNED; num_points],
            live_count: initial_k,
            next_id: initial_k as u32,
            graph,
            runtime,
            pool,
            rng: StdRng::seed_from_u64(params.seed),
            telemetry: BuildTelemetry::default(),
            scratch_pool: Vec::new(),
            scratch_cands: Vec::new(),
            scratch_neighbors: Vec::new(),
        })
    }

    /// Number of live clusters.
    pub fn num_clusters(&self) -> usize {
        self.live_count
    }

    /// Read-only access to the build telemetry accumulated so far (splits,
    /// reassignments, latencies).
    pub fn telemetry(&self) -> &BuildTelemetry {
        &self.telemetry
    }

    /// Current size of every live cluster (points assigned to it), in no
    /// particular order. Useful for inspecting the final size distribution.
    pub fn cluster_sizes(&self) -> Vec<usize> {
        self.centroid_vecs
            .iter()
            .enumerate()
            .filter_map(|(cid, slot)| slot.as_ref().map(|_| self.lists[cid].len()))
            .collect()
    }

    /// Final refinement pass: replace every live centroid with the mean of the
    /// points currently assigned to it, L2-normalized when
    /// [`OnlineParams::normalize_centroids`] is set.
    ///
    /// The streaming build routes and splits incrementally, so a centroid can
    /// drift away from the exact mean of its *final* membership. Recomputing
    /// each centroid as that mean — one batch Lloyd update at fixed assignments —
    /// tightens the clustering (never increases the L2 residual for the current
    /// partition) without moving any point to a different cluster. Empty
    /// clusters are left unchanged.
    ///
    /// Only the centroid vectors change; the inverted lists (and thus the
    /// flushed partition) are untouched. Call this after streaming and before
    /// [`flush`](Self::flush).
    pub fn refine_centroids(&mut self) {
        let dim = self.dim;
        let normalize = self.params.normalize_centroids;
        let mut mean = vec![0.0f32; dim];
        for cid in 0..self.centroid_vecs.len() {
            if self.centroid_vecs[cid].is_none() {
                continue;
            }
            let members = &self.lists[cid];
            if members.is_empty() {
                continue;
            }
            mean.iter_mut().for_each(|m| *m = 0.0);
            for &pid in members {
                let row = self.points.row(pid as usize);
                for (m, &x) in mean.iter_mut().zip(row) {
                    *m += x;
                }
            }
            let inv = 1.0 / members.len() as f32;
            for m in mean.iter_mut() {
                *m *= inv;
            }
            if normalize {
                let norm = mean.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    let inv_norm = 1.0 / norm;
                    for m in mean.iter_mut() {
                        *m *= inv_norm;
                    }
                }
            }
            if let Some(cv) = self.centroid_vecs[cid].as_mut() {
                cv.copy_from_slice(&mean);
            }
        }
    }

    /// Clustering residual: the sum of squared distances from every assigned
    /// point to its centroid. Lower is a tighter clustering.
    pub fn residual(&self) -> f64 {
        let mut sum = 0.0f64;
        for (cid, slot) in self.centroid_vecs.iter().enumerate() {
            if let Some(cv) = slot {
                for &pid in &self.lists[cid] {
                    sum += sq_l2(self.points.row(pid as usize), cv) as f64;
                }
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

        self.lists[c as usize].push(pid);
        self.assignments[p] = c;
        self.telemetry.total_inserts += 1;

        if self.lists[c as usize].len() > self.params.split_threshold
            && self.live_count < self.params.target_clusters
        {
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
        self.nearest_live(pid)
            .ok_or_else(|| GraphIvfError::invalid("no live centroid available for assignment"))
    }

    /// Split cluster `c` into two child centroids via a local 2-means, then
    /// reassign the points of `c` and of `c`'s graph-neighboring clusters among
    /// the two children and the neighboring centroids.
    fn split(&mut self, c: u32) -> Result<()> {
        let cu = c as usize;
        let split_start = Instant::now();

        // Take C's members out; if too small to split, restore and bail.
        let members = std::mem::take(&mut self.lists[cu]);
        if members.len() < 2 {
            self.lists[cu] = members;
            return Ok(());
        }
        let cluster_size = members.len();

        // 1. Two child centroids from a local 2-means over C.
        let two_means_start = Instant::now();
        let two = self.two_means(&members)?;
        let two_means_us = two_means_start.elapsed().as_micros() as u64;
        let child1: Box<[f32]> = two.row(0).to_vec().into_boxed_slice();
        let child2: Box<[f32]> = two.row(1).to_vec().into_boxed_slice();

        // 2. Live graph neighbors of c (before deleting it).
        let mut neighbors = std::mem::take(&mut self.scratch_neighbors);
        centroids::neighbors(&self.graph, c, &mut neighbors)?;
        neighbors.retain(|&x| {
            x != c
                && (x as usize) < self.centroid_vecs.len()
                && self.centroid_vecs[x as usize].is_some()
        });
        let num_neighbors = neighbors.len();

        // 3. Allocate the two child ids and update the centroid table.
        let id1 = self.alloc_id()?;
        let id2 = self.alloc_id()?;
        self.centroid_vecs[id1 as usize] = Some(child1);
        self.centroid_vecs[id2 as usize] = Some(child2);
        self.centroid_vecs[cu] = None; // retire c

        // 4. Mutate the graph: delete c, insert the two children.
        centroids::delete_centroid(&self.graph, &self.runtime, c)?;
        {
            let v1 = self.centroid_vecs[id1 as usize].as_ref().expect("just set");
            centroids::insert_centroid(&self.graph, &self.runtime, id1, v1)?;
        }
        {
            let v2 = self.centroid_vecs[id2 as usize].as_ref().expect("just set");
            centroids::insert_centroid(&self.graph, &self.runtime, id2, v2)?;
        }
        self.live_count += 1; // -1 (c) + 2 (children)

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
            let taken = std::mem::take(&mut self.lists[nc as usize]);
            pool.extend_from_slice(&taken);
        }
        // c's list is already empty (members were taken); keep it empty.

        // 7. Reassign every pooled point to its nearest candidate centroid and
        //    rebuild the affected inverted lists.
        let reassign_start = Instant::now();
        for &pid in &pool {
            let best = self.nearest_among(pid, &cands);
            self.lists[best as usize].push(pid);
            self.assignments[pid as usize] = best;
        }
        let reassign_us = reassign_start.elapsed().as_micros() as u64;
        let num_reassigned = pool.len();

        // Return the scratch buffers for reuse.
        self.scratch_pool = pool;
        self.scratch_cands = cands;
        self.scratch_neighbors = neighbors;

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
            live_after: self.live_count,
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

    /// Nearest candidate centroid to point `pid` by squared-L2 (candidates are
    /// centroid ids; retired ids are skipped).
    fn nearest_among(&self, pid: u32, cands: &[u32]) -> u32 {
        let p = self.points.row(pid as usize);
        let mut best = cands[0];
        let mut best_d = f32::INFINITY;
        for &cand in cands {
            if let Some(cv) = &self.centroid_vecs[cand as usize] {
                let d = sq_l2(p, cv);
                if d < best_d {
                    best_d = d;
                    best = cand;
                }
            }
        }
        best
    }

    /// Brute-force nearest live centroid to point `pid` by squared-L2, scanning
    /// every occupied centroid slot. Used only as a robustness fallback when the
    /// centroid graph search returns no live result; returns `None` if no live
    /// centroid exists.
    fn nearest_live(&self, pid: u32) -> Option<u32> {
        let p = self.points.row(pid as usize);
        let mut best = None;
        let mut best_d = f32::INFINITY;
        for (id, slot) in self.centroid_vecs.iter().enumerate() {
            if let Some(cv) = slot {
                let d = sq_l2(p, cv);
                if d < best_d {
                    best_d = d;
                    best = Some(id as u32);
                }
            }
        }
        best
    }

    /// Allocate the next centroid id, erroring if capacity is exhausted.
    fn alloc_id(&mut self) -> Result<u32> {
        let id = self.next_id;
        if (id as usize) >= self.centroid_vecs.len() {
            return Err(GraphIvfError::invalid(
                "centroid capacity exceeded; increase target_clusters",
            ));
        }
        self.next_id += 1;
        Ok(id)
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
        let dim = self.dim;
        let num_points = self.points.nrows();
        if stored.nrows() != num_points {
            return Err(GraphIvfError::invalid(format!(
                "stored corpus has {} rows but clustering corpus has {num_points}",
                stored.nrows()
            )));
        }

        // Dense remap of live centroid ids to a contiguous 0..k range.
        let live: Vec<usize> = (0..self.centroid_vecs.len())
            .filter(|&c| self.centroid_vecs[c].is_some())
            .collect();
        let k = live.len();
        let mut remap = vec![UNASSIGNED; self.centroid_vecs.len()];
        let mut cbuf = vec![0.0f32; k * dim];
        for (new, &old) in live.iter().enumerate() {
            remap[old] = new as u32;
            cbuf[new * dim..(new + 1) * dim]
                .copy_from_slice(self.centroid_vecs[old].as_ref().expect("live"));
        }
        let centroids_mat = Matrix::try_from(cbuf.into_boxed_slice(), k, dim)
            .map_err(|_| GraphIvfError::invalid("centroid matrix shape mismatch"))?;

        // Dense per-point assignments.
        let mut dense = vec![0u32; num_points];
        for (pid, slot) in dense.iter_mut().enumerate() {
            let c = self.assignments[pid];
            if c == UNASSIGNED {
                return Err(GraphIvfError::invalid(
                    "cannot flush: some points have not been inserted",
                ));
            }
            *slot = remap[c as usize];
        }

        // Write centroids (always f32).
        let centroids_path = with_suffix(prefix, CENTROIDS_SUFFIX);
        let mut centroids_file = std::fs::File::create(&centroids_path)?;
        write_bin(centroids_mat.as_view(), &mut centroids_file)
            .map_err(|e| GraphIvfError::malformed(e.to_string()))?;

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
    use crate::{GraphIvfIndex, SearchParams};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    fn mat(data: Vec<f32>, nrows: usize, ncols: usize) -> Matrix<f32> {
        Matrix::try_from(data.into_boxed_slice(), nrows, ncols).unwrap()
    }

    fn params(target: usize, threshold: usize) -> OnlineParams {
        OnlineParams {
            target_clusters: target,
            split_threshold: threshold,
            assign_l: 32,
            two_means_iters: 10,
            graph: GraphParams::default(),
            metric: Metric::L2,
            normalize_centroids: false,
            refine_centroids: false,
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
        c.centroid_vecs.iter().filter_map(|s| s.clone()).collect()
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

        // Neighbors are readable and non-empty for a connected node.
        let mut nbrs = Vec::new();
        centroids::neighbors(&graph, 0, &mut nbrs).unwrap();
        assert!(!nbrs.is_empty());
    }

    // ----- clusterer invariants -----

    /// Every inserted point is accounted for exactly once in a live cluster.
    fn assert_invariants(c: &OnlineClusterer, inserted: usize) {
        // live_count matches the centroid table.
        let live = c.centroid_vecs.iter().filter(|s| s.is_some()).count();
        assert_eq!(live, c.live_count);
        assert!(c.live_count <= c.params.target_clusters);

        // Sum of live list lengths == inserted count; retired ids hold nothing.
        let mut total = 0usize;
        for (cid, slot) in c.centroid_vecs.iter().enumerate() {
            if slot.is_none() {
                assert!(c.lists[cid].is_empty(), "retired centroid has points");
            } else {
                total += c.lists[cid].len();
            }
        }
        assert_eq!(total, inserted);

        // Every assigned point points to a live centroid.
        for pid in 0..inserted {
            let a = c.assignments[pid];
            assert_ne!(a, UNASSIGNED);
            assert!(c.centroid_vecs[a as usize].is_some());
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
            assert!(e.num_reassigned >= e.cluster_size); // pool includes C
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
        // target < initial
        let initial = mat(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0], 3, 2);
        assert!(OnlineClusterer::new(points.clone(), initial, params(2, 10)).is_err());
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
        for cv in c.centroid_vecs.iter().flatten() {
            let is_corpus_point = (0..points.nrows()).any(|r| points.row(r) == cv.as_ref());
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

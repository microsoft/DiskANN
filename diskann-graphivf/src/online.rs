/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Online (incremental) graph-IVF clustering with split-and-reassign.
//!
//! [`OnlineClusterer`] builds the IVF partition incrementally instead of in a
//! single batch Lloyd pass. Points are routed to their nearest centroid via a
//! mutable centroid graph; when a cluster grows past a threshold it is split,
//! and the points of the split cluster together with the points of its
//! graph-neighboring clusters are reassigned among the new and neighboring
//! centroids.
//!
//! There is one write path, [`insert_batch`](OnlineClusterer::insert_batch):
//! route the batch, then jointly split whichever clusters overflowed and
//! reassign each split region. A batch large enough to be worth the dispatch
//! routes across the thread pool; reassignment is always a GEMM.
//!
//! Points can also be removed with
//! [`delete_batch`](OnlineClusterer::delete_batch), the mirror image: drop the
//! points from their inverted lists, then retire whichever clusters fell below
//! `merge_threshold`. Retiring *dissolves* a cluster — it leaves the centroid
//! graph and its members are scattered onto their nearest survivors by the same
//! GEMM the split path uses — so a split is `+1` live cluster and a merge is
//! `-1`. Splits are insert-driven and merges are delete-driven, and neither
//! triggers the other, so the two cannot cascade.
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
use diskann_vector::PreprocessedDistanceFunction;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use tokio::runtime::Runtime;

use crate::{
    centroids::{self, MutableCentroidGraph},
    cluster::{self, sq_l2},
    index::{with_suffix, CENTROIDS_SUFFIX, LISTS_SUFFIX, META_SUFFIX},
    params::{EmptyClusterPolicy, OnlineParams, SearchParams},
    storage::{self, Layout},
    GraphIvfError, Result,
};

use diskann::{utils::VectorRepr, ANNError};

/// Sentinel in [`OnlineClusterer::assignments`] for a point that is not
/// currently in the index — either never inserted, or deleted. Shared with the
/// list writer, which skips rows carrying it.
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
/// The mutable centroid graph accumulates soft-deleted (tombstoned) slots as
/// clusters split — near the target cluster count roughly half the graph can be
/// tombstones. A narrow beam can then occasionally exhaust its frontier on
/// tombstoned nodes and return no live centroid, so the search is retried with
/// `wide_l` before giving up and, as a last resort, falls back to a brute-force
/// scan over the live centroids. Successful narrow-beam routes are unchanged.
fn route_one(
    graph: &MutableCentroidGraph,
    runtime: &Runtime,
    table: &CentroidTable,
    point: &[f32],
    base_l: usize,
    wide_l: usize,
) -> Result<u32> {
    let mut ids = [0u32; 1];
    let mut dist = [0.0f32; 1];
    for l in [base_l, wide_l] {
        if centroids::search_mut(graph, runtime, point, l, &mut ids, &mut dist)? > 0 {
            return Ok(ids[0]);
        }
    }
    table
        .nearest(point)
        .ok_or_else(|| GraphIvfError::invalid("no live centroid available for assignment"))
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
///
/// A batched insert splits every overflowing cluster together, so it emits one
/// event per split parent that all share the batch's `insert_index` and
/// `live_after`, and divide the batch's joint k-means time between them.
#[derive(Debug, Clone, Copy)]
pub struct SplitEvent {
    /// Number of inserts completed (inclusive) when this split fired. Serves as
    /// the build-progress timestamp, in `[1, corpus_len]`. Every split of one
    /// batch shares the timestamp of that batch's last point.
    pub insert_index: u64,
    /// The centroid id that was split (and retired).
    pub cluster: u32,
    /// Size of the split cluster at split time (the overflow that triggered it).
    pub cluster_size: usize,
    /// Number of live neighbor clusters drawn into the reassignment, besides the
    /// split's own two children.
    pub num_neighbors: usize,
    /// Points that actually changed cluster in this split's reassignment pass.
    /// A point re-examined but routed back to the centroid it already held is
    /// not counted; every member of the split (retired) cluster is, since it
    /// must move to a child.
    pub num_reassigned: usize,
    /// Live centroid count immediately after the split (net `+1` per split).
    pub live_after: usize,
    /// Wall-clock of the 2-means, in microseconds. When a batch split several
    /// clusters at once this is the parent's share of the joint k-means,
    /// prorated by member count.
    pub two_means_us: u64,
    /// Wall-clock of the reassignment pass, in microseconds.
    pub reassign_us: u64,
    /// Wall-clock of the whole split (2-means + graph mutation + reassign), in
    /// microseconds.
    pub total_us: u64,
}

/// One cluster-merge event, recorded as it happens during a delete.
///
/// The counterpart of [`SplitEvent`]. A split takes one cell over
/// `split_threshold` and re-clusters it into two; a merge takes one cell under
/// `merge_threshold` and dissolves it, scattering its members onto the nearest
/// survivors. Both retire the cell they consumed and both place points with the
/// same GEMM — a split is `+1` live cluster, a merge is `-1`.
///
/// The asymmetry is that a split *fits* new centroids, so a neighbor's points
/// may now belong elsewhere and have to be re-examined. A merge fits nothing:
/// removing a centroid cannot change a surviving point's
/// nearest-among-survivors, so only the victim's own members move. That is why
/// there is no `kmeans_us` here and no neighbor-member accounting.
///
/// Splits and merges are recorded separately because they are driven by
/// different operations — splits by [`insert_batch`](OnlineClusterer::insert_batch),
/// merges by [`delete_batch`](OnlineClusterer::delete_batch) — and a single
/// timeline over both would need a shared clock the two do not have.
///
/// A batched delete retires every underfull cluster it can, so it emits one
/// event per retirement that all share the batch's `op_index` and `live_after`.
#[derive(Debug, Clone, Copy)]
pub struct MergeEvent {
    /// Total operations (inserts + deletes) completed when this merge fired.
    /// Every merge of one batch shares the timestamp of that batch.
    pub op_index: u64,
    /// The underfull centroid that was dissolved (retired).
    pub victim: u32,
    /// Points `victim` still held when it was dissolved (the underflow that
    /// triggered the merge). Every one of them is re-placed.
    pub victim_size: usize,
    /// Number of surviving clusters offered as landing sites for those points.
    pub num_neighbors: usize,
    /// Points that actually changed cluster in this merge's reassignment pass.
    pub num_reassigned: usize,
    /// Live centroid count immediately after the merge (net `-1` per merge).
    pub live_after: usize,
    /// Wall-clock of the centroid-graph search for landing sites, in
    /// microseconds.
    pub search_us: u64,
    /// Wall-clock of the reassignment pass, in microseconds.
    pub reassign_us: u64,
    /// Wall-clock of the whole merge (search + reassign), in microseconds.
    /// Excludes the graph mutation, which is shared by the whole batch.
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
    /// Total points deleted.
    pub total_deletes: u64,
    /// Total merges performed. Each retires one cluster.
    pub total_merges: u64,
    /// Total points that actually changed cluster, summed across all merges.
    /// The mirror of [`total_reassigned`](Self::total_reassigned), except that
    /// a merge only ever re-places the retired cluster's own members, so every
    /// one of them is counted and nothing else is.
    pub total_merge_reassigned: u64,
    /// Cumulative time removing points from their inverted lists, microseconds.
    /// Excludes the merge handling a delete may trigger.
    pub delete_us: u64,
    /// Cumulative time in merge handling (graph mutation + search + reassign),
    /// microseconds.
    pub merge_us: u64,
    /// Ordered per-split records (see [`SplitEvent`]).
    pub splits: Vec<SplitEvent>,
    /// Ordered per-merge records (see [`MergeEvent`]).
    pub merges: Vec<MergeEvent>,
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

    /// Write the per-merge timeline to `path` as CSV (one row per retired
    /// cluster, with a header). Columns match the fields of [`MergeEvent`].
    ///
    /// Kept separate from [`write_csv`](Self::write_csv) so that adding merges
    /// does not change the split CSV's schema, which downstream analysis
    /// consumes positionally.
    ///
    /// # Errors
    ///
    /// Returns any I/O error from creating or writing the file.
    pub fn write_merges_csv(&self, path: &Path) -> std::io::Result<()> {
        use std::fmt::Write as _;
        let mut out = String::with_capacity(64 + self.merges.len() * 48);
        out.push_str(
            "op_index,victim,victim_size,num_neighbors,num_reassigned,\
             live_after,search_us,reassign_us,total_us\n",
        );
        for e in &self.merges {
            let _ = writeln!(
                out,
                "{},{},{},{},{},{},{},{},{}",
                e.op_index,
                e.victim,
                e.victim_size,
                e.num_neighbors,
                e.num_reassigned,
                e.live_after,
                e.search_us,
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

    /// Number of ids still available to [`alloc`](Self::alloc).
    fn alloc_budget(&self) -> usize {
        self.vecs.len().saturating_sub(self.next_id as usize)
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

    /// Remove `victims` from centroid `cid`'s inverted list, resetting their
    /// reverse assignments to [`UNASSIGNED`] so the points may be re-inserted
    /// later.
    ///
    /// `victims` must be sorted, deduplicated, and all currently assigned to
    /// `cid`. The list is filtered in a single pass regardless of how many
    /// points are being removed, which is what keeps a batched delete linear in
    /// list length rather than `O(victims * list_len)`.
    fn remove_sorted(&mut self, cid: u32, victims: &[u32]) {
        debug_assert!(victims.windows(2).all(|w| w[0] < w[1]));
        self.lists[cid as usize].retain(|pid| victims.binary_search(pid).is_err());
        for &pid in victims {
            self.assignments[pid as usize] = UNASSIGNED;
        }
    }

    /// Remove and return centroid `cid`'s inverted list, leaving it empty. The
    /// reverse assignments of those points are left untouched — the caller is
    /// expected to reassign every returned point.
    fn take_members(&mut self, cid: u32) -> Vec<u32> {
        std::mem::take(&mut self.lists[cid as usize])
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
    /// Scratch reused by the GEMM reassignment: the contiguous candidate-centroid
    /// matrix.
    scratch_cvecs: Vec<f32>,
    /// Scratch reused by the GEMM reassignment: the contiguous point tile.
    scratch_tile: Vec<f32>,
    /// Scratch reused by the GEMM reassignment: the per-point argmin output.
    scratch_best: Vec<u32>,
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
            scratch_cvecs: Vec::new(),
            scratch_tile: Vec::new(),
            scratch_best: Vec::new(),
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

    /// Insert a batch of points, routing each to its nearest centroid and then
    /// splitting whichever clusters that pushed past `split_threshold`.
    ///
    /// A batch is processed in phases rather than point by point:
    ///
    /// 1. **Route.** Splits are deferred to the end of the batch, so routing is
    ///    read-only with respect to the centroid graph and a batch large enough
    ///    to be worth the dispatch runs across the whole thread pool.
    /// 2. **Split jointly.** Every routed-to cluster that now overflows is
    ///    collected; their inverted lists are unioned and re-clustered in a
    ///    *single* k-means into twice as many centroids (two per overflowing
    ///    parent). With one parent this is exactly a local 2-means; with several
    ///    it lets clusters that overflowed together — which are usually adjacent
    ///    — resolve their shared boundaries jointly instead of greedily.
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
    /// in the index, or if routing, clustering, or graph mutation fails.
    pub fn insert_batch(&mut self, pids: &[u32]) -> Result<()> {
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

        // 2. Append every point to its routed cluster. Both failure modes —
        //    already present, and named twice — were ruled out above.
        for (&pid, &cid) in pids.iter().zip(routes.iter()) {
            let prev = self.partition.assign(pid, cid);
            debug_assert_eq!(prev, UNASSIGNED);
        }
        self.telemetry.total_inserts += pids.len() as u64;

        // 3. The routed-to clusters that overflowed. Only clusters that received
        //    a point this batch can have grown, so the scan is over the batch's
        //    distinct routes rather than the whole partition.
        routes.sort_unstable();
        routes.dedup();
        let mut parents = routes;
        parents.retain(|&c| self.partition.list_len(c) > self.params.split_threshold);
        if parents.is_empty() {
            return Ok(());
        }

        // 4. Admission control. Each parent is replaced by two children, so a
        //    split costs two ids and adds one live cluster. When the batch would
        //    breach either bound, the most overfull clusters are split first and
        //    the rest wait for a later batch.
        let mut admitted = parents.len().min(self.table.alloc_budget() / 2);
        if let Some(k) = self.params.max_clusters {
            admitted = admitted.min(k.saturating_sub(self.table.live_count()));
        }
        if admitted == 0 {
            return Ok(());
        }
        if admitted < parents.len() {
            parents.sort_unstable_by(|&a, &b| {
                self.partition
                    .list_len(b)
                    .cmp(&self.partition.list_len(a))
                    .then(a.cmp(&b))
            });
            parents.truncate(admitted);
            parents.sort_unstable();
        }

        self.split_batch(&parents)
    }

    /// Delete a batch of points, then merge whichever clusters that emptied
    /// past [`merge_threshold`](OnlineParams::merge_threshold).
    ///
    /// The mirror image of [`insert_batch`](Self::insert_batch), in the same
    /// three phases: group the batch by the cluster each point currently sits
    /// in, filter those inverted lists one pass each, then handle whichever
    /// clusters ended up the wrong size.
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
    /// currently present in the index, or if the merge pass fails.
    pub fn delete_batch(&mut self, pids: &[u32]) -> Result<()> {
        if pids.is_empty() {
            return Ok(());
        }
        let delete_start = Instant::now();
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

        // 2. Filter each touched cluster's list in a single pass.
        let mut touched: Vec<u32> = Vec::new();
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
            touched.push(cid);
            start = end;
        }
        self.telemetry.total_deletes += deleted as u64;
        self.telemetry.delete_us += delete_start.elapsed().as_micros() as u64;

        // 3. Merge the touched clusters that are now underfull. Only clusters
        //    that lost a point this batch can have shrunk, so the scan is over
        //    the batch's distinct clusters rather than the partition.
        if !self.params.merges_enabled() {
            return Ok(());
        }
        let mut victims: Vec<u32> = touched
            .into_iter()
            .filter(|&c| self.partition.list_len(c) < self.params.merge_threshold)
            .collect();
        if victims.is_empty() {
            return Ok(());
        }

        // Admission control, the mirror of the split's. A merge retires a live
        // cluster and allocates nothing, so the `min_clusters` floor is the
        // only bound — unlike a split, it cannot run out of centroid ids.
        let floor = self.params.effective_min_clusters();
        let admitted = victims
            .len()
            .min(self.table.live_count().saturating_sub(floor));
        if admitted == 0 {
            return Ok(());
        }
        // Emptiest first, so that when the floor admits only some of them the
        // clusters that have lost the most are the ones that go.
        victims.sort_unstable_by_key(|&c| (self.partition.list_len(c), c));
        victims.truncate(admitted);

        self.merge_batch(&victims)
    }

    /// The `want` live centroids nearest `c`'s own centroid vector, in
    /// ascending distance and excluding `c` itself.
    ///
    /// Used by [`split_batch`](Self::split_batch) to define its working region:
    /// one cluster plus its centroid-space neighborhood, searched *before* any
    /// retirement so it sees the pre-mutation graph. The result is therefore
    /// only a candidate list — the caller re-checks liveness after publishing
    /// its own structural change.
    ///
    /// `k = want + 1` reserves the slot `c` itself takes at distance zero.
    fn region_neighbors(&mut self, c: u32, want: usize) -> Result<Vec<u32>> {
        let query = self
            .table
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
    /// Anchored on a vector rather than a centroid id so that
    /// [`merge_batch`](Self::merge_batch) can search *after* retiring its
    /// victims — with the victims already out of the graph, the result can only
    /// contain survivors, which is what makes the exclusion bookkeeping
    /// unnecessary there.
    fn nearest_live_centroids(&mut self, query: &[f32], want: usize) -> Result<Vec<u32>> {
        let search_k = want.max(1);
        let search_l = self.params.reassign_l.max(search_k);

        let mut ids = std::mem::take(&mut self.scratch_neighbors);
        let mut dist = std::mem::take(&mut self.scratch_dist);
        ids.clear();
        ids.resize(search_k, 0);
        dist.clear();
        dist.resize(search_k, 0.0);

        let found = centroids::search_mut(
            &self.graph,
            &self.runtime,
            query,
            search_l,
            &mut ids,
            &mut dist,
        )?;
        let mut out = ids[..found].to_vec();

        self.scratch_neighbors = ids;
        self.scratch_dist = dist;

        out.retain(|&x| self.table.is_live(x));
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
        let graph = &self.graph;
        let table = &self.table;
        let points = &self.points;
        let base_l = self.params.assign_l.max(1);
        let wide_l = base_l.saturating_mul(8).max(512);

        if pids.len() <= ROUTE_CHUNK {
            for (pid, slot) in pids.iter().zip(out.iter_mut()) {
                *slot = route_one(
                    graph,
                    &self.runtime,
                    table,
                    points.row(*pid as usize),
                    base_l,
                    wide_l,
                )?;
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
                    *slot = route_one(graph, &runtime, table, point, base_l, wide_l)?;
                }
                Ok(())
            })
    }

    /// Split every cluster in `parents` at once.
    ///
    /// The members of all `l` parents are unioned and re-clustered by a *single*
    /// k-means into `2l` children, rather than by `l` independent 2-means. The
    /// union is seeded with two members drawn from each parent, so the run starts
    /// from the same configuration `l` separate 2-means would, but the parents'
    /// boundaries are then free to move against one another — clusters that
    /// overflow together in one batch are usually adjacent, and the joint pass
    /// resolves them jointly instead of greedily.
    ///
    /// Once the children exist, each parent's region is reassigned in turn (see
    /// [`reassign_gemm`](Self::reassign_gemm)). `parents` must be sorted, live,
    /// hold at least two members each, and fit the id budget and cluster cap —
    /// [`insert_batch`](Self::insert_batch) guarantees all four.
    fn split_batch(&mut self, parents: &[u32]) -> Result<()> {
        let split_start = Instant::now();
        let dim = self.dim;
        let l = parents.len();

        // The `reassign_neighbors` live centroids nearest each parent, found by
        // searching the parent's own centroid vector. The searches run before the
        // parents are retired and the children inserted, so they see the
        // pre-split centroids; the parent itself (distance 0) is dropped, and its
        // own two children join the candidate set once they exist. `k = s + 1`
        // reserves the slot the parent takes.
        let s = self.params.reassign_neighbors;
        let mut neighbor_sets: Vec<Vec<u32>> = Vec::with_capacity(l);
        for &c in parents {
            neighbor_sets.push(self.region_neighbors(c, s)?);
        }

        // Union every parent's members; `spans[i]` is parent i's range in `x`.
        let mut x: Vec<u32> = Vec::new();
        let mut spans: Vec<(usize, usize)> = Vec::with_capacity(l);
        for &c in parents {
            let start = x.len();
            let members = self.partition.take_members(c);
            x.extend_from_slice(&members);
            spans.push((start, x.len()));
        }
        let m = x.len();

        // 1. One k-means over the union into `2l` children.
        let kmeans_start = Instant::now();
        let mut buf = vec![0.0f32; m * dim];
        for (i, &pid) in x.iter().enumerate() {
            buf[i * dim..(i + 1) * dim].copy_from_slice(self.points.row(pid as usize));
        }
        let data = Matrix::try_from(buf.into_boxed_slice(), m, dim)
            .map_err(|_| GraphIvfError::invalid("split sub-matrix shape mismatch"))?;

        // Seed each parent's two children with two distinct members of that
        // parent, matching how a single split seeds its local 2-means.
        let mut seed = vec![0.0f32; 2 * l * dim];
        for (i, &(lo, hi)) in spans.iter().enumerate() {
            let span = hi - lo;
            debug_assert!(span >= 2);
            let a = self.rng.random_range(0..span);
            let mut b = self.rng.random_range(0..span);
            if b == a {
                b = (a + 1) % span;
            }
            seed[2 * i * dim..(2 * i + 1) * dim].copy_from_slice(data.row(lo + a));
            seed[(2 * i + 1) * dim..(2 * i + 2) * dim].copy_from_slice(data.row(lo + b));
        }
        let mut children = Matrix::try_from(seed.into_boxed_slice(), 2 * l, dim)
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
        let kmeans_us = kmeans_start.elapsed().as_micros() as u64;

        // 2. Publish the structural change: allocate the children, retire the
        //    parents, and mirror both into the centroid graph.
        let mut child_ids = Vec::with_capacity(2 * l);
        for i in 0..2 * l {
            child_ids.push(
                self.table
                    .alloc(children.row(i).to_vec().into_boxed_slice())?,
            );
        }
        for &c in parents {
            self.table.retire(c);
            centroids::delete_centroid(&self.graph, &self.runtime, c)?;
        }
        for &id in &child_ids {
            let v = self.table.get(id).expect("just allocated");
            centroids::insert_centroid(&self.graph, &self.runtime, id, v)?;
        }
        let live_after = self.table.live_count();

        // 3. Reassign each split region in turn. The candidates are the parent's
        //    own two children plus the neighbors picked before the mutation, less
        //    any neighbor that was itself a parent of this batch and has since
        //    been retired — that region is covered by its own turn.
        for (i, &parent) in parents.iter().enumerate() {
            let (lo, hi) = spans[i];
            let cluster_size = hi - lo;

            let mut cands = std::mem::take(&mut self.scratch_cands);
            cands.clear();
            cands.extend(
                neighbor_sets[i]
                    .iter()
                    .copied()
                    .filter(|&c| self.table.is_live(c)),
            );
            let num_neighbors = cands.len();
            cands.push(child_ids[2 * i]);
            cands.push(child_ids[2 * i + 1]);

            // Candidate points: the parent's own members plus everything its
            // surviving neighbors hold. The children's lists are still empty —
            // no other region can have placed a point on them — so the k-means
            // assignment is not applied separately; reassignment places every
            // member against the neighbors too, which strictly refines it.
            let mut pool = std::mem::take(&mut self.scratch_pool);
            pool.clear();
            pool.extend_from_slice(&x[lo..hi]);
            for &c in &cands[..num_neighbors] {
                let taken = self.partition.take_members(c);
                pool.extend_from_slice(&taken);
            }

            let reassign_start = Instant::now();
            let num_reassigned = self.reassign_gemm(&cands, &pool)?;
            let reassign_us = reassign_start.elapsed().as_micros() as u64;

            self.scratch_pool = pool;
            self.scratch_cands = cands;

            // The joint k-means covered every parent at once; charge each parent
            // the share of it proportional to the points it contributed.
            let two_means_us = if m == 0 {
                0
            } else {
                kmeans_us * cluster_size as u64 / m as u64
            };
            self.telemetry.total_splits += 1;
            self.telemetry.total_reassigned += num_reassigned as u64;
            self.telemetry.splits.push(SplitEvent {
                insert_index: self.telemetry.total_inserts,
                cluster: parent,
                cluster_size,
                num_neighbors,
                num_reassigned,
                live_after,
                two_means_us,
                reassign_us,
                total_us: two_means_us + reassign_us,
            });
        }

        self.telemetry.split_us += split_start.elapsed().as_micros() as u64;
        Ok(())
    }

    /// Retire every cluster in `victims`, scattering each one's members onto
    /// their nearest surviving centroids.
    ///
    /// The counterpart of [`split_batch`](Self::split_batch), but deliberately
    /// not its mirror. A split has to re-examine the neighbors' points because
    /// it *fits* new centroids, and a point that was nearest its own centroid
    /// may now be nearer a child. A merge fits nothing, and removing a centroid
    /// cannot change a surviving point's nearest-among-survivors — so the
    /// neighbors' members are provably already where they belong, and only the
    /// victim's own points are re-placed. That is the whole operator: no
    /// k-means, no allocation, no region union.
    ///
    /// Because nothing is fitted, a merge consumes no centroid id. Deletes are
    /// therefore free against the
    /// [`centroid_capacity`](OnlineParams::centroid_capacity) budget, which
    /// only splits draw down.
    ///
    /// Every victim is retired before any point is placed, which is what lets
    /// the searches run unguarded: with the victims already out of the graph,
    /// [`nearest_live_centroids`](Self::nearest_live_centroids) can only return
    /// survivors, so a batch retiring several mutually adjacent clusters needs
    /// no claiming or exclusion bookkeeping at all.
    ///
    /// `victims` must be live, distinct, and leave at least one cluster
    /// standing; [`delete_batch`](Self::delete_batch) guarantees all of these.
    fn merge_batch(&mut self, victims: &[u32]) -> Result<()> {
        let merge_start = Instant::now();
        let s = self.params.reassign_neighbors;

        // 1. Snapshot what is about to disappear: each victim's centroid vector
        //    (the anchor its landing sites are searched from) and its members.
        //    Both have to be read before the retirement below invalidates them.
        let doomed: Vec<(u32, Vec<f32>, Vec<u32>)> = victims
            .iter()
            .map(|&v| {
                let anchor = self.table.get(v).expect("victim is live").to_vec();
                (v, anchor, self.partition.take_members(v))
            })
            .collect();

        // 2. Retire the whole batch up front, so no later search can offer a
        //    centroid that is itself on its way out.
        for &(v, _, _) in &doomed {
            self.table.retire(v);
            centroids::delete_centroid(&self.graph, &self.runtime, v)?;
        }
        let live_after = self.table.live_count();

        // 3. Scatter each victim's members over the survivors nearest its
        //    anchor. `doomed` owns the members, so the GEMM can read them
        //    directly without staging through scratch.
        let op_index = self.telemetry.total_inserts + self.telemetry.total_deletes;
        for (victim, anchor, members) in &doomed {
            let search_start = Instant::now();
            let cands = self.nearest_live_centroids(anchor, s)?;
            let search_us = search_start.elapsed().as_micros() as u64;
            if cands.is_empty() {
                // `delete_batch` holds the live count above `min_clusters >= 1`,
                // so the graph always has somewhere to put these points. Failing
                // loudly beats silently orphaning them.
                return Err(GraphIvfError::invalid(format!(
                    "retiring cluster {victim} found no surviving centroid for its {} points",
                    members.len()
                )));
            }

            let reassign_start = Instant::now();
            let num_reassigned = self.reassign_gemm(&cands, members)?;
            let reassign_us = reassign_start.elapsed().as_micros() as u64;

            self.telemetry.total_merges += 1;
            self.telemetry.total_merge_reassigned += num_reassigned as u64;
            self.telemetry.merges.push(MergeEvent {
                op_index,
                victim: *victim,
                victim_size: members.len(),
                num_neighbors: cands.len(),
                num_reassigned,
                live_after,
                search_us,
                reassign_us,
                total_us: search_us + reassign_us,
            });
        }

        self.telemetry.merge_us += merge_start.elapsed().as_micros() as u64;
        Ok(())
    }

    /// Reassign every point of `pool` to its nearest centroid in `cands`,
    /// returning how many actually changed cluster.
    ///
    /// Distances are computed as `‖p‖² - 2p·c + ‖c‖²` over a `tile x |cands|`
    /// GEMM instead of a scalar loop per point, which is what makes a large
    /// `reassign_neighbors` affordable. The caller must have emptied every
    /// candidate's inverted list first (all of `pool` is re-appended here).
    fn reassign_gemm(&mut self, cands: &[u32], pool: &[u32]) -> Result<usize> {
        if cands.is_empty() || pool.is_empty() {
            return Ok(0);
        }
        let dim = self.dim;
        let nc = cands.len();

        let mut cvecs = std::mem::take(&mut self.scratch_cvecs);
        let mut tile = std::mem::take(&mut self.scratch_tile);
        let mut best = std::mem::take(&mut self.scratch_best);
        cvecs.clear();
        cvecs.resize(nc * dim, 0.0);
        for (i, &c) in cands.iter().enumerate() {
            cvecs[i * dim..(i + 1) * dim]
                .copy_from_slice(self.table.get(c).expect("candidate is live"));
        }
        let rows = REASSIGN_TILE.min(pool.len());
        tile.clear();
        tile.resize(rows * dim, 0.0);
        best.clear();
        best.resize(rows, 0);

        let mut num_reassigned = 0usize;
        for chunk in pool.chunks(rows) {
            let n = chunk.len();
            for (i, &pid) in chunk.iter().enumerate() {
                tile[i * dim..(i + 1) * dim].copy_from_slice(self.points.row(pid as usize));
            }
            compute_closest_centers(
                &tile[..n * dim],
                n,
                dim,
                &cvecs,
                nc,
                1,
                &mut best[..n],
                None,
                None,
                self.pool.as_ref(),
            )?;
            for (i, &pid) in chunk.iter().enumerate() {
                let cid = cands[best[i] as usize];
                if self.partition.assign(pid, cid) != cid {
                    num_reassigned += 1;
                }
            }
        }

        self.scratch_cvecs = cvecs;
        self.scratch_tile = tile;
        self.scratch_best = best;
        Ok(num_reassigned)
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

        // Write inverted lists from the stored representation and the metadata.
        let stored_dim = stored.ncols();
        let lists_path = with_suffix(prefix, LISTS_SUFFIX);
        let (counts, offsets) = storage::write_lists_stored::<T>(&lists_path, stored, &dense, k)?;
        let layout = Layout {
            dim: stored_dim,
            metric: self.params.metric,
            element_size: std::mem::size_of::<T>(),
            num_points: live_points,
            graph: self.params.graph,
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
    pub fn searcher(&self) -> Result<OnlineSearcher<'_>> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .map_err(ANNError::from)?;
        Ok(OnlineSearcher {
            clusterer: self,
            runtime,
            cids: Vec::new(),
            cdist: Vec::new(),
            candidates: Vec::new(),
            scanned: 0,
        })
    }
}

/// A single-threaded query handle into a live [`OnlineClusterer`].
///
/// Answers queries against the in-memory `f32` corpus, so it measures the
/// partition the online build has arrived at *without* the quantization error
/// the flushed index would add. Recall read through this handle is therefore an
/// upper bound on the flushed index's, and the two are directly comparable only
/// for an `f32` index.
///
/// Not shareable across threads — open one handle per worker.
pub struct OnlineSearcher<'a> {
    clusterer: &'a OnlineClusterer,
    runtime: Runtime,
    cids: Vec<u32>,
    cdist: Vec<f32>,
    /// Reused across queries so a steady-state search allocates nothing.
    candidates: Vec<(u32, f32)>,
    scanned: u64,
}

impl std::fmt::Debug for OnlineSearcher<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnlineSearcher")
            .field("dim", &self.clusterer.dim)
            .field("num_clusters", &self.clusterer.num_clusters())
            .finish_non_exhaustive()
    }
}

impl OnlineSearcher<'_> {
    /// Corpus vectors scored across every query this handle has answered.
    ///
    /// Probed lists are scanned exhaustively, so this is the exact work the
    /// partition imposed. It is the denominator recall should be read against:
    /// a fixed `nlist` buys a different amount of scan as the partition
    /// changes, so recall-versus-`nlist` and recall-versus-scan can move in
    /// opposite directions.
    pub fn points_scanned(&self) -> u64 {
        self.scanned
    }

    /// Return the `k` approximate nearest neighbors of `query` as `(id,
    /// distance)` pairs sorted by ascending distance, where `id` is the corpus
    /// row index. Fewer than `k` are returned when the probed lists hold fewer
    /// live points.
    ///
    /// Mirrors [`GraphIvfIndex`](crate::GraphIvfIndex)'s search: probe the
    /// `nlist` nearest centroids, then scan their inverted lists exhaustively.
    /// Two deliberate differences: scoring is over the `f32` corpus rather than
    /// a stored quantized copy, and there is no I/O.
    ///
    /// Centroid navigation is always L2 — the centroid graph is built that way
    /// regardless of `metric` — while candidate scoring honors `metric`. Under
    /// [`Metric::InnerProduct`](crate::Metric) the returned distance is the
    /// negated inner product, keeping the "smaller is better" ordering.
    ///
    /// # Errors
    ///
    /// Returns an error if `k` is zero, if `query` has the wrong dimension, if
    /// `params` is invalid for the current cluster count, or if the centroid
    /// graph search fails.
    pub fn search(
        &mut self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
    ) -> Result<Vec<(u32, f32)>> {
        let c = self.clusterer;
        params.validate(c.num_clusters())?;
        if k == 0 {
            return Err(GraphIvfError::invalid("k must be non-zero"));
        }
        if query.len() != c.dim {
            return Err(GraphIvfError::invalid(format!(
                "query has dim {} but index has dim {}",
                query.len(),
                c.dim
            )));
        }

        // 1. Nearest `nlist` centroids, via the same graph an insert routes on.
        self.cids.clear();
        self.cids.resize(params.nlist, 0);
        self.cdist.clear();
        self.cdist.resize(params.nlist, 0.0);
        let n = centroids::search_mut(
            &c.graph,
            &self.runtime,
            query,
            params.effective_l(),
            &mut self.cids,
            &mut self.cdist,
        )?;

        // 2. Exhaustively score the probed lists.
        let scorer = f32::query_distance(query, c.params.metric.search_metric());
        self.candidates.clear();
        for &cid in &self.cids[..n] {
            for &pid in c.partition.members(cid) {
                self.candidates
                    .push((pid, scorer.evaluate_similarity(c.points.row(pid as usize))));
            }
        }
        self.scanned += self.candidates.len() as u64;

        // Every metric here is "smaller is better": L2 and cosine are
        // squared-L2, and the inner-product distance is negated.
        if self.candidates.len() > k {
            self.candidates
                .select_nth_unstable_by(k - 1, |a, b| a.1.total_cmp(&b.1));
            self.candidates.truncate(k);
        }
        self.candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        Ok(self.candidates.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraphIvfIndex, GraphParams};
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
            num_threads: 2,
            ..Default::default()
        }
    }

    /// [`params`] with merging enabled. `split` is raised to satisfy the
    /// hysteresis requirement when the caller asks for a tight merge floor.
    fn merge_params(target: usize, split: usize, merge: usize) -> OnlineParams {
        OnlineParams {
            split_threshold: split.max(2 * merge),
            merge_threshold: merge,
            ..params(target, split)
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
        let live: Vec<u32> = (0..inserted as u32).collect();
        assert_live_invariants(c, &live);
    }

    /// Exactly the points in `live` are held, each once, by a live cluster, and
    /// no other point is. This is the delete-aware form of
    /// [`assert_invariants`], which is the special case `live == 0..inserted`.
    fn assert_live_invariants(c: &OnlineClusterer, live: &[u32]) {
        // live_count matches the centroid table.
        let live_clusters = c.table.iter_live().count();
        assert_eq!(live_clusters, c.table.live_count());
        if let Some(k) = c.params.max_clusters {
            assert!(c.table.live_count() <= k);
        }
        assert!(c.table.live_count() <= c.table.capacity());

        // Sum of live list lengths == live count; retired ids hold nothing.
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
        assert_eq!(total, live.len());

        // Every live point sits on a live centroid, and every other point on
        // none.
        for pid in 0..c.points.nrows() as u32 {
            let a = c.partition.assignment(pid);
            if live.contains(&pid) {
                assert_ne!(a, UNASSIGNED, "live point {pid} is unassigned");
                assert!(c.table.is_live(a), "point {pid} sits on a retired centroid");
            } else {
                assert_eq!(a, UNASSIGNED, "absent point {pid} is still assigned");
            }
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
            c.insert_batch(&[pid]).unwrap();
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
            c.insert_batch(&[pid]).unwrap();
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
            c.insert_batch(&[pid]).unwrap();
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
            c.insert_batch(&[pid]).unwrap();
        }
        assert_invariants(&c, nn);

        // Far more than the single seed centroid; roughly `~ 2 * nn / threshold`.
        assert!(c.num_clusters() > 10, "got {}", c.num_clusters());
        let mean = nn as f64 / c.num_clusters() as f64;
        assert!(mean <= 21.0, "mean cluster size {mean} exceeds threshold");
    }

    #[test]
    fn batched_inserts_preserve_invariants_and_split() {
        // The batched path defers splitting to the end of a batch, so several
        // clusters overflow at once and are re-clustered together. The partition
        // it lands on differs from the streaming path's, but the structural
        // invariants and the +1-live-cluster-per-split accounting are the same.
        let mut rng = StdRng::seed_from_u64(13);
        let (nn, dim) = (900usize, 8usize);
        let mut v = vec![0.0f32; nn * dim];
        for x in v.iter_mut() {
            *x = rng.random_range(-1.0..1.0);
        }
        let points = mat(v, nn, dim);

        let mut ib = vec![0.0f32; 4 * dim];
        for i in 0..4 {
            ib[i * dim..(i + 1) * dim].copy_from_slice(points.row(rng.random_range(0..nn)));
        }
        let initial = mat(ib, 4, dim);

        let mut p = params(64, 40);
        p.centroid_capacity = 4 * nn;

        let mut c = OnlineClusterer::new(points.clone(), initial, p).unwrap();
        let ids: Vec<u32> = (0..nn as u32).collect();
        for batch in ids.chunks(128) {
            c.insert_batch(batch).unwrap();
        }

        assert_invariants(&c, nn);
        assert!(c.num_clusters() > 4, "batches should overflow clusters");
        assert_eq!(
            c.num_clusters(),
            4 + c.telemetry().total_splits as usize,
            "every split retires one parent and allocates two children"
        );
        assert_eq!(c.telemetry().total_inserts, nn as u64);

        // A batch's splits are re-clustered jointly, so several events share one
        // timestamp, but the timeline is still ordered.
        let mut prev = 0u64;
        for e in &c.telemetry().splits {
            assert!(e.insert_index >= prev);
            prev = e.insert_index;
        }

        // Local assignment can only be worse than the optimal one for the
        // centroid set it produced.
        let opt = optimal_residual(&points, &live_centroids(&c));
        assert!(c.residual() >= opt - 1e-3);
    }

    #[test]
    fn batched_inserts_respect_max_clusters() {
        // Admission control has to hold even when a single batch overflows more
        // clusters than the cap has room for.
        let mut rng = StdRng::seed_from_u64(17);
        let (nn, dim) = (800usize, 4usize);
        let mut v = vec![0.0f32; nn * dim];
        for x in v.iter_mut() {
            *x = rng.random_range(-1.0..1.0);
        }
        let points = mat(v, nn, dim);
        let initial = mat(points.row(0).to_vec(), 1, dim);

        let mut p = params(6, 10);
        p.centroid_capacity = 4 * nn;

        let mut c = OnlineClusterer::new(points, initial, p).unwrap();
        let ids: Vec<u32> = (0..nn as u32).collect();
        for batch in ids.chunks(200) {
            c.insert_batch(batch).unwrap();
        }
        assert_invariants(&c, nn);
        assert!(c.num_clusters() <= 6, "got {}", c.num_clusters());
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
            c.insert_batch(&[pid]).unwrap();
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
            c.insert_batch(&[pid]).unwrap();
        }

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

    // ----- deletes and merges -----

    /// Four tight groups of `per` points at the corners of a 30x30 square,
    /// returned with the corners themselves as the initial centroid set. Every
    /// point is unambiguously nearest its own corner, so routing is exact and
    /// the starting partition is known: group `i` occupies cluster `i`.
    fn four_groups(per: usize) -> (Matrix<f32>, Matrix<f32>) {
        const CORNERS: [[f32; 2]; 4] = [[0.0, 0.0], [30.0, 0.0], [0.0, 30.0], [30.0, 30.0]];
        let mut rng = StdRng::seed_from_u64(4242);
        let mut v = Vec::with_capacity(4 * per * 2);
        for c in CORNERS {
            for _ in 0..per {
                v.push(c[0] + rng.random_range(-0.5..0.5));
                v.push(c[1] + rng.random_range(-0.5..0.5));
            }
        }
        let points = mat(v, 4 * per, 2);
        let initial = mat(CORNERS.iter().flatten().copied().collect(), 4, 2);
        (points, initial)
    }

    #[test]
    fn delete_removes_points_without_merging() {
        // Merging disabled: deleting only shrinks lists, never the cluster set.
        let (points, initial) = four_groups(5);
        let mut c = OnlineClusterer::new(points, initial, params(8, 10_000)).unwrap();
        c.insert_batch(&(0..20u32).collect::<Vec<_>>()).unwrap();
        assert_eq!(c.cluster_sizes(), vec![5, 5, 5, 5]);

        // Delete across two clusters at once, out of order, to exercise the
        // group-by-cluster path.
        c.delete_batch(&[12, 0, 3, 11]).unwrap();

        let live: Vec<u32> = (0..20u32).filter(|p| ![0, 3, 11, 12].contains(p)).collect();
        assert_live_invariants(&c, &live);
        assert_eq!(c.num_clusters(), 4, "merging is off; no cluster dissolves");
        assert_eq!(c.cluster_sizes().iter().sum::<usize>(), 16);
        assert_eq!(c.telemetry().total_deletes, 4);
        assert_eq!(c.telemetry().total_merges, 0);
    }

    #[test]
    fn delete_batch_is_idempotent_within_a_batch() {
        // A pid repeated inside one batch is deduplicated rather than being
        // counted twice or corrupting the list.
        let (points, initial) = four_groups(5);
        let mut c = OnlineClusterer::new(points, initial, params(8, 10_000)).unwrap();
        c.insert_batch(&(0..20u32).collect::<Vec<_>>()).unwrap();

        c.delete_batch(&[7, 7, 7]).unwrap();
        let live: Vec<u32> = (0..20u32).filter(|&p| p != 7).collect();
        assert_live_invariants(&c, &live);
        assert_eq!(c.telemetry().total_deletes, 1);
    }

    #[test]
    fn delete_rejects_absent_and_out_of_range_points() {
        let (points, initial) = four_groups(5);
        let mut c = OnlineClusterer::new(points, initial, params(8, 10_000)).unwrap();
        c.insert_batch(&(0..10u32).collect::<Vec<_>>()).unwrap();

        // Never inserted.
        assert!(c.delete_batch(&[15]).is_err());
        // Past the end of the corpus.
        assert!(c.delete_batch(&[100]).is_err());
        // Already deleted.
        c.delete_batch(&[2]).unwrap();
        assert!(c.delete_batch(&[2]).is_err());

        // A rejected batch leaves the index exactly as it was.
        let live: Vec<u32> = (0..10u32).filter(|&p| p != 2).collect();
        assert_live_invariants(&c, &live);
    }

    #[test]
    fn insert_rejects_points_already_present() {
        let (points, initial) = four_groups(5);
        let mut c = OnlineClusterer::new(points, initial, params(8, 10_000)).unwrap();
        c.insert_batch(&[0, 1, 2]).unwrap();

        assert!(c.insert_batch(&[1]).is_err(), "re-insert must be rejected");
        assert!(
            c.insert_batch(&[5, 5]).is_err(),
            "a pid twice in one batch must be rejected"
        );
        assert_live_invariants(&c, &[0, 1, 2]);
    }

    #[test]
    fn deleted_point_can_be_reinserted() {
        // Delete/insert of the same pid is the churn pattern a streaming
        // runbook produces; the point must come back on a live cluster.
        let (points, initial) = four_groups(5);
        let mut c = OnlineClusterer::new(points, initial, params(8, 10_000)).unwrap();
        c.insert_batch(&(0..20u32).collect::<Vec<_>>()).unwrap();

        c.delete_batch(&[4, 9]).unwrap();
        c.insert_batch(&[9, 4]).unwrap();

        assert_live_invariants(&c, &(0..20u32).collect::<Vec<_>>());
        // Points 4 and 9 belong to groups 0 and 1, and route back there.
        assert_eq!(c.partition.assignment(4), c.partition.assignment(0));
        assert_eq!(c.partition.assignment(9), c.partition.assignment(5));
        assert_eq!(c.telemetry().total_inserts, 22);
        assert_eq!(c.telemetry().total_deletes, 2);
    }

    #[test]
    fn underflow_retires_the_cluster_and_scatters_it_onto_survivors() {
        let (points, initial) = four_groups(5);
        let budget_before = {
            let c =
                OnlineClusterer::new(points.clone(), initial.clone(), merge_params(8, 10_000, 3))
                    .unwrap();
            c.table.alloc_budget()
        };

        let mut c = OnlineClusterer::new(points, initial, merge_params(8, 10_000, 3)).unwrap();
        c.insert_batch(&(0..20u32).collect::<Vec<_>>()).unwrap();
        assert_eq!(c.num_clusters(), 4);

        // Group 0 drops to 2 members, below the merge threshold of 3.
        c.delete_batch(&[0, 1, 2]).unwrap();

        let live: Vec<u32> = (3..20u32).collect();
        assert_live_invariants(&c, &live);
        assert_eq!(
            c.num_clusters(),
            3,
            "the cell is gone and nothing replaces it: net -1"
        );
        assert_eq!(
            c.table.alloc_budget(),
            budget_before,
            "retiring a cluster fits nothing, so it consumes no id"
        );

        // The starved cluster's points land on a live cluster, not dropped.
        let a3 = c.partition.assignment(3);
        assert_ne!(a3, UNASSIGNED);
        assert!(c.table.is_live(a3));
        assert_eq!(c.cluster_sizes().iter().sum::<usize>(), 17);

        let t = c.telemetry();
        assert_eq!(t.total_deletes, 3);
        assert_eq!(t.total_merges, 1);
        assert_eq!(t.merges.len(), 1);
        let e = t.merges[0];
        assert_eq!(e.victim_size, 2, "what was left of the starved cluster");
        assert_eq!(e.live_after, 3);
        assert_eq!(e.op_index, 23, "20 inserts followed by 3 deletes");
        assert_eq!(
            e.num_reassigned, 2,
            "only the victim's own members are re-placed"
        );
    }

    #[test]
    fn retiring_adjacent_clusters_never_lands_a_point_on_a_retired_cell() {
        // Groups 0 and 1 are each other's nearest neighbors and both starve in
        // the same batch. Retiring the whole batch before placing any point is
        // what stops one victim's members landing on the other.
        let (points, initial) = four_groups(5);
        let mut c = OnlineClusterer::new(points, initial, merge_params(8, 10_000, 3)).unwrap();
        c.insert_batch(&(0..20u32).collect::<Vec<_>>()).unwrap();

        c.delete_batch(&[0, 1, 2, 5, 6, 7]).unwrap();

        let live: Vec<u32> = (0..20u32)
            .filter(|p| ![0, 1, 2, 5, 6, 7].contains(p))
            .collect();
        assert_live_invariants(&c, &live);
        assert_eq!(
            c.telemetry().total_merges,
            2,
            "both victims go in one batch"
        );
        assert_eq!(c.num_clusters(), 2);
        assert_eq!(c.cluster_sizes().iter().sum::<usize>(), 14);
    }

    #[test]
    fn merges_run_with_an_exhausted_id_budget() {
        // `centroid_capacity == initial` leaves no allocations at all. Only
        // splits draw on that budget, so an underfull cluster still retires.
        let (points, initial) = four_groups(5);
        let p = OnlineParams {
            centroid_capacity: 4,
            ..merge_params(8, 10_000, 3)
        };
        let mut c = OnlineClusterer::new(points, initial, p).unwrap();
        c.insert_batch(&(0..20u32).collect::<Vec<_>>()).unwrap();
        assert_eq!(c.table.alloc_budget(), 0);

        c.delete_batch(&[0, 1, 2]).unwrap();

        assert_live_invariants(&c, &(3..20u32).collect::<Vec<_>>());
        assert_eq!(c.telemetry().total_merges, 1);
        assert_eq!(c.num_clusters(), 3);
    }

    #[test]
    fn merges_stop_at_the_min_clusters_floor() {
        // The same starvation, but the floor forbids giving up any cluster.
        let (points, initial) = four_groups(5);
        let p = OnlineParams {
            min_clusters: 4,
            ..merge_params(8, 10_000, 3)
        };
        let mut c = OnlineClusterer::new(points, initial, p).unwrap();
        c.insert_batch(&(0..20u32).collect::<Vec<_>>()).unwrap();

        c.delete_batch(&[0, 1, 2]).unwrap();

        assert_live_invariants(&c, &(3..20u32).collect::<Vec<_>>());
        assert_eq!(c.num_clusters(), 4, "the floor holds the cluster open");
        assert_eq!(c.telemetry().total_merges, 0);
        assert_eq!(c.cluster_sizes(), vec![2, 5, 5, 5]);
    }

    #[test]
    fn rejects_merge_threshold_without_hysteresis() {
        // A merge floor at more than half the split ceiling means a freshly
        // split cluster is immediately a merge candidate.
        let (points, initial) = four_groups(5);
        let mut p = params(8, 30);
        p.merge_threshold = 20;
        assert!(OnlineClusterer::new(points.clone(), initial.clone(), p).is_err());

        // Exactly half is the tightest setting that is accepted.
        p.merge_threshold = 15;
        assert!(OnlineClusterer::new(points, initial, p).is_ok());
    }

    #[test]
    fn merge_telemetry_csv_has_one_row_per_merge() {
        // Groups 0 and 3 sit on opposite corners, so the two retirements are
        // independent and each scatters onto a different survivor.
        let (points, initial) = four_groups(5);
        let mut c = OnlineClusterer::new(points, initial, merge_params(8, 10_000, 3)).unwrap();
        c.insert_batch(&(0..20u32).collect::<Vec<_>>()).unwrap();
        c.delete_batch(&[0, 1, 2, 15, 16, 17]).unwrap();

        let t = c.telemetry();
        assert_eq!(t.total_merges, 2);

        let dir = std::env::temp_dir().join(format!("graphivf_merge_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let csv = dir.join("merges.csv");
        t.write_merges_csv(&csv).unwrap();
        let text = std::fs::read_to_string(&csv).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            lines[0],
            "op_index,victim,victim_size,num_neighbors,num_reassigned,\
             live_after,search_us,reassign_us,total_us"
        );
        assert_eq!(lines.len(), 1 + t.merges.len());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn flush_after_deletes_drops_them_and_keeps_original_ids() {
        // The flushed index holds exactly the live points, still labelled by
        // their corpus row, so groundtruth computed over the corpus lines up.
        let (points, n) = two_blobs(50, 3);
        let initial = mat(vec![10.0, 10.0], 1, 2);
        let mut c = OnlineClusterer::new(points.clone(), initial, params(2, 25)).unwrap();
        c.insert_batch(&(0..n as u32).collect::<Vec<_>>()).unwrap();

        let removed: Vec<u32> = (0..10u32).collect();
        c.delete_batch(&removed).unwrap();

        let dir = std::env::temp_dir().join(format!("graphivf_del_flush_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let prefix = dir.join("idx");
        c.flush(&prefix, c.points.as_view()).unwrap();

        let index = GraphIvfIndex::<f32>::load(&prefix, 2).unwrap();
        let mut searcher = index.searcher().unwrap();
        let sp = SearchParams {
            nlist: index.num_clusters(),
            centroid_search_l: 16,
        };
        // Scanning every list returns the whole live corpus and nothing else.
        let results = searcher.search(&[0.0f32, 0.0], n, &sp).unwrap();
        assert_eq!(results.len(), n - removed.len());
        for (id, _) in &results {
            assert!(!removed.contains(id), "deleted point {id} was returned");
            assert!((*id as usize) < n);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ----- online search -----

    /// Exact top-`k` over `live`, as ids in ascending-distance order.
    fn brute_force(points: &Matrix<f32>, live: &[u32], q: &[f32], k: usize) -> Vec<u32> {
        let mut v: Vec<(u32, f64)> = live
            .iter()
            .map(|&p| (p, sqd(points.row(p as usize), q)))
            .collect();
        v.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        v.into_iter().take(k).map(|(p, _)| p).collect()
    }

    #[test]
    fn online_search_probing_everything_is_exact() {
        // Probing every list turns the search into a full scan, so it must
        // reproduce brute force exactly.
        let (points, n) = two_blobs(50, 31);
        let initial = mat(vec![10.0, 10.0], 1, 2);
        let mut c = OnlineClusterer::new(points.clone(), initial, params(4, 25)).unwrap();
        c.insert_batch(&(0..n as u32).collect::<Vec<_>>()).unwrap();

        let sp = SearchParams {
            nlist: c.num_clusters(),
            centroid_search_l: 16,
        };
        let live: Vec<u32> = (0..n as u32).collect();
        let mut s = c.searcher().unwrap();
        for q in [[0.0f32, 0.0], [20.0, 20.0], [10.0, 10.0]] {
            let got: Vec<u32> = s
                .search(&q, 10, &sp)
                .unwrap()
                .into_iter()
                .map(|r| r.0)
                .collect();
            assert_eq!(got, brute_force(&points, &live, &q, 10));
        }
    }

    #[test]
    fn online_search_after_deletes_and_merges_sees_only_live_points() {
        let (points, initial) = four_groups(5);
        let mut c =
            OnlineClusterer::new(points.clone(), initial, merge_params(8, 10_000, 3)).unwrap();
        c.insert_batch(&(0..20u32).collect::<Vec<_>>()).unwrap();
        // Starves group 0 into a merge, so the query's own cluster is gone.
        c.delete_batch(&[0, 1, 2]).unwrap();

        let live: Vec<u32> = (3..20u32).collect();
        let sp = SearchParams {
            nlist: c.num_clusters(),
            centroid_search_l: 16,
        };
        let mut s = c.searcher().unwrap();
        let got: Vec<u32> = s
            .search(&[0.0f32, 0.0], 20, &sp)
            .unwrap()
            .into_iter()
            .map(|r| r.0)
            .collect();

        assert_eq!(got.len(), live.len(), "deleted points must not be returned");
        assert_eq!(got, brute_force(&points, &live, &[0.0, 0.0], 20));
    }

    #[test]
    fn points_scanned_counts_every_probed_list_member() {
        // The scan count is what recall should be read against, so it has to be
        // the exact list volume the probe touched, not an estimate from nlist.
        let (points, initial) = four_groups(5);
        let mut c = OnlineClusterer::new(points, initial, params(8, 10_000)).unwrap();
        c.insert_batch(&(0..20u32).collect::<Vec<_>>()).unwrap();

        let mut s = c.searcher().unwrap();
        assert_eq!(s.points_scanned(), 0);

        let sp = SearchParams {
            nlist: 1,
            centroid_search_l: 8,
        };
        s.search(&[0.0f32, 0.0], 3, &sp).unwrap();
        assert_eq!(s.points_scanned(), 5, "one group's list, not the top-3");

        // Accumulates across queries, and grows with the probe width.
        let sp = SearchParams {
            nlist: 4,
            centroid_search_l: 8,
        };
        s.search(&[0.0f32, 0.0], 3, &sp).unwrap();
        assert_eq!(
            s.points_scanned(),
            25,
            "5 from the first query, then all 20"
        );
    }

    #[test]
    fn online_search_returns_fewer_than_k_when_lists_are_short() {
        let (points, initial) = four_groups(5);
        let mut c = OnlineClusterer::new(points, initial, params(8, 10_000)).unwrap();
        c.insert_batch(&(0..20u32).collect::<Vec<_>>()).unwrap();

        let sp = SearchParams {
            nlist: 1,
            centroid_search_l: 8,
        };
        let mut s = c.searcher().unwrap();
        let got = s.search(&[0.0f32, 0.0], 50, &sp).unwrap();
        assert_eq!(got.len(), 5, "one list holds one group");
        assert!(got.windows(2).all(|w| w[0].1 <= w[1].1));
    }

    #[test]
    fn online_search_rejects_bad_queries() {
        let (points, initial) = four_groups(5);
        let mut c = OnlineClusterer::new(points, initial, params(8, 10_000)).unwrap();
        c.insert_batch(&(0..20u32).collect::<Vec<_>>()).unwrap();

        let sp = SearchParams {
            nlist: 2,
            centroid_search_l: 8,
        };
        let mut s = c.searcher().unwrap();
        assert!(s.search(&[0.0, 0.0], 0, &sp).is_err(), "k must be non-zero");
        assert!(s.search(&[0.0, 0.0, 0.0], 5, &sp).is_err(), "wrong dim");
        let too_many = SearchParams {
            nlist: 99,
            centroid_search_l: 99,
        };
        assert!(s.search(&[0.0, 0.0], 5, &too_many).is_err());
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
            c.insert_batch(&[pid]).unwrap();
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

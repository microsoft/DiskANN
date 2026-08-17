/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Tunable parameters for building and searching a graph-IVF index.

use diskann_vector::distance::Metric as VectorMetric;

/// Distance metric.
///
/// `L2` and `Cosine` both score by squared-L2; cosine has the same ordering for
/// unit vectors. A plain static build normalizes its corpus copy. Callers must
/// supply normalized queries, and build paths that store pre-encoded or online
/// corpus rows verbatim require those rows to be normalized before construction.
/// `InnerProduct` is a *hybrid* metric intended for maximum-inner-product (MIPS)
/// datasets: the index is still **built** (clustering, centroid assignment)
/// under squared-L2. After an
/// index is flushed and loaded, both centroids and inverted-list points are
/// scored by inner product (larger is better). Live [`OnlineSearcher`] queries
/// retain L2 navigation in the mutable centroid graph while scoring list
/// candidates by inner product.
///
/// [`OnlineSearcher`]: crate::OnlineSearcher
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Metric {
    /// Squared Euclidean distance.
    L2,
    /// Cosine similarity over L2-normalized vectors, ranked by squared-L2.
    Cosine,
    /// Maximum inner product. Build (clustering + assignment) uses squared-L2.
    /// A loaded index scores centroids and list points by inner product; live
    /// online search keeps L2 centroid navigation and scores list points by
    /// inner product.
    InnerProduct,
}

impl Metric {
    pub(crate) fn as_u8(self) -> u8 {
        match self {
            Metric::L2 => 0,
            Metric::Cosine => 1,
            Metric::InnerProduct => 2,
        }
    }

    pub(crate) fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Metric::L2),
            1 => Some(Metric::Cosine),
            2 => Some(Metric::InnerProduct),
            _ => None,
        }
    }

    /// Whether a static build should L2-normalize its clustering corpus.
    pub(crate) fn normalizes(self) -> bool {
        matches!(self, Metric::Cosine)
    }

    /// The [`diskann_vector`] distance used to score list candidates and to
    /// construct/navigate the immutable centroid graph when an index is loaded.
    ///
    /// `InnerProduct` scores by (negated) inner product so queries reach the
    /// maximum-inner-product neighbors; `L2` and `Cosine` score by squared-L2.
    /// Cosine corpus and query vectors must be normalized before scoring, making
    /// L2 order equivalent to cosine. Clustering and the mutable online centroid
    /// graph always use squared-L2 regardless of this value, so live online
    /// search uses this distance only for candidate scoring.
    pub(crate) fn search_metric(self) -> VectorMetric {
        match self {
            Metric::InnerProduct => VectorMetric::InnerProduct,
            Metric::L2 | Metric::Cosine => VectorMetric::L2,
        }
    }
}

/// Construction parameters for the in-memory centroid graph.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GraphParams {
    /// Pruned out-degree (`R`).
    pub degree: usize,
    /// Maximum out-degree as a multiple of `degree` (slack, `>= 1.0`).
    pub slack: f32,
    /// Search-list size used during graph construction (`L`).
    pub l_build: usize,
    /// Pruning alpha (`>= 1.0`).
    pub alpha: f32,
}

impl Default for GraphParams {
    fn default() -> Self {
        Self {
            degree: 32,
            slack: 1.2,
            l_build: 64,
            alpha: 1.2,
        }
    }
}

/// Strategy for assigning corpus points to their nearest centroid during the
/// k-means (Lloyd's) iterations that refine the centroids.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum AssignMethod {
    /// Exact brute-force nearest-centroid assignment (GEMM-based). Cost is
    /// `O(num_points * num_clusters * dim)` per iteration; the historical
    /// default and the most accurate option.
    #[default]
    Exact,
    /// Graph-accelerated approximate nearest-centroid assignment. Builds an
    /// in-memory graph over the centroids and searches it for each point,
    /// optionally re-ranking the top `rerank` candidates exactly. Scales to
    /// large `num_clusters` where the exact scan is intractable.
    ///
    /// The graph built here is scratch for the k-means refinement and is
    /// unrelated to the one [`CentroidRouting::Graph`] may build, so it carries
    /// its own recipe and beam.
    Graph {
        /// Rebuild the centroid graph every this many iterations (`1` rebuilds
        /// every iteration). Clamped to `>= 1`.
        rebuild_every: usize,
        /// Number of graph candidates to re-rank exactly per point (`1` trusts
        /// the graph's nearest result directly). Clamped to `>= 1`.
        rerank: usize,
        /// Construction parameters for the refinement graph.
        #[serde(default)]
        graph: GraphParams,
        /// Search-list size for the per-point walk during refinement.
        #[serde(default = "default_assign_l")]
        assign_l: usize,
    },
}

/// How the index finds the centroids nearest to a vector.
///
/// This is a single index-wide choice: it governs every centroid lookup the
/// index performs — routing points on insert, locating neighbors when a cluster
/// splits or is dissolved, and selecting the clusters a query probes. Keeping it
/// to one setting means the clusters a query probes are found the same way the
/// points in them were routed, so the two can never disagree about what
/// "nearest" means.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CentroidSearch {
    /// Navigate a DiskANN graph built over the centroids.
    ///
    /// Sub-linear in the cluster count and the only option that stays practical
    /// as clusters grow into the millions, at the cost of occasionally missing a
    /// nearest cluster. On an index that churns, the graph also accumulates
    /// tombstones, and the beam has to be widened
    /// ([`SearchParams::centroid_search_alpha`]) to compensate.
    #[default]
    Graph,
    /// Score every live centroid.
    ///
    /// Exact by construction: the nearest clusters are never missed, retired
    /// clusters can never be returned, and no beam parameter applies. Scoring is
    /// done with a batched matrix multiply, so the cost is
    /// `O(num_clusters * dim)` per vector at memory-bandwidth speed rather than
    /// per-candidate speed — practical well beyond the point where a naive scan
    /// is not, but still linear in the cluster count.
    ///
    /// No centroid graph is built or maintained in this mode, since nothing
    /// would search it.
    Exact,
}

/// Centroid routing for a batch build: the [`CentroidSearch`] mode together
/// with the knobs that mode actually consumes.
///
/// [`CentroidSearch`] alone is enough to *open* an existing index, because a
/// loaded graph is rebuilt from the recipe in the index metadata. Building one
/// additionally needs a beam width and a graph recipe, and both are meaningless
/// under [`Exact`](Self::Exact). Keeping them inside the
/// [`Graph`](Self::Graph) variant makes supplying an ignored knob impossible
/// rather than merely useless.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "mode", rename_all = "lowercase", deny_unknown_fields)]
pub enum CentroidRouting {
    /// Navigate a DiskANN graph built over the centroids.
    Graph {
        /// Centroid graph construction parameters.
        #[serde(default)]
        graph: GraphParams,
        /// Search-list size used when assigning corpus points to centroids.
        #[serde(default = "default_assign_l")]
        assign_l: usize,
    },
    /// Score every live centroid with a batched matrix multiply.
    Exact,
}

impl Default for CentroidRouting {
    fn default() -> Self {
        Self::Graph {
            graph: GraphParams::default(),
            assign_l: default_assign_l(),
        }
    }
}

fn default_assign_l() -> usize {
    32
}

impl CentroidRouting {
    /// Routing for an index being opened.
    ///
    /// The graph recipe comes from the index metadata rather than the caller,
    /// and the assignment beam is left at its default because opening an index
    /// assigns nothing — [`assign`](crate::GraphIvfIndex) already ran at build
    /// time and the result is on disk.
    pub(crate) fn for_load(mode: CentroidSearch, graph: GraphParams) -> Self {
        match mode {
            CentroidSearch::Graph => Self::Graph {
                graph,
                assign_l: default_assign_l(),
            },
            CentroidSearch::Exact => Self::Exact,
        }
    }

    /// Graph recipe recorded in the index metadata so a later
    /// [`load`](crate::GraphIvfIndex::load) with [`CentroidSearch::Graph`] can
    /// rebuild one over the flushed centroids.
    ///
    /// An exact build never constructs a graph, so it has no recipe of its own
    /// to record and the defaults stand in.
    pub(crate) fn stored_graph_params(self) -> GraphParams {
        match self {
            Self::Graph { graph, .. } => graph,
            Self::Exact => GraphParams::default(),
        }
    }

    fn validate(&self) -> crate::Result<()> {
        use crate::GraphIvfError as E;
        if let Self::Graph { graph, assign_l } = self {
            if *assign_l == 0 {
                return Err(E::invalid("assign_l must be non-zero"));
            }
            if graph.degree == 0 || graph.l_build == 0 {
                return Err(E::invalid("graph degree and l_build must be non-zero"));
            }
        }
        Ok(())
    }
}

/// Centroid routing for an online build.
///
/// Distinct from [`CentroidRouting`] because a streaming build also searches
/// the centroids when a cluster splits or dissolves, which a batch build never
/// does, so the `Graph` variant carries one more beam width.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OnlineCentroidRouting {
    /// Navigate a mutable DiskANN graph maintained over the live centroids.
    Graph {
        /// Centroid graph construction parameters.
        graph: GraphParams,
        /// Search-list size used to route each inserted point.
        assign_l: usize,
        /// Search-list size for split-neighbor and dissolve-survivor selection.
        /// Raised internally to fit the requested candidates and exclusions.
        reassign_l: usize,
    },
    /// Score every live centroid with a batched matrix multiply.
    Exact,
}

impl Default for OnlineCentroidRouting {
    fn default() -> Self {
        Self::Graph {
            graph: GraphParams::default(),
            assign_l: default_assign_l(),
            reassign_l: default_assign_l(),
        }
    }
}

impl OnlineCentroidRouting {
    /// See [`CentroidRouting::stored_graph_params`].
    pub(crate) fn stored_graph_params(self) -> GraphParams {
        match self {
            Self::Graph { graph, .. } => graph,
            Self::Exact => GraphParams::default(),
        }
    }

    /// Beam width for routing one point, and the widened retry that follows it.
    /// `None` in exact mode, where no beam applies.
    pub(crate) fn route_beams(self) -> Option<(usize, usize)> {
        match self {
            Self::Graph { assign_l, .. } => {
                let base = assign_l.max(1);
                Some((base, base.saturating_mul(8).max(512)))
            }
            Self::Exact => None,
        }
    }

    /// Beam width for split-neighbor and dissolve-survivor selection, floored to
    /// return at least `want` candidates. `None` in exact mode.
    pub(crate) fn neighbor_beam(self, want: usize) -> Option<usize> {
        match self {
            Self::Graph { reassign_l, .. } => Some(reassign_l.max(want)),
            Self::Exact => None,
        }
    }

    pub(crate) fn validate(&self) -> crate::Result<()> {
        use crate::GraphIvfError as E;
        if let Self::Graph {
            graph,
            assign_l,
            reassign_l,
        } = self
        {
            if *assign_l == 0 || *reassign_l == 0 {
                return Err(E::invalid("assign_l and reassign_l must be non-zero"));
            }
            if graph.degree == 0 || graph.l_build == 0 {
                return Err(E::invalid("graph degree and l_build must be non-zero"));
            }
        }
        Ok(())
    }
}

/// Policy for centroids whose cluster received no points in a Lloyd's iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum EmptyClusterPolicy {
    /// Zero the centroid (legacy behavior of the brute-force k-means driver).
    Zero,
    /// Keep the centroid at its previous position.
    #[default]
    PreserveOld,
    /// Move the centroid onto the corpus point farthest from its assigned
    /// centroid, splitting the most spread-out cluster.
    ReseedFarthest,
}

/// Parameters controlling an index build.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct BuildParams {
    /// Number of clusters / centroids (`k`).
    pub num_clusters: usize,
    /// Distance metric.
    pub metric: Metric,
    /// Number of corpus points to sample for k-means training.
    pub sample_size: usize,
    /// Number of Lloyd's iterations for k-means.
    pub kmeans_iters: usize,
    /// How the built index finds nearest centroids when assigning the corpus,
    /// and the knobs that mode needs.
    #[serde(default)]
    pub routing: CentroidRouting,
    /// Number of worker threads to use during the build.
    pub num_threads: usize,
    /// RNG seed for sampling and k-means (for reproducibility).
    pub seed: u64,
    /// Nearest-centroid assignment strategy used during the k-means refinement.
    pub assign_method: AssignMethod,
    /// Policy for clusters that become empty during k-means refinement.
    pub empty_clusters: EmptyClusterPolicy,
    /// L2-normalize every centroid onto the unit sphere after each Lloyd's
    /// update. Useful for unit-normalized corpora where the raw cluster mean
    /// (which shrinks inward) is a worse angular representative than its
    /// projection back onto the sphere.
    pub normalize_centroids: bool,
}

impl BuildParams {
    pub(crate) fn validate(&self, num_points: usize, dim: usize) -> crate::Result<()> {
        use crate::GraphIvfError as E;
        if dim == 0 {
            return Err(E::invalid("dim must be non-zero"));
        }
        if num_points == 0 {
            return Err(E::invalid("corpus is empty"));
        }
        if self.num_clusters == 0 {
            return Err(E::invalid("num_clusters must be non-zero"));
        }
        if self.num_clusters > num_points {
            return Err(E::invalid(format!(
                "num_clusters ({}) cannot exceed number of points ({num_points})",
                self.num_clusters
            )));
        }
        if self.sample_size < self.num_clusters {
            return Err(E::invalid(format!(
                "sample_size ({}) must be >= num_clusters ({})",
                self.sample_size, self.num_clusters
            )));
        }
        if self.num_threads == 0 {
            return Err(E::invalid("num_threads must be non-zero"));
        }
        self.routing.validate()?;
        Ok(())
    }

    /// Sample size actually used, clamped to the corpus size and the k-means++
    /// limit of `2^23` points.
    ///
    /// [`CentroidInit::Forgy`] takes an unclamped `samples` count, so callers
    /// constructing one by hand should route the configured sample size through
    /// here first.
    ///
    /// [`CentroidInit::Forgy`]: crate::CentroidInit::Forgy
    pub fn effective_sample_size(&self, num_points: usize) -> usize {
        const KMEANSPP_MAX: usize = 1 << 23;
        self.sample_size.min(num_points).min(KMEANSPP_MAX)
    }
}

/// Parameters for an online (incremental) [`OnlineClusterer`] build.
///
/// [`OnlineClusterer`]: crate::OnlineClusterer
#[derive(Debug, Clone, Copy)]
pub struct OnlineParams {
    /// Optional cap on the number of live clusters. `Some(k)` stops splitting
    /// once `k` live clusters exist, reproducing a fixed target granularity.
    /// `None` lets the partition grow driven solely by `split_threshold`:
    /// splitting continues for every inserted point and the final cluster count
    /// emerges from the data and threshold. This is independent of
    /// [`centroid_capacity`](Self::centroid_capacity), the hard resource bound.
    pub max_clusters: Option<usize>,
    /// Total centroid id slots the internal mutable centroid graph (and the
    /// id-indexed side tables) pre-allocate. Every split permanently retires the
    /// parent id and allocates two children, so the ids consumed over a build is
    /// `initial + 2 * splits`; size this to roughly `2 *` the expected final
    /// live-cluster count. Splitting stops when the slots are exhausted,
    /// whatever [`max_clusters`](Self::max_clusters) says.
    pub centroid_capacity: usize,
    /// A cluster is split once it holds strictly more than this many points.
    /// Must be `>= 2`.
    pub split_threshold: usize,
    /// Number of nearest centroid clusters (besides the two children) drawn in
    /// as reassignment candidates when a cluster is split, and the maximum
    /// survivor landing sites considered when a cluster is dissolved. Must be
    /// `>= 1`.
    ///
    /// A candidate *count*, so it applies whichever routing mode is in use.
    pub reassign_neighbors: usize,
    /// Lloyd iterations for split k-means (two children per admitted parent).
    pub two_means_iters: usize,
    /// A cluster is retired once deletes drop it below this many points. `0`
    /// disables merging entirely: deletes still remove points, but the
    /// partition only ever gains clusters.
    ///
    /// Retiring dissolves the cluster: it is removed from the centroid graph
    /// and its members are scattered onto their nearest survivors. No centroid
    /// is fitted and no id is consumed, so deletes are free against the
    /// [`centroid_capacity`](Self::centroid_capacity) budget.
    ///
    /// Must leave a hysteresis gap below
    /// [`split_threshold`](Self::split_threshold) — `2 * merge_threshold <=
    /// split_threshold` — so that the children of a fresh split cannot
    /// immediately be merged again.
    pub merge_threshold: usize,
    /// Floor on the live cluster count: a merge is skipped if it would take the
    /// partition below this. Clamped to `>= 1` internally — orphans need at
    /// least one surviving cluster to land on.
    pub min_clusters: usize,
    /// How the clusterer finds nearest centroids — for routing, splitting,
    /// merging, and query-time cluster selection alike — and the knobs that
    /// mode needs.
    pub routing: OnlineCentroidRouting,
    /// Candidate-scoring metric for live search and the metric recorded in the
    /// flushed index metadata. Clustering and centroid-graph navigation always
    /// use squared-L2 (as in a batch build).
    pub metric: Metric,
    /// L2-normalize warmup and split-child centroids (for unit-normalized
    /// corpora).
    pub normalize_centroids: bool,
    /// Worker threads for split k-means, routing, and graph construction.
    pub num_threads: usize,
    /// RNG seed for split seeding (reproducibility).
    pub seed: u64,
}

impl Default for OnlineParams {
    /// Insert-only defaults: splitting enabled, merging off
    /// (`merge_threshold: 0`).
    ///
    /// Exists so callers can set the handful of fields they care about with
    /// `..Default::default()` rather than restating every knob.
    fn default() -> Self {
        Self {
            max_clusters: None,
            centroid_capacity: 1024,
            split_threshold: 256,
            reassign_neighbors: 8,
            two_means_iters: 10,
            merge_threshold: 0,
            min_clusters: 1,
            routing: OnlineCentroidRouting::default(),
            metric: Metric::L2,
            normalize_centroids: false,
            num_threads: 1,
            seed: 0,
        }
    }
}

impl OnlineParams {
    /// Live-cluster floor, never below one.
    pub(crate) fn effective_min_clusters(&self) -> usize {
        self.min_clusters.max(1)
    }

    /// Whether delete-driven cluster retirement is enabled.
    pub(crate) fn merges_enabled(&self) -> bool {
        self.merge_threshold > 0
    }
}

/// Floor on the centroid-graph search list returned by
/// [`SearchParams::effective_l`].
///
/// A greedy graph walk narrower than this has too little room to recover from a
/// poor entry point, and a beam this small is cheap regardless, so scaling below
/// it buys nothing.
pub const MIN_CENTROID_SEARCH_L: usize = 128;

/// Default [`SearchParams::centroid_search_alpha`].
///
/// Chosen by measuring centroid selection against an exact scan over a 30M
/// streaming runbook: 4.0 holds ~98% of the exactly-nearest clusters from 6k to
/// 60k clusters, while 1.5 holds only ~63% and costs nearly seven points of
/// end-to-end recall@50 to save well under a factor of two in latency.
pub const DEFAULT_CENTROID_SEARCH_ALPHA: f32 = 4.0;

/// Parameters controlling a single search.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct SearchParams {
    /// Number of nearest clusters to probe (inverted lists to fetch).
    pub nlist: usize,
    /// Centroid-graph search list as a multiple of `nlist`.
    ///
    /// The centroid beam is charged to every query, so a fixed size has to be
    /// picked for the largest `nlist` a workload will ever ask for and then
    /// overpays for every smaller one. On an index that grows or churns there is
    /// no single good value: sized for the peak it dominates the query at small
    /// cluster counts, sized for the start it silently degrades to a truncated
    /// walk later. Expressing the beam as a multiple of the request keeps the
    /// overshoot proportional at any index size.
    ///
    /// Graph-search accuracy tracks this ratio and is otherwise insensitive to
    /// how many clusters exist, so one value holds across the whole life of an
    /// index. It is also the parameter recall is most sensitive to: clusters
    /// missed here are missed before a single point is scanned, and no amount of
    /// scanning recovers them.
    ///
    /// Must be at least 1.0 — a search list shorter than `nlist` cannot return
    /// `nlist` clusters.
    pub centroid_search_alpha: f32,
}

impl SearchParams {
    /// Probe `nlist` clusters with the default oversampling.
    pub const fn new(nlist: usize) -> Self {
        Self {
            nlist,
            centroid_search_alpha: DEFAULT_CENTROID_SEARCH_ALPHA,
        }
    }

    pub(crate) fn validate(&self, num_clusters: usize) -> crate::Result<()> {
        use crate::GraphIvfError as E;
        if self.nlist == 0 {
            return Err(E::invalid("nlist must be non-zero"));
        }
        if self.nlist > num_clusters {
            return Err(E::invalid(format!(
                "nlist ({}) cannot exceed num_clusters ({num_clusters})",
                self.nlist
            )));
        }
        // Rejects NaN as well, which would otherwise reach `effective_l` and
        // cast to zero.
        if !self.centroid_search_alpha.is_finite() || self.centroid_search_alpha < 1.0 {
            return Err(E::invalid(format!(
                "centroid_search_alpha ({}) must be finite and at least 1.0",
                self.centroid_search_alpha
            )));
        }
        Ok(())
    }

    /// Search-list size to use: `alpha * nlist`, floored at
    /// [`MIN_CENTROID_SEARCH_L`].
    pub(crate) fn effective_l(&self) -> usize {
        // Float-to-int casts saturate in Rust, so an extreme alpha clamps to
        // usize::MAX rather than wrapping to a tiny beam.
        let scaled = (f64::from(self.centroid_search_alpha) * self.nlist as f64).ceil() as usize;
        scaled.max(MIN_CENTROID_SEARCH_L)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_u8_round_trips() {
        for m in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
            assert_eq!(Metric::from_u8(m.as_u8()), Some(m));
        }
        assert_eq!(Metric::from_u8(3), None);
        assert!(!Metric::L2.normalizes());
        assert!(Metric::Cosine.normalizes());
    }

    fn valid_build() -> BuildParams {
        BuildParams {
            num_clusters: 8,
            metric: Metric::L2,
            sample_size: 100,
            kmeans_iters: 10,
            routing: CentroidRouting::default(),
            num_threads: 2,
            seed: 0,
            assign_method: AssignMethod::Exact,
            empty_clusters: EmptyClusterPolicy::PreserveOld,
            normalize_centroids: false,
        }
    }
    #[test]
    fn build_validate_accepts_good_params() {
        assert!(valid_build().validate(100, 4).is_ok());
    }

    #[test]
    fn build_validate_rejects_bad_params() {
        assert!(valid_build().validate(100, 0).is_err()); // zero dim
        assert!(valid_build().validate(0, 4).is_err()); // empty corpus

        let mut p = valid_build();
        p.num_clusters = 0;
        assert!(p.validate(100, 4).is_err());

        let mut p = valid_build();
        p.num_clusters = 200; // exceeds num_points
        assert!(p.validate(100, 4).is_err());

        let mut p = valid_build();
        p.sample_size = 4; // < num_clusters
        assert!(p.validate(100, 4).is_err());

        let mut p = valid_build();
        p.routing = CentroidRouting::Graph {
            graph: GraphParams::default(),
            assign_l: 0,
        };
        assert!(p.validate(100, 4).is_err());

        let mut p = valid_build();
        p.num_threads = 0;
        assert!(p.validate(100, 4).is_err());

        let mut p = valid_build();
        p.routing = CentroidRouting::Graph {
            graph: GraphParams {
                degree: 0,
                ..GraphParams::default()
            },
            assign_l: 16,
        };
        assert!(p.validate(100, 4).is_err());
    }

    #[test]
    fn effective_sample_size_clamps() {
        let mut p = valid_build();
        p.sample_size = 1_000;
        // Clamped down to the corpus size.
        assert_eq!(p.effective_sample_size(100), 100);
        // Honored when within bounds.
        assert_eq!(p.effective_sample_size(10_000), 1_000);
        // Clamped to the k-means++ cap.
        p.sample_size = (1 << 23) + 5;
        assert_eq!(p.effective_sample_size(usize::MAX), 1 << 23);
    }

    #[test]
    fn search_validate_and_effective_l() {
        // Below the floor the beam is pinned to it.
        let p = SearchParams::new(4);
        assert!(p.validate(8).is_ok());
        assert_eq!(p.effective_l(), MIN_CENTROID_SEARCH_L);

        // Above the floor the beam scales with nlist and rounds up.
        let p2 = SearchParams {
            nlist: 1_000,
            centroid_search_alpha: 1.5,
        };
        assert_eq!(p2.effective_l(), 1_500);
        let p3 = SearchParams {
            nlist: 999,
            centroid_search_alpha: 1.5,
        };
        assert_eq!(p3.effective_l(), 1_499);

        // nlist must be non-zero and within the cluster count.
        assert!(SearchParams::new(0).validate(8).is_err());
        assert!(SearchParams::new(9).validate(8).is_err());

        // Alpha must be finite and leave room for nlist results.
        for bad in [0.5, f32::NAN, f32::INFINITY] {
            let p = SearchParams {
                nlist: 4,
                centroid_search_alpha: bad,
            };
            assert!(p.validate(8).is_err(), "alpha {bad} should be rejected");
        }
    }
}

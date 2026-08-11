/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Tunable parameters for building and searching a graph-IVF index.

use diskann_vector::distance::Metric as VectorMetric;

/// Distance metric.
///
/// `L2` and `Cosine` reduce everything to squared-L2 (`Cosine` additionally
/// L2-normalizes vectors at build and query time). `InnerProduct` is a *hybrid*
/// metric intended for maximum-inner-product (MIPS) datasets: the index is
/// still **built** (clustering, centroid assignment) under squared-L2, but at
/// **search** time both the centroids and the inverted-list points are scored
/// by inner product (larger is better).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Metric {
    /// Squared Euclidean distance.
    L2,
    /// Cosine similarity (vectors are L2-normalized).
    Cosine,
    /// Maximum inner product. Build (clustering + assignment) uses squared-L2;
    /// search scores centroids and inverted-list points by inner product.
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

    /// Whether vectors must be L2-normalized for this metric.
    pub(crate) fn normalizes(self) -> bool {
        matches!(self, Metric::Cosine)
    }

    /// The [`diskann_vector`] distance used at **search** time — both for
    /// scoring candidates and for navigating the centroid graph.
    ///
    /// `InnerProduct` scores by (negated) inner product so queries reach the
    /// maximum-inner-product neighbors; `L2` and `Cosine` score by squared-L2
    /// (`Cosine` vectors are normalized at build and query time, making L2 order
    /// equivalent to cosine). Clustering and the build-time centroid graph
    /// always use squared-L2 regardless of this value.
    pub(crate) fn search_metric(self) -> VectorMetric {
        match self {
            Metric::InnerProduct => VectorMetric::InnerProduct,
            Metric::L2 | Metric::Cosine => VectorMetric::L2,
        }
    }
}

/// Construction parameters for the in-memory centroid graph.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
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
    Graph {
        /// Rebuild the centroid graph every this many iterations (`1` rebuilds
        /// every iteration). Clamped to `>= 1`.
        rebuild_every: usize,
        /// Number of graph candidates to re-rank exactly per point (`1` trusts
        /// the graph's nearest result directly). Clamped to `>= 1`.
        rerank: usize,
    },
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
    /// Search-list size used when assigning corpus points to centroids.
    pub assign_l: usize,
    /// Centroid graph construction parameters.
    pub graph: GraphParams,
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
        if self.assign_l == 0 {
            return Err(E::invalid("assign_l must be non-zero"));
        }
        if self.num_threads == 0 {
            return Err(E::invalid("num_threads must be non-zero"));
        }
        if self.graph.degree == 0 || self.graph.l_build == 0 {
            return Err(E::invalid("graph degree and l_build must be non-zero"));
        }
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
    /// Centroid-graph search-list size used to route each inserted point.
    pub assign_l: usize,
    /// Number of nearest centroid clusters (besides the two children) drawn in
    /// as reassignment candidates when a cluster is split. Must be `>= 1`.
    ///
    /// This replaces the earlier policy of using the split centroid's direct
    /// centroid-graph out-edges: the candidates are instead the `s` nearest live
    /// centroids to the split centroid, found by searching the centroid graph.
    pub reassign_neighbors: usize,
    /// Centroid-graph search-list size for the nearest-centroid search that
    /// selects the [`reassign_neighbors`](Self::reassign_neighbors) candidate
    /// clusters. Larger values make the selection more accurate at higher cost;
    /// it is clamped up to `reassign_neighbors + 1` internally. A good default
    /// is `max(reassign_neighbors, assign_l)`.
    pub reassign_l: usize,
    /// Number of 2-means iterations used to split a cluster.
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
    /// Centroid-graph construction parameters.
    pub graph: GraphParams,
    /// Metric recorded in the flushed index metadata. Clustering and graph
    /// navigation always use squared-L2 (as in a batch build); this only
    /// controls how the *loaded* index scores at search time.
    pub metric: Metric,
    /// L2-normalize the two child centroids after a split (for unit-normalized
    /// corpora).
    pub normalize_centroids: bool,
    /// Worker threads for the internal 2-means and graph construction.
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
            assign_l: 32,
            reassign_neighbors: 8,
            reassign_l: 32,
            two_means_iters: 10,
            merge_threshold: 0,
            min_clusters: 1,
            graph: GraphParams::default(),
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

/// Parameters controlling a single search.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct SearchParams {
    /// Number of nearest clusters to probe (inverted lists to fetch).
    pub nlist: usize,
    /// Search-list size for the centroid graph search (`>= nlist`).
    pub centroid_search_l: usize,
}

impl SearchParams {
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
        Ok(())
    }

    /// Search-list size to use, never smaller than `nlist`.
    pub(crate) fn effective_l(&self) -> usize {
        self.centroid_search_l.max(self.nlist)
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
            assign_l: 16,
            graph: GraphParams::default(),
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
        p.assign_l = 0;
        assert!(p.validate(100, 4).is_err());

        let mut p = valid_build();
        p.num_threads = 0;
        assert!(p.validate(100, 4).is_err());

        let mut p = valid_build();
        p.graph.degree = 0;
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
        let p = SearchParams {
            nlist: 4,
            centroid_search_l: 2,
        };
        assert!(p.validate(8).is_ok());
        // effective_l is never smaller than nlist.
        assert_eq!(p.effective_l(), 4);

        let p2 = SearchParams {
            nlist: 4,
            centroid_search_l: 10,
        };
        assert_eq!(p2.effective_l(), 10);

        // nlist must be non-zero and within the cluster count.
        let zero = SearchParams {
            nlist: 0,
            centroid_search_l: 8,
        };
        assert!(zero.validate(8).is_err());
        let too_many = SearchParams {
            nlist: 9,
            centroid_search_l: 9,
        };
        assert!(too_many.validate(8).is_err());
    }
}

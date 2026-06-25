/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Tunable parameters for building and searching a graph-IVF index.

/// Distance metric. Internally everything is reduced to squared-L2; `Cosine`
/// is realized by L2-normalizing vectors at build and query time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Metric {
    /// Squared Euclidean distance.
    L2,
    /// Cosine similarity (vectors are L2-normalized).
    Cosine,
}

impl Metric {
    pub(crate) fn as_u8(self) -> u8 {
        match self {
            Metric::L2 => 0,
            Metric::Cosine => 1,
        }
    }

    pub(crate) fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Metric::L2),
            1 => Some(Metric::Cosine),
            _ => None,
        }
    }

    /// Whether vectors must be L2-normalized for this metric.
    pub(crate) fn normalizes(self) -> bool {
        matches!(self, Metric::Cosine)
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
    pub(crate) fn effective_sample_size(&self, num_points: usize) -> usize {
        const KMEANSPP_MAX: usize = 1 << 23;
        self.sample_size.min(num_points).min(KMEANSPP_MAX)
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
        for m in [Metric::L2, Metric::Cosine] {
            assert_eq!(Metric::from_u8(m.as_u8()), Some(m));
        }
        assert_eq!(Metric::from_u8(2), None);
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

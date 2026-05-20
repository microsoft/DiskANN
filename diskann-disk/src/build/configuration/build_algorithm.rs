/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Build algorithm selection for graph index construction.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Selects the graph construction algorithm for index building.
///
/// - [`Vamana`](BuildAlgorithm::Vamana): the default incremental insert + prune
///   builder. Its tuning knobs (`l_build`, `alpha`) live on the outer
///   `DiskIndexBuildParameters` / index configuration because they're shared
///   with the search-time prune.
/// - [`PiPNN`](BuildAlgorithm::PiPNN): partition-based batch builder
///   (arXiv:2602.21247). All of its tuning knobs are PiPNN-specific and
///   are carried on the variant itself so they don't pollute the outer
///   config when Vamana is selected.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "algorithm")]
#[non_exhaustive]
pub enum BuildAlgorithm {
    /// Default Vamana graph construction. Uses `l_build` and `alpha` from
    /// the outer index configuration.
    #[default]
    Vamana,

    /// PiPNN: Pick-in-Partitions Nearest Neighbors (arXiv:2602.21247).
    /// All PiPNN-specific knobs are inlined into this variant.
    PiPNN {
        /// Maximum leaf partition size.
        #[serde(default = "default_c_max")]
        c_max: usize,
        /// Minimum cluster size before merging.
        #[serde(default = "default_c_min")]
        c_min: usize,
        /// Sampling fraction for RBC leaders.
        #[serde(default = "default_p_samp")]
        p_samp: f64,
        /// Fanout at each partitioning level. `fanout[level]` is the number of
        /// leaders each point joins at level `level`.
        #[serde(default = "default_fanout")]
        fanout: Vec<usize>,
        /// k-NN within each leaf (graph edges added per leaf-point).
        #[serde(default = "default_leaf_k", alias = "k")]
        leaf_k: usize,
        /// Number of independent partitioning passes (graph is the union).
        #[serde(default = "default_replicas")]
        replicas: usize,
        /// Maximum reservoir size per node in HashPrune.
        #[serde(default = "default_l_max")]
        l_max: usize,
        /// Number of LSH hyperplanes for HashPrune.
        #[serde(default = "default_num_hash_planes")]
        num_hash_planes: usize,
        /// Whether to apply a final RobustPrune-style diversity pass.
        #[serde(default)]
        final_prune: bool,
        /// Diversity factor for `final_prune` (DiskANN's alpha). Ignored if
        /// `final_prune` is false. Larger values keep more candidates that
        /// would otherwise be occluded.
        #[serde(default = "default_final_prune_alpha")]
        final_prune_alpha: f32,
        /// After `final_prune`, fill each node's neighbor list back up to
        /// `max_degree` with occluded candidates rather than leaving it
        /// sparse. Ignored if `final_prune` is false.
        #[serde(default = "default_saturate_after_prune")]
        saturate_after_prune: bool,
        /// Maximum number of leaders sampled at each partitioning level.
        /// Acts as a ceiling on `p_samp * cluster_size`.
        #[serde(default = "default_leader_cap")]
        leader_cap: usize,
    },
}

impl fmt::Display for BuildAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildAlgorithm::Vamana => write!(f, "Vamana"),
            BuildAlgorithm::PiPNN {
                c_max,
                leaf_k,
                replicas,
                ..
            } => {
                write!(
                    f,
                    "PiPNN(c_max={}, leaf_k={}, replicas={})",
                    c_max, leaf_k, replicas
                )
            }
        }
    }
}

impl BuildAlgorithm {
    /// Convert PiPNN build parameters to a `PiPNNConfig`. Returns `None` for
    /// the Vamana variant.
    ///
    /// `max_degree`, `metric`, and `num_threads` come from the outer DiskANN
    /// index configuration (they're shared with the search-time runtime and
    /// must agree). All other knobs are read from the variant fields, so
    /// PiPNN users can tune them via JSON without affecting Vamana defaults.
    #[cfg(feature = "pipnn")]
    pub fn to_pipnn_config(
        &self,
        max_degree: usize,
        metric: diskann_vector::distance::Metric,
        num_threads: usize,
    ) -> Option<diskann_pipnn::PiPNNConfig> {
        let BuildAlgorithm::PiPNN {
            c_max,
            c_min,
            p_samp,
            fanout,
            leaf_k,
            replicas,
            l_max,
            num_hash_planes,
            final_prune,
            final_prune_alpha,
            saturate_after_prune,
            leader_cap,
        } = self
        else {
            return None;
        };

        Some(diskann_pipnn::PiPNNConfig {
            c_max: *c_max,
            c_min: *c_min,
            p_samp: *p_samp,
            fanout: fanout.clone(),
            k: *leaf_k,
            max_degree,
            replicas: *replicas,
            l_max: *l_max,
            num_hash_planes: *num_hash_planes,
            metric,
            final_prune: *final_prune,
            alpha: *final_prune_alpha,
            num_threads,
            leader_cap: *leader_cap,
            saturate_after_prune: *saturate_after_prune,
        })
    }
}

// ─── Defaults (also used by serde when fields are omitted in JSON) ───

fn default_c_max() -> usize {
    1024
}
fn default_c_min() -> usize {
    256
}
fn default_p_samp() -> f64 {
    0.005
}
fn default_fanout() -> Vec<usize> {
    vec![10, 3]
}
fn default_leaf_k() -> usize {
    3
}
fn default_replicas() -> usize {
    1
}
fn default_l_max() -> usize {
    128
}
fn default_num_hash_planes() -> usize {
    12
}
fn default_final_prune_alpha() -> f32 {
    1.2
}
fn default_saturate_after_prune() -> bool {
    true
}
fn default_leader_cap() -> usize {
    1000
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_pipnn() -> BuildAlgorithm {
        BuildAlgorithm::PiPNN {
            c_max: 2048,
            c_min: 512,
            p_samp: 0.1,
            fanout: vec![5, 3],
            leaf_k: 4,
            replicas: 2,
            l_max: 256,
            num_hash_planes: 12,
            final_prune: false,
            final_prune_alpha: default_final_prune_alpha(),
            saturate_after_prune: default_saturate_after_prune(),
            leader_cap: default_leader_cap(),
        }
    }

    #[test]
    fn test_build_algorithm_default_is_vamana() {
        let algo = BuildAlgorithm::default();
        assert_eq!(algo, BuildAlgorithm::Vamana);
    }

    #[test]
    fn test_build_algorithm_display_vamana() {
        assert_eq!(format!("{}", BuildAlgorithm::Vamana), "Vamana");
    }

    #[test]
    fn test_build_algorithm_display_pipnn() {
        assert_eq!(
            format!("{}", sample_pipnn()),
            "PiPNN(c_max=2048, leaf_k=4, replicas=2)"
        );
    }

    #[test]
    fn test_build_algorithm_serde_roundtrip_vamana() {
        let algo = BuildAlgorithm::Vamana;
        let json = serde_json::to_string(&algo).unwrap();
        let back: BuildAlgorithm = serde_json::from_str(&json).unwrap();
        assert_eq!(algo, back);
    }

    #[test]
    fn test_build_algorithm_serde_roundtrip_pipnn() {
        let algo = sample_pipnn();
        let json = serde_json::to_string(&algo).unwrap();
        let back: BuildAlgorithm = serde_json::from_str(&json).unwrap();
        assert_eq!(algo, back);
    }

    #[test]
    fn test_build_algorithm_serde_pipnn_defaults() {
        let json = r#"{"algorithm":"PiPNN"}"#;
        let back: BuildAlgorithm = serde_json::from_str(json).unwrap();
        let expected = BuildAlgorithm::PiPNN {
            c_max: default_c_max(),
            c_min: default_c_min(),
            p_samp: default_p_samp(),
            fanout: default_fanout(),
            leaf_k: default_leaf_k(),
            replicas: default_replicas(),
            l_max: default_l_max(),
            num_hash_planes: default_num_hash_planes(),
            final_prune: false,
            final_prune_alpha: default_final_prune_alpha(),
            saturate_after_prune: default_saturate_after_prune(),
            leader_cap: default_leader_cap(),
        };
        assert_eq!(back, expected);
    }

    /// `leaf_k` accepts `"k"` as a serde alias for backwards compat with
    /// users that mirrored `PiPNNConfig::k` in their JSON.
    #[test]
    fn test_build_algorithm_serde_k_alias() {
        let json = r#"{"algorithm":"PiPNN","k":7}"#;
        let back: BuildAlgorithm = serde_json::from_str(json).unwrap();
        if let BuildAlgorithm::PiPNN { leaf_k, .. } = back {
            assert_eq!(leaf_k, 7);
        } else {
            panic!("expected PiPNN variant");
        }
    }

    #[test]
    #[cfg(feature = "pipnn")]
    fn test_to_pipnn_config_vamana_returns_none() {
        assert!(BuildAlgorithm::Vamana
            .to_pipnn_config(64, diskann_vector::distance::Metric::L2, 16)
            .is_none());
    }

    #[test]
    #[cfg(feature = "pipnn")]
    fn test_to_pipnn_config_pipnn_returns_some() {
        let algo = BuildAlgorithm::PiPNN {
            c_max: 512,
            c_min: 128,
            p_samp: 0.01,
            fanout: vec![8],
            leaf_k: 5,
            replicas: 1,
            l_max: 128,
            num_hash_planes: 12,
            final_prune: true,
            final_prune_alpha: 1.3,
            saturate_after_prune: false,
            leader_cap: 500,
        };
        let cfg = algo
            .to_pipnn_config(64, diskann_vector::distance::Metric::L2, 16)
            .unwrap();
        assert_eq!(cfg.c_max, 512);
        assert_eq!(cfg.k, 5); // leaf_k → k
        assert_eq!(cfg.max_degree, 64);
        assert_eq!(cfg.alpha, 1.3);
        assert_eq!(cfg.leader_cap, 500);
        assert!(!cfg.saturate_after_prune);
    }
}

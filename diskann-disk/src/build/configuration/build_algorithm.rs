/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Build algorithm selection for graph index construction.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Selects the graph construction algorithm for index building.
///
/// - `Vamana`: The default incremental insert + prune algorithm.
/// - `PiPNN`: Partition-based batch builder (arXiv:2602.21247).
///   Significantly faster build times at comparable graph quality.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "algorithm")]
pub enum BuildAlgorithm {
    /// Default Vamana graph construction.
    #[default]
    Vamana,

    /// PiPNN: Pick-in-Partitions Nearest Neighbors.
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
        /// Fanout at each partitioning level.
        #[serde(default = "default_fanout")]
        fanout: Vec<usize>,
        /// k-NN within each leaf.
        #[serde(default = "default_leaf_k")]
        leaf_k: usize,
        /// Number of independent partitioning passes.
        #[serde(default = "default_replicas")]
        replicas: usize,
        /// Maximum reservoir size per node in HashPrune.
        #[serde(default = "default_l_max")]
        l_max: usize,
        /// Number of LSH hyperplanes for HashPrune.
        #[serde(default = "default_num_hash_planes")]
        num_hash_planes: usize,
        /// Whether to apply a final RobustPrune pass.
        #[serde(default)]
        final_prune: bool,
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
    /// Convert PiPNN build parameters to a PiPNNConfig.
    /// `max_degree`, `metric`, and `alpha` come from the DiskANN index configuration.
    #[cfg(feature = "pipnn")]
    pub fn to_pipnn_config(
        &self,
        max_degree: usize,
        metric: diskann_vector::distance::Metric,
        alpha: f32,
    ) -> Option<diskann_pipnn::PiPNNConfig> {
        match self {
            BuildAlgorithm::PiPNN {
                c_max, c_min, p_samp, fanout, leaf_k, replicas,
                l_max, num_hash_planes, final_prune,
            } => Some(diskann_pipnn::PiPNNConfig {
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
                alpha,
            }),
            _ => None,
        }
    }
}

fn default_c_max() -> usize {
    1024
}
fn default_c_min() -> usize {
    256
}
fn default_p_samp() -> f64 {
    0.05
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_algorithm_default_is_vamana() {
        let algo = BuildAlgorithm::default();
        assert_eq!(algo, BuildAlgorithm::Vamana, "default BuildAlgorithm should be Vamana");
    }

    #[test]
    fn test_build_algorithm_display_vamana() {
        let algo = BuildAlgorithm::Vamana;
        let display = format!("{}", algo);
        assert_eq!(display, "Vamana", "Vamana display should be 'Vamana'");
    }

    #[test]
    fn test_build_algorithm_display_pipnn() {
        let algo = BuildAlgorithm::PiPNN {
            c_max: 2048,
            c_min: 512,
            p_samp: 0.1,
            fanout: vec![5, 3],
            leaf_k: 4,
            replicas: 2,
            l_max: 256,
            num_hash_planes: 12,
            final_prune: false,
        };
        let display = format!("{}", algo);
        assert_eq!(
            display,
            "PiPNN(c_max=2048, leaf_k=4, replicas=2)",
            "PiPNN display should include c_max, leaf_k, and replicas"
        );
    }

    #[test]
    fn test_build_algorithm_serde_roundtrip_vamana() {
        let algo = BuildAlgorithm::Vamana;
        let json = serde_json::to_string(&algo).expect("serialize Vamana should succeed");
        let deserialized: BuildAlgorithm =
            serde_json::from_str(&json).expect("deserialize Vamana should succeed");
        assert_eq!(algo, deserialized, "Vamana should roundtrip through serde_json");
    }

    #[test]
    fn test_build_algorithm_serde_roundtrip_pipnn() {
        let algo = BuildAlgorithm::PiPNN {
            c_max: 2048,
            c_min: 512,
            p_samp: 0.1,
            fanout: vec![5, 3],
            leaf_k: 4,
            replicas: 2,
            l_max: 256,
            num_hash_planes: 8,
            final_prune: true,
        };
        let json = serde_json::to_string(&algo).expect("serialize PiPNN should succeed");
        let deserialized: BuildAlgorithm =
            serde_json::from_str(&json).expect("deserialize PiPNN should succeed");
        assert_eq!(algo, deserialized, "PiPNN with all fields should roundtrip through serde_json");
    }

    #[test]
    fn test_build_algorithm_serde_pipnn_defaults() {
        // Deserialize PiPNN with only the algorithm tag -- all fields should use defaults.
        let json = r#"{"algorithm":"PiPNN"}"#;
        let deserialized: BuildAlgorithm =
            serde_json::from_str(json).expect("PiPNN with defaults should deserialize");

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
        };
        assert_eq!(
            deserialized, expected,
            "deserializing PiPNN with missing fields should use default values"
        );
    }

    #[test]
    fn test_build_algorithm_partial_eq() {
        let v1 = BuildAlgorithm::Vamana;
        let v2 = BuildAlgorithm::Vamana;
        assert_eq!(v1, v2, "two Vamana instances should be equal");

        let p1 = BuildAlgorithm::PiPNN {
            c_max: 1024,
            c_min: 256,
            p_samp: 0.05,
            fanout: vec![10, 3],
            leaf_k: 3,
            replicas: 1,
            l_max: 128,
            num_hash_planes: 12,
            final_prune: false,
        };
        let p2 = p1.clone();
        assert_eq!(p1, p2, "cloned PiPNN should equal original");

        assert_ne!(v1, p1, "Vamana and PiPNN should not be equal");

        let p3 = BuildAlgorithm::PiPNN {
            c_max: 2048, // different
            c_min: 256,
            p_samp: 0.05,
            fanout: vec![10, 3],
            leaf_k: 3,
            replicas: 1,
            l_max: 128,
            num_hash_planes: 12,
            final_prune: false,
        };
        assert_ne!(p1, p3, "PiPNN with different c_max should not be equal");
    }

    #[test]
    #[cfg(feature = "pipnn")]
    fn test_to_pipnn_config_vamana_returns_none() {
        let algo = BuildAlgorithm::Vamana;
        assert!(algo.to_pipnn_config(64, diskann_vector::distance::Metric::L2, 1.2).is_none());
    }

    #[test]
    #[cfg(feature = "pipnn")]
    fn test_to_pipnn_config_pipnn_returns_some() {
        let algo = BuildAlgorithm::PiPNN {
            c_max: 512, c_min: 128, p_samp: 0.01, fanout: vec![8],
            leaf_k: 5, replicas: 1, l_max: 128, num_hash_planes: 12,
            final_prune: true,
        };
        let config = algo.to_pipnn_config(64, diskann_vector::distance::Metric::L2, 1.2);
        assert!(config.is_some());
        let config = config.unwrap();
        assert_eq!(config.c_max, 512);
        assert_eq!(config.k, 5); // leaf_k maps to k
        assert_eq!(config.max_degree, 64);
        assert_eq!(config.alpha, 1.2);
    }
}

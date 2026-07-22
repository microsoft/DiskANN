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
///   builder. Graph-wide tuning knobs such as `l_build` and `alpha` live on the
///   outer index configuration.
/// - [`PiPNN`](BuildAlgorithm::PiPNN): one-shot partition-based batch builder
///   (arXiv:2602.21247). Carries a [`diskann_pipnn::PiPNNConfig`] with its
///   algorithm-specific partition and candidate-generation knobs. PiPNN is
///   one-shot; the disk wrapper selects Vamana instead when the estimated peak
///   would exceed its build-memory limit.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "algorithm")]
#[non_exhaustive]
pub enum BuildAlgorithm {
    /// Default Vamana graph construction.
    #[default]
    Vamana,

    /// PiPNN: Pick-in-Partitions Nearest Neighbors (arXiv:2602.21247).
    #[cfg(feature = "pipnn")]
    PiPNN(diskann_pipnn::PiPNNConfig),
}

impl fmt::Display for BuildAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildAlgorithm::Vamana => write!(f, "Vamana"),
            #[cfg(feature = "pipnn")]
            BuildAlgorithm::PiPNN(cfg) => {
                write!(f, "PiPNN({:?})", cfg)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_build_algorithm_serde_roundtrip_vamana() {
        let algo = BuildAlgorithm::Vamana;
        let json = serde_json::to_string(&algo).unwrap();
        let back: BuildAlgorithm = serde_json::from_str(&json).unwrap();
        assert_eq!(algo, back);
    }

    #[cfg(feature = "pipnn")]
    mod pipnn {
        use super::*;

        fn sample() -> diskann_pipnn::PiPNNConfig {
            diskann_pipnn::PiPNNConfig {
                c_max: 2048,
                c_min: 512,
                p_samp: 0.1,
                fanout: vec![5, 3],
                k: 4,
                replicas: 2,
                l_max: 256,
                num_hash_planes: 12,
                final_prune: false,
                skip_hash_prune: false,
            }
        }

        #[test]
        fn display_includes_inner_config() {
            let algo = BuildAlgorithm::PiPNN(sample());
            let s = format!("{}", algo);
            assert!(s.starts_with("PiPNN("));
            assert!(s.contains("c_max: 2048"));
        }

        #[test]
        fn serde_roundtrip() {
            let algo = BuildAlgorithm::PiPNN(sample());
            let json = serde_json::to_string(&algo).unwrap();
            let back: BuildAlgorithm = serde_json::from_str(&json).unwrap();
            assert_eq!(algo, back);
        }

        #[test]
        fn serde_pipnn_rejects_obsolete_fields() {
            let json = r#"{"algorithm":"PiPNN","leaf_k":7,"final_prune_alpha":1.5}"#;
            assert!(serde_json::from_str::<BuildAlgorithm>(json).is_err());
        }

        #[test]
        fn serde_pipnn_defaults() {
            let json = r#"{"algorithm":"PiPNN"}"#;
            let back: BuildAlgorithm = serde_json::from_str(json).unwrap();
            assert_eq!(
                back,
                BuildAlgorithm::PiPNN(diskann_pipnn::PiPNNConfig::default())
            );
        }

        #[test]
        fn serde_pipnn_inline_fields() {
            let json = r#"{"algorithm":"PiPNN","c_max":512,"c_min":128,"k":5}"#;
            let back: BuildAlgorithm = serde_json::from_str(json).unwrap();
            if let BuildAlgorithm::PiPNN(cfg) = back {
                assert_eq!(cfg.c_max, 512);
                assert_eq!(cfg.c_min, 128);
                assert_eq!(cfg.k, 5);
                // Other fields fall back to PiPNNConfig::default().
                assert_eq!(cfg.l_max, diskann_pipnn::PiPNNConfig::default().l_max);
            } else {
                panic!("expected PiPNN variant");
            }
        }
    }
}

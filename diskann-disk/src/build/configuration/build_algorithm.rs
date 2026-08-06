/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Graph-build algorithm selection and its JSON-facing configuration.

use std::fmt;

use serde::{Deserialize, Serialize};

/// PiPNN parameters in the JSON build configuration.
///
/// The common index configuration supplies graph degree, alpha, metric, thread
/// count, and memory limit.
#[cfg(feature = "pipnn")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PiPNNParameters {
    /// Maximum number of points in a leaf.
    pub c_max: usize,
    /// Minimum leaf size used by global small-leaf merging.
    pub c_min: usize,
    /// Fraction of a cluster sampled as leaders.
    pub p_samp: f64,
    /// Number of nearest leaders retained at each partition level.
    pub fanout: Vec<usize>,
    /// Number of nearest neighbors selected within each leaf.
    pub k: usize,
    /// Number of independent partition passes.
    pub replicas: usize,
    /// HashPrune policy, or `None` for exact direct-candidate accumulation.
    pub hash_prune: Option<HashPruneParameters>,
}

/// JSON-facing HashPrune parameters.
#[cfg(feature = "pipnn")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HashPruneParameters {
    /// Number of random-hyperplane sketch dimensions.
    pub num_hash_planes: usize,
    /// Maximum candidates retained per point.
    pub l_max: usize,
    /// Apply shared Vamana RobustPrune after reservoir extraction.
    pub final_prune: bool,
}

#[cfg(feature = "pipnn")]
impl Default for HashPruneParameters {
    fn default() -> Self {
        Self {
            num_hash_planes: 12,
            l_max: 64,
            final_prune: true,
        }
    }
}

#[cfg(feature = "pipnn")]
impl From<&HashPruneParameters> for diskann::graph::pipnn::HashPruneConfig {
    fn from(config: &HashPruneParameters) -> Self {
        Self {
            num_hash_planes: config.num_hash_planes,
            l_max: config.l_max,
            final_prune: config.final_prune,
        }
    }
}

#[cfg(feature = "pipnn")]
impl Default for PiPNNParameters {
    fn default() -> Self {
        Self {
            c_max: 256,
            c_min: 16,
            p_samp: 0.005,
            fanout: vec![8, 3],
            k: 2,
            replicas: 1,
            hash_prune: None,
        }
    }
}

#[cfg(feature = "pipnn")]
impl From<&PiPNNParameters> for diskann::graph::pipnn::PiPNNConfig {
    fn from(config: &PiPNNParameters) -> Self {
        Self {
            c_max: config.c_max,
            c_min: config.c_min,
            p_samp: config.p_samp,
            fanout: config.fanout.clone(),
            leaf_k: config.k,
            replicas: config.replicas,
        }
    }
}

/// Graph construction algorithm for index building.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "algorithm")]
#[non_exhaustive]
pub enum BuildAlgorithm {
    /// Default Vamana graph construction.
    #[default]
    Vamana,

    /// PiPNN one-shot partition-based graph construction.
    #[cfg(feature = "pipnn")]
    PiPNN(PiPNNParameters),
}

impl fmt::Display for BuildAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vamana => write!(f, "Vamana"),
            #[cfg(feature = "pipnn")]
            Self::PiPNN(config) => write!(f, "PiPNN({config:?})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_vamana() {
        assert_eq!(BuildAlgorithm::default(), BuildAlgorithm::Vamana);
    }

    #[test]
    fn vamana_serde_roundtrip() {
        let json = serde_json::to_string(&BuildAlgorithm::Vamana).unwrap();
        assert_eq!(
            serde_json::from_str::<BuildAlgorithm>(&json).unwrap(),
            BuildAlgorithm::Vamana
        );
    }

    #[cfg(feature = "pipnn")]
    #[test]
    fn pipnn_serde_uses_inline_defaults_and_rejects_unknown_fields() {
        let algorithm: BuildAlgorithm = serde_json::from_str(
            r#"{"algorithm":"PiPNN","c_max":512,"c_min":64,"fanout":[10,3],"k":3}"#,
        )
        .unwrap();
        let BuildAlgorithm::PiPNN(config) = algorithm else {
            panic!("expected PiPNN");
        };
        assert_eq!(config.c_max, 512);
        assert_eq!(config.c_min, 64);
        assert_eq!(config.fanout, [10, 3]);
        assert_eq!(config.k, 3);
        assert_eq!(config.replicas, 1);
        assert_eq!(config.hash_prune, None);

        let explicit: BuildAlgorithm =
            serde_json::from_str(r#"{"algorithm":"PiPNN","hash_prune":{}}"#).unwrap();
        let BuildAlgorithm::PiPNN(explicit) = explicit else {
            panic!("expected PiPNN");
        };
        assert_eq!(explicit.hash_prune, Some(HashPruneParameters::default()));

        assert!(
            serde_json::from_str::<BuildAlgorithm>(r#"{"algorithm":"PiPNN","l_max":72}"#).is_err()
        );
    }
}

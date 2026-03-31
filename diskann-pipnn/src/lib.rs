/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! PiPNN (Pick-in-Partitions Nearest Neighbors) index builder.
//!
//! Implements the PiPNN algorithm from arXiv:2602.21247, which builds graph-based
//! ANN indexes significantly faster than Vamana/HNSW by:
//! 1. Partitioning the dataset into overlapping clusters via Randomized Ball Carving
//! 2. Building local graphs within each leaf cluster using GEMM-based all-pairs distance
//! 3. Merging edges from overlapping partitions using HashPrune (LSH-based online pruning)

pub mod builder;
pub mod gemm;
pub mod hash_prune;
pub mod leaf_build;
pub mod partition;
pub mod quantize;

use diskann_vector::distance::Metric;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during PiPNN index construction.
#[derive(Debug, Error)]
pub enum PiPNNError {
    #[error("configuration error: {0}")]
    Config(String),

    #[error("data dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error(
        "data length mismatch: expected {expected} elements ({npoints} x {ndims}), got {actual}"
    )]
    DataLengthMismatch {
        expected: usize,
        actual: usize,
        npoints: usize,
        ndims: usize,
    },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for PiPNN operations.
pub type PiPNNResult<T> = Result<T, PiPNNError>;

/// Custom serde module for `Metric`, which does not derive Serialize/Deserialize.
/// Serializes as a string representation (e.g. "l2", "cosine").
mod metric_serde {
    use diskann_vector::distance::Metric;
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(metric: &Metric, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(metric.as_str())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Metric, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse::<Metric>().map_err(serde::de::Error::custom)
    }
}

/// Configuration for the PiPNN index builder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiPNNConfig {
    /// Number of LSH hyperplanes for HashPrune.
    pub num_hash_planes: usize,
    /// Maximum leaf partition size.
    pub c_max: usize,
    /// Minimum cluster size before merging.
    pub c_min: usize,
    /// Sampling fraction for RBC leaders.
    pub p_samp: f64,
    /// Fanout at each partitioning level (overlap factor).
    pub fanout: Vec<usize>,
    /// k for k-NN in leaf building.
    pub k: usize,
    /// Maximum graph degree (R).
    pub max_degree: usize,
    /// Number of independent partitioning passes (replicas).
    pub replicas: usize,
    /// Maximum reservoir size per node in HashPrune.
    pub l_max: usize,
    /// Distance metric.
    #[serde(with = "metric_serde")]
    pub metric: Metric,
    /// Whether to apply a final diversity-prune pass (occlusion-based, similar to RobustPrune).
    pub final_prune: bool,
    /// Alpha (occlusion factor) for final diversity prune. Same as DiskANN's `alpha` parameter.
    /// Higher values yield sparser graphs. Default: 1.2 (matches DiskANN default).
    pub alpha: f32,
    /// Number of threads to use. 0 means use all available cores.
    #[serde(default)]
    pub num_threads: usize,
}

impl PiPNNConfig {
    /// Validate the configuration, returning an error if any parameter is invalid.
    pub fn validate(&self) -> PiPNNResult<()> {
        if self.c_max == 0 {
            return Err(PiPNNError::Config("c_max must be > 0".into()));
        }
        if self.c_min == 0 {
            return Err(PiPNNError::Config("c_min must be > 0".into()));
        }
        if self.c_min > self.c_max {
            return Err(PiPNNError::Config(format!(
                "c_min ({}) must be <= c_max ({})",
                self.c_min, self.c_max
            )));
        }
        if self.max_degree == 0 {
            return Err(PiPNNError::Config("max_degree must be > 0".into()));
        }
        if self.k == 0 {
            return Err(PiPNNError::Config("k must be > 0".into()));
        }
        if self.replicas == 0 {
            return Err(PiPNNError::Config("replicas must be > 0".into()));
        }
        if self.l_max == 0 {
            return Err(PiPNNError::Config("l_max must be > 0".into()));
        }
        if !self.p_samp.is_finite() {
            return Err(PiPNNError::Config("p_samp must be finite".into()));
        }
        if self.p_samp <= 0.0 || self.p_samp > 1.0 {
            return Err(PiPNNError::Config(format!(
                "p_samp ({}) must be in (0.0, 1.0]",
                self.p_samp
            )));
        }
        if self.fanout.is_empty() {
            return Err(PiPNNError::Config("fanout must not be empty".into()));
        }
        if self.fanout.contains(&0) {
            return Err(PiPNNError::Config("all fanout values must be > 0".into()));
        }
        if self.num_hash_planes == 0 || self.num_hash_planes > 16 {
            return Err(PiPNNError::Config(format!(
                "num_hash_planes ({}) must be in [1, 16]",
                self.num_hash_planes
            )));
        }
        if self.alpha < 1.0 {
            return Err(PiPNNError::Config(format!(
                "alpha ({}) must be >= 1.0",
                self.alpha
            )));
        }
        if !self.alpha.is_finite() {
            return Err(PiPNNError::Config("alpha must be finite".into()));
        }
        if self.metric == Metric::InnerProduct {
            return Err(PiPNNError::Config(
                "InnerProduct metric is not supported by PiPNN; use L2, Cosine, or CosineNormalized".into(),
            ));
        }
        Ok(())
    }
}

impl Default for PiPNNConfig {
    fn default() -> Self {
        Self {
            num_hash_planes: 12,
            c_max: 1024,
            c_min: 256,
            p_samp: 0.005,
            fanout: vec![10, 3],
            k: 3,
            max_degree: 64,
            replicas: 1,
            l_max: 128,
            metric: Metric::L2,
            final_prune: false,
            alpha: 1.2,
            num_threads: 0,
        }
    }
}

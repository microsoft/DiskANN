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
pub mod hash_prune;
pub mod leaf_build;
pub mod partition;
pub(crate) mod rayon_util;

use std::num::NonZeroUsize;

use diskann_vector::distance::Metric;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during PiPNN index construction.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PiPNNError {
    /// PiPNNConfig validation failed (e.g. `c_max < c_min`, `alpha < 1.0`).
    #[error("configuration error: {0}")]
    Config(String),

    /// `data.len() != npoints * ndims` at a builder entry point.
    #[error(
        "data length mismatch: expected {expected} elements ({npoints} x {ndims}), got {actual}"
    )]
    DataLengthMismatch {
        expected: usize,
        actual: usize,
        npoints: usize,
        ndims: usize,
    },

    /// I/O error from `PiPNNGraph::save_graph`.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for PiPNN operations.
pub type PiPNNResult<T> = Result<T, PiPNNError>;

/// Configuration for the PiPNN index builder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
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
    #[serde(alias = "leaf_k")]
    pub k: usize,
    /// Number of independent partitioning passes (replicas).
    pub replicas: usize,
    /// Maximum reservoir size per node in HashPrune.
    pub l_max: usize,
    /// Whether to apply a final diversity-prune pass (occlusion-based, similar to RobustPrune).
    pub final_prune: bool,
    /// Alpha (occlusion factor) for final diversity prune. Same as DiskANN's `alpha` parameter.
    /// Higher values yield sparser graphs. Default: 1.2 (matches DiskANN default).
    #[serde(alias = "final_prune_alpha")]
    pub alpha: f32,
    /// Maximum leaders per partition level. Default: 1000 (paper recommendation).
    pub leader_cap: usize,
    /// Whether to saturate after final prune (fill remaining degree slots with
    /// closest non-selected candidates). Default: true.
    pub saturate_after_prune: bool,
}

impl PiPNNConfig {
    /// Validate the configuration, returning an error if any parameter is invalid.
    pub fn validate(&self) -> PiPNNResult<()> {
        // Partition-layer rules: delegate to the owning module.
        partition::PartitionConfig::validate_params(
            self.c_max,
            self.c_min,
            self.p_samp,
            &self.fanout,
            self.leader_cap,
        )?;
        if self.k == 0 {
            return Err(PiPNNError::Config("k must be > 0".into()));
        }
        if self.replicas == 0 {
            return Err(PiPNNError::Config("replicas must be > 0".into()));
        }
        if self.l_max == 0 {
            return Err(PiPNNError::Config("l_max must be > 0".into()));
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
            replicas: 1,
            l_max: 128,
            final_prune: false,
            alpha: 1.2,
            leader_cap: 1000,
            saturate_after_prune: true,
        }
    }
}

/// Validated input to [`builder::build_typed`].
#[derive(Debug, Clone)]
pub struct PiPNNBuildContext {
    config: PiPNNConfig,
    max_degree: NonZeroUsize,
    metric: Metric,
    num_threads: usize,
}

impl PiPNNBuildContext {
    /// `num_threads = 0` means use all available cores.
    pub fn new(
        config: PiPNNConfig,
        max_degree: NonZeroUsize,
        metric: Metric,
        num_threads: usize,
    ) -> PiPNNResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            max_degree,
            metric,
            num_threads,
        })
    }

    pub fn config(&self) -> &PiPNNConfig {
        &self.config
    }

    pub fn max_degree(&self) -> NonZeroUsize {
        self.max_degree
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn num_threads(&self) -> usize {
        self.num_threads
    }
}

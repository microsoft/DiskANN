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
pub(crate) mod cpu_dispatch;
pub(crate) mod hash_prune;
pub(crate) mod leaf_build;
pub(crate) mod partition;
pub(crate) mod partition_inner;
pub(crate) mod rayon_util;

use std::num::NonZeroUsize;

use diskann_vector::distance::Metric;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during PiPNN index construction.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PiPNNError {
    /// PiPNN configuration validation failed.
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

    /// A `VectorRepr::as_f32_into` call failed during build. The contained
    /// message preserves the underlying error's `Display` text, since the
    /// associated `T::Error` type is erased at the public boundary.
    #[error("vector conversion failed: {0}")]
    Conversion(String),

    /// Shared Vamana RobustPrune rejected the candidate pool.
    #[error("robust prune failed: {0}")]
    Prune(String),
}

/// Result type for PiPNN operations.
pub type PiPNNResult<T> = Result<T, PiPNNError>;

/// Configuration for the PiPNN index builder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
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
    /// Number of independent partitioning passes (replicas).
    pub replicas: usize,
    /// Maximum reservoir size per node in HashPrune.
    pub l_max: usize,
    /// Whether to apply a final diversity-prune pass (occlusion-based, similar to RobustPrune).
    pub final_prune: bool,
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
            crate::partition::LEADER_CAP,
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
        if self.l_max > hash_prune::MAX_RESERVOIR_LEN {
            return Err(PiPNNError::Config(format!(
                "l_max ({}) exceeds the structural bound {} (HotSlot.len / \
                 farthest_idx are u8)",
                self.l_max,
                hash_prune::MAX_RESERVOIR_LEN,
            )));
        }
        if self.num_hash_planes == 0 || self.num_hash_planes > 16 {
            return Err(PiPNNError::Config(format!(
                "num_hash_planes ({}) must be in [1, 16]",
                self.num_hash_planes
            )));
        }
        Ok(())
    }
}

impl Default for PiPNNConfig {
    /// Production-measured defaults: lighter `c_max`/`l_max` than the paper's
    /// initial recommendation, but matched on recall in the project's
    /// benchmark sweeps (see CLAUDE.md). Halves HP RSS vs the paper config at
    /// equal quality. High-dimensional embeddings (768d+) or non-Euclidean
    /// metrics may still want a workload-specific sweep.
    fn default() -> Self {
        Self {
            num_hash_planes: 14,
            c_max: 256,
            c_min: 16,
            p_samp: 0.005,
            fanout: vec![8, 3],
            k: 2,
            replicas: 1,
            l_max: 64,
            final_prune: true,
        }
    }
}

/// Validated input to [`builder::build_typed`].
#[derive(Debug, Clone)]
pub struct PiPNNBuildContext {
    config: PiPNNConfig,
    max_degree: NonZeroUsize,
    alpha: f32,
    metric: Metric,
    num_threads: usize,
}

impl PiPNNBuildContext {
    /// `num_threads = 0` means use all available cores.
    pub fn new(
        config: PiPNNConfig,
        max_degree: NonZeroUsize,
        alpha: f32,
        metric: Metric,
        num_threads: usize,
    ) -> PiPNNResult<Self> {
        config.validate()?;
        if !alpha.is_finite() || alpha < 1.0 {
            return Err(PiPNNError::Config(format!(
                "alpha ({alpha}) must be finite and >= 1.0"
            )));
        }
        Ok(Self {
            config,
            max_degree,
            alpha,
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

    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn num_threads(&self) -> usize {
        self.num_threads
    }
}

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

pub mod gemm;
pub mod hash_prune;
pub mod leaf_build;
pub mod partition;
pub mod builder;
pub mod quantize;

use diskann_vector::distance::Metric;

/// Configuration for the PiPNN index builder.
#[derive(Debug, Clone)]
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
    pub metric: Metric,
    /// Whether to apply a final RobustPrune pass.
    pub final_prune: bool,
    /// If set, quantize vectors to this many bits before building.
    /// Only 1-bit is currently supported.
    pub quantize_bits: Option<usize>,
}

impl Default for PiPNNConfig {
    fn default() -> Self {
        Self {
            num_hash_planes: 12,
            c_max: 1024,
            c_min: 256,
            p_samp: 0.05,
            fanout: vec![10, 3],
            k: 3,
            max_degree: 64,
            replicas: 1,
            l_max: 128,
            metric: Metric::L2,
            final_prune: false,
            quantize_bits: None,
        }
    }
}

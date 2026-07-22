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
//! 3. Merging overlapping leaf edges with either HashPrune or exact deduplication
//! 4. Optionally applying DiskANN's shared Vamana RobustPrune kernel

pub mod builder;
pub(crate) mod cpu_dispatch;
pub(crate) mod direct_candidates;
pub(crate) mod graph_prune;
pub(crate) mod hash_prune;
pub(crate) mod leaf_build;
pub(crate) mod partition;
pub(crate) mod partition_inner;
pub(crate) mod rayon_util;

use std::num::NonZeroUsize;

use diskann::{ANNError, ANNResult};
use diskann_vector::distance::Metric;
use serde::{Deserialize, Serialize};

/// Measured proportional allocator/runtime headroom for large builds.
const PEAK_MEMORY_HEADROOM_PERCENT: u128 = 108;

/// Measured allocator floor for each Rayon worker on smaller builds.
const PEAK_MEMORY_PER_WORKER_HEADROOM: u128 = 28 * 1024 * 1024;
const PEAK_MEMORY_PROCESS_HEADROOM: u128 = 16 * 1024 * 1024;

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
    /// Whether to apply Vamana's shared RobustPrune to overfull HashPrune reservoirs.
    pub final_prune: bool,
    /// Bypass HashPrune and send deduplicated leaf candidates directly to
    /// Vamana's shared RobustPrune.
    pub skip_hash_prune: bool,
}

impl PiPNNConfig {
    /// Validate the configuration, returning an error if any parameter is invalid.
    pub fn validate(&self) -> ANNResult<()> {
        if self.c_max == 0 {
            return Err(config_error("c_max must be > 0"));
        }
        if self.c_min == 0 {
            return Err(config_error("c_min must be > 0"));
        }
        if self.c_min > self.c_max {
            return Err(config_error(format!(
                "c_min ({}) must be <= c_max ({})",
                self.c_min, self.c_max
            )));
        }
        if !self.p_samp.is_finite() || self.p_samp <= 0.0 || self.p_samp > 1.0 {
            return Err(config_error(format!(
                "p_samp ({}) must be finite and in (0.0, 1.0]",
                self.p_samp
            )));
        }
        if self.fanout.is_empty() {
            return Err(config_error("fanout must not be empty"));
        }
        if self.fanout.contains(&0) {
            return Err(config_error("all fanout values must be > 0"));
        }
        if let Some(&fanout) = self
            .fanout
            .iter()
            .find(|&&fanout| fanout > partition::MAX_FANOUT)
        {
            return Err(config_error(format!(
                "fanout value {fanout} exceeds MAX_FANOUT ({})",
                partition::MAX_FANOUT
            )));
        }
        if self.k == 0 {
            return Err(config_error("k must be > 0"));
        }
        if self.replicas == 0 {
            return Err(config_error("replicas must be > 0"));
        }
        if self.skip_hash_prune {
            if !self.final_prune {
                return Err(config_error("skip_hash_prune requires final_prune = true"));
            }
        } else {
            if self.l_max == 0 {
                return Err(config_error("l_max must be > 0"));
            }
            if self.l_max > hash_prune::MAX_RESERVOIR_LEN {
                return Err(config_error(format!(
                    "l_max ({}) exceeds the structural bound {} (HotSlot.len / \
                     farthest_idx are u8)",
                    self.l_max,
                    hash_prune::MAX_RESERVOIR_LEN,
                )));
            }
            if self.num_hash_planes == 0 || self.num_hash_planes > 16 {
                return Err(config_error(format!(
                    "num_hash_planes ({}) must be in [1, 16]",
                    self.num_hash_planes
                )));
            }
        }
        Ok(())
    }

    /// Estimate peak resident bytes for a one-shot build.
    ///
    /// The estimate covers the larger of partitioning, leaf-build/HashPrune,
    /// and graph-finalization residency, then adds allocator headroom measured
    /// on 100K, 1M, and 10M BigANN FP16 builds. Leaf IDs use the configured
    /// fanout product because profiling confirmed an actual 30x overlap for
    /// `fanout=[10, 3]` on BigANN10M.
    pub fn estimated_peak_memory_bytes(
        &self,
        npoints: usize,
        dimensions: usize,
        element_size: usize,
        num_threads: usize,
    ) -> Option<usize> {
        let workers = if num_threads == 0 {
            std::thread::available_parallelism().map_or(1, NonZeroUsize::get)
        } else {
            num_threads
        } as u128;
        let scan_lanes = (self.l_max as u128).div_ceil(32).checked_mul(32)?.max(32);
        let leaf_copies = self
            .fanout
            .iter()
            .try_fold(self.replicas as u128, |n, &fanout| {
                n.checked_mul(fanout as u128)
            })?;
        let dataset = (dimensions as u128).checked_mul(element_size as u128)?;
        let sketches = if self.skip_hash_prune {
            0
        } else {
            (self.num_hash_planes as u128).checked_mul(4)?
        };
        let leaf_ids = leaf_copies.checked_mul(4)?;
        let (merge_metadata, merge_storage) = if self.skip_hash_prune {
            // Each membership offers at most 2*k directed edges. The direct
            // accumulator deduplicates as it inserts, so its Vec can grow to
            // the next power-of-two capacity for that per-point upper bound.
            let offered = leaf_copies.checked_mul(self.k as u128)?.checked_mul(2)?;
            let capacity = offered.checked_next_power_of_two()?;
            (
                std::mem::size_of::<parking_lot::Mutex<diskann::graph::AdjacencyList<u32>>>()
                    as u128,
                capacity.checked_mul(std::mem::size_of::<u32>() as u128)?,
            )
        } else {
            (16, scan_lanes.checked_mul(8)?)
        };
        let leaf_build_per_point = dataset
            .checked_add(merge_metadata)?
            .checked_add(sketches)?
            .checked_add(merge_storage)?
            .checked_add(leaf_ids)?;
        // Partition result clusters are materialized while per-thread partial
        // clusters are still live, so budget two copies of accumulated IDs.
        let partition_per_point = dataset
            .checked_add(16)?
            .checked_add(sketches)?
            .checked_add(leaf_ids.checked_mul(2)?)?;
        let finalization_per_point = if self.skip_hash_prune {
            dataset
                // During conversion both outer row arrays coexist, but each
                // candidate Vec is moved into the output without copying its
                // payload. RobustPrune then reuses that Vec for the result.
                .checked_add(merge_metadata)?
                .checked_add(std::mem::size_of::<Vec<u32>>() as u128)?
                .checked_add(merge_storage)?
        } else {
            let output_ids = (self.l_max as u128).checked_mul(4)?;
            let retained_cold_lanes =
                scan_lanes.checked_mul(if self.final_prune { 4 } else { 6 })?;
            dataset
                .checked_add(16)?
                .checked_add(retained_cold_lanes)?
                .checked_add(24)?
                .checked_add(output_ids)?
        };

        let leaf_size = self.c_max.min(npoints).max(1) as u128;
        let leaf_scratch = leaf_size
            .checked_mul(dimensions as u128)?
            .checked_mul(4)?
            .checked_add(leaf_size.checked_mul(leaf_size)?.checked_mul(5)?)?
            .checked_add(leaf_size.checked_mul(self.k as u128)?.checked_mul(24)?)?
            .checked_add(
                leaf_size
                    .checked_mul(self.num_hash_planes as u128)?
                    .checked_mul(4)?,
            )?
            .checked_add(leaf_size.checked_mul(20)?)?
            .checked_add(68)?;
        let partition_rows = (npoints as u128).min(1024);
        let partition_leaders = (npoints as u128).min(1000);
        let partition_scratch = partition_leaders
            .checked_mul(dimensions as u128)?
            .checked_mul(4)?
            .checked_add(
                partition_rows
                    .checked_mul(dimensions as u128)?
                    .checked_mul(4)?,
            )?
            .checked_add(
                partition_rows
                    .checked_mul(partition_leaders)?
                    .checked_mul(4)?,
            )?;
        let points = npoints as u128;
        let leaf_build = points
            .checked_mul(leaf_build_per_point)?
            .checked_add(workers.checked_mul(leaf_scratch)?)?;
        let partition = points
            .checked_mul(partition_per_point)?
            .checked_add(workers.checked_mul(partition_scratch)?)?;
        let finalization = points.checked_mul(finalization_per_point)?;
        let structural_peak = leaf_build.max(partition).max(finalization);
        let proportional = structural_peak
            .checked_mul(PEAK_MEMORY_HEADROOM_PERCENT)?
            .div_ceil(100);
        let allocator_floor = structural_peak
            .checked_add(PEAK_MEMORY_PROCESS_HEADROOM)?
            .checked_add(workers.checked_mul(PEAK_MEMORY_PER_WORKER_HEADROOM)?)?;
        let bytes = proportional.max(allocator_floor);
        usize::try_from(bytes).ok()
    }
}

impl Default for PiPNNConfig {
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
            skip_hash_prune: false,
        }
    }
}

/// Validated input to [`builder::build_typed`].
#[derive(Debug, Clone)]
pub struct PiPNNBuildContext {
    pub(crate) config: PiPNNConfig,
    pub(crate) max_degree: NonZeroUsize,
    pub(crate) alpha: f32,
    pub(crate) metric: Metric,
    pub(crate) num_threads: usize,
}

impl PiPNNBuildContext {
    /// `num_threads = 0` means use all available cores.
    pub fn new(
        config: PiPNNConfig,
        max_degree: NonZeroUsize,
        alpha: f32,
        metric: Metric,
        num_threads: usize,
    ) -> ANNResult<Self> {
        config.validate()?;
        if !alpha.is_finite() || alpha < 1.0 {
            return Err(config_error(format!(
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
}

#[track_caller]
pub(crate) fn config_error(message: impl std::fmt::Display) -> ANNError {
    ANNError::log_index_config_error("PiPNN".into(), message.to_string())
}

#[cfg(test)]
mod tests;

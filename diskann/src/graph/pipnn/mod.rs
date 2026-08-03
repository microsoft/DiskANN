/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Provider-independent PiPNN graph construction.
//!
//! PiPNN means **Pick-in-Partitions Nearest Neighbors**. It builds a graph for
//! approximate nearest-neighbor search: every input vector becomes one graph
//! vertex, and its adjacency list stores other vectors worth visiting during a
//! later query. This crate constructs that adjacency; it does not execute queries.
//!
//! Incremental builders such as Vamana find construction candidates by running
//! beam search against a partially built graph: they repeatedly follow graph
//! edges to discover nearby vertices, causing random memory access. PiPNN removes
//! that search from construction and uses three bulk stages instead:
//!
//! 1. **Partition.** Randomized Ball Carving samples points called *leaders*.
//!    Every point is assigned to its nearest `fanout` leaders. Assigning to more
//!    than one leader makes child groups overlap. Oversized groups are processed
//!    recursively until bounded groups called *leaves* remain.
//! 2. **Pick within leaves.** Vectors in one leaf are contiguous enough for one
//!    dense general matrix multiplication (GEMM) to compute all pair dot
//!    products. Each point picks its nearest leaf companions; selected pairs
//!    become candidate graph edges.
//! 3. **Merge and prune.** Candidate edges from overlapping leaves are combined
//!    into one unique list per source. Vamana RobustPrune then selects a bounded,
//!    directionally diverse adjacency list.
//!
//! ```text
//! dataset points
//!      │
//!      v
//! sample leaders + point/leader GEMM
//!      │
//!      v
//! choose nearest leaders ──> overlapping child groups ──> recurse ──> leaves
//!                                                                      │
//!                                                    leaf all-pairs GEMM
//!                                                                      │
//!                                                                      v
//!                                                     pick local neighbors
//!                                                                      │
//!                                                                      v
//!                                                     merge/prune edges
//!                                                                      │
//!                                                                      v
//!                                                          search graph
//! ```
//!
//! This module keeps GEMM separate from score selection: callers compute dense
//! dot-product matrices, then the kernels documented below convert those dots to
//! metric scores and retain top candidates. A *point* is a vector being assigned
//! during partitioning; a *leader* names a child group. In leaf selection,
//! *source* names the point whose output list is being built and *target* names
//! another point in that same leaf.
//!
//! The crate owns overlapping partition generation, leaf-local nearest-neighbor
//! construction, candidate merging, and graph-degree finalization. The caller
//! supplies a contiguous dataset view, DiskANN graph policy, and the Rayon pool.
//! Providers, start/frozen points, quantization, persistence, and search remain
//! outside this algorithm boundary.
//!
//! Numerical kernels include:
//!
//! - [`partition_kernel::PartitionKernel`] converts point-by-leader dot-product
//!   tiles into nearest leader positions.
//! - [`leaf_kernel::LeafKernel`] scans each leaf's lower-triangular dot-product
//!   matrix once and retains nearest non-self neighbors for both endpoints.
//!
//! Both handles are prepared once per build metric and reused across stripes or
//! leaves. Each output view supplies call-specific fanout or neighbor width.
//! Preparation selects the runtime architecture and returns a direct function
//! pointer; repeated calls do not repeat ISA or metric dispatch.
//!
//! # Main modules and structures
//!
//! ## Public build API
//!
//! - [`PiPNNConfig`] holds Randomized Ball Carving, fanout, leaf size, local `k`,
//!   and replication parameters.
//! - [`PiPNNBuildContext`] validates that algorithm parameters, graph pruning
//!   policy, metric, and caller-owned Rayon pool agree.
//! - [`build_graph`] runs the full pipeline over a borrowed row-major dataset and
//!   returns one dataset-ID adjacency list per input point.
//!
//! ## [`partition_kernel`]
//!
//! Partition callers compute point-by-leader dots with GEMM.
//! [`partition_kernel::PartitionInput`] bundles that tile with typed
//! [`partition_kernel::PartitionScales`]. A prepared
//! [`partition_kernel::PartitionKernel`] writes sorted leader-local positions to
//! caller-owned output. Fanout is output column count and is bounded by
//! [`partition_kernel::MAX_PARTITION_FANOUT`]. Module documentation describes
//! scale units, validation, `process_points`, and tracker insertion.
//!
//! ## [`leaf_kernel`]
//!
//! Leaf callers compute a lower-triangular point-by-point dot matrix with
//! `sgemm_aat_lower`. [`leaf_kernel::LeafKernelWorkspace`] owns reusable
//! per-worker scratch, and
//! [`leaf_kernel::LeafKernel`] writes sorted [`leaf_kernel::LeafNeighbor`] values
//! to caller-owned output. [`leaf_kernel::leaf_neighbor_count`] derives each
//! leaf's width from point count and requested `k`. Module documentation explains
//! fixed-width selection, `process_pairs`, and stable endpoint insertion.
//!
//! ## Private pipeline stages
//!
//! - `partitioning` recursively samples leaders, invokes the partition kernel,
//!   scatters points into overlapping children, and returns bounded leaves.
//! - `leaf_build` gathers each leaf, computes its Gram matrix, invokes the leaf
//!   kernel, translates local positions to dataset IDs, and merges candidates.
//! - `finalization` applies shared Vamana RobustPrune to overfull candidate lists.
//! - `kernel_metric` owns norm preparation, exact ranking inputs, and numerical
//!   edge cases. The graph build selects one concrete metric for both kernels.
//!
//! # Typical use
//!
//! 1. Construct [`PiPNNConfig`] and DiskANN graph [`Config`].
//! 2. Create [`PiPNNBuildContext`] with metric and caller-owned Rayon pool.
//! 3. Call [`build_graph`] with one row-major [`MatrixView`] of dataset vectors.
//! 4. The outer index builder chooses start/frozen points and serializes returned
//!    adjacency; those policies are intentionally not part of this crate.
//!
//! Stage outputs are owned values. Leaves move into candidate construction;
//! candidate lists move into finalization. Ownership releases each stage's large
//! scratch before the next outer allocation.
//!
//! # Ownership and performance boundary
//!
//! Kernels borrow all matrices and mutate only caller-owned output/scratch. They
//! do not own providers, thread pools, GEMM buffers, graph IDs, or persistence.
//! Partition traversal performs one score per point-leader pair; leaf traversal
//! performs one score per unordered point pair. Detailed complexity and scratch
//! costs are documented in each module. PiPNN itself never names instruction
//! sets; `diskann-wide` owns architecture selection.

mod kernel_metric;
mod simd;

mod finalization;
mod leaf_build;
mod leaf_kernel;
mod partition_kernel;
mod partitioning;

use crate::{
    graph::{AdjacencyList, Config},
    utils::VectorRepr,
    ANNError, ANNResult,
};
use diskann_utils::views::MatrixView;
use diskann_vector::distance::Metric;
use rayon::ThreadPool;

/// Configuration of PiPNN's partitioning and local-neighbor algorithm.
///
/// Graph degree, pruning policy, and alpha belong to DiskANN's graph
/// configuration and are supplied separately through [`PiPNNBuildContext`].
#[derive(Clone, Debug, PartialEq)]
pub struct PiPNNConfig {
    /// Maximum number of points in a leaf.
    pub c_max: usize,
    /// Minimum leaf size used by global small-leaf merging.
    pub c_min: usize,
    /// Fraction of a cluster sampled as partition leaders.
    pub p_samp: f64,
    /// Number of nearest leaders retained at each overlapping partition level.
    pub fanout: Vec<usize>,
    /// Number of nearest neighbors selected within each leaf.
    pub k: usize,
    /// Number of independent partition passes over the dataset.
    pub replicas: usize,
}

impl PiPNNConfig {
    /// Validate the algorithm-specific partition and leaf-build parameters.
    pub fn validate(&self) -> ANNResult<()> {
        if self.c_max == 0 {
            return Err(config_error("c_max must be greater than zero"));
        }
        if self.c_min == 0 {
            return Err(config_error("c_min must be greater than zero"));
        }
        if self.c_min > self.c_max {
            return Err(config_error(format!(
                "c_min ({}) must not exceed c_max ({})",
                self.c_min, self.c_max
            )));
        }
        if !self.p_samp.is_finite() || !(0.0..=1.0).contains(&self.p_samp) || self.p_samp == 0.0 {
            return Err(config_error(format!(
                "p_samp ({}) must be finite and in (0, 1]",
                self.p_samp
            )));
        }
        if self.fanout.is_empty() {
            return Err(config_error("fanout must not be empty"));
        }
        if let Some(&fanout) = self
            .fanout
            .iter()
            .find(|&&fanout| !(1..=partition_kernel::MAX_PARTITION_FANOUT).contains(&fanout))
        {
            return Err(config_error(format!(
                "fanout ({fanout}) must be in [1, {}]",
                partition_kernel::MAX_PARTITION_FANOUT
            )));
        }
        if self.k == 0 {
            return Err(config_error("k must be greater than zero"));
        }
        if self.replicas == 0 {
            return Err(config_error("replicas must be greater than zero"));
        }
        Ok(())
    }
}

/// Validated, borrowed policy and execution context for one PiPNN graph build.
#[derive(Debug)]
pub struct PiPNNBuildContext<'a> {
    pub(crate) config: PiPNNConfig,
    pub(crate) graph: &'a Config,
    pub(crate) metric: Metric,
    pub(crate) pool: &'a ThreadPool,
}

impl<'a> PiPNNBuildContext<'a> {
    /// Validate and combine PiPNN configuration with outer graph policy.
    pub fn new(
        config: PiPNNConfig,
        graph: &'a Config,
        metric: Metric,
        pool: &'a ThreadPool,
    ) -> ANNResult<Self> {
        config.validate()?;
        if !graph.alpha().is_finite() || graph.alpha() < 1.0 {
            return Err(config_error(format!(
                "graph alpha ({}) must be finite and at least 1",
                graph.alpha()
            )));
        }
        if graph.prune_kind() != metric.into() {
            return Err(config_error(format!(
                "graph prune kind {:?} is incompatible with metric {metric:?}",
                graph.prune_kind()
            )));
        }

        Ok(Self {
            config,
            graph,
            metric,
            pool,
        })
    }
}

/// Build PiPNN adjacency for real points in `data`.
///
/// This is the core algorithm boundary. Search entry-point selection, frozen nodes,
/// providers, serialization, and index writers belong to the outer build pipelines.
/// For raw `u8` and `i8` vectors, `CosineNormalized` is evaluated as `Cosine` because
/// those representations are converted to f32 scratch but are not unit-normalized.
pub fn build_graph<T>(
    data: MatrixView<'_, T>,
    context: &PiPNNBuildContext<'_>,
) -> ANNResult<Vec<AdjacencyList<u32>>>
where
    T: VectorRepr + Send + Sync + 'static,
{
    context.pool.install(|| build_graph_inner(data, context))
}

fn build_graph_inner<T>(
    data: MatrixView<'_, T>,
    context: &PiPNNBuildContext<'_>,
) -> ANNResult<Vec<AdjacencyList<u32>>>
where
    T: VectorRepr + Send + Sync + 'static,
{
    if data.nrows() == 0 {
        return Err(ANNError::log_dimension_mismatch_error(
            "PiPNN requires at least one data point".into(),
        ));
    }
    if data.ncols() == 0 {
        return Err(ANNError::log_dimension_mismatch_error(
            "PiPNN requires at least one data dimension".into(),
        ));
    }
    if data.nrows() > u32::MAX as usize {
        return Err(config_error(format!(
            "dataset point count ({}) exceeds the u32 graph ID limit",
            data.nrows()
        )));
    }
    data.nrows().checked_mul(data.ncols()).ok_or_else(|| {
        ANNError::log_dimension_mismatch_error(format!(
            "PiPNN dataset shape {} x {} overflows usize",
            data.nrows(),
            data.ncols()
        ))
    })?;
    // Integer source vectors are not guaranteed unit-normalized after conversion,
    // so their normalized-cosine request must use the norm-aware formula.
    let metric = effective_metric::<T>(context.metric);

    let leaves = tracing::info_span!("pipnn.partition")
        .in_scope(|| partitioning::partition(data, context.config.clone(), metric))?;
    // `leaves` is consumed here. Workers borrow individual ID lists during the
    // parallel pass, and the complete partition allocation drops on return.
    let candidates = tracing::info_span!("pipnn.leaf_build").in_scope(|| {
        leaf_build::build_leaf_candidates(data, leaves, context.config.k, metric)
            .map_err(ANNError::opaque)
    })?;
    // Finalization consumes candidate lists and reuses their allocations for the
    // resulting adjacency where possible.
    tracing::info_span!("pipnn.finalization")
        .in_scope(|| finalization::prune_overfull(data, candidates, context.graph, metric))
}

fn effective_metric<T: 'static>(metric: Metric) -> Metric {
    use std::any::TypeId;

    if metric == Metric::CosineNormalized
        && (TypeId::of::<T>() == TypeId::of::<u8>() || TypeId::of::<T>() == TypeId::of::<i8>())
    {
        Metric::Cosine
    } else {
        metric
    }
}

#[track_caller]
fn config_error(message: impl std::fmt::Display) -> ANNError {
    ANNError::log_index_config_error("PiPNN".into(), message.to_string())
}

#[cfg(test)]
mod tests;

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Numerical kernels for provider-independent PiPNN graph construction.
//!
//! PiPNN assigns points to overlapping leader partitions. It computes one
//! lower-triangular all-pairs matrix per bounded leaf. It then merges selected
//! leaf neighbors into graph candidates.
//!
//! - [`partition_kernel`] ranks leader-column positions. Output width is runtime
//!   fanout. A scratch vector retains ranked leaders and reuses its allocation.
//! - [`leaf_kernel`] scans each pair in the strict lower triangle one time. It
//!   updates both endpoints. Each point keeps the requested number of neighbors.
//!   This number cannot exceed the number of other points in the leaf.
//! - `kernel_metric` owns norm preparation and exact metric ranking inputs.
//!
//! The graph build selects concrete architecture `A` and metric `M` types once.
//! Both kernels validate views and metric norm layouts before unchecked SIMD
//! access. They mutate only caller-owned output and scratch storage.

mod finalization;
mod kernel_metric;
mod simd;
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

/// Build PiPNN adjacency for real rows in `data`.
///
/// This is the core algorithm boundary. Search entry-point selection, frozen nodes,
/// providers, serialization, and index writers belong to the outer build pipelines.
/// For raw `u8` and `i8` rows, `CosineNormalized` is evaluated as `Cosine` because
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
            "PiPNN requires at least one data row".into(),
        ));
    }
    if data.ncols() == 0 {
        return Err(ANNError::log_dimension_mismatch_error(
            "PiPNN requires at least one data dimension".into(),
        ));
    }
    if data.nrows() > u32::MAX as usize {
        return Err(config_error(format!(
            "dataset row count ({}) exceeds the u32 graph ID limit",
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
    let metric = effective_metric::<T>(context.metric);

    let leaves = tracing::info_span!("pipnn.partition")
        .in_scope(|| partitioning::partition(data, &context.config, metric))?;
    let candidates = tracing::info_span!("pipnn.leaf_build").in_scope(|| {
        leaf_build::build_leaf_candidates(data, &leaves, context.config.k, metric)
            .map_err(ANNError::opaque)
    })?;
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

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
//! - `leaf_metric` and `partition_metric` fill portable ranking buffers.
//!
//! The graph build selects concrete architecture `A` and metric `M` types once.
//! Metric computation is architecture-neutral. Kernels use `A` for SIMD ranking.

mod finalization;
mod leaf_build;
mod leaf_kernel;
mod leaf_metric;
mod partition_kernel;
mod partition_metric;
mod partitioning;
mod simd;

use crate::{
    graph::{AdjacencyList, Config},
    utils::VectorRepr,
    ANNError, ANNResult,
};
use diskann_utils::views::MatrixView;
use diskann_vector::distance::Metric;
use rayon::ThreadPool;

pub(super) struct L2;
pub(super) struct Cosine;
pub(super) struct CosineNormalized;
pub(super) struct InnerProduct;

/// Convert one dot product and two norms to cosine distance.
///
/// Treat a zero or subnormal norm as zero similarity. This rule takes precedence
/// over the dot value. Clamp finite similarity to the cosine range. Otherwise,
/// a NaN input produces a NaN distance.
#[inline(always)]
fn cosine_distance(dot: f32, source_norm: f32, target_norm: f32) -> f32 {
    if source_norm < f32::MIN_POSITIVE.sqrt() || target_norm < f32::MIN_POSITIVE.sqrt() {
        1.0
    } else {
        1.0 - (dot / (source_norm * target_norm)).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod cosine_distance_contract_tests {
    use super::cosine_distance;
    use rstest::rstest;

    mod cosine_distance_tests {
        use super::*;

        #[test]
        fn zero_norm_takes_precedence_over_a_nan_dot_product() {
            // Given
            let dot = f32::NAN;
            let source_norm = 0.0_f32;
            let target_norm = 1.0_f32;
            let zero_similarity = 0.0_f32;
            let expected = 1.0 - zero_similarity;

            // When
            let actual = cosine_distance(dot, source_norm, target_norm);

            // Then
            assert_eq!(actual, expected);
        }

        #[rstest]
        #[case::zero_source(0.0, 1.0)]
        #[case::zero_target(1.0, 0.0)]
        #[case::subnormal_source(f32::MIN_POSITIVE.sqrt() / 2.0, 1.0)]
        #[case::subnormal_target(1.0, f32::MIN_POSITIVE.sqrt() / 2.0)]
        #[trace]
        fn zero_or_subnormal_norm_produces_unit_distance(
            #[case] source_norm: f32,
            #[case] target_norm: f32,
        ) {
            // Given
            let dot = 0.0_f32;
            let zero_similarity = 0.0_f32;
            let expected = 1.0 - zero_similarity;

            // When
            let actual = cosine_distance(dot, source_norm, target_norm);

            // Then
            assert_eq!(actual, expected);
        }

        #[test]
        fn minimum_normal_norm_uses_normalized_similarity() {
            // Given
            let source_norm = f32::MIN_POSITIVE.sqrt();
            let target_norm = 1.0_f32;
            let expected_similarity = 0.5_f32;
            let dot = expected_similarity * source_norm * target_norm;
            let expected = 1.0 - expected_similarity;

            // When
            let actual = cosine_distance(dot, source_norm, target_norm);

            // Then
            assert_eq!(actual, expected);
        }

        #[rstest]
        #[case::above_one(1.0)]
        #[case::below_negative_one(-1.0)]
        #[trace]
        fn finite_similarity_outside_the_cosine_range_is_clamped(#[case] bounded_similarity: f32) {
            // Given
            let source_norm = 2.0_f32;
            let target_norm = 2.0_f32;
            let norm_product = source_norm * target_norm;
            let rounding_excess = f32::EPSILON * norm_product;
            let dot = bounded_similarity * (norm_product + rounding_excess);
            let expected = 1.0 - bounded_similarity;

            // When
            let actual = cosine_distance(dot, source_norm, target_norm);

            // Then
            assert_eq!(actual, expected);
        }

        #[rstest]
        #[case::nan_dot(f32::NAN, 1.0, 1.0)]
        #[case::nan_source_norm(0.0, f32::NAN, 1.0)]
        #[case::nan_target_norm(0.0, 1.0, f32::NAN)]
        #[trace]
        fn nan_without_a_zero_norm_produces_nan_distance(
            #[case] dot: f32,
            #[case] source_norm: f32,
            #[case] target_norm: f32,
        ) {
            // Given: the case supplies one NaN. The other values do not select the zero-norm rule.

            // When
            let actual = cosine_distance(dot, source_norm, target_norm);

            // Then
            assert!(actual.is_nan());
        }
    }
}

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
    fn validate(&self) -> ANNResult<()> {
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

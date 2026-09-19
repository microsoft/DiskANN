/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Provider-independent [PiPNN](https://arxiv.org/html/2602.21247v1) graph construction.
//!
//! PiPNN builds graph candidates in three steps. A leader is a sampled dataset
//! point that acts as the center of one child partition. A leaf is a bounded
//! child partition used for local neighbor selection.
//!
//! 1. `partitioning` samples leaders and makes overlapping leaves of at most
//!    `c_max` points. Points without rankable leaders contribute no assignments.
//! 2. `leaf_build` computes one ranking-distance buffer for each leaf. It
//!    selects local neighbors and merges their global point IDs.
//! 3. `finalization` applies Vamana RobustPrune to each candidate list that is
//!    longer than the graph degree.
//!
//! `diskann-wide` selects architecture `A` for the ranking kernels. One match
//! selects metric marker `M`. Metric computation stays architecture-neutral.
//! Ranking kernels reapply architecture features inside Rayon jobs.
//!
//! [`PiPNNConfig`] contains partition and local-neighbor parameters.
//! [`PiPNNBuildContext`] borrows graph policy and a Rayon pool. [`build_graph`]
//! borrows one contiguous [`MatrixView`]. It returns one adjacency list for each
//! input point.
//!
//! The function does not load providers or select start and frozen points. It
//! also does not quantize, serialize, or search the graph.
//!
//! Partition and leaf work use separate reusable buffers. The build consumes
//! each output before it creates another graph representation.

mod conversion;
mod finalization;
mod leaf_build;
mod leaf_kernel;
mod leaf_metric;
mod partition_kernel;
mod partition_metric;
mod partitioning;
mod simd;
mod topk;

use crate::{
    ANNError, ANNResult,
    graph::{AdjacencyList, Config},
    utils::VectorRepr,
};
use diskann_utils::views::MatrixView;
use diskann_vector::distance::Metric;
use diskann_wide::arch::{self, Target2};
use rayon::ThreadPool;

use self::{leaf_metric::LeafMetric, partition_metric::PartitionMetric, simd::PiPNNSIMDSchema};

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
mod cosine_distance_tests {
    use super::cosine_distance;
    use rstest::rstest;

    #[rstest]
    #[case::same_direction(6.0, 2.0, 3.0, 0.0)]
    #[case::opposite_directions(-6.0, 2.0, 3.0, 2.0)]
    #[case::orthogonal(0.0, 2.0, 3.0, 1.0)]
    #[case::positive_similarity(3.0, 2.0, 3.0, 0.5)]
    #[case::negative_similarity(-3.0, 2.0, 3.0, 1.5)]
    fn distance_uses_both_vector_norms(
        #[case] dot: f32,
        #[case] source_norm: f32,
        #[case] target_norm: f32,
        #[case] expected: f32,
    ) {
        assert_eq!(cosine_distance(dot, source_norm, target_norm), expected);
    }

    #[rstest]
    #[case::above_one(1.0 + f32::EPSILON, 0.0)]
    #[case::below_minus_one(-1.0 - f32::EPSILON, 2.0)]
    fn rounding_outside_the_similarity_range_is_clamped(
        #[case] similarity: f32,
        #[case] expected: f32,
    ) {
        assert_eq!(cosine_distance(similarity, 1.0, 1.0), expected);
    }

    #[rstest]
    #[case::zero(0.0)]
    #[case::below_cutoff(f32::from_bits(f32::MIN_POSITIVE.sqrt().to_bits() - 1))]
    fn a_small_norm_takes_precedence_over_the_dot_product(
        #[case] small_norm: f32,
        #[values(0.75, f32::NAN)] dot: f32,
        #[values(false, true)] small_source: bool,
    ) {
        let (source_norm, target_norm) = if small_source {
            (small_norm, 1.0)
        } else {
            (1.0, small_norm)
        };

        assert_eq!(cosine_distance(dot, source_norm, target_norm), 1.0);
    }

    #[rstest]
    #[case::source(true)]
    #[case::target(false)]
    fn a_norm_at_the_cutoff_still_contributes_similarity(#[case] source_at_cutoff: bool) {
        let norm = f32::MIN_POSITIVE.sqrt();
        let (source_norm, target_norm) = if source_at_cutoff {
            (norm, 1.0)
        } else {
            (1.0, norm)
        };

        assert_eq!(cosine_distance(norm / 4.0, source_norm, target_norm), 0.75);
    }

    #[rstest]
    #[case::dot(f32::NAN, 2.0, 3.0)]
    #[case::source_norm(1.0, f32::NAN, 3.0)]
    #[case::target_norm(1.0, 2.0, f32::NAN)]
    fn nan_propagates_when_neither_norm_is_small(
        #[case] dot: f32,
        #[case] source_norm: f32,
        #[case] target_norm: f32,
    ) {
        assert!(cosine_distance(dot, source_norm, target_norm).is_nan());
    }
}

/// PiPNN partition and leaf-selection policy.
///
/// DiskANN graph policy separately supplies degree, alpha, and prune metric.
#[derive(Clone, Debug, PartialEq)]
pub struct PiPNNConfig {
    /// Maximum number of points in a leaf.
    pub c_max: usize,
    /// Minimum leaf size used by global small-leaf merging.
    pub c_min: usize,
    /// Fraction of a cluster sampled as child-partition centers.
    pub p_samp: f64,
    /// Number of nearest centers assigned at each recursive partition level.
    /// Levels after this schedule assign each point to one center.
    pub fanout: Vec<usize>,
    /// Number of nearest neighbors selected within each leaf.
    pub leaf_k: usize,
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
        if !(0.0 < self.p_samp && self.p_samp <= 1.0) {
            return Err(config_error(format!(
                "p_samp ({}) must be in (0, 1]",
                self.p_samp
            )));
        }
        if self.fanout.is_empty() {
            return Err(config_error("fanout must not be empty"));
        }
        if self.fanout.contains(&0) {
            return Err(config_error("fanout values must be greater than zero"));
        }
        if self.leaf_k == 0 {
            return Err(config_error("leaf_k must be greater than zero"));
        }
        if self.replicas == 0 {
            return Err(config_error("replicas must be greater than zero"));
        }
        Ok(())
    }
}

/// PiPNN policy and borrowed execution resources for one graph build.
#[derive(Debug)]
pub struct PiPNNBuildContext<'a> {
    config: PiPNNConfig,
    graph: &'a Config,
    metric: Metric,
    pool: &'a ThreadPool,
}

impl<'a> PiPNNBuildContext<'a> {
    /// Check and combine PiPNN configuration with DiskANN graph policy.
    pub fn new(
        config: PiPNNConfig,
        graph: &'a Config,
        metric: Metric,
        pool: &'a ThreadPool,
    ) -> ANNResult<Self> {
        config.validate()?;
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

/// Build one PiPNN adjacency list for each point in `data`.
///
/// This graph contains only real dataset points. Start-point selection and index
/// serialization are separate operations.
///
/// Raw `u8` and `i8` vectors are not unit-normalized after conversion to `f32`.
/// The build therefore uses norm-aware cosine for these two input types.
pub fn build_graph<T>(
    data: MatrixView<'_, T>,
    context: &PiPNNBuildContext<'_>,
) -> ANNResult<Vec<AdjacencyList<u32>>>
where
    T: VectorRepr,
{
    context
        .pool
        .install(|| validate_and_dispatch_build(data, context))
}

/// Check dataset bounds and select the architecture and metric implementation.
fn validate_and_dispatch_build<T>(
    data: MatrixView<'_, T>,
    context: &PiPNNBuildContext<'_>,
) -> ANNResult<Vec<AdjacencyList<u32>>>
where
    T: VectorRepr,
{
    if data.nrows() == 0 {
        return Err(ANNError::message("PiPNN requires at least one data point"));
    }
    if data.ncols() == 0 {
        return Err(ANNError::message(
            "PiPNN requires at least one data dimension",
        ));
    }
    if data.nrows() > u32::MAX as usize {
        return Err(config_error(format!(
            "dataset point count ({}) exceeds the u32 graph ID limit",
            data.nrows()
        )));
    }
    arch::dispatch2_no_features(BuildGraph, data, context)
}

struct BuildGraph;

impl<A, T> Target2<A, ANNResult<Vec<AdjacencyList<u32>>>, MatrixView<'_, T>, &PiPNNBuildContext<'_>>
    for BuildGraph
where
    A: PiPNNSIMDSchema,
    T: VectorRepr,
{
    fn run(
        self,
        arch: A,
        data: MatrixView<'_, T>,
        context: &PiPNNBuildContext<'_>,
    ) -> ANNResult<Vec<AdjacencyList<u32>>> {
        // Converting raw integer coordinates does not normalize their vectors.
        let metric = effective_metric::<T>(context.metric);
        match metric {
            Metric::L2 => build_graph_for::<A, L2, T>(arch, data, context, metric),
            Metric::Cosine => build_graph_for::<A, Cosine, T>(arch, data, context, metric),
            Metric::CosineNormalized => {
                build_graph_for::<A, CosineNormalized, T>(arch, data, context, metric)
            }
            Metric::InnerProduct => {
                build_graph_for::<A, InnerProduct, T>(arch, data, context, metric)
            }
        }
    }
}

/// Run the PiPNN graph pipeline for one selected metric implementation.
///
/// The function builds overlapping leaves, merges direct candidates, and applies
/// final graph-degree pruning.
fn build_graph_for<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    context: &PiPNNBuildContext<'_>,
    metric: Metric,
) -> ANNResult<Vec<AdjacencyList<u32>>>
where
    A: PiPNNSIMDSchema,
    M: LeafMetric + PartitionMetric,
    T: VectorRepr,
{
    let leaves = tracing::info_span!("pipnn.partition")
        .in_scope(|| partitioning::partition::<A, M, T>(arch, data, &context.config))?;
    // Leaf jobs borrow individual ID lists. This call consumes the leaf vector,
    // so its complete allocation drops when leaf construction returns.
    let candidates = tracing::info_span!("pipnn.leaf_build").in_scope(|| {
        leaf_build::build_leaf_candidates::<A, M, T>(arch, data, leaves, context.config.leaf_k)
            .map_err(ANNError::new)
    })?;
    // Finalization consumes each candidate list. It reuses that list's allocation
    // for the final adjacency when the graph policy permits it.
    tracing::info_span!("pipnn.finalization")
        .in_scope(|| finalization::prune_overfull(data, candidates, context.graph, metric))
}

fn effective_metric<T: VectorRepr>(metric: Metric) -> Metric {
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
    ANNError::message(format!("PiPNN configuration: {message}"))
}

#[cfg(test)]
mod effective_metric_tests {
    use super::*;
    use half::f16;
    use rstest::rstest;

    #[rstest]
    #[case::l2(Metric::L2)]
    #[case::cosine(Metric::Cosine)]
    #[case::normalized_cosine(Metric::CosineNormalized)]
    #[case::inner_product(Metric::InnerProduct)]
    fn floating_point_formats_keep_the_requested_metric(#[case] metric: Metric) {
        // Given / When: floating-point callers define whether their data is normalized.
        let actual = (
            effective_metric::<f32>(metric),
            effective_metric::<f16>(metric),
        );

        // Then: both supported floating-point formats preserve the request.
        assert_eq!(actual, (metric, metric));
    }

    #[rstest]
    #[case::l2(Metric::L2)]
    #[case::cosine(Metric::Cosine)]
    #[case::inner_product(Metric::InnerProduct)]
    fn integer_formats_keep_metrics_that_do_not_assume_normalization(#[case] metric: Metric) {
        // Given / When
        let actual = (
            effective_metric::<u8>(metric),
            effective_metric::<i8>(metric),
        );

        // Then: only normalized cosine needs the integer-data fallback.
        assert_eq!(actual, (metric, metric));
    }
}
#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "deterministic test fixture construction must abort on invalid setup"
)]
mod build_graph_tests {
    use super::{PiPNNBuildContext, PiPNNConfig, build_graph};
    use crate::graph::config::{self, MaxDegree};
    use diskann_utils::views::MatrixView;
    use diskann_vector::distance::Metric;
    use rstest::rstest;

    fn pipnn_config() -> PiPNNConfig {
        PiPNNConfig {
            c_max: 4,
            c_min: 1,
            p_samp: 0.5,
            fanout: vec![2],
            leaf_k: 1,
            replicas: 1,
        }
    }

    fn graph_config(metric: Metric, degree: usize) -> crate::graph::Config {
        config::Builder::new_with(degree, MaxDegree::same(), 8, metric.into(), |builder| {
            builder.alpha(1.2);
        })
        .build()
        .unwrap()
    }

    fn pool(threads: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
    }

    fn rows(graph: Vec<crate::graph::AdjacencyList<u32>>) -> Vec<Vec<u32>> {
        graph.into_iter().map(Vec::from).collect()
    }

    #[test]
    fn single_leaf_merges_neighbors_in_both_directions() {
        // Given: unequal gaps give nearest choices 0->1, 1->0, 2->1, 3->2.
        // Symmetric insertion adds the reverse edge for each choice.
        let values = [0.0_f32, 1.0, 3.0, 7.0];
        let data = MatrixView::try_from(&values[..], 4, 1).unwrap();
        let expected = [vec![1], vec![0, 2], vec![1, 3], vec![2]];
        let graph = graph_config(Metric::L2, 2);
        let pool = pool(2);
        let context = PiPNNBuildContext::new(pipnn_config(), &graph, Metric::L2, &pool).unwrap();

        // When
        let actual = build_graph(data, &context).unwrap();

        // Then
        assert_eq!(rows(actual), expected);
    }

    #[test]
    fn omits_non_rankable_candidates_without_invalid_ids() {
        // Given: three points force partitioning at c_max=2. The NaN point
        // has no rankable leader or neighbor, so only the finite points connect.
        let values = [0.0_f32, 1.0, f32::NAN];
        let data = MatrixView::try_from(&values[..], 3, 1).unwrap();
        let graph = graph_config(Metric::InnerProduct, 2);
        let pool = pool(1);
        let config = PiPNNConfig {
            c_max: 2,
            c_min: 1,
            p_samp: 1.0,
            fanout: vec![2],
            leaf_k: 1,
            replicas: 1,
        };
        let context = PiPNNBuildContext::new(config, &graph, Metric::InnerProduct, &pool).unwrap();
        let expected = [vec![1], vec![0], vec![]];

        // When
        let actual = build_graph(data, &context).unwrap();

        // Then
        assert_eq!(rows(actual), expected);
    }

    #[test]
    fn finalization_keeps_the_nearest_neighbor_when_degree_is_one() {
        // Given: selecting both peers makes each row overfull. With degree one,
        // RobustPrune keeps the nearest: 0->1, 1->0, 2->1.
        let values = [0.0_f32, 1.0, 3.0];
        let data = MatrixView::try_from(&values[..], 3, 1).unwrap();
        let expected = [vec![1], vec![0], vec![1]];
        let graph = graph_config(Metric::L2, 1);
        let pool = pool(2);
        let config = PiPNNConfig {
            c_max: 3,
            c_min: 1,
            p_samp: 0.5,
            fanout: vec![2],
            leaf_k: 2,
            replicas: 1,
        };
        let context = PiPNNBuildContext::new(config, &graph, Metric::L2, &pool).unwrap();

        // When
        let actual = build_graph(data, &context).unwrap();

        // Then
        assert_eq!(rows(actual), expected);
    }

    #[rstest]
    #[case::no_points(0, 4, "PiPNN requires at least one data point")]
    #[case::no_dimensions(4, 0, "PiPNN requires at least one data dimension")]
    fn empty_input_shape_returns_an_error(
        #[case] points: usize,
        #[case] dimensions: usize,
        #[case] message: &str,
    ) {
        // Given
        let data = MatrixView::try_from(&[] as &[f32], points, dimensions).unwrap();
        let graph = graph_config(Metric::L2, 2);
        let pool = pool(1);
        let context = PiPNNBuildContext::new(pipnn_config(), &graph, Metric::L2, &pool).unwrap();

        // When
        let error = build_graph(data, &context).unwrap_err();

        // Then
        assert!(error.to_string().contains(message), "{error}");
    }

    #[rstest]
    #[case::l2(Metric::L2, [vec![1, 2], vec![0], vec![0]])]
    #[case::inner_product(Metric::InnerProduct, [vec![1], vec![0, 2], vec![1]])]
    fn selected_metric_changes_the_neighbor_graph(
        #[case] metric: Metric,
        #[case] expected: [Vec<u32>; 3],
    ) {
        // Given: squared L2 scores for pairs (0,1),(0,2),(1,2) are 4,0.8,6.4;
        // negative-dot scores are -3,-0.6,-1.8. The nearest choices differ.
        let values = [1.0_f32, 0.0, 3.0, 0.0, 0.6, 0.8];
        let data = MatrixView::try_from(&values[..], 3, 2).unwrap();
        let graph = graph_config(metric, 2);
        let pool = pool(1);
        let context = PiPNNBuildContext::new(pipnn_config(), &graph, metric, &pool).unwrap();

        // When
        let actual = build_graph(data, &context).unwrap();

        // Then
        assert_eq!(rows(actual), expected);
    }

    #[rstest]
    #[case::cosine(Metric::Cosine)]
    #[case::normalized_cosine(Metric::CosineNormalized)]
    fn cosine_metrics_rank_unit_vectors_by_direction(#[case] metric: Metric) {
        // Given: pairwise dots are 0.6,-1,-0.6, so 0 chooses 1, 1 chooses 0,
        // and 2 chooses 1. Symmetrization gives the expected three-node chain.
        let values = [1.0_f32, 0.0, 0.6, 0.8, -1.0, 0.0];
        let data = MatrixView::try_from(&values[..], 3, 2).unwrap();
        let expected = [vec![1], vec![0, 2], vec![1]];
        let graph = graph_config(metric, 2);
        let pool = pool(1);
        let context = PiPNNBuildContext::new(pipnn_config(), &graph, metric, &pool).unwrap();

        // When
        let actual = build_graph(data, &context).unwrap();

        // Then
        assert_eq!(rows(actual), expected);
    }

    #[test]
    fn singleton_has_no_self_neighbor() {
        // Given: a singleton has zero non-self candidates even though leaf_k=1.
        let values = [3.0_f32, 4.0];
        let data = MatrixView::try_from(&values[..], 1, 2).unwrap();
        let graph = graph_config(Metric::L2, 2);
        let pool = pool(1);
        let context = PiPNNBuildContext::new(pipnn_config(), &graph, Metric::L2, &pool).unwrap();

        // When
        let actual = build_graph(data, &context).unwrap();

        // Then
        assert_eq!(rows(actual), [Vec::<u32>::new()]);
    }

    #[rstest]
    #[case::unsigned([1_u8, 0, 100, 1, 2, 0, 0, 1, 1, 1, 200, 2, 2, 1, 1, 2])]
    #[case::signed([1_i8, 0, 100, 1, 2, 0, 0, 1, 1, 1, 120, 2, 2, 1, 1, 2])]
    fn integer_normalized_cosine_matches_cosine<T: crate::utils::VectorRepr>(
        #[case] values: [T; 16],
    ) {
        // Given: unequal norms distinguish cosine from raw dot ranking.
        // Keep one leaf so partition randomness cannot hide metric selection.
        let data = MatrixView::try_from(&values[..], 8, 2).unwrap();
        let pool = pool(2);
        let build = |metric| {
            let graph = graph_config(metric, 2);
            let config = PiPNNConfig {
                c_max: 8,
                c_min: 1,
                p_samp: 0.5,
                fanout: vec![2],
                leaf_k: 1,
                replicas: 1,
            };
            let context = PiPNNBuildContext::new(config, &graph, metric, &pool).unwrap();
            rows(build_graph(data, &context).unwrap())
        };
        let expected = build(Metric::Cosine);

        // When
        let actual = build(Metric::CosineNormalized);

        // Then
        assert_eq!(actual, expected);
    }

    #[rstest]
    #[case::one_worker(1)]
    #[case::four_workers(4)]
    fn recursive_build_is_deterministic_for_a_fixed_pool(#[case] threads: usize) {
        // Given: distinct points require multiple partition levels and two replicas.
        let values: Vec<f32> = (0..96).flat_map(|i| [i as f32, (i * i) as f32]).collect();
        let data = MatrixView::try_from(values.as_slice(), 96, 2).unwrap();
        let graph = graph_config(Metric::L2, 8);
        let pool = pool(threads);
        let config = PiPNNConfig {
            c_max: 16,
            c_min: 4,
            p_samp: 0.25,
            fanout: vec![3, 2],
            leaf_k: 3,
            replicas: 2,
        };
        let context = PiPNNBuildContext::new(config, &graph, Metric::L2, &pool).unwrap();

        let expected = build_graph(data, &context).unwrap();

        // When
        let actual = build_graph(data, &context).unwrap();

        // Then
        assert_eq!(actual, expected);
    }
}
#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "deterministic test fixture construction must abort on invalid setup"
)]
mod config_tests {
    use super::{PiPNNBuildContext, PiPNNConfig};
    use crate::graph::config::{self, MaxDegree};
    use diskann_vector::distance::Metric;
    use rstest::rstest;

    fn pipnn_config() -> PiPNNConfig {
        PiPNNConfig {
            c_max: 512,
            c_min: 64,
            p_samp: 0.01,
            fanout: vec![10, 3],
            leaf_k: 2,
            replicas: 1,
        }
    }

    fn graph_config(metric: Metric, alpha: f32) -> crate::graph::Config {
        config::Builder::new_with(64, MaxDegree::same(), 72, metric.into(), |builder| {
            builder.alpha(alpha);
        })
        .build()
        .unwrap()
    }

    fn pool() -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
    }

    #[rstest]
    #[case::zero_maximum(PiPNNConfig { c_max: 0, ..pipnn_config() }, "c_max must be greater than zero")]
    #[case::zero_minimum(PiPNNConfig { c_min: 0, ..pipnn_config() }, "c_min must be greater than zero")]
    #[case::minimum_exceeds_maximum(PiPNNConfig { c_min: 513, ..pipnn_config() }, "c_min (513) must not exceed c_max (512)")]
    #[case::zero_sampling(PiPNNConfig { p_samp: 0.0, ..pipnn_config() }, "p_samp (0) must be in (0, 1]")]
    #[case::negative_sampling(PiPNNConfig { p_samp: -0.5, ..pipnn_config() }, "p_samp (-0.5) must be in (0, 1]")]
    #[case::sampling_above_one(PiPNNConfig { p_samp: 1.5, ..pipnn_config() }, "p_samp (1.5) must be in (0, 1]")]
    #[case::nan_sampling(PiPNNConfig { p_samp: f64::NAN, ..pipnn_config() }, "p_samp (NaN) must be in (0, 1]")]
    #[case::infinite_sampling(PiPNNConfig { p_samp: f64::INFINITY, ..pipnn_config() }, "p_samp (inf) must be in (0, 1]")]
    #[case::empty_fanout(PiPNNConfig { fanout: vec![], ..pipnn_config() }, "fanout must not be empty")]
    #[case::zero_fanout(PiPNNConfig { fanout: vec![1, 0], ..pipnn_config() }, "fanout values must be greater than zero")]
    #[case::zero_neighbors(PiPNNConfig { leaf_k: 0, ..pipnn_config() }, "leaf_k must be greater than zero")]
    #[case::zero_replicas(PiPNNConfig { replicas: 0, ..pipnn_config() }, "replicas must be greater than zero")]
    fn invalid_policy_reports_the_rejected_parameter(
        #[case] config: PiPNNConfig,
        #[case] message: &str,
    ) {
        // Given: one invalid field in an otherwise valid policy.
        let expected = format!("PiPNN configuration: {message}");

        // When
        let error = config.validate().unwrap_err();

        // Then
        assert!(error.to_string().contains(&expected), "{error}");
    }

    #[test]
    fn context_validates_the_partition_policy() {
        // Given: verify the public constructor invokes the policy validation.
        let config = PiPNNConfig {
            replicas: 0,
            ..pipnn_config()
        };
        let graph = graph_config(Metric::L2, 1.2);
        let pool = pool();

        // When
        let error = PiPNNBuildContext::new(config, &graph, Metric::L2, &pool).unwrap_err();

        // Then
        assert!(
            error
                .to_string()
                .contains("PiPNN configuration: replicas must be greater than zero"),
            "{error}",
        );
    }

    #[test]
    fn rejects_graph_policy_for_a_different_metric() {
        // Given: inner-product occlusion cannot prune an L2 build.
        let graph = graph_config(Metric::InnerProduct, 1.2);
        let pool = pool();
        let expected = format!(
            "PiPNN configuration: graph prune kind {:?} is incompatible with metric L2",
            graph.prune_kind(),
        );

        // When
        let error = PiPNNBuildContext::new(pipnn_config(), &graph, Metric::L2, &pool).unwrap_err();

        // Then
        assert!(error.to_string().contains(&expected), "{error}");
    }
}

#[cfg(all(test, not(miri)))]
mod test_support {
    use diskann_vector::distance::Metric;

    pub(super) fn dense_points(rows: usize, dimensions: usize, seed: u64) -> Vec<f32> {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        // Multiples of 1/8 keep unnormalized products exact at the tested sizes.
        let mut rng = StdRng::seed_from_u64(seed);
        let mut values: Vec<_> = (0..rows * dimensions)
            .map(|_| rng.random_range(-16..=16) as f32 / 8.0)
            .collect();
        for (point, row) in values.chunks_exact_mut(dimensions).enumerate() {
            // A substantial final coordinate makes an omitted dimension visible
            // even after normalization. It also guarantees nonzero norms.
            row[dimensions - 1] = 8.0 + (point % 7) as f32;
        }
        values
    }

    // These scalar definitions use the actual f32 inputs, with f64 arithmetic.
    // Callers supply finite vectors with nonzero norms. L2 includes the point norm.
    pub(super) fn distance(metric: Metric, point: &[f32], target: &[f32]) -> f64 {
        match metric {
            Metric::L2 => point
                .iter()
                .zip(target)
                .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
                .sum(),
            Metric::InnerProduct | Metric::CosineNormalized => -point
                .iter()
                .zip(target)
                .map(|(&x, &y)| f64::from(x) * f64::from(y))
                .sum::<f64>(),
            Metric::Cosine => {
                let dot: f64 = point
                    .iter()
                    .zip(target)
                    .map(|(&x, &y)| f64::from(x) * f64::from(y))
                    .sum();
                let norm = |row: &[f32]| {
                    row.iter()
                        .map(|&x| f64::from(x).powi(2))
                        .sum::<f64>()
                        .sqrt()
                };
                1.0 - (dot / (norm(point) * norm(target))).clamp(-1.0, 1.0)
            }
        }
    }

    pub(super) fn normalize(values: &mut [f32], dimensions: usize) {
        for row in values.chunks_exact_mut(dimensions) {
            let norm = row
                .iter()
                .map(|&x| f64::from(x).powi(2))
                .sum::<f64>()
                .sqrt();
            for value in row {
                *value = (f64::from(*value) / norm) as f32;
            }
        }
    }

    // Put the second coordinate in the last dimension so omitting a tail changes ranking.
    pub(super) fn packed_points(
        coordinates: &[[f32; 2]],
        dimensions: usize,
        unit_norm: bool,
    ) -> Vec<f32> {
        let mut values = vec![0.0; coordinates.len() * dimensions];
        for (row, &[x, y]) in values.chunks_exact_mut(dimensions).zip(coordinates) {
            row[0] = x;
            row[dimensions - 1] = y;
        }
        if unit_norm {
            normalize(&mut values, dimensions);
        }
        values
    }

    #[test]
    fn dense_fixtures_are_replayable_and_exercise_the_last_dimension() {
        let values = dense_points(3, 17, 1287);

        assert_eq!(values, dense_points(3, 17, 1287));
        assert_eq!(values.len(), 3 * 17);
        assert_ne!(&values[..17], &values[17..34]);
        assert_eq!([values[16], values[33], values[50]], [8.0, 9.0, 10.0]);
        assert!(values[..16].iter().filter(|&&x| x != 0.0).count() > 8);
    }

    #[rstest::rstest]
    #[case::squared_l2(Metric::L2, 18.0)]
    #[case::negative_dot(Metric::InnerProduct, -2.0)]
    #[case::cosine(Metric::Cosine, 0.7830695421813438)]
    fn scalar_reference_matches_hand_calculated_distances(
        #[case] metric: Metric,
        #[case] expected: f64,
    ) {
        // [1, 2] and [4, -1]: squared difference 18, dot 2, squared norms 5 and 17.
        assert!((distance(metric, &[1.0, 2.0], &[4.0, -1.0]) - expected).abs() < 1.0e-14);
    }

    #[test]
    fn normalized_points_keep_their_direction_and_last_coordinate() {
        let actual = packed_points(&[[3.0, 4.0], [0.0, -2.0]], 3, true);
        assert_eq!(actual, [0.6, 0.0, 0.8, 0.0, 0.0, -1.0]);
        assert_eq!(
            distance(Metric::CosineNormalized, &actual[..3], &actual[3..]),
            f64::from(0.8_f32)
        );
    }
}

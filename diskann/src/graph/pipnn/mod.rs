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
//!    selects local neighbors. The direct path merges their global point IDs.
//!    The HashPrune path sends weighted edges to bounded point reservoirs.
//! 3. `finalization` applies Vamana RobustPrune to direct candidates. It also
//!    prunes HashPrune candidates when `final_prune` is true.
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

mod bf16;
mod conversion;
mod finalization;
mod hash_prune;
mod leaf_build;
mod leaf_kernel;
mod leaf_metric;
mod lsh;
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
    #[case::zero_source(0.0, 0.0, 1.0)]
    #[case::zero_target(0.0, 1.0, 0.0)]
    #[case::subnormal_source(f32::MIN_POSITIVE.sqrt() / 2.0, f32::MIN_POSITIVE.sqrt() / 2.0, 1.0)]
    #[case::subnormal_target(f32::MIN_POSITIVE.sqrt() / 2.0, 1.0, f32::MIN_POSITIVE.sqrt() / 2.0)]
    #[case::zero_norm_before_nan_dot(f32::NAN, 0.0, 1.0)]
    fn small_norm_produces_unit_distance(
        #[case] dot: f32,
        #[case] source_norm: f32,
        #[case] target_norm: f32,
    ) {
        assert_eq!(cosine_distance(dot, source_norm, target_norm), 1.0);
    }

    #[test]
    fn minimum_normal_norm_uses_normalized_similarity() {
        let norm = f32::MIN_POSITIVE.sqrt();
        assert_eq!(cosine_distance(0.5 * norm, norm, 1.0), 0.5);
    }

    #[rstest]
    #[case::above_one(1.0 + f32::EPSILON, 0.0)]
    #[case::below_negative_one(-1.0 - f32::EPSILON, 2.0)]
    fn finite_similarity_outside_the_cosine_range_is_clamped(
        #[case] similarity: f32,
        #[case] expected: f32,
    ) {
        assert_eq!(cosine_distance(similarity, 1.0, 1.0), expected);
    }

    #[rstest]
    #[case::nan_dot(f32::NAN, 1.0, 1.0)]
    #[case::nan_source_norm(0.0, f32::NAN, 1.0)]
    #[case::nan_target_norm(0.0, 1.0, f32::NAN)]
    fn nan_without_a_zero_norm_produces_nan_distance(
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

/// HashPrune policy for bounded candidate reservoirs.
#[derive(Clone, Debug, PartialEq)]
pub struct HashPruneConfig {
    /// Number of random-hyperplane bits in each relative-direction hash.
    pub num_hash_planes: usize,
    /// Maximum number of direction buckets retained for each source point.
    pub l_max: usize,
    /// Apply Vamana RobustPrune after reservoir extraction.
    pub final_prune: bool,
}

impl HashPruneConfig {
    /// Check the structural HashPrune limits.
    pub fn validate(&self) -> ANNResult<()> {
        if !(1..=lsh::MAX_PLANES).contains(&self.num_hash_planes) {
            return Err(config_error(format!(
                "num_hash_planes ({}) must be in [1, {}]",
                self.num_hash_planes,
                lsh::MAX_PLANES
            )));
        }
        if !(1..=hash_prune::MAX_RESERVOIR_LEN).contains(&self.l_max) {
            return Err(config_error(format!(
                "l_max ({}) must be in [1, {}]",
                self.l_max,
                hash_prune::MAX_RESERVOIR_LEN
            )));
        }
        Ok(())
    }

    /// Check that the reservoir and hash space can hold `degree` neighbors.
    pub fn validate_for_degree(&self, degree: usize) -> ANNResult<()> {
        self.validate()?;
        let hash_capacity = 1usize
            .checked_shl(self.num_hash_planes as u32)
            .unwrap_or(usize::MAX);
        let candidate_capacity = self.l_max.min(hash_capacity);
        if candidate_capacity < degree {
            return Err(config_error(format!(
                "HashPrune capacity min(l_max={}, hash buckets={hash_capacity}) must be at least \
                 the graph degree ({degree})",
                self.l_max
            )));
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
    hash_prune: Option<HashPruneConfig>,
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
            hash_prune: None,
        })
    }

    /// Enable HashPrune candidate merging for this build.
    pub fn with_hash_prune(mut self, config: HashPruneConfig) -> ANNResult<Self> {
        config.validate_for_degree(self.graph.pruned_degree().get())?;
        self.hash_prune = Some(config);
        Ok(self)
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
/// The function builds overlapping leaves and runs the configured candidate
/// merge. It prunes direct candidates to graph degree. It prunes HashPrune
/// candidates when `final_prune` is true.
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
    match &context.hash_prune {
        None => {
            // Leaf jobs borrow individual ID lists. This call consumes the leaf
            // vector, so its allocation drops when leaf construction returns.
            let candidates = tracing::info_span!("pipnn.leaf_build").in_scope(|| {
                leaf_build::build_leaf_candidates::<A, M, T>(
                    arch,
                    data,
                    leaves,
                    context.config.leaf_k,
                )
                .map_err(ANNError::new)
            })?;
            tracing::info_span!("pipnn.finalization")
                .in_scope(|| finalization::prune_overfull(data, candidates, context.graph, metric))
        }
        Some(config) => {
            // `HashPrune` lives until all leaf jobs finish. A leaf job locks only
            // one source reservoir at a time.
            let hash_prune =
                hash_prune::HashPrune::new(data, config.num_hash_planes, config.l_max, 42)?;
            // This call consumes the leaves. Each weighted CSR list exists only
            // during its leaf job. The reservoirs retain the selected edges.
            tracing::info_span!("pipnn.leaf_build").in_scope(|| {
                leaf_build::add_hash_prune_candidates::<A, M, T>(
                    arch,
                    data,
                    leaves,
                    context.config.leaf_k,
                    &hash_prune,
                )
                .map_err(ANNError::new)
            })?;
            if config.final_prune {
                let candidates = hash_prune.into_candidate_lists();
                tracing::info_span!("pipnn.finalization").in_scope(|| {
                    finalization::prune_overfull(data, candidates, context.graph, metric)
                })
            } else {
                Ok(hash_prune.into_nearest_lists(context.graph.pruned_degree().get()))
            }
        }
    }
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
    use super::{HashPruneConfig, PiPNNBuildContext, PiPNNConfig, build_graph};
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
    #[rstest]
    #[case::one_worker(1)]
    #[case::four_workers(4)]
    fn hash_prune_build_keeps_nearest_neighbors_in_each_direction(#[case] threads: usize) {
        // Given: two replicas offer the same leaf concurrently. Opposite
        // directions on a line have complementary hashes. Each ray retains
        // its nearest point. The reverse edge from source 0 reaches source 1
        // before source 1 offers point 2. Within-degree rows keep that order.
        let values = [-3.0_f32, 0.0, 1.0];
        let data = MatrixView::column_vector(&values[..]);
        let expected = [vec![1], vec![0, 2], vec![1]];
        let graph = graph_config(Metric::L2, 2);
        let pool = pool(threads);
        let config = PiPNNConfig {
            c_max: 3,
            c_min: 1,
            p_samp: 1.0,
            fanout: vec![1],
            leaf_k: 2,
            replicas: 2,
        };
        let context = PiPNNBuildContext::new(config, &graph, Metric::L2, &pool)
            .unwrap()
            .with_hash_prune(HashPruneConfig {
                num_hash_planes: 8,
                l_max: 16,
                final_prune: true,
            })
            .unwrap();

        // When
        let actual = build_graph(data, &context).unwrap();

        // Then
        assert_eq!(rows(actual), expected);
    }
}
#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "deterministic test fixture construction must abort on invalid setup"
)]
mod config_tests {
    use super::{HashPruneConfig, PiPNNBuildContext, PiPNNConfig};
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
    fn graph_config_with_degree(metric: Metric, alpha: f32, degree: usize) -> crate::graph::Config {
        config::Builder::new_with(degree, MaxDegree::same(), 72, metric.into(), |builder| {
            builder.alpha(alpha);
        })
        .build()
        .unwrap()
    }

    #[rstest]
    #[case::zero_hash_planes(HashPruneConfig {
        num_hash_planes: 0,
        l_max: 64,
        final_prune: true,
    })]
    #[case::too_many_hash_planes(HashPruneConfig {
        num_hash_planes: 17,
        l_max: 64,
        final_prune: true,
    })]
    #[case::zero_l_max(HashPruneConfig {
        num_hash_planes: 8,
        l_max: 0,
        final_prune: true,
    })]
    #[case::l_max_above_storage_limit(HashPruneConfig {
        num_hash_planes: 8,
        l_max: 256,
        final_prune: true,
    })]
    fn invalid_hash_prune_parameter_is_rejected(#[case] invalid_config: HashPruneConfig) {
        let graph = graph_config(Metric::L2, 1.2);
        let pool = pool();
        let context = PiPNNBuildContext::new(pipnn_config(), &graph, Metric::L2, &pool).unwrap();

        assert!(context.with_hash_prune(invalid_config).is_err());
    }

    #[test]
    fn candidate_capacity_below_graph_degree_is_rejected() {
        let graph = graph_config(Metric::L2, 1.2);
        let pool = pool();
        let context = PiPNNBuildContext::new(pipnn_config(), &graph, Metric::L2, &pool).unwrap();
        let below_degree_capacity = HashPruneConfig {
            num_hash_planes: 8,
            l_max: 63,
            final_prune: true,
        };

        assert!(context.with_hash_prune(below_degree_capacity).is_err());
    }

    #[test]
    fn candidate_capacity_equal_to_graph_degree_is_accepted() {
        let graph = graph_config(Metric::L2, 1.2);
        let pool = pool();
        let context = PiPNNBuildContext::new(pipnn_config(), &graph, Metric::L2, &pool).unwrap();
        let equal_degree_capacity = HashPruneConfig {
            num_hash_planes: 8,
            l_max: 64,
            final_prune: true,
        };

        context.with_hash_prune(equal_degree_capacity).unwrap();
    }

    #[test]
    fn hash_bucket_capacity_equal_to_graph_degree_is_accepted() {
        let pool = pool();
        let graph = graph_config_with_degree(Metric::L2, 1.2, 2);
        let context = PiPNNBuildContext::new(pipnn_config(), &graph, Metric::L2, &pool).unwrap();
        let two_hash_buckets = HashPruneConfig {
            num_hash_planes: 1,
            l_max: 64,
            final_prune: true,
        };

        context.with_hash_prune(two_hash_buckets).unwrap();
    }

    #[test]
    fn hash_bucket_capacity_below_graph_degree_is_rejected() {
        let pool = pool();
        let graph = graph_config_with_degree(Metric::L2, 1.2, 3);
        let context = PiPNNBuildContext::new(pipnn_config(), &graph, Metric::L2, &pool).unwrap();
        let two_hash_buckets = HashPruneConfig {
            num_hash_planes: 1,
            l_max: 64,
            final_prune: true,
        };

        assert!(context.with_hash_prune(two_hash_buckets).is_err());
    }
}

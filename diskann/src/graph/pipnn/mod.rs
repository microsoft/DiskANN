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
use diskann_utils::views::{MatrixView, MutMatrixView};
use diskann_vector::distance::Metric;
use diskann_wide::arch::{self, Target2};
use rayon::ThreadPool;

use self::{leaf_metric::LeafMetric, partition_metric::PartitionMetric, simd::Simd};

/// Squared Euclidean distance.
pub(super) struct L2;
/// Cosine distance, `1 - cos(x, y)`, for vectors of any norm.
pub(super) struct Cosine;
/// Cosine distance for unit vectors, computed from the dot product only.
pub(super) struct CosineNormalized;
/// Negated inner product: a larger dot product is nearer.
pub(super) struct InnerProduct;

/// Convert one dot product and two vector norms to cosine distance.
///
/// A norm below `√f32::MIN_POSITIVE` counts as zero and gives distance 1 for any
/// dot value, NaN included. Below this cutoff, the product of two norms can
/// underflow. The similarity is clamped to `[-1, 1]` to remove rounding error.
/// When neither norm is below the cutoff, a NaN dot or norm gives a NaN distance.
#[inline(always)]
fn cosine_distance(dot: f32, source_norm: f32, target_norm: f32) -> f32 {
    if source_norm < f32::MIN_POSITIVE.sqrt() || target_norm < f32::MIN_POSITIVE.sqrt() {
        1.0
    } else {
        1.0 - (dot / (source_norm * target_norm)).clamp(-1.0, 1.0)
    }
}

/// Return an error if a kernel output does not have one row per input point.
///
/// The top-k functions check this shape only in debug builds. The kernel entry
/// points check it first, so a bad output shape is an error in every build.
fn check_output_rows(points: usize, rows: usize) -> ANNResult<()> {
    if rows == points {
        Ok(())
    } else {
        Err(ANNError::message(format!(
            "invalid kernel output row count {rows} for {points} points"
        )))
    }
}

/// Borrow a `rows x columns` prefix of reusable distance storage.
///
/// The storage grows to the largest shape that it serves and never shrinks.
/// A shape whose element count overflows `usize` is an error, not a wrapped size.
fn distance_scratch(
    storage: &mut Vec<f32>,
    rows: usize,
    columns: usize,
) -> ANNResult<MutMatrixView<'_, f32>> {
    let len = rows.checked_mul(columns).ok_or_else(|| {
        ANNError::message(format!(
            "distance matrix size overflows for {rows} x {columns}"
        ))
    })?;
    if storage.len() < len {
        storage.resize(len, 0.0);
    }
    Ok(MutMatrixView::try_from(&mut storage[..len], rows, columns)?)
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
    fn distance_is_one_minus_the_dot_divided_by_both_norms(
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
    fn a_norm_below_the_cutoff_gives_distance_one_for_any_dot(#[case] small_norm: f32) {
        // A NaN dot must not change the result, and either norm can be the small one.
        for dot in [0.75, f32::NAN] {
            for (source_norm, target_norm) in [(small_norm, 1.0), (1.0, small_norm)] {
                assert_eq!(
                    cosine_distance(dot, source_norm, target_norm),
                    1.0,
                    "dot={dot}, norms=({source_norm:e}, {target_norm:e})"
                );
            }
        }
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
    fn nan_propagates_when_neither_norm_is_below_the_cutoff(
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
    A: Simd,
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
    A: Simd,
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
mod build_graph_tests {
    use super::{HashPruneConfig, PiPNNBuildContext, PiPNNConfig, build_graph};
    use crate::graph::config::{self, MaxDegree};
    use diskann_utils::views::MatrixView;
    use diskann_vector::distance::Metric;
    use rstest::rstest;

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

#[cfg(test)]
mod test_support {
    use super::simd::Simd;
    use diskann_vector::distance::Metric;

    // Ignore member order while preserving group positions and duplicate counts.
    // Positions identify leaders or graph sources; sorting the outer list would
    // hide assignments to the wrong group. Copies keep the observations intact.
    pub(super) fn sorted_members_per_row(rows: &[Vec<u32>]) -> Vec<Vec<u32>> {
        rows.iter()
            .map(|row| {
                let mut members = row.clone();
                members.sort_unstable();
                members
            })
            .collect()
    }

    #[test]
    fn membership_comparison_preserves_group_positions_and_duplicate_counts() {
        let rows = [vec![9, 3, 9], vec![], vec![2, 1], vec![2, 1]];

        assert_eq!(
            sorted_members_per_row(&rows),
            [vec![3, 9, 9], vec![], vec![1, 2], vec![1, 2]]
        );
    }

    /// A test body that runs once for each architecture.
    pub(super) trait ArchCheck {
        fn check<A: Simd>(&self, arch: A);
    }

    /// Run `test` with `Scalar` and with each SIMD architecture that this CPU supports.
    ///
    /// The emulated `Scalar` lanes always run. `V3` runs on AVX2 hardware, `V4` runs
    /// on AVX-512 hardware or under Miri, and `Neon` runs on aarch64. Production
    /// selects one of these architectures at run time, so each one needs its own run.
    pub(super) fn for_each_arch(test: &impl ArchCheck) {
        test.check(diskann_wide::arch::Scalar);
        #[cfg(target_arch = "x86_64")]
        {
            use diskann_wide::arch::x86_64::{V3, V4};
            if let Some(arch) = V3::new_checked() {
                test.check(arch);
            }
            if let Some(arch) = V4::new_checked_miri() {
                test.check(arch);
            }
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
            test.check(arch);
        }
    }

    /// Return the largest error between a kernel distance and [`distance`] for
    /// [`dense_points`] inputs.
    ///
    /// L2 and inner product are exact for these inputs. Cosine rounds only in square
    /// roots and division. Normalized cosine also rounds each normalized coordinate,
    /// so its bound grows with the dimension.
    pub(super) fn dense_tolerance(metric: Metric, dimensions: usize) -> f64 {
        match metric {
            Metric::L2 | Metric::InnerProduct => 0.0,
            Metric::Cosine => 16.0 * f64::from(f32::EPSILON),
            Metric::CosineNormalized => {
                // gamma = n*u / (1 - n*u) bounds product and reduction rounding.
                let roundoff = dimensions as f64 * f64::from(f32::EPSILON);
                roundoff / (1.0 - roundoff)
            }
        }
    }

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

    // These scalar definitions match the DiskANN metric distances. They use the
    // actual f32 inputs with f64 arithmetic. Callers supply finite vectors with
    // nonzero norms. L2 includes the point norm.
    pub(super) fn distance(metric: Metric, point: &[f32], target: &[f32]) -> f64 {
        let dot = |x: &[f32], y: &[f32]| {
            x.iter()
                .zip(y)
                .map(|(&x, &y)| f64::from(x) * f64::from(y))
                .sum::<f64>()
        };
        match metric {
            Metric::L2 => point
                .iter()
                .zip(target)
                .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
                .sum(),
            Metric::InnerProduct => -dot(point, target),
            Metric::CosineNormalized => 1.0 - dot(point, target),
            Metric::Cosine => {
                let norm = |row: &[f32]| dot(row, row).sqrt();
                1.0 - (dot(point, target) / (norm(point) * norm(target))).clamp(-1.0, 1.0)
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
    fn dense_fixtures_are_deterministic_and_nonzero_in_the_last_dimension() {
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
    #[case::one_minus_dot(Metric::CosineNormalized, -1.0)]
    #[case::cosine(Metric::Cosine, 0.7830695421813438)]
    fn scalar_reference_matches_hand_calculated_distances(
        #[case] metric: Metric,
        #[case] expected: f64,
    ) {
        // [1, 2] and [4, -1]: squared difference 18, dot 2, squared norms 5 and 17.
        assert!((distance(metric, &[1.0, 2.0], &[4.0, -1.0]) - expected).abs() < 1.0e-14);
    }

    #[test]
    fn normalized_packed_points_put_the_second_coordinate_last() {
        let actual = packed_points(&[[3.0, 4.0], [0.0, -2.0]], 3, true);
        assert_eq!(actual, [0.6, 0.0, 0.8, 0.0, 0.0, -1.0]);
        assert_eq!(
            distance(Metric::CosineNormalized, &actual[..3], &actual[3..]),
            1.0 + f64::from(0.8_f32)
        );
    }
}

#[cfg(all(test, not(miri)))]
mod construction_tests {
    use super::*;
    use crate::graph::config;
    use half::f16;
    use rstest::rstest;

    fn partition_policy() -> PiPNNConfig {
        PiPNNConfig {
            c_max: 4,
            c_min: 1,
            p_samp: 1.0,
            fanout: vec![2],
            leaf_k: 1,
            replicas: 1,
        }
    }

    fn graph_policy(degree: usize, metric: Metric) -> Result<Config, config::ConfigError> {
        config::Builder::new_with(degree, config::MaxDegree::same(), 16, metric.into(), |b| {
            b.alpha(1.0);
        })
        .build()
    }

    #[rstest]
    #[case::zero_maximum(|c: &mut PiPNNConfig| c.c_max = 0, "c_max must be greater than zero")]
    #[case::zero_minimum(|c: &mut PiPNNConfig| c.c_min = 0, "c_min must be greater than zero")]
    #[case::inverted_leaf_bounds(|c: &mut PiPNNConfig| c.c_min = 5, "c_min (5) must not exceed c_max (4)")]
    #[case::zero_probability(|c: &mut PiPNNConfig| c.p_samp = 0.0, "p_samp (0) must be in (0, 1]")]
    #[case::negative_probability(|c: &mut PiPNNConfig| c.p_samp = -0.5, "p_samp (-0.5) must be in (0, 1]")]
    #[case::probability_above_one(|c: &mut PiPNNConfig| c.p_samp = 1.0 + f64::EPSILON, "must be in (0, 1]")]
    #[case::nan_probability(|c: &mut PiPNNConfig| c.p_samp = f64::NAN, "p_samp (NaN) must be in (0, 1]")]
    #[case::positive_infinite_probability(|c: &mut PiPNNConfig| c.p_samp = f64::INFINITY, "p_samp (inf) must be in (0, 1]")]
    #[case::negative_infinite_probability(|c: &mut PiPNNConfig| c.p_samp = f64::NEG_INFINITY, "p_samp (-inf) must be in (0, 1]")]
    #[case::missing_fanout(|c: &mut PiPNNConfig| c.fanout.clear(), "fanout must not be empty")]
    #[case::zero_later_fanout(|c: &mut PiPNNConfig| c.fanout = vec![2, 0, 1], "fanout values must be greater than zero")]
    #[case::zero_leaf_neighbors(|c: &mut PiPNNConfig| c.leaf_k = 0, "leaf_k must be greater than zero")]
    #[case::zero_replicas(|c: &mut PiPNNConfig| c.replicas = 0, "replicas must be greater than zero")]
    fn invalid_configuration_identifies_the_rejected_condition(
        #[case] invalidate: fn(&mut PiPNNConfig),
        #[case] expected_message: &str,
    ) {
        let mut config = partition_policy();
        invalidate(&mut config);

        let error = config.validate().unwrap_err();

        let message = error.to_string();
        assert!(message.contains(expected_message), "{message}");
    }

    #[rstest]
    #[case::smallest_positive_probability(f64::from_bits(1))]
    #[case::all_points_sampled(1.0)]
    fn equal_leaf_bounds_and_valid_probability_endpoints_are_accepted(#[case] p_samp: f64) {
        let config = PiPNNConfig {
            c_min: 4,
            p_samp,
            ..partition_policy()
        };

        config.validate().unwrap();
    }

    #[test]
    fn a_build_context_rejects_an_invalid_partition_policy() {
        let config = PiPNNConfig {
            leaf_k: 0,
            ..partition_policy()
        };
        let graph = graph_policy(2, Metric::L2).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        let error = PiPNNBuildContext::new(config, &graph, Metric::L2, &pool).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("leaf_k must be greater than zero"),
            "{error}"
        );
    }

    #[rstest]
    #[case::cosine_uses_triangle_pruning(Metric::L2, Metric::Cosine)]
    #[case::normalized_cosine_uses_triangle_pruning(Metric::L2, Metric::CosineNormalized)]
    #[case::inner_product_uses_occluding_pruning(Metric::InnerProduct, Metric::InnerProduct)]
    fn a_build_context_accepts_metrics_with_the_same_pruning_kind(
        #[case] graph_metric: Metric,
        #[case] requested_metric: Metric,
    ) {
        let graph = graph_policy(2, graph_metric).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        PiPNNBuildContext::new(partition_policy(), &graph, requested_metric, &pool).unwrap();
    }

    #[rstest]
    #[case::inner_product_with_triangle_pruning(Metric::L2, Metric::InnerProduct)]
    #[case::l2_with_occluding_pruning(Metric::InnerProduct, Metric::L2)]
    fn a_build_context_rejects_incompatible_pruning_kinds(
        #[case] graph_metric: Metric,
        #[case] requested_metric: Metric,
    ) {
        let graph = graph_policy(2, graph_metric).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        let error = PiPNNBuildContext::new(partition_policy(), &graph, requested_metric, &pool)
            .unwrap_err();

        assert!(
            error.to_string().contains("incompatible with metric"),
            "{error}"
        );
    }

    #[rstest]
    #[case::no_points(0, 2, "at least one data point")]
    #[case::no_dimensions(2, 0, "at least one data dimension")]
    fn an_empty_dataset_axis_is_rejected_before_building(
        #[case] rows: usize,
        #[case] columns: usize,
        #[case] expected_message: &str,
    ) {
        let data = MatrixView::try_from(&[] as &[f32], rows, columns).unwrap();
        let graph = graph_policy(2, Metric::L2).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let context =
            PiPNNBuildContext::new(partition_policy(), &graph, Metric::L2, &pool).unwrap();

        let error = build_graph(data, &context).unwrap_err();

        assert!(error.to_string().contains(expected_message), "{error}");
    }

    #[rstest]
    #[case::float32([0.0_f32, 1.0, 4.0])]
    #[case::float16([0.0_f32, 1.0, 4.0].map(f16::from_f32))]
    #[case::signed_integer([0_i8, 1, 4])]
    #[case::unsigned_integer([0_u8, 1, 4])]
    fn native_vector_types_build_the_expected_neighbor_graph<T: VectorRepr>(
        #[case] values: [T; 3],
    ) {
        // Nearest choices 0 -> 1, 1 -> 0 and 2 -> 1 become a symmetric chain.
        let data = MatrixView::try_from(&values[..], 3, 1).unwrap();
        let graph = graph_policy(2, Metric::L2).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let context =
            PiPNNBuildContext::new(partition_policy(), &graph, Metric::L2, &pool).unwrap();

        let actual = build_graph(data, &context).unwrap();
        let actual: Vec<_> = actual.into_iter().map(Vec::from).collect();

        assert_eq!(
            test_support::sorted_members_per_row(&actual),
            [vec![1], vec![0, 2], vec![1]]
        );
    }

    #[rstest]
    #[case::l2(Metric::L2, vec![vec![2], vec![2], vec![0, 1, 3], vec![2]])]
    #[case::cosine(Metric::Cosine, vec![vec![1], vec![0], vec![3], vec![2]])]
    #[case::normalized_cosine(Metric::CosineNormalized, vec![vec![1], vec![0], vec![3], vec![2]])]
    #[case::inner_product(Metric::InnerProduct, vec![vec![1], vec![0, 3], vec![3], vec![1, 2]])]
    fn the_requested_metric_determines_leaf_neighbors(
        #[case] metric: Metric,
        #[case] expected: Vec<Vec<u32>>,
    ) {
        // L2's nearest choices are [2, 2, 0, 2]; cosine pairs similar directions
        // 0 <-> 1 and 2 <-> 3; dot products give [1, 3, 3, 2].
        // The second coordinate sits in the last dimension of an embedding.
        let dimensions = 1537;
        let values = test_support::packed_points(
            &[[1.0, 0.0], [5.0, 2.0], [1.0, 3.0], [0.0, 9.0]],
            dimensions,
            metric == Metric::CosineNormalized,
        );
        let data = MatrixView::try_from(values.as_slice(), 4, dimensions).unwrap();
        let graph = graph_policy(3, metric).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let context = PiPNNBuildContext::new(partition_policy(), &graph, metric, &pool).unwrap();

        let actual = build_graph(data, &context).unwrap();
        let actual: Vec<_> = actual.into_iter().map(Vec::from).collect();

        assert_eq!(
            test_support::sorted_members_per_row(&actual),
            test_support::sorted_members_per_row(&expected)
        );
    }

    #[rstest]
    #[case::l2(Metric::L2, vec![vec![2], vec![], vec![0], vec![]])]
    #[case::cosine(Metric::Cosine, vec![vec![1], vec![0], vec![3], vec![2]])]
    #[case::normalized_cosine(Metric::CosineNormalized, vec![vec![1], vec![0], vec![3], vec![2]])]
    #[case::inner_product(Metric::InnerProduct, vec![vec![1], vec![0], vec![3], vec![2]])]
    fn splitting_uses_the_requested_metric_to_group_points(
        #[case] metric: Metric,
        #[case] expected: Vec<Vec<u32>>,
    ) {
        // Four points force splitting at c_max=2. Every point is sampled, so
        // leader order cannot change memberships. With L2, leader 0 gets {0,2}
        // and leader 2 gets all four points, which then split into singletons.
        // Cosine pairs directions {0,1} and {2,3}. Inner product picks leaders
        // {1,2} for points 0/1 and {2,3} for points 2/3; the oversized leader-2
        // cluster then splits into those same pairs using fanout one.
        let values = test_support::packed_points(
            &[[1.0, 0.0], [6.0, 1.0], [2.0, 4.0], [0.0, 9.0]],
            2,
            metric == Metric::CosineNormalized,
        );
        let data = MatrixView::try_from(values.as_slice(), 4, 2).unwrap();
        let config = PiPNNConfig {
            c_max: 2,
            ..partition_policy()
        };
        // Every two-point leaf contributes its only pair; degree three retains
        // every edge. The graph therefore exposes partition membership directly.
        let graph = graph_policy(3, metric).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let context = PiPNNBuildContext::new(config, &graph, metric, &pool).unwrap();

        let actual = build_graph(data, &context).unwrap();
        let actual: Vec<_> = actual.into_iter().map(Vec::from).collect();

        assert_eq!(
            test_support::sorted_members_per_row(&actual),
            test_support::sorted_members_per_row(&expected)
        );
    }

    #[rstest]
    #[case::signed_leaf_ranking([1_i8, 0, 4, 1, 1, 2, 0, 9], 1, 3)]
    #[case::unsigned_leaf_ranking([1_u8, 0, 4, 1, 1, 2, 0, 9], 1, 3)]
    #[case::signed_final_pruning([1_i8, 0, 4, 1, 1, 2, 0, 9], 3, 1)]
    #[case::unsigned_final_pruning([1_u8, 0, 4, 1, 1, 2, 0, 9], 3, 1)]
    fn normalized_cosine_requests_on_raw_integers_use_vector_norms<T: VectorRepr>(
        #[case] values: [T; 8],
        #[case] leaf_k: usize,
        #[case] degree: usize,
    ) {
        // Cosine pairs 0 <-> 1 and 2 <-> 3. Unnormalized dot products instead
        // connect 1 to 3, so treating raw integers as unit vectors changes edges.
        let data = MatrixView::try_from(&values[..], 4, 2).unwrap();
        let config = PiPNNConfig {
            leaf_k,
            ..partition_policy()
        };
        let graph = graph_policy(degree, Metric::CosineNormalized).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let context =
            PiPNNBuildContext::new(config, &graph, Metric::CosineNormalized, &pool).unwrap();

        let actual = build_graph(data, &context).unwrap();

        assert_eq!(
            actual.into_iter().map(Vec::from).collect::<Vec<_>>(),
            [vec![1], vec![0], vec![3], vec![2]]
        );
    }

    #[test]
    fn final_pruning_enforces_graph_degree_after_leaf_candidates_are_merged() {
        let values = [0.0_f32, 1.0, 4.0];
        let data = MatrixView::try_from(&values[..], 3, 1).unwrap();
        let config = PiPNNConfig {
            leaf_k: 2,
            ..partition_policy()
        };
        let graph = graph_policy(1, Metric::L2).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let context = PiPNNBuildContext::new(config, &graph, Metric::L2, &pool).unwrap();

        let actual = build_graph(data, &context).unwrap();

        assert_eq!(
            actual.into_iter().map(Vec::from).collect::<Vec<_>>(),
            [vec![1], vec![0], vec![1]]
        );
    }

    #[rstest]
    #[case::serial_single_replica(1, 1)]
    #[case::parallel_replicas(3, 2)]
    fn overlapping_partitions_build_the_same_deduplicated_graph(
        #[case] workers: usize,
        #[case] replicas: usize,
    ) {
        // Sampling every point with fanout 2 makes two overlapping copies
        // of each close pair. Neither replicas nor worker count may duplicate edges.
        let values = [0.0_f32, 1.0, 10.0, 11.0];
        let data = MatrixView::try_from(&values[..], 4, 1).unwrap();
        let config = PiPNNConfig {
            c_max: 2,
            replicas,
            ..partition_policy()
        };
        let graph = graph_policy(2, Metric::L2).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        let context = PiPNNBuildContext::new(config, &graph, Metric::L2, &pool).unwrap();

        let actual = build_graph(data, &context).unwrap();

        assert_eq!(
            actual.into_iter().map(Vec::from).collect::<Vec<_>>(),
            [vec![1], vec![0], vec![3], vec![2]]
        );
    }

    #[test]
    fn a_single_point_has_one_empty_adjacency_list() {
        let values = [3.0_f32];
        let data = MatrixView::try_from(&values[..], 1, 1).unwrap();
        let graph = graph_policy(2, Metric::L2).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let context =
            PiPNNBuildContext::new(partition_policy(), &graph, Metric::L2, &pool).unwrap();

        let actual = build_graph(data, &context).unwrap();

        assert_eq!(
            actual.into_iter().map(Vec::from).collect::<Vec<_>>(),
            [Vec::<u32>::new()]
        );
    }

    #[test]
    fn stalled_partitioning_remains_a_typed_build_error() {
        let values = [1.0_f32; 5];
        let data = MatrixView::try_from(&values[..], 5, 1).unwrap();
        let config = PiPNNConfig {
            c_max: 2,
            fanout: vec![1],
            ..partition_policy()
        };
        let graph = graph_policy(2, Metric::L2).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let context = PiPNNBuildContext::new(config, &graph, Metric::L2, &pool).unwrap();

        let error = build_graph(data, &context).unwrap_err();

        assert!(matches!(
            error.downcast_ref::<partitioning::PartitionError>(),
            Some(partitioning::PartitionError::IterationLimit { size: 5, .. })
        ));
    }
}

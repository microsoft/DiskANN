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
//! 1. `partitioning` samples leaders and makes overlapping leaves. Each leaf has
//!    at most `c_max` points.
//! 2. `leaf_build` computes a lower-triangular Gram matrix for each leaf. It
//!    selects local neighbors. The direct path merges their global point IDs.
//!    The HashPrune path sends weighted edges to bounded point reservoirs.
//! 3. `finalization` applies Vamana RobustPrune to direct candidates. It also
//!    prunes HashPrune candidates when `final_prune` is true.
//!
//! `diskann-wide` selects architecture `A`. One match selects metric marker `M`.
//! The build passes both concrete types through all replicas, recursive
//! partitions, stripes, and leaves. The numerical loops do not dispatch again.
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

mod kernel_metric;
mod simd;

mod bf16;
mod finalization;
mod hash_prune;
mod leaf_build;
mod leaf_kernel;
mod lsh;
mod partition_kernel;
mod partitioning;
mod rabitq1;

use crate::{
    ANNError, ANNResult,
    graph::{AdjacencyList, Config},
    utils::VectorRepr,
};
use diskann_utils::views::MatrixView;
use diskann_vector::distance::Metric;
use diskann_wide::arch::{self, Target1};
use rayon::ThreadPool;

use self::{
    kernel_metric::{Cosine, CosineNormalized, InnerProduct, L2, LeafMetric, PartitionMetric},
    simd::PiPNNSIMDSchema,
};

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
    pub(crate) config: PiPNNConfig,
    pub(crate) graph: &'a Config,
    pub(crate) metric: Metric,
    pub(crate) pool: &'a ThreadPool,
    hash_prune: Option<HashPruneConfig>,
    rabitq1_seed: Option<u64>,
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
            rabitq1_seed: None,
        })
    }

    /// Use build-only spherical RaBitQ1 distances.
    pub fn with_rabitq1(mut self, seed: u64) -> ANNResult<Self> {
        self.rabitq1_seed = Some(seed);
        Ok(self)
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
    T: VectorRepr + Send + Sync + 'static,
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
    T: VectorRepr + Send + Sync + 'static,
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
    // Conversion does not make integer vectors unit length. Use the norm-aware
    // cosine formula for these vectors.
    let metric = effective_metric::<T>(context.metric);
    arch::dispatch1_no_features(
        RunBuildGraph,
        BuildGraphCall {
            data,
            context,
            metric,
        },
    )
}

struct BuildGraphCall<'data, 'context, 'policy, T> {
    data: MatrixView<'data, T>,
    context: &'context PiPNNBuildContext<'policy>,
    metric: Metric,
}

struct RunBuildGraph;

impl<A, T> Target1<A, ANNResult<Vec<AdjacencyList<u32>>>, BuildGraphCall<'_, '_, '_, T>>
    for RunBuildGraph
where
    A: PiPNNSIMDSchema,
    T: VectorRepr + Send + Sync + 'static,
{
    fn run(
        self,
        arch: A,
        call: BuildGraphCall<'_, '_, '_, T>,
    ) -> ANNResult<Vec<AdjacencyList<u32>>> {
        match call.metric {
            Metric::L2 => build_graph_for::<A, L2, T>(arch, call.data, call.context, Metric::L2),
            Metric::Cosine => {
                build_graph_for::<A, Cosine, T>(arch, call.data, call.context, Metric::Cosine)
            }
            Metric::CosineNormalized => build_graph_for::<A, CosineNormalized, T>(
                arch,
                call.data,
                call.context,
                Metric::CosineNormalized,
            ),
            Metric::InnerProduct => build_graph_for::<A, InnerProduct, T>(
                arch,
                call.data,
                call.context,
                Metric::InnerProduct,
            ),
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
    T: VectorRepr + Send + Sync + 'static,
{
    let rabitq1 = context
        .rabitq1_seed
        .map(|seed| rabitq1::Store::train(data, metric, seed).map_err(ANNError::new))
        .transpose()?;
    let scorer = rabitq1.as_ref();
    let leaves = tracing::info_span!("pipnn.partition")
        .in_scope(|| partitioning::partition::<A, M, T>(arch, data, &context.config, scorer))?;
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
                    scorer,
                )
                .map_err(ANNError::new)
            })?;
            tracing::info_span!("pipnn.finalization").in_scope(|| match scorer {
                Some(store) => {
                    finalization::prune_overfull_rabitq1(arch, store, candidates, context.graph)
                }
                None => finalization::prune_overfull(data, candidates, context.graph, metric),
            })
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
                    scorer,
                )
                .map_err(ANNError::new)
            })?;
            if config.final_prune {
                let candidates = hash_prune.into_candidate_lists();
                tracing::info_span!("pipnn.finalization").in_scope(|| match scorer {
                    Some(store) => {
                        finalization::prune_overfull_rabitq1(arch, store, candidates, context.graph)
                    }
                    None => finalization::prune_overfull(data, candidates, context.graph, metric),
                })
            } else {
                Ok(hash_prune.into_nearest_lists(context.graph.pruned_degree().get()))
            }
        }
    }
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
    ANNError::message(format!("PiPNN configuration: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use rstest::rstest;

    #[test]
    fn integer_vectors_use_cosine_when_normalized_cosine_is_requested() {
        assert_eq!(
            effective_metric::<u8>(Metric::CosineNormalized),
            Metric::Cosine
        );
        assert_eq!(
            effective_metric::<i8>(Metric::CosineNormalized),
            Metric::Cosine
        );
    }

    #[rstest]
    fn metric_selection_is_unchanged_for_float_vectors(
        #[values(
            Metric::L2,
            Metric::Cosine,
            Metric::CosineNormalized,
            Metric::InnerProduct
        )]
        metric: Metric,
    ) {
        assert_eq!(effective_metric::<f32>(metric), metric);
        assert_eq!(effective_metric::<f16>(metric), metric);
    }

    #[rstest]
    fn integer_vectors_keep_non_normalized_metric_selection(
        #[values(Metric::L2, Metric::Cosine, Metric::InnerProduct)] metric: Metric,
    ) {
        assert_eq!(effective_metric::<u8>(metric), metric);
        assert_eq!(effective_metric::<i8>(metric), metric);
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
    use half::f16;
    use rand::{Rng, SeedableRng, rngs::StdRng};
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

    fn thread_pool(threads: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
    }

    fn adjacency_rows(graph: Vec<crate::graph::AdjacencyList<u32>>) -> Vec<Vec<u32>> {
        graph.into_iter().map(Vec::from).collect()
    }

    fn deterministic_point_values(points: usize, dimensions: usize) -> Vec<f32> {
        (0..points)
            .flat_map(|point| {
                (0..dimensions)
                    .map(move |dimension| point as f32 + dimension as f32 / dimensions as f32)
            })
            .collect()
    }

    fn assert_graph_invariants(
        graph: &[crate::graph::AdjacencyList<u32>],
        points: usize,
        degree: usize,
    ) {
        assert_eq!(graph.len(), points);
        for (source, row) in graph.iter().enumerate() {
            assert!(row.len() <= degree);
            let mut sorted = row.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), row.len());
            assert!(
                row.iter()
                    .all(|&id| (id as usize) < points && id as usize != source)
            );
        }
    }

    #[test]
    fn single_leaf_build_maps_local_neighbors_to_dataset_ids() {
        // Given
        let point_values = [0.0_f32, 1.0, 2.0, 3.0];
        let data = MatrixView::try_from(&point_values[..], 4, 1).unwrap();
        let graph = graph_config(Metric::L2, 2);
        let pool = thread_pool(2);
        let context = PiPNNBuildContext::new(pipnn_config(), &graph, Metric::L2, &pool).unwrap();
        let expected_adjacency = [vec![1], vec![0, 2], vec![1, 3], vec![2]];

        // When
        let actual_adjacency = adjacency_rows(build_graph(data, &context).unwrap());

        // Then
        assert_eq!(actual_adjacency, expected_adjacency);
    }

    #[test]
    fn degree_one_pruning_keeps_adjacent_neighbors_on_a_line() {
        // Given
        let point_values = [0.0_f32, 1.0, 2.0, 3.0];
        let data = MatrixView::try_from(&point_values[..], 4, 1).unwrap();
        let graph = graph_config(Metric::L2, 1);
        let pool = thread_pool(2);
        let context = PiPNNBuildContext::new(pipnn_config(), &graph, Metric::L2, &pool).unwrap();
        let expected_adjacency = [vec![1], vec![2], vec![3], vec![2]];

        // When
        let actual_adjacency = adjacency_rows(build_graph(data, &context).unwrap());

        // Then
        assert_eq!(actual_adjacency, expected_adjacency);
    }

    #[test]
    fn non_rankable_points_leave_empty_adjacency_without_sentinel_ids() {
        let values = [0.0_f32, 1.0, f32::NAN];
        let data = MatrixView::try_from(&values[..], 3, 1).unwrap();
        let graph = graph_config(Metric::InnerProduct, 2);
        let pool = thread_pool(1);
        let config = PiPNNConfig {
            c_max: 2,
            c_min: 1,
            p_samp: 1.0,
            fanout: vec![2],
            leaf_k: 1,
            replicas: 1,
        };
        let context = PiPNNBuildContext::new(config, &graph, Metric::InnerProduct, &pool).unwrap();
        let expected_rankable_adjacency = [vec![1], vec![0], vec![]];

        let actual_graph = build_graph(data, &context).unwrap();

        assert_graph_invariants(&actual_graph, 3, 2);
        assert_eq!(adjacency_rows(actual_graph), expected_rankable_adjacency);
    }

    #[test]
    fn single_leaf_adjacency_is_bounded_by_the_graph_degree() {
        let data = [0.0_f32, 1.0, 2.0, 3.0, 4.0];
        let data = MatrixView::try_from(&data[..], 5, 1).unwrap();
        let graph = graph_config(Metric::L2, 1);
        let pool = thread_pool(2);
        let config = PiPNNConfig {
            c_max: 5,
            c_min: 1,
            p_samp: 0.5,
            fanout: vec![2],
            leaf_k: 4,
            replicas: 1,
        };
        let context = PiPNNBuildContext::new(config, &graph, Metric::L2, &pool).unwrap();

        let actual_graph = build_graph(data, &context).unwrap();

        assert_graph_invariants(&actual_graph, 5, 1);
        assert!(actual_graph.iter().all(|row| row.len() == 1));
    }

    #[rstest]
    #[case::zero_points(0, 4)]
    #[case::zero_dimensions(4, 0)]
    fn empty_dataset_dimension_is_rejected(#[case] point_count: usize, #[case] dimensions: usize) {
        // Given
        let graph = graph_config(Metric::L2, 2);
        let pool = thread_pool(1);
        let context = PiPNNBuildContext::new(pipnn_config(), &graph, Metric::L2, &pool).unwrap();
        let empty_data = MatrixView::try_from(&[] as &[f32], point_count, dimensions).unwrap();

        // When
        let result = build_graph(empty_data, &context);

        // Then
        assert!(result.is_err());
    }

    fn assert_graph_build_succeeds<T: crate::utils::VectorRepr + Send + Sync + 'static>(
        values: &[T],
        metric: Metric,
    ) {
        let data = MatrixView::try_from(values, 6, 2).unwrap();
        let graph = graph_config(metric, 2);
        let pool = thread_pool(2);
        let context = PiPNNBuildContext::new(pipnn_config(), &graph, metric, &pool).unwrap();
        let actual_graph = build_graph(data, &context).unwrap();
        assert_graph_invariants(&actual_graph, 6, 2);
    }

    #[rstest]
    fn f32_graph_build_succeeds_with_each_metric(
        #[values(
            Metric::L2,
            Metric::Cosine,
            Metric::CosineNormalized,
            Metric::InnerProduct
        )]
        metric: Metric,
    ) {
        let diagonal = std::f32::consts::FRAC_1_SQRT_2;
        let unit_vectors = [
            1.0_f32, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, diagonal, diagonal, -diagonal, -diagonal,
        ];

        assert_graph_build_succeeds(&unit_vectors, metric);
    }

    #[test]
    fn f16_graph_build_succeeds_with_l2() {
        let values = [
            1.0_f32, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0, 1.0, 1.0, 2.0, 2.0,
        ];
        assert_graph_build_succeeds(&values.map(f16::from_f32), Metric::L2);
    }

    #[test]
    fn u8_graph_build_succeeds_with_l2() {
        let values = [1_u8, 0, 0, 1, 2, 0, 0, 2, 1, 1, 2, 2];
        assert_graph_build_succeeds(&values, Metric::L2);
    }

    #[test]
    fn i8_graph_build_succeeds_with_l2() {
        let values = [1_i8, 0, 0, 1, -1, 0, 0, -1, 1, 1, -1, -1];
        assert_graph_build_succeeds(&values, Metric::L2);
    }

    #[test]
    fn integer_vector_graphs_match_cosine_when_normalized_cosine_is_requested() {
        fn assert_integer_graphs_match_cosine<
            T: crate::utils::VectorRepr + Send + Sync + 'static,
        >(
            values: &[T],
        ) {
            let data = MatrixView::try_from(values, 8, 2).unwrap();
            let pool = thread_pool(2);
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
                adjacency_rows(build_graph(data, &context).unwrap())
            };
            assert_eq!(build(Metric::CosineNormalized), build(Metric::Cosine));
        }

        assert_integer_graphs_match_cosine(&[1_u8, 0, 2, 0, 0, 1, 0, 2, 1, 1, 2, 1, 1, 2, 2, 2]);
        assert_integer_graphs_match_cosine(&[
            1_i8, 0, -1, 0, 0, 1, 0, -1, 1, 1, -1, -1, 1, -1, -1, 1,
        ]);
    }

    #[test]
    fn graph_build_is_deterministic_for_a_fixed_pool_size() {
        let data = deterministic_point_values(96, 4);
        let data = MatrixView::try_from(&data[..], 96, 4).unwrap();
        let graph = graph_config(Metric::L2, 8);
        let pool = thread_pool(4);
        let config = PiPNNConfig {
            c_max: 16,
            c_min: 4,
            p_samp: 0.25,
            fanout: vec![3, 2],
            leaf_k: 3,
            replicas: 2,
        };
        let context = PiPNNBuildContext::new(config, &graph, Metric::L2, &pool).unwrap();

        let first = build_graph(data, &context).unwrap();
        let second = build_graph(data, &context).unwrap();

        assert_eq!(first, second);
        assert_graph_invariants(&first, 96, 8);
    }

    #[test]
    fn graph_build_preserves_invariants_across_fixed_seed_inputs() {
        let mut rng = StdRng::seed_from_u64(0x857a_d38b_44c2_0f11);
        for case in 0..24 {
            let points = rng.random_range(4..=32);
            let dimensions = rng.random_range(1..=8);
            let c_max = rng.random_range(4..=points.min(12));
            let c_min = rng.random_range(1..=c_max);
            let degree = rng.random_range(1..=points.min(8));
            let values: Vec<f32> = (0..points * dimensions)
                .map(|_| rng.random_range(-10.0..10.0))
                .collect();
            let data = MatrixView::try_from(&values[..], points, dimensions).unwrap();
            let graph = graph_config(Metric::L2, degree);
            let pool = thread_pool(2);
            let config = PiPNNConfig {
                c_max,
                c_min,
                p_samp: 0.5,
                fanout: vec![2],
                leaf_k: rng.random_range(1..=7),
                replicas: rng.random_range(1..=2),
            };
            let context = PiPNNBuildContext::new(config, &graph, Metric::L2, &pool).unwrap();

            let actual_graph = build_graph(data, &context)
                .unwrap_or_else(|error| panic!("randomized case {case} failed: {error}"));
            assert_graph_invariants(&actual_graph, points, degree);
        }
    }

    #[test]
    fn parallel_hash_prune_build_is_set_invariant() {
        let points = 64;
        let dimensions = 4;
        let values = deterministic_point_values(points, dimensions);
        let data = MatrixView::try_from(values.as_slice(), points, dimensions).unwrap();
        let graph = graph_config(Metric::L2, 8);
        let pool = thread_pool(4);
        let config = PiPNNConfig {
            c_max: 16,
            c_min: 4,
            p_samp: 0.25,
            fanout: vec![3, 2],
            leaf_k: 3,
            replicas: 2,
        };
        let hash_prune = HashPruneConfig {
            num_hash_planes: 8,
            l_max: 16,
            final_prune: true,
        };
        let build = || {
            let context = PiPNNBuildContext::new(config.clone(), &graph, Metric::L2, &pool)
                .unwrap()
                .with_hash_prune(hash_prune.clone())
                .unwrap();
            build_graph(data, &context).unwrap()
        };

        let first = build();
        let second = build();
        let canonicalize = |graph: &[crate::graph::AdjacencyList<u32>]| {
            graph
                .iter()
                .map(|row| {
                    let mut ids = row.to_vec();
                    ids.sort_unstable();
                    ids
                })
                .collect::<Vec<_>>()
        };

        // Parallel finalization can order equal candidates differently. Compare
        // the retained neighbor sets.
        assert_eq!(canonicalize(&first), canonicalize(&second));
        assert_graph_invariants(&first, points, 8);
        assert!(first.iter().any(|row| !row.is_empty()));
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
        graph_config_with_degree(metric, alpha, 64)
    }

    fn graph_config_with_degree(metric: Metric, alpha: f32, degree: usize) -> crate::graph::Config {
        config::Builder::new_with(degree, MaxDegree::same(), 72, metric.into(), |builder| {
            builder.alpha(alpha);
        })
        .build()
        .unwrap()
    }

    fn two_thread_pool() -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
    }

    #[rstest]
    #[case::zero_c_max(PiPNNConfig { c_max: 0, ..pipnn_config() })]
    #[case::zero_c_min(PiPNNConfig { c_min: 0, ..pipnn_config() })]
    #[case::c_min_above_c_max(PiPNNConfig { c_min: 513, ..pipnn_config() })]
    #[case::zero_sampling_probability(PiPNNConfig { p_samp: 0.0, ..pipnn_config() })]
    #[case::negative_sampling_probability(PiPNNConfig { p_samp: -0.01, ..pipnn_config() })]
    #[case::sampling_probability_above_one(PiPNNConfig { p_samp: 1.01, ..pipnn_config() })]
    #[case::nan_sampling_probability(PiPNNConfig { p_samp: f64::NAN, ..pipnn_config() })]
    #[case::empty_fanout(PiPNNConfig { fanout: Vec::new(), ..pipnn_config() })]
    #[case::zero_later_fanout(PiPNNConfig { fanout: vec![1, 0], ..pipnn_config() })]
    #[case::zero_leaf_k(PiPNNConfig { leaf_k: 0, ..pipnn_config() })]
    #[case::zero_replicas(PiPNNConfig { replicas: 0, ..pipnn_config() })]
    fn invalid_algorithm_parameter_is_rejected(#[case] invalid_config: PiPNNConfig) {
        let graph = graph_config(Metric::L2, 1.2);
        let pool = two_thread_pool();

        PiPNNBuildContext::new(invalid_config, &graph, Metric::L2, &pool)
            .expect_err("invalid PiPNN config must be rejected");
    }

    #[test]
    fn graph_policy_for_a_different_metric_is_rejected() {
        let graph = graph_config(Metric::InnerProduct, 1.2);
        let pool = two_thread_pool();

        let error = PiPNNBuildContext::new(pipnn_config(), &graph, Metric::L2, &pool).unwrap_err();

        assert!(error.to_string().contains("prune kind"));
    }

    #[rstest]
    #[case::below_one(0.9)]
    #[case::nan(f32::NAN)]
    #[case::infinity(f32::INFINITY)]
    fn build_context_accepts_alpha_allowed_by_graph_config(#[case] alpha: f32) {
        let pool = two_thread_pool();
        let graph = graph_config(Metric::L2, alpha);

        PiPNNBuildContext::new(pipnn_config(), &graph, Metric::L2, &pool).unwrap();
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
        let pool = two_thread_pool();
        let context = PiPNNBuildContext::new(pipnn_config(), &graph, Metric::L2, &pool).unwrap();

        assert!(context.with_hash_prune(invalid_config).is_err());
    }

    #[test]
    fn candidate_capacity_below_graph_degree_is_rejected() {
        let graph = graph_config(Metric::L2, 1.2);
        let pool = two_thread_pool();
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
        let pool = two_thread_pool();
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
        let pool = two_thread_pool();
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
        let pool = two_thread_pool();
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

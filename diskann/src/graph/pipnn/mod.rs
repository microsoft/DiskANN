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
//!    selects local neighbors and merges their global point IDs.
//! 3. `finalization` applies Vamana RobustPrune to each candidate list that is
//!    longer than the graph degree.
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

mod finalization;
mod leaf_build;
mod leaf_kernel;
mod partition_kernel;
mod partitioning;

use crate::{
    ANNError, ANNResult,
    graph::{AdjacencyList, Config},
    utils::VectorRepr,
};
use diskann_utils::views::MatrixView;
use diskann_vector::distance::Metric;
use diskann_wide::{
    Architecture, SIMDMask, SIMDSelect, SIMDVector,
    arch::{self, Target1},
};
use rayon::ThreadPool;

use self::kernel_metric::{Cosine, CosineNormalized, InnerProduct, KernelMetric, L2};

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
    /// Number of nearest neighbors selected within each leaf (`1..=3`).
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
        if !(1..=leaf_kernel::MAX_LEAF_NEIGHBORS).contains(&self.leaf_k) {
            return Err(config_error(format!(
                "leaf_k ({}) must be in [1, {}]",
                self.leaf_k,
                leaf_kernel::MAX_LEAF_NEIGHBORS
            )));
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
    pub(crate) config: PiPNNConfig,
    pub(crate) graph: &'a Config,
    pub(crate) metric: Metric,
    pub(crate) pool: &'a ThreadPool,
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
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
    T: VectorRepr + Send + Sync + 'static,
{
    fn run(
        self,
        arch: A,
        call: BuildGraphCall<'_, '_, '_, T>,
    ) -> ANNResult<Vec<AdjacencyList<u32>>> {
        match call.metric {
            Metric::L2 => build_graph_for::<A, L2, T>(arch, call.data, call.context),
            Metric::Cosine => build_graph_for::<A, Cosine, T>(arch, call.data, call.context),
            Metric::CosineNormalized => {
                build_graph_for::<A, CosineNormalized, T>(arch, call.data, call.context)
            }
            Metric::InnerProduct => {
                build_graph_for::<A, InnerProduct, T>(arch, call.data, call.context)
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
) -> ANNResult<Vec<AdjacencyList<u32>>>
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
    M: KernelMetric,
    T: VectorRepr + Send + Sync + 'static,
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
        .in_scope(|| finalization::prune_overfull(data, candidates, context.graph, M::METRIC))
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

    #[test]
    fn integer_normalized_cosine_uses_unnormalized_cosine() {
        for metric in [
            Metric::L2,
            Metric::Cosine,
            Metric::CosineNormalized,
            Metric::InnerProduct,
        ] {
            let expected = if metric == Metric::CosineNormalized {
                Metric::Cosine
            } else {
                metric
            };
            assert_eq!(effective_metric::<u8>(metric), expected);
            assert_eq!(effective_metric::<i8>(metric), expected);
            assert_eq!(effective_metric::<f32>(metric), metric);
            assert_eq!(effective_metric::<f16>(metric), metric);
        }
    }
}
#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "deterministic test fixture construction must abort on invalid setup"
)]
mod build_graph_tests {
    use super::{PiPNNBuildContext, PiPNNConfig, build_graph, leaf_kernel};
    use crate::graph::config::{self, MaxDegree};
    use diskann_utils::views::MatrixView;
    use diskann_vector::distance::Metric;
    use half::f16;
    use rand::{Rng, SeedableRng, rngs::StdRng};

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
    fn builds_a_single_leaf_graph_for_real_dataset_ids() {
        let data = [0.0_f32, 1.0, 2.0, 3.0];
        let data = MatrixView::try_from(&data[..], 4, 1).unwrap();
        let graph = graph_config(Metric::L2, 2);
        let pool = pool(2);
        let context = PiPNNBuildContext::new(pipnn_config(), &graph, Metric::L2, &pool).unwrap();

        let actual = build_graph(data, &context).unwrap();

        assert_eq!(rows(actual), [vec![1], vec![0, 2], vec![1, 3], vec![2]]);

        let graph = graph_config(Metric::L2, 1);
        let context = PiPNNBuildContext::new(pipnn_config(), &graph, Metric::L2, &pool).unwrap();

        let pruned = build_graph(data, &context).unwrap();

        assert_graph_invariants(&pruned, 4, 1);
        for (source, neighbors) in pruned.iter().enumerate() {
            assert_eq!(source.abs_diff(neighbors[0] as usize), 1);
        }
    }

    #[test]
    fn prunes_overfull_single_leaf_candidates_to_the_graph_degree() {
        let data = [0.0_f32, 1.0, 2.0, 3.0, 4.0];
        let data = MatrixView::try_from(&data[..], 5, 1).unwrap();
        let graph = graph_config(Metric::L2, 1);
        let pool = pool(2);
        let config = PiPNNConfig {
            c_max: 5,
            c_min: 1,
            p_samp: 0.5,
            fanout: vec![2],
            leaf_k: leaf_kernel::MAX_LEAF_NEIGHBORS,
            replicas: 1,
        };
        let context = PiPNNBuildContext::new(config, &graph, Metric::L2, &pool).unwrap();

        let actual = build_graph(data, &context).unwrap();

        assert_graph_invariants(&actual, 5, 1);
        assert!(actual.iter().all(|row| row.len() == 1));
    }

    #[test]
    fn rejects_empty_dataset_dimensions_at_the_public_boundary() {
        let graph = graph_config(Metric::L2, 2);
        let pool = pool(1);
        let context = PiPNNBuildContext::new(pipnn_config(), &graph, Metric::L2, &pool).unwrap();

        let no_rows = MatrixView::try_from(&[] as &[f32], 0, 4).unwrap();
        let no_columns = MatrixView::try_from(&[] as &[f32], 4, 0).unwrap();

        assert!(build_graph(no_rows, &context).is_err());
        assert!(build_graph(no_columns, &context).is_err());
    }

    #[test]
    fn supports_every_source_type_and_metric() {
        fn build<T: crate::utils::VectorRepr + Send + Sync + 'static>(
            values: &[T],
            metric: Metric,
        ) {
            let data = MatrixView::try_from(values, 6, 2).unwrap();
            let graph = graph_config(metric, 2);
            let pool = pool(2);
            let context = PiPNNBuildContext::new(pipnn_config(), &graph, metric, &pool).unwrap();
            let actual = build_graph(data, &context).unwrap();
            assert_graph_invariants(&actual, 6, 2);
        }

        let values = [
            1.0_f32, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, 0.5, 0.5, -0.5, -0.5,
        ];
        for metric in [
            Metric::L2,
            Metric::Cosine,
            Metric::CosineNormalized,
            Metric::InnerProduct,
        ] {
            build(&values, metric);
        }
        build(&values.map(f16::from_f32), Metric::L2);
        build(&[1_u8, 0, 0, 1, 2, 0, 0, 2, 1, 1, 2, 2], Metric::L2);
        build(&[1_i8, 0, 0, 1, -1, 0, 0, -1, 1, 1, -1, -1], Metric::L2);
    }

    #[test]
    fn integer_normalized_cosine_matches_cosine() {
        fn assert_match<T: crate::utils::VectorRepr + Send + Sync + 'static>(values: &[T]) {
            let data = MatrixView::try_from(values, 8, 2).unwrap();
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
            assert_eq!(build(Metric::CosineNormalized), build(Metric::Cosine));
        }

        assert_match(&[1_u8, 0, 100, 1, 2, 0, 0, 1, 1, 1, 200, 2, 2, 1, 1, 2]);
        assert_match(&[1_i8, 0, 100, 1, 2, 0, 0, 1, 1, 1, 120, 2, 2, 1, 1, 2]);
    }

    #[test]
    fn is_deterministic_for_a_fixed_pool_size() {
        let data: Vec<f32> = (0..96 * 4)
            .map(|value| ((value * 17 + 3) % 101) as f32)
            .collect();
        let data = MatrixView::try_from(&data[..], 96, 4).unwrap();
        let graph = graph_config(Metric::L2, 8);
        let pool = pool(4);
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
    fn fixed_seed_randomized_sweeps_preserve_graph_invariants() {
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
            let pool = pool(2);
            let config = PiPNNConfig {
                c_max,
                c_min,
                p_samp: 0.5,
                fanout: vec![2],
                leaf_k: rng.random_range(1..=3),
                replicas: rng.random_range(1..=2),
            };
            let context = PiPNNBuildContext::new(config, &graph, Metric::L2, &pool).unwrap();

            let actual = build_graph(data, &context)
                .unwrap_or_else(|error| panic!("randomized case {case} failed: {error}"));
            assert_graph_invariants(&actual, points, degree);
        }
    }
}
#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "deterministic test fixture construction must abort on invalid setup"
)]
mod config_tests {
    use super::{PiPNNBuildContext, PiPNNConfig, leaf_kernel};
    use crate::graph::config::{self, MaxDegree};
    use diskann_vector::distance::Metric;

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

    #[test]
    fn rejects_each_invalid_algorithm_parameter() {
        let graph = graph_config(Metric::L2, 1.2);
        let pool = pool();
        let mut cases = [
            PiPNNConfig {
                c_max: 0,
                ..pipnn_config()
            },
            PiPNNConfig {
                c_min: 0,
                ..pipnn_config()
            },
            PiPNNConfig {
                c_min: 513,
                ..pipnn_config()
            },
            PiPNNConfig {
                p_samp: 0.0,
                ..pipnn_config()
            },
            PiPNNConfig {
                p_samp: -0.01,
                ..pipnn_config()
            },
            PiPNNConfig {
                p_samp: 1.01,
                ..pipnn_config()
            },
            PiPNNConfig {
                p_samp: f64::NAN,
                ..pipnn_config()
            },
            PiPNNConfig {
                fanout: Vec::new(),
                ..pipnn_config()
            },
            PiPNNConfig {
                fanout: vec![1, 0],
                ..pipnn_config()
            },
            PiPNNConfig {
                leaf_k: 0,
                ..pipnn_config()
            },
            PiPNNConfig {
                leaf_k: leaf_kernel::MAX_LEAF_NEIGHBORS + 1,
                ..pipnn_config()
            },
            PiPNNConfig {
                replicas: 0,
                ..pipnn_config()
            },
        ];

        for config in &mut cases {
            PiPNNBuildContext::new(config.clone(), &graph, Metric::L2, &pool)
                .expect_err("invalid PiPNN config must be rejected");
        }
    }

    #[test]
    fn rejects_graph_policy_for_a_different_metric() {
        let graph = graph_config(Metric::InnerProduct, 1.2);
        let pool = pool();

        let error = PiPNNBuildContext::new(pipnn_config(), &graph, Metric::L2, &pool).unwrap_err();

        assert!(error.to_string().contains("prune kind"));
    }

    #[test]
    fn does_not_add_alpha_validation_beyond_graph_config() {
        let pool = pool();
        for alpha in [0.9, f32::NAN, f32::INFINITY] {
            let graph = graph_config(Metric::L2, alpha);
            PiPNNBuildContext::new(pipnn_config(), &graph, Metric::L2, &pool).unwrap();
        }
    }
}

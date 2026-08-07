/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Provider-independent [PiPNN](https://arxiv.org/html/2602.21247v1) graph construction.
//!
//! PiPNN means **Pick-in-Partitions Nearest Neighbors**. It builds a graph for
//! approximate nearest-neighbor search: every input vector becomes one graph
//! vertex, and its adjacency list stores other vectors worth visiting during a
//! later query. This module constructs that adjacency; it does not execute queries.
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
//! - `kernel_metric` owns metric formulas, scale units, numerical edge cases, and
//!   one-time runtime-to-concrete metric selection shared by both kernels.
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
    /// Number of nearest neighbors selected within each leaf (`1..=3`).
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
        if !(1..=leaf_kernel::MAX_LEAF_NEIGHBORS).contains(&self.k) {
            return Err(config_error(format!(
                "k ({}) must be in [1, {}]",
                self.k,
                leaf_kernel::MAX_LEAF_NEIGHBORS
            )));
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
    data.nrows().checked_mul(data.ncols()).ok_or_else(|| {
        ANNError::message(format!(
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
            .map_err(ANNError::new)
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
            k: 1,
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
            k: leaf_kernel::MAX_LEAF_NEIGHBORS,
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
                    k: 1,
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
            k: 3,
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
                k: rng.random_range(1..=3),
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
            k: 2,
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
                fanout: vec![17],
                ..pipnn_config()
            },
            PiPNNConfig {
                k: 0,
                ..pipnn_config()
            },
            PiPNNConfig {
                k: leaf_kernel::MAX_LEAF_NEIGHBORS + 1,
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

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Provider-independent [PiPNN](https://arxiv.org/html/2602.21247v1) graph construction.
//!
//! PiPNN builds a graph without graph search, in three steps. A leaf is a small
//! cluster of at most `c_max` points; the build selects neighbors only inside
//! leaves.
//!
//! 1. `partitioning` splits the dataset into overlapping leaves.
//! 2. `leaf_build` selects the `leaf_k` nearest neighbors of each point inside
//!    each leaf. The direct merge adds them to one candidate list per point.
//!    With [`HashPruneConfig`], `hash_prune` adds them to one bounded reservoir
//!    per point instead. A reservoir keeps at most `l_max` candidates, at most
//!    one for each direction hash.
//! 3. `finalization` prunes each list that is longer than the graph degree with
//!    Vamana RobustPrune. Without `final_prune`, a HashPrune build keeps the
//!    nearest reservoir entries instead.
//!
//! [`build_graph`] selects the SIMD architecture and the metric once, so the
//! steps compile for one architecture and one metric. It borrows one contiguous
//! [`MatrixView`] and returns one adjacency list for each point. Each step takes
//! ownership of the previous output and frees it when it returns.
//!
//! The build does not load providers or select start points. It also does not
//! quantize, serialize, or search the graph.

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

use std::num::NonZeroUsize;

use crate::{
    ANNError, ANNResult,
    graph::{AdjacencyList, Config, config::PruneKind},
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

/// Error for invalid PiPNN parameters.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum PiPNNConfigError {
    #[error("p_samp ({0}) must be in (0, 1]")]
    SamplingFraction(f64),
    #[error("fanout must not be empty")]
    EmptyFanout,
    #[error("graph prune kind {prune_kind:?} does not match metric {metric:?}")]
    PruneKind {
        prune_kind: PruneKind,
        metric: Metric,
    },
    #[error("num_hash_planes ({0}) must be in [1, {max}]", max = lsh::MAX_PLANES)]
    HashPlanes(usize),
    #[error("l_max ({0}) must be in [1, {max}]", max = hash_prune::MAX_RESERVOIR_LEN)]
    ReservoirLength(usize),
    #[error(
        "HashPrune capacity min(l_max={l_max}, hash buckets={hash_buckets}) must be at least \
         the graph degree ({degree})"
    )]
    HashPruneCapacity {
        l_max: usize,
        hash_buckets: usize,
        degree: usize,
    },
}

crate::convert_error!(PiPNNConfigError);

/// PiPNN partition and leaf parameters.
///
/// The DiskANN graph [`Config`] supplies the degree, alpha, and prune kind.
#[derive(Clone, Debug, PartialEq)]
pub struct PiPNNConfig {
    /// Maximum number of points in a leaf.
    pub c_max: NonZeroUsize,
    /// Fraction of the points of a cluster that a split samples as leaders, in
    /// `(0, 1]`.
    pub p_samp: f64,
    /// Number of nearest leaders that each point joins, for each split level.
    /// Levels after this schedule use one leader. The schedule must not be empty.
    pub fanout: Vec<NonZeroUsize>,
    /// Number of nearest neighbors that each point selects inside a leaf.
    pub leaf_k: NonZeroUsize,
    /// Number of independent partitionings of the dataset.
    pub replicas: NonZeroUsize,
}

impl PiPNNConfig {
    /// Check the parameters that the field types do not constrain.
    pub fn validate(&self) -> Result<(), PiPNNConfigError> {
        if !(0.0 < self.p_samp && self.p_samp <= 1.0) {
            return Err(PiPNNConfigError::SamplingFraction(self.p_samp));
        }
        if self.fanout.is_empty() {
            return Err(PiPNNConfigError::EmptyFanout);
        }
        Ok(())
    }
}

/// HashPrune parameters.
///
/// Each edge `source -> target` gets a direction hash with one bit per random
/// hyperplane. The reservoir of `source` keeps the nearest target for each hash,
/// up to `l_max` targets.
#[derive(Clone, Debug, PartialEq)]
pub struct HashPruneConfig {
    /// Number of random hyperplanes, one hash bit each.
    pub num_hash_planes: usize,
    /// Maximum number of candidates in the reservoir of a point.
    pub l_max: usize,
    /// Prune the reservoir candidates with Vamana RobustPrune. Without this
    /// step, each point keeps its nearest candidates up to the graph degree.
    pub final_prune: bool,
}

impl HashPruneConfig {
    /// Check the parameter ranges, and check that a reservoir can hold `degree`
    /// candidates.
    ///
    /// A reservoir holds at most one candidate per hash, so it holds at most
    /// `2^num_hash_planes` candidates.
    pub fn validate(&self, degree: usize) -> Result<(), PiPNNConfigError> {
        if !(1..=lsh::MAX_PLANES).contains(&self.num_hash_planes) {
            return Err(PiPNNConfigError::HashPlanes(self.num_hash_planes));
        }
        if !(1..=hash_prune::MAX_RESERVOIR_LEN).contains(&self.l_max) {
            return Err(PiPNNConfigError::ReservoirLength(self.l_max));
        }
        let hash_buckets = 1 << self.num_hash_planes;
        if self.l_max.min(hash_buckets) < degree {
            return Err(PiPNNConfigError::HashPruneCapacity {
                l_max: self.l_max,
                hash_buckets,
                degree,
            });
        }
        Ok(())
    }
}

/// PiPNN parameters, graph policy, and the Rayon pool for one graph build.
#[derive(Debug)]
pub struct PiPNNBuildContext<'a> {
    config: PiPNNConfig,
    hash_prune: Option<HashPruneConfig>,
    graph: &'a Config,
    metric: Metric,
    pool: &'a ThreadPool,
}

impl<'a> PiPNNBuildContext<'a> {
    /// Check `config` and combine it with the graph policy.
    ///
    /// Final pruning applies the prune kind of `graph` to distances of `metric`,
    /// so the two must match.
    pub fn new(
        config: PiPNNConfig,
        graph: &'a Config,
        metric: Metric,
        pool: &'a ThreadPool,
    ) -> Result<Self, PiPNNConfigError> {
        config.validate()?;
        let prune_kind = graph.prune_kind();
        if prune_kind != metric.into() {
            return Err(PiPNNConfigError::PruneKind { prune_kind, metric });
        }

        Ok(Self {
            config,
            hash_prune: None,
            graph,
            metric,
            pool,
        })
    }

    /// Merge the leaf candidates with HashPrune reservoirs.
    pub fn with_hash_prune(mut self, config: HashPruneConfig) -> Result<Self, PiPNNConfigError> {
        config.validate(self.graph.pruned_degree().get())?;
        self.hash_prune = Some(config);
        Ok(self)
    }
}

/// Build one adjacency list for each point of `data`.
///
/// The graph links dataset points only; the caller selects start points and
/// serializes the index. `CosineNormalized` on `u8` or `i8` data builds with
/// `Cosine`, because converted integer vectors do not have unit norm.
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
        return Err(ANNError::message(format!(
            "PiPNN dataset point count ({}) exceeds the u32 graph ID limit",
            data.nrows()
        )));
    }
    arch::dispatch2_no_features(BuildGraph, data, context)
}

/// Runs the build for the architecture that `diskann-wide` selects at run time.
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

/// Run the three build steps for architecture `A` and metric `M`.
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
    // The leaf step takes the leaves by value, so their memory is free before
    // final pruning starts.
    let leaf_k = context.config.leaf_k.get();
    let candidates = match &context.hash_prune {
        None => tracing::info_span!("pipnn.leaf_build").in_scope(|| {
            leaf_build::build_leaf_candidates::<A, M, T>(arch, data, leaves, leaf_k)
        })?,
        Some(config) => {
            let reservoirs =
                hash_prune::HashPrune::new(data, config.num_hash_planes, config.l_max, 42)?;
            tracing::info_span!("pipnn.leaf_build").in_scope(|| {
                leaf_build::add_hash_prune_candidates::<A, M, T>(
                    arch,
                    data,
                    leaves,
                    leaf_k,
                    &reservoirs,
                )
            })?;
            if !config.final_prune {
                return Ok(reservoirs.into_nearest_lists(context.graph.pruned_degree().get()));
            }
            reservoirs.into_candidate_lists()
        }
    };
    // Final pruning cuts each candidate list in place.
    Ok(tracing::info_span!("pipnn.finalization")
        .in_scope(|| finalization::prune_overfull(data, candidates, context.graph, metric)))
}

/// Return the metric that the build uses for element type `T`. See [`build_graph`].
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

#[cfg(test)]
mod test_support {
    use super::simd::Simd;
    use diskann_vector::distance::Metric;

    // Sort the members of each row but keep the row order. A row position is a
    // leader column or a graph source, so sorting the rows would hide a wrong
    // assignment. Repeated members stay visible.
    pub(super) fn sorted_members_per_row(rows: &[Vec<u32>]) -> Vec<Vec<u32>> {
        rows.iter()
            .map(|row| {
                let mut members = row.clone();
                members.sort_unstable();
                members
            })
            .collect()
    }

    pub(super) fn nz(value: usize) -> std::num::NonZeroUsize {
        std::num::NonZeroUsize::new(value).unwrap()
    }

    /// Build a Rayon pool with `threads` workers.
    pub(super) fn thread_pool(threads: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
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

#[cfg(test)]
mod construction_tests {
    use super::*;
    use crate::graph::config;
    use half::f16;
    use test_support::{nz, sorted_members_per_row, thread_pool};

    fn partition_policy() -> PiPNNConfig {
        PiPNNConfig {
            c_max: nz(4),
            p_samp: 1.0,
            fanout: vec![nz(2)],
            leaf_k: nz(1),
            replicas: nz(1),
        }
    }

    fn graph_policy(degree: usize, metric: Metric) -> Result<Config, config::ConfigError> {
        config::Builder::new_with(degree, config::MaxDegree::same(), 16, metric.into(), |b| {
            b.alpha(1.0);
        })
        .build()
    }

    /// Build the graph of `values` with `threads` workers.
    fn build<T: VectorRepr>(
        values: &[T],
        dimensions: usize,
        config: PiPNNConfig,
        degree: usize,
        metric: Metric,
        threads: usize,
    ) -> Vec<Vec<u32>> {
        let data = MatrixView::try_from(values, values.len() / dimensions, dimensions).unwrap();
        let graph = graph_policy(degree, metric).unwrap();
        let pool = thread_pool(threads);
        let context = PiPNNBuildContext::new(config, &graph, metric, &pool).unwrap();
        let graph = build_graph(data, &context).unwrap();
        graph.into_iter().map(Vec::from).collect()
    }

    #[test]
    fn invalid_sampling_fractions_and_an_empty_fanout_are_rejected() {
        for p_samp in [
            0.0,
            -0.5,
            1.0 + f64::EPSILON,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            let config = PiPNNConfig {
                p_samp,
                ..partition_policy()
            };

            let error = config.validate().unwrap_err();

            assert!(
                matches!(error, PiPNNConfigError::SamplingFraction(value) if value.to_bits() == p_samp.to_bits()),
                "p_samp {p_samp}: {error}"
            );
        }
        let config = PiPNNConfig {
            fanout: vec![],
            ..partition_policy()
        };
        assert_eq!(config.validate(), Err(PiPNNConfigError::EmptyFanout));
    }

    #[test]
    fn sampling_fraction_endpoints_are_accepted() {
        for p_samp in [f64::from_bits(1), 1.0] {
            let config = PiPNNConfig {
                p_samp,
                ..partition_policy()
            };

            assert_eq!(config.validate(), Ok(()), "p_samp {p_samp}");
        }
    }

    #[test]
    fn a_build_context_checks_the_configuration() {
        let config = PiPNNConfig {
            fanout: vec![],
            ..partition_policy()
        };
        let graph = graph_policy(2, Metric::L2).unwrap();
        let pool = thread_pool(1);

        let error = PiPNNBuildContext::new(config, &graph, Metric::L2, &pool).unwrap_err();

        assert_eq!(error, PiPNNConfigError::EmptyFanout);
    }

    #[test]
    fn a_build_context_requires_the_pruning_kind_of_its_metric() {
        // L2, cosine and normalized cosine share triangle pruning. Inner product
        // uses occluding pruning.
        let pool = thread_pool(1);
        for (graph_metric, metric, compatible) in [
            (Metric::L2, Metric::Cosine, true),
            (Metric::L2, Metric::CosineNormalized, true),
            (Metric::InnerProduct, Metric::InnerProduct, true),
            (Metric::L2, Metric::InnerProduct, false),
            (Metric::InnerProduct, Metric::L2, false),
        ] {
            let graph = graph_policy(2, graph_metric).unwrap();

            let result = PiPNNBuildContext::new(partition_policy(), &graph, metric, &pool);

            let case = format!("graph {graph_metric:?}, metric {metric:?}");
            match result {
                Ok(_) => assert!(compatible, "{case}"),
                Err(error) => assert_eq!(
                    (compatible, error),
                    (
                        false,
                        PiPNNConfigError::PruneKind {
                            prune_kind: graph_metric.into(),
                            metric
                        }
                    ),
                    "{case}"
                ),
            }
        }
    }

    #[test]
    fn an_empty_dataset_axis_is_rejected_before_building() {
        let graph = graph_policy(2, Metric::L2).unwrap();
        let pool = thread_pool(1);
        let context =
            PiPNNBuildContext::new(partition_policy(), &graph, Metric::L2, &pool).unwrap();

        for (rows, columns, expected_message) in [
            (0, 2, "at least one data point"),
            (2, 0, "at least one data dimension"),
        ] {
            let data = MatrixView::try_from(&[] as &[f32], rows, columns).unwrap();

            let error = build_graph(data, &context).unwrap_err();

            assert!(error.to_string().contains(expected_message), "{error}");
        }
    }

    #[test]
    fn native_vector_types_build_the_expected_neighbor_graph() {
        // Nearest choices 0 -> 1, 1 -> 0 and 2 -> 1 become a symmetric chain.
        fn check<T: VectorRepr>(values: [T; 3]) {
            let actual = build(&values, 1, partition_policy(), 2, Metric::L2, 2);

            assert_eq!(
                sorted_members_per_row(&actual),
                [vec![1], vec![0, 2], vec![1]],
                "{}",
                std::any::type_name::<T>()
            );
        }

        check([0.0_f32, 1.0, 4.0]);
        check([0.0_f32, 1.0, 4.0].map(f16::from_f32));
        check([0_i8, 1, 4]);
        check([0_u8, 1, 4]);
    }

    #[test]
    fn the_requested_metric_determines_leaf_neighbors() {
        // L2's nearest choices are [2, 2, 0, 2]; cosine pairs similar directions
        // 0 <-> 1 and 2 <-> 3; dot products give [1, 3, 3, 2].
        // The second coordinate sits in the last dimension of an embedding.
        let dimensions = 1537;
        for (metric, expected) in [
            (Metric::L2, vec![vec![2], vec![2], vec![0, 1, 3], vec![2]]),
            (Metric::Cosine, vec![vec![1], vec![0], vec![3], vec![2]]),
            (
                Metric::CosineNormalized,
                vec![vec![1], vec![0], vec![3], vec![2]],
            ),
            (
                Metric::InnerProduct,
                vec![vec![1], vec![0, 3], vec![3], vec![1, 2]],
            ),
        ] {
            let values = test_support::packed_points(
                &[[1.0, 0.0], [5.0, 2.0], [1.0, 3.0], [0.0, 9.0]],
                dimensions,
                metric == Metric::CosineNormalized,
            );

            let actual = build(&values, dimensions, partition_policy(), 3, metric, 2);

            assert_eq!(
                sorted_members_per_row(&actual),
                sorted_members_per_row(&expected),
                "{metric:?}"
            );
        }
    }

    #[test]
    fn splitting_uses_the_requested_metric_to_group_points() {
        // Four points force splitting at c_max=2. Every point is sampled, so
        // leader order cannot change memberships. With L2, leader 0 gets {0,2}
        // and leader 2 gets all four points, which then split into singletons.
        // Cosine pairs directions {0,1} and {2,3}. Inner product picks leaders
        // {1,2} for points 0/1 and {2,3} for points 2/3; the oversized leader-2
        // cluster then splits into those same pairs using fanout one.
        // Every two-point leaf contributes its only pair; degree three retains
        // every edge. The graph therefore exposes partition membership directly.
        let config = PiPNNConfig {
            c_max: nz(2),
            ..partition_policy()
        };
        for (metric, expected) in [
            (Metric::L2, vec![vec![2], vec![], vec![0], vec![]]),
            (Metric::Cosine, vec![vec![1], vec![0], vec![3], vec![2]]),
            (
                Metric::CosineNormalized,
                vec![vec![1], vec![0], vec![3], vec![2]],
            ),
            (
                Metric::InnerProduct,
                vec![vec![1], vec![0], vec![3], vec![2]],
            ),
        ] {
            let values = test_support::packed_points(
                &[[1.0, 0.0], [6.0, 1.0], [2.0, 4.0], [0.0, 9.0]],
                2,
                metric == Metric::CosineNormalized,
            );

            let actual = build(&values, 2, config.clone(), 3, metric, 2);

            assert_eq!(
                sorted_members_per_row(&actual),
                sorted_members_per_row(&expected),
                "{metric:?}"
            );
        }
    }

    #[test]
    fn normalized_cosine_requests_on_raw_integers_use_vector_norms() {
        // Cosine pairs 0 <-> 1 and 2 <-> 3. Unnormalized dot products instead
        // connect 1 to 3, so treating raw integers as unit vectors changes edges.
        // The first build checks leaf ranking. The second build selects three
        // leaf neighbors and checks final pruning to one.
        fn check<T: VectorRepr>(values: [T; 8]) {
            for (leaf_k, degree) in [(1, 3), (3, 1)] {
                let config = PiPNNConfig {
                    leaf_k: nz(leaf_k),
                    ..partition_policy()
                };

                let actual = build(&values, 2, config, degree, Metric::CosineNormalized, 2);

                assert_eq!(
                    actual,
                    [vec![1], vec![0], vec![3], vec![2]],
                    "{}, leaf_k={leaf_k}, degree={degree}",
                    std::any::type_name::<T>()
                );
            }
        }

        check([1_i8, 0, 4, 1, 1, 2, 0, 9]);
        check([1_u8, 0, 4, 1, 1, 2, 0, 9]);
    }

    #[test]
    fn final_pruning_cuts_merged_candidates_to_the_degree() {
        let config = PiPNNConfig {
            leaf_k: nz(2),
            ..partition_policy()
        };

        let actual = build(&[0.0_f32, 1.0, 4.0], 1, config, 1, Metric::L2, 2);

        assert_eq!(actual, [vec![1], vec![0], vec![1]]);
    }

    #[test]
    fn overlapping_partitions_build_the_same_deduplicated_graph() {
        // Sampling every point with fanout 2 makes two overlapping copies
        // of each close pair. Neither replicas nor worker count may duplicate edges.
        for (workers, replicas) in [(1, 1), (3, 2)] {
            let config = PiPNNConfig {
                c_max: nz(2),
                replicas: nz(replicas),
                ..partition_policy()
            };

            let actual = build(
                &[0.0_f32, 1.0, 10.0, 11.0],
                1,
                config,
                2,
                Metric::L2,
                workers,
            );

            assert_eq!(
                actual,
                [vec![1], vec![0], vec![3], vec![2]],
                "{workers} workers, {replicas} replicas"
            );
        }
    }

    #[test]
    fn a_single_point_has_one_empty_adjacency_list() {
        let actual = build(&[3.0_f32], 1, partition_policy(), 2, Metric::L2, 1);

        assert_eq!(actual, [Vec::<u32>::new()]);
    }

    #[test]
    fn multi_level_builds_give_valid_rows_for_every_metric() {
        // With 1,000 points and c_max 32, the build splits over several levels,
        // makes overlapping leaves, merges their candidates on four workers and
        // prunes lists above the degree. Each row must be non-empty, within the
        // degree, and free of duplicates and self edges.
        let (points, dimensions, degree) = (1000, 16, 8);
        let config = PiPNNConfig {
            c_max: nz(32),
            p_samp: 0.05,
            fanout: vec![nz(4), nz(2)],
            leaf_k: nz(2),
            replicas: nz(1),
        };
        for metric in [
            Metric::L2,
            Metric::Cosine,
            Metric::CosineNormalized,
            Metric::InnerProduct,
        ] {
            let mut values = test_support::dense_points(points, dimensions, 1290);
            if metric == Metric::CosineNormalized {
                test_support::normalize(&mut values, dimensions);
            }

            let actual = build(&values, dimensions, config.clone(), degree, metric, 4);

            for (point, neighbors) in actual.iter().enumerate() {
                let mut distinct = neighbors.clone();
                distinct.sort_unstable();
                distinct.dedup();
                assert!(
                    !neighbors.is_empty()
                        && neighbors.len() <= degree
                        && distinct.len() == neighbors.len()
                        && !neighbors.contains(&(point as u32)),
                    "{metric:?}, point {point}: {neighbors:?}"
                );
            }
        }
    }

    #[test]
    fn copies_of_one_vector_build_leaf_local_edges() {
        // Every leader is a copy, so each point ties at every split. The build
        // splits the six copies into two leaves of three points.
        let actual = build(&[1.0_f32; 6], 1, partition_policy(), 2, Metric::L2, 2);

        for (point, neighbors) in actual.iter().enumerate() {
            assert!(!neighbors.is_empty(), "point {point}");
            assert!(
                neighbors
                    .iter()
                    .all(|&neighbor| neighbor / 3 == point as u32 / 3),
                "point {point}: {neighbors:?}"
            );
        }
    }

    #[test]
    fn hash_prune_parameters_must_hold_the_graph_degree() {
        let pool = thread_pool(1);
        for (num_hash_planes, l_max, degree, expected) in [
            (0, 64, 2, Err(PiPNNConfigError::HashPlanes(0))),
            (17, 64, 2, Err(PiPNNConfigError::HashPlanes(17))),
            (8, 0, 2, Err(PiPNNConfigError::ReservoirLength(0))),
            (8, 256, 2, Err(PiPNNConfigError::ReservoirLength(256))),
            (16, 255, 2, Ok(())),
            (8, 64, 64, Ok(())),
            (
                8,
                63,
                64,
                Err(PiPNNConfigError::HashPruneCapacity {
                    l_max: 63,
                    hash_buckets: 256,
                    degree: 64,
                }),
            ),
            // One plane gives two hashes, so a reservoir holds two candidates.
            (1, 64, 2, Ok(())),
            (
                1,
                64,
                3,
                Err(PiPNNConfigError::HashPruneCapacity {
                    l_max: 64,
                    hash_buckets: 2,
                    degree: 3,
                }),
            ),
        ] {
            let graph = graph_policy(degree, Metric::L2).unwrap();
            let context =
                PiPNNBuildContext::new(partition_policy(), &graph, Metric::L2, &pool).unwrap();
            let config = HashPruneConfig {
                num_hash_planes,
                l_max,
                final_prune: true,
            };

            let actual = context.with_hash_prune(config).map(|_| ());

            assert_eq!(
                actual, expected,
                "planes={num_hash_planes}, l_max={l_max}, degree={degree}"
            );
        }
    }

    #[test]
    fn hash_prune_keeps_the_nearest_neighbor_in_each_direction() {
        // Two replicas add the same leaf in parallel. On a line, the two
        // directions from a point have complementary hashes, so each reservoir
        // keeps the nearest point on each side. Degree two keeps both.
        let values = [-3.0_f32, 0.0, 1.0];
        let data = MatrixView::column_vector(&values[..]);
        let graph = graph_policy(2, Metric::L2).unwrap();
        let config = PiPNNConfig {
            c_max: nz(3),
            fanout: vec![nz(1)],
            leaf_k: nz(2),
            replicas: nz(2),
            ..partition_policy()
        };
        for (threads, final_prune) in [(1, true), (4, true), (4, false)] {
            let pool = thread_pool(threads);
            let context = PiPNNBuildContext::new(config.clone(), &graph, Metric::L2, &pool)
                .unwrap()
                .with_hash_prune(HashPruneConfig {
                    num_hash_planes: 8,
                    l_max: 16,
                    final_prune,
                })
                .unwrap();

            let actual = build_graph(data, &context).unwrap();

            let actual: Vec<Vec<u32>> = actual.into_iter().map(Vec::from).collect();
            assert_eq!(
                sorted_members_per_row(&actual),
                [vec![1], vec![0, 2], vec![1]],
                "{threads} threads, final_prune={final_prune}"
            );
        }
    }
}

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

/// Error for invalid PiPNN parameters.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum PiPNNConfigError {
    #[error("p_samp ({0}) must be in (0, 1]")]
    SamplingFraction(f64),
    #[error("fanout must not be empty")]
    EmptyFanout,
    #[error(
        "max_depth ({max_depth}) must be at least the number of fanout levels ({fanout_levels})"
    )]
    MaxDepth {
        max_depth: usize,
        fanout_levels: usize,
    },
    #[error("max_leaders ({0}) must be at least 2")]
    MaxLeaders(usize),
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
///
/// The default configuration is `c_max = 512`, `p_samp = 0.005`,
/// `max_leaders = 1000`, `fanout = [10, 3]`, `max_depth = 10`, `leaf_k = 2`
/// and `replicas = 1`.
/// Each point then joins about 30 leaves, the product of the fanout values.
/// Larger `c_max`, fanout values, `leaf_k` and `replicas` all give each point
/// more or nearer candidates, which raises recall and makes the build slower.
#[derive(Clone, Debug, PartialEq)]
pub struct PiPNNConfig {
    /// Maximum number of points in a leaf. A larger leaf holds more of the true
    /// neighbors of its points, so the selected neighbors are nearer. The leaf
    /// distance work grows with the square of the leaf size.
    pub c_max: NonZeroUsize,
    /// Fraction of the points of a cluster that a split samples as leaders, in
    /// `(0, 1]`. More leaders give smaller children and fewer split levels, but
    /// each split costs `points x leaders x dimensions` multiply-adds. A split
    /// samples at least 2 and at most `max_leaders` leaders.
    pub p_samp: f64,
    /// Maximum number of leaders in one split, at least 2. A split costs
    /// `points x leaders x dimensions` multiply-adds, so the cap bounds the
    /// first splits of a large dataset, where `p_samp` alone samples many
    /// leaders: 10M points at `p_samp = 0.005` would give 50,000. The PiPNN
    /// paper uses about 1,000.
    pub max_leaders: NonZeroUsize,
    /// Number of nearest leaders that each point joins, for each split level.
    /// Levels after this schedule use one leader. The schedule must not be empty.
    /// Larger values make leaves overlap more, so each point gets candidates
    /// from more leaves. The number of leaves and the leaf build time grow with
    /// the product of the values.
    pub fanout: Vec<NonZeroUsize>,
    /// Maximum number of split levels. A cluster that is still larger than
    /// `c_max` after `max_depth` splits is cut into equal leaves in point
    /// order, and the build logs a warning with the number of points it cut.
    /// Such leaves group points arbitrarily, which can lower recall. The value
    /// must be at least the number of `fanout` levels.
    pub max_depth: NonZeroUsize,
    /// Number of nearest neighbors that each point selects inside a leaf. Each
    /// selected pair adds a candidate in both directions. A larger value gives
    /// more candidates and makes final pruning slower.
    pub leaf_k: NonZeroUsize,
    /// Number of independent partitionings of the dataset. Each replica repeats
    /// the partition and leaf work with new leaders, which adds candidates and
    /// build time.
    pub replicas: NonZeroUsize,
    /// Seed of leader sampling. The same seed gives the same leaves. `None`
    /// draws a seed from the operating system, so each build samples new leaves.
    pub seed: Option<u64>,
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
        if self.max_depth.get() < self.fanout.len() {
            return Err(PiPNNConfigError::MaxDepth {
                max_depth: self.max_depth.get(),
                fanout_levels: self.fanout.len(),
            });
        }
        if self.max_leaders.get() < 2 {
            return Err(PiPNNConfigError::MaxLeaders(self.max_leaders.get()));
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
    dimensions: NonZeroUsize,
    pool: &'a ThreadPool,
}

impl<'a> PiPNNBuildContext<'a> {
    /// Check `config` and combine it with the graph policy.
    ///
    /// Final pruning uses the prune kind of `graph`, as Vamana does. Build
    /// `graph` with the prune kind of `metric` (`metric.into()`).
    ///
    /// `dimensions` is the length of each row after conversion to `f32`.
    /// Compressed element types such as MinMax store more elements per row,
    /// so it can differ from `data.ncols()`.
    pub fn new(
        config: PiPNNConfig,
        graph: &'a Config,
        metric: Metric,
        dimensions: NonZeroUsize,
        pool: &'a ThreadPool,
    ) -> Result<Self, PiPNNConfigError> {
        config.validate()?;
        Ok(Self {
            config,
            hash_prune: None,
            graph,
            metric,
            dimensions,
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
/// Each step converts the rows it reads to `f32` with
/// [`VectorRepr::as_f32_into`], so any element type that converts to `f32`
/// works, including compressed ones.
///
/// The graph links dataset points only; the caller selects start points and
/// serializes the index.
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
        let metric = context.metric;
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
    let dimensions = context.dimensions.get();
    let leaves = tracing::info_span!("pipnn.partition")
        .in_scope(|| partitioning::partition::<A, M, T>(arch, data, dimensions, &context.config))?;
    // The leaf step takes the leaves by value, so their memory is free before
    // final pruning starts.
    let leaf_k = context.config.leaf_k.get();
    let candidates = match &context.hash_prune {
        None => tracing::info_span!("pipnn.leaf_build").in_scope(|| {
            leaf_build::build_leaf_candidates::<A, M, T>(arch, data, dimensions, leaves, leaf_k)
        })?,
        Some(config) => {
            let reservoirs = hash_prune::HashPrune::new(
                data,
                dimensions,
                config.num_hash_planes,
                config.l_max,
                42,
            )?;
            tracing::info_span!("pipnn.leaf_build").in_scope(|| {
                leaf_build::add_hash_prune_candidates::<A, M, T>(
                    arch,
                    data,
                    dimensions,
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

#[cfg(test)]
mod test_support {
    use std::collections::BTreeSet;

    use super::simd::Simd;
    use diskann_vector::distance::Metric;

    /// Return the directed edges `(source, target)` of `rows`, where row `i`
    /// lists the neighbors of point `i`. A repeated edge panics, so tests that
    /// compare edge sets still catch duplicate candidates.
    pub(super) fn edges(rows: &[Vec<u32>]) -> BTreeSet<(u32, u32)> {
        let mut edges = BTreeSet::new();
        for (source, row) in rows.iter().enumerate() {
            for &target in row {
                assert!(
                    edges.insert((source as u32, target)),
                    "repeated edge {source} -> {target}"
                );
            }
        }
        edges
    }

    /// Return both directions of each undirected pair.
    pub(super) fn both_directions<const N: usize>(pairs: [(u32, u32); N]) -> BTreeSet<(u32, u32)> {
        pairs
            .into_iter()
            .flat_map(|(a, b)| [(a, b), (b, a)])
            .collect()
    }

    pub(super) fn nz(value: usize) -> std::num::NonZeroUsize {
        std::num::NonZeroUsize::new(value).unwrap()
    }

    /// Build a Rayon pool with `n_threads` threads.
    pub(super) fn thread_pool(n_threads: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .unwrap()
    }

    #[test]
    #[should_panic(expected = "repeated edge 0 -> 1")]
    fn a_repeated_edge_panics() {
        // Point 0 repeats point 1; an edge-set comparison must reject the repeat.
        edges(&[vec![1, 1], vec![]]);
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
    use crate::{
        graph::{DiskANNIndex, config, search::Knn, test::provider as test_provider},
        neighbor::{BackInserter, Neighbor},
        test::tokio::current_thread_runtime,
    };
    use diskann_utils::{io::read_bin, test_data_root, views::Matrix};
    use diskann_vector::DistanceFunction;
    use half::f16;
    use std::fs::File;
    use test_support::{both_directions, edges, nz, thread_pool};

    const SIFT: &str = "sift/siftsmall_learn_256pts.fbin";
    const SIFT_NORMALIZED: &str = "sift/siftsmall_learn_256pts_normalized.fbin";

    /// Every point is a leader (`p_samp = 1.0`), so leader sampling picks the
    /// same set for any seed.
    fn partition_policy() -> PiPNNConfig {
        PiPNNConfig {
            c_max: nz(4),
            p_samp: 1.0,
            max_leaders: nz(1000),
            fanout: vec![nz(2)],
            max_depth: nz(10),
            leaf_k: nz(1),
            replicas: nz(1),
            seed: Some(1290),
        }
    }

    /// Use alpha one and no degree slack. PiPNN does not use the build search list.
    fn graph_policy(degree: usize, metric: Metric) -> Config {
        let search_list_size = 16;
        let alpha = 1.0;
        config::Builder::new_with(
            degree,
            config::MaxDegree::same(),
            search_list_size,
            metric.into(),
            |b| {
                b.alpha(alpha);
            },
        )
        .build()
        .unwrap()
    }

    fn load_sift(file: &str) -> Matrix<f32> {
        read_bin::<f32>(&mut File::open(test_data_root().join(file)).unwrap()).unwrap()
    }

    /// 256 x 0.05 = 12.8 rounds up to 13 leaders, so the fanout-10 level runs
    /// and the leaves overlap.
    fn sift_pipnn_config() -> PiPNNConfig {
        PiPNNConfig {
            c_max: nz(64),
            p_samp: 0.05,
            max_leaders: nz(1000),
            fanout: vec![nz(10), nz(3)],
            max_depth: nz(10),
            leaf_k: nz(2),
            replicas: nz(1),
            seed: Some(1290),
        }
    }

    /// Degree 16 with the default alpha. PiPNN does not read the build search
    /// list size.
    fn sift_graph(metric: Metric) -> Config {
        let degree = 16;
        let l_build = 64;
        config::Builder::new(
            degree,
            config::MaxDegree::default_slack(),
            l_build,
            metric.into(),
        )
        .build()
        .unwrap()
    }

    /// Return the point nearest to the mean of `data`, under squared L2.
    fn medoid(data: MatrixView<'_, f32>) -> usize {
        let point_count = data.nrows();
        let dimensions = data.ncols();
        let mut mean = vec![0.0_f64; dimensions];
        for row in data.row_iter() {
            for (sum, &value) in mean.iter_mut().zip(row) {
                *sum += f64::from(value);
            }
        }
        let mean: Vec<f32> = mean
            .into_iter()
            .map(|sum| (sum / point_count as f64) as f32)
            .collect();
        let l2 = f32::distance(Metric::L2, Some(dimensions));
        data.row_iter()
            .enumerate()
            .map(|(id, row)| (id, l2.evaluate_similarity(row, mean.as_slice())))
            .min_by(|left, right| left.1.total_cmp(&right.1))
            .unwrap()
            .0
    }

    /// Build the PiPNN graph of `build_data` on one thread.
    fn build_sift_graph<T: VectorRepr>(
        build_data: MatrixView<'_, T>,
        graph: &Config,
        metric: Metric,
    ) -> Vec<AdjacencyList<u32>> {
        let config = sift_pipnn_config();
        let dimensions = NonZeroUsize::new(build_data.ncols()).unwrap();
        let n_threads = 1;
        let pool = thread_pool(n_threads);
        let context = PiPNNBuildContext::new(config, graph, metric, dimensions, &pool).unwrap();
        build_graph(build_data, &context).unwrap()
    }

    /// Load `data` and its PiPNN graph into the test provider. One start node
    /// points to the medoid, as the disk path starts at the medoid; the provider
    /// leaves start nodes out of search results.
    fn searchable_index(
        data: MatrixView<'_, f32>,
        graph: Config,
        metric: Metric,
        adjacency: Vec<AdjacencyList<u32>>,
    ) -> DiskANNIndex<test_provider::Provider> {
        let medoid_id = medoid(data);
        let start_id = data.nrows() as u32;
        let degree = graph.pruned_degree().get();
        let provider_config = test_provider::Config::new(
            metric,
            degree,
            test_provider::StartPoint::new(start_id, data.row(medoid_id).to_vec()),
        )
        .unwrap();
        let start_neighbors = AdjacencyList::from_iter_untrusted([medoid_id as u32]);
        let points = data
            .row_iter()
            .zip(adjacency)
            .enumerate()
            .map(|(id, (row, neighbors))| (id as u32, row.to_vec(), neighbors));
        let provider = test_provider::Provider::new_from(
            provider_config,
            [(start_id, start_neighbors)],
            points,
        )
        .unwrap();
        DiskANNIndex::new(graph, provider, None)
    }

    /// Assert that a search for each point of `data` returns its ten nearest
    /// distances, in order. An exact top 1 for every point also shows that
    /// every point is reachable from the start node.
    fn assert_exact_top_10(
        index: &DiskANNIndex<test_provider::Provider>,
        data: MatrixView<'_, f32>,
        metric: Metric,
        case: &str,
    ) {
        let search_l = 32;
        let top_k = 10;
        let point_count = data.nrows();
        let dimensions = data.ncols();
        let runtime = current_thread_runtime();
        let distance = f32::distance(metric, Some(dimensions));
        let mut mismatches = 0;
        let mut example = String::new();

        for (query_id, query) in data.row_iter().enumerate() {
            let mut expected: Vec<f32> = data
                .row_iter()
                .map(|point| distance.evaluate_similarity(query, point))
                .collect();
            expected.sort_unstable_by(f32::total_cmp);
            expected.truncate(top_k);
            let mut neighbors = vec![Neighbor::<u32>::default(); top_k];
            let stats = runtime
                .block_on(index.search(
                    Knn::new_default(search_l).unwrap(),
                    &test_provider::Strategy::new(),
                    &test_provider::Context::new(),
                    query,
                    &mut BackInserter::new(neighbors.as_mut_slice()),
                ))
                .unwrap();
            let actual: Vec<f32> = neighbors
                .iter()
                .take(stats.result_count as usize)
                .map(|neighbor| *neighbor.distance())
                .collect();
            // Equal distances can have different IDs, so compare distances in order.
            if actual != expected {
                mismatches += 1;
                if mismatches == 1 {
                    example = format!("query {query_id}: expected {expected:?}, actual {actual:?}");
                }
            }
        }

        assert_eq!(
            mismatches, 0,
            "{case}: {mismatches}/{point_count} queries differ; first: {example}"
        );
    }

    #[test]
    fn invalid_sampling_fractions_are_rejected() {
        // Fractions outside (0, 1] cannot select a valid sample of leaders.
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
    }

    #[test]
    fn an_empty_fanout_is_rejected() {
        // An empty fanout list has no first level.
        let config = PiPNNConfig {
            fanout: vec![],
            ..partition_policy()
        };
        assert_eq!(config.validate(), Err(PiPNNConfigError::EmptyFanout));
    }

    #[test]
    fn sampling_fraction_endpoints_are_accepted() {
        // The smallest positive fraction and the inclusive upper bound are valid.
        let smallest_positive = f64::from_bits(1);
        for p_samp in [smallest_positive, 1.0] {
            let config = PiPNNConfig {
                p_samp,
                ..partition_policy()
            };

            assert_eq!(config.validate(), Ok(()), "p_samp {p_samp}");
        }
    }

    #[test]
    fn a_leader_cap_below_two_is_rejected() {
        // A split needs two leaders.
        let config = PiPNNConfig {
            max_leaders: nz(1),
            ..partition_policy()
        };

        assert_eq!(config.validate(), Err(PiPNNConfigError::MaxLeaders(1)));
    }

    #[test]
    fn a_max_depth_below_the_fanout_levels_is_rejected() {
        // Two fanout levels cannot fit within one split level.
        let config = PiPNNConfig {
            fanout: vec![nz(10), nz(3)],
            max_depth: nz(1),
            ..partition_policy()
        };

        let actual = config.validate();

        assert_eq!(
            actual,
            Err(PiPNNConfigError::MaxDepth {
                max_depth: 1,
                fanout_levels: 2
            })
        );
    }

    #[test]
    fn a_build_context_checks_the_configuration() {
        // An empty fanout must fail when the context is created, before a build.
        let degree = 2;
        let n_threads = 1;
        let config = PiPNNConfig {
            fanout: vec![],
            ..partition_policy()
        };
        let dimensions = nz(1);
        let graph = graph_policy(degree, Metric::L2);
        let pool = thread_pool(n_threads);

        let error =
            PiPNNBuildContext::new(config, &graph, Metric::L2, dimensions, &pool).unwrap_err();

        assert_eq!(error, PiPNNConfigError::EmptyFanout);
    }

    #[test]
    fn an_empty_dataset_axis_is_rejected_before_building() {
        // Each shape has one empty axis; validation must reject it before a kernel runs.
        let degree = 2;
        let n_threads = 1;
        let dimensions = nz(2);
        let graph = graph_policy(degree, Metric::L2);
        let pool = thread_pool(n_threads);
        let context =
            PiPNNBuildContext::new(partition_policy(), &graph, Metric::L2, dimensions, &pool)
                .unwrap();

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
    fn every_point_finds_its_exact_top_10_for_each_metric() {
        // Raw SIFT rows serve L2 and cosine. Normalized cosine and inner product
        // need unit vectors, so they use the normalized file.
        for (metric, file) in [
            (Metric::L2, SIFT),
            (Metric::Cosine, SIFT),
            (Metric::CosineNormalized, SIFT_NORMALIZED),
            (Metric::InnerProduct, SIFT_NORMALIZED),
        ] {
            let data = load_sift(file);
            let graph = sift_graph(metric);

            let adjacency = build_sift_graph(data.as_view(), &graph, metric);
            let index = searchable_index(data.as_view(), graph, metric, adjacency);

            assert_exact_top_10(&index, data.as_view(), metric, &format!("{metric:?}"));
        }
    }

    #[test]
    fn every_point_finds_its_exact_top_10_for_each_vector_type() {
        let data = load_sift(SIFT);
        // These bounds make f16 and u8 conversion exact, so every build must
        // find the same distances against the original f32 rows.
        assert!(
            data.as_slice()
                .iter()
                .all(|&value| value.fract() == 0.0 && (0.0..=255.0).contains(&value)),
            "raw SIFT values must be integers in 0..=255 for exact f16 and u8 conversion"
        );
        let half = data.map(|&value| f16::from_f32(value));
        let unsigned = data.map(|&value| value as u8);
        // i8 uses every value minus 128; a shift keeps every L2 distance.
        let shift = 128.0;
        let shifted = data.map(|&value| value - shift);
        let signed = shifted.map(|&value| value as i8);
        let metric = Metric::L2;
        let graph = sift_graph(metric);

        let adjacency = build_sift_graph(data.as_view(), &graph, metric);
        let index = searchable_index(data.as_view(), graph.clone(), metric, adjacency);
        assert_exact_top_10(&index, data.as_view(), metric, std::any::type_name::<f32>());

        let adjacency = build_sift_graph(half.as_view(), &graph, metric);
        let index = searchable_index(data.as_view(), graph.clone(), metric, adjacency);
        assert_exact_top_10(&index, data.as_view(), metric, std::any::type_name::<f16>());

        let adjacency = build_sift_graph(unsigned.as_view(), &graph, metric);
        let index = searchable_index(data.as_view(), graph.clone(), metric, adjacency);
        assert_exact_top_10(&index, data.as_view(), metric, std::any::type_name::<u8>());

        let adjacency = build_sift_graph(signed.as_view(), &graph, metric);
        let index = searchable_index(shifted.as_view(), graph, metric, adjacency);
        assert_exact_top_10(
            &index,
            shifted.as_view(),
            metric,
            std::any::type_name::<i8>(),
        );
    }

    #[test]
    fn hash_prune_parameters_must_hold_the_graph_degree() {
        let n_threads = 1;
        let dimensions = nz(1);
        let pool = thread_pool(n_threads);
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
            let graph = graph_policy(degree, Metric::L2);
            let context =
                PiPNNBuildContext::new(partition_policy(), &graph, Metric::L2, dimensions, &pool)
                    .unwrap();
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
        let dimensions = NonZeroUsize::new(data.ncols()).unwrap();
        let degree = 2;
        let graph = graph_policy(degree, Metric::L2);
        let config = PiPNNConfig {
            c_max: nz(3),
            fanout: vec![nz(1)],
            leaf_k: nz(2),
            replicas: nz(2),
            ..partition_policy()
        };
        for (n_threads, final_prune) in [(1, true), (4, true), (4, false)] {
            let pool = thread_pool(n_threads);
            let context =
                PiPNNBuildContext::new(config.clone(), &graph, Metric::L2, dimensions, &pool)
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
                edges(&actual),
                both_directions([(0, 1), (1, 2)]),
                "{n_threads} threads, final_prune={final_prune}"
            );
        }
    }
}

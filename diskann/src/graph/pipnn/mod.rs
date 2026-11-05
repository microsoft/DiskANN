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
//!    each leaf and merges them into one candidate list per point.
//! 3. `finalization` prunes each list that is longer than the graph degree with
//!    Vamana RobustPrune.
//!
//! [`build_graph`] selects the SIMD architecture and the metric once and puts
//! them, with the element type, behind trait objects, so the steps compile
//! once. It reads rows through [`Dataset`] and returns one adjacency list for
//! each point. Each step takes ownership of the previous output and
//! frees it when it returns. The steps run as Rayon jobs in the thread pool
//! of the caller.
//!
//! The build does not load providers or select start points. It also does not
//! quantize, serialize, or search the graph.

mod dataset;
mod finalization;
mod kernels;
mod leaf_build;
mod leaf_kernel;
mod leaf_metric;
mod partition_kernel;
mod partition_metric;
mod partitioning;
mod simd;
mod topk;

use std::{
    num::{NonZeroU16, NonZeroUsize},
    time::Instant,
};

use crate::{
    ANNError, ANNResult,
    graph::{AdjacencyList, config},
    tracked_debug,
    utils::VectorRepr,
};

pub use self::dataset::{Dataset, Element, Rows};

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
pub enum ConfigError {
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
}

crate::convert_error!(ConfigError);

/// Error for a dataset that a build cannot start from.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum DatasetError {
    #[error("PiPNN requires at least one data point")]
    NoPoints,
    #[error("PiPNN requires at least one data dimension")]
    NoDimensions,
    #[error("PiPNN dataset point count ({0}) exceeds the u32 graph ID limit")]
    TooManyPoints(usize),
}

crate::convert_error!(DatasetError);

/// PiPNN build parameters: the partition parameters, the leaf `k`, and the
/// RobustPrune policy of final pruning. The metric belongs to the
/// [`Dataset`], and the occlusion rule of pruning follows the metric.
///
/// [`Config::builder`] checks the parameters once, so every build step can
/// rely on them. The default configuration is `c_max = 512`, `p_samp = 0.005`,
/// `max_leaders = 1000`, `fanout = [10, 3]`, `max_depth = 30` and `leaf_k = 2`.
/// Each point then joins about 30 leaves, the product of the fanout values.
/// Larger `c_max`, fanout values and `leaf_k` all give each point more or
/// nearer candidates, which raises recall and makes the build slower.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    partition: PartitionConfig,
    leaf_k: NonZeroUsize,
    prune: PruneConfig,
}

/// The parameters that the partition step reads.
#[derive(Clone, Debug, PartialEq)]
struct PartitionConfig {
    c_max: NonZeroUsize,
    p_samp: f64,
    max_leaders: NonZeroUsize,
    fanout: Vec<NonZeroUsize>,
    max_depth: NonZeroUsize,
    seed: u64,
}

/// The parameters that final pruning reads.
#[derive(Clone, Debug, PartialEq)]
struct PruneConfig {
    degree: NonZeroUsize,
    alpha: f32,
    max_occlusion_size: NonZeroU16,
}

impl Config {
    /// Start a configuration that prunes each point to `degree` neighbors.
    pub fn builder(degree: NonZeroUsize) -> Builder {
        Builder::new(degree)
    }

    /// Maximum number of points in a leaf.
    pub fn c_max(&self) -> NonZeroUsize {
        self.partition.c_max
    }

    /// Fraction of the points of a cluster that a split samples as leaders.
    pub fn p_samp(&self) -> f64 {
        self.partition.p_samp
    }

    /// Maximum number of leaders in one split.
    pub fn max_leaders(&self) -> NonZeroUsize {
        self.partition.max_leaders
    }

    /// Number of nearest leaders that each point joins, for each split level.
    pub fn fanout(&self) -> &[NonZeroUsize] {
        &self.partition.fanout
    }

    /// Maximum number of split levels.
    pub fn max_depth(&self) -> NonZeroUsize {
        self.partition.max_depth
    }

    /// Seed of leader sampling. A build with the same seed and data samples
    /// the same leaves, so a caller can log it to repeat a build.
    pub fn seed(&self) -> u64 {
        self.partition.seed
    }

    /// Number of nearest neighbors that each point selects inside a leaf.
    pub fn leaf_k(&self) -> NonZeroUsize {
        self.leaf_k
    }

    /// Number of neighbors that final pruning keeps for each point.
    pub fn degree(&self) -> NonZeroUsize {
        self.prune.degree
    }

    /// RobustPrune alpha of final pruning.
    pub fn alpha(&self) -> f32 {
        self.prune.alpha
    }

    /// Number of nearest candidates that final pruning considers for one point.
    pub fn max_occlusion_size(&self) -> NonZeroU16 {
        self.prune.max_occlusion_size
    }
}

/// Default parameters of [`Builder::new`].
mod defaults {
    use std::num::NonZeroUsize;

    pub(super) const C_MAX: NonZeroUsize = NonZeroUsize::new(512).unwrap();
    pub(super) const P_SAMP: f64 = 0.005;
    pub(super) const MAX_LEADERS: NonZeroUsize = NonZeroUsize::new(1000).unwrap();
    pub(super) const FANOUT: [NonZeroUsize; 2] = [
        NonZeroUsize::new(10).unwrap(),
        NonZeroUsize::new(3).unwrap(),
    ];
    pub(super) const MAX_DEPTH: NonZeroUsize = NonZeroUsize::new(30).unwrap();
    pub(super) const LEAF_K: NonZeroUsize = NonZeroUsize::new(2).unwrap();
}

/// Builder for [`Config`].
///
/// Each parameter has a default except the degree. [`Builder::build`] rejects
/// parameters that the field types do not constrain.
#[derive(Clone, Debug)]
pub struct Builder {
    degree: NonZeroUsize,
    c_max: NonZeroUsize,
    p_samp: f64,
    max_leaders: NonZeroUsize,
    fanout: Vec<NonZeroUsize>,
    max_depth: NonZeroUsize,
    leaf_k: NonZeroUsize,
    seed: Option<u64>,
    alpha: f32,
    max_occlusion_size: NonZeroU16,
}

impl Builder {
    /// Start from the defaults for `degree`.
    pub fn new(degree: NonZeroUsize) -> Self {
        Self {
            degree,
            c_max: defaults::C_MAX,
            p_samp: defaults::P_SAMP,
            max_leaders: defaults::MAX_LEADERS,
            fanout: defaults::FANOUT.to_vec(),
            max_depth: defaults::MAX_DEPTH,
            leaf_k: defaults::LEAF_K,
            seed: None,
            alpha: config::defaults::ALPHA,
            max_occlusion_size: config::defaults::MAX_OCCLUSION_SIZE,
        }
    }

    /// Start from the prune policy of `graph`: its pruned degree, alpha and
    /// occlusion size. PiPNN reads no other part of the graph configuration.
    pub fn from_graph(graph: &config::Config) -> Self {
        Self::new(graph.pruned_degree())
            .alpha(graph.alpha())
            .max_occlusion_size(graph.max_occlusion_size_u16())
    }

    /// Maximum number of points in a leaf. A larger leaf holds more of the true
    /// neighbors of its points, so the selected neighbors are nearer. The leaf
    /// distance work grows with the square of the leaf size.
    pub fn c_max(mut self, c_max: NonZeroUsize) -> Self {
        self.c_max = c_max;
        self
    }

    /// Fraction of the points of a cluster that a split samples as leaders, in
    /// `(0, 1]`. More leaders give smaller children and fewer split levels, but
    /// each split costs `points x leaders x dimensions` multiply-adds. A split
    /// samples at least 2 and at most `max_leaders` leaders.
    pub fn p_samp(mut self, p_samp: f64) -> Self {
        self.p_samp = p_samp;
        self
    }

    /// Maximum number of leaders in one split, at least 2. A split costs
    /// `points x leaders x dimensions` multiply-adds, so the cap bounds the
    /// first splits of a large dataset, where `p_samp` alone samples many
    /// leaders: 10M points at `p_samp = 0.005` would give 50,000. The PiPNN
    /// paper uses about 1,000.
    pub fn max_leaders(mut self, max_leaders: NonZeroUsize) -> Self {
        self.max_leaders = max_leaders;
        self
    }

    /// Number of nearest leaders that each point joins, for each split level.
    /// Levels after this schedule use one leader. The schedule must not be empty.
    /// Larger values make leaves overlap more, so each point gets candidates
    /// from more leaves. The number of leaves and the leaf build time grow with
    /// the product of the values.
    /// Put larger values first: a large first fanout sends each point to more
    /// distinct regions of the dataset and gives more distinct candidates.
    pub fn fanout(mut self, fanout: Vec<NonZeroUsize>) -> Self {
        self.fanout = fanout;
        self
    }

    /// Maximum number of split levels. A cluster that is still larger than
    /// `c_max` after `max_depth` splits is cut into equal leaves in point
    /// order, and the build logs a warning with the number of points it cut.
    /// Such leaves group points arbitrarily, which can lower recall. The value
    /// must be at least the number of `fanout` levels.
    pub fn max_depth(mut self, max_depth: NonZeroUsize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Number of nearest neighbors that each point selects inside a leaf. Each
    /// selected pair adds a candidate in both directions. A larger value gives
    /// more candidates and makes final pruning slower.
    pub fn leaf_k(mut self, leaf_k: NonZeroUsize) -> Self {
        self.leaf_k = leaf_k;
        self
    }

    /// Seed of leader sampling. The same seed gives the same leaves. Without
    /// a seed, [`Builder::build`] draws one at random, so each configuration
    /// samples new leaves.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// RobustPrune alpha of final pruning. A larger alpha occludes fewer
    /// candidates and keeps longer edges.
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Number of nearest candidates that final pruning considers for one point,
    /// as Vamana does. RobustPrune addresses candidates with `u16` positions.
    pub fn max_occlusion_size(mut self, max_occlusion_size: NonZeroU16) -> Self {
        self.max_occlusion_size = max_occlusion_size;
        self
    }

    /// Check the parameters that the field types do not constrain and return
    /// the configuration.
    pub fn build(self) -> Result<Config, ConfigError> {
        if !(0.0 < self.p_samp && self.p_samp <= 1.0) {
            return Err(ConfigError::SamplingFraction(self.p_samp));
        }
        if self.fanout.is_empty() {
            return Err(ConfigError::EmptyFanout);
        }
        if self.max_depth.get() < self.fanout.len() {
            return Err(ConfigError::MaxDepth {
                max_depth: self.max_depth.get(),
                fanout_levels: self.fanout.len(),
            });
        }
        if self.max_leaders.get() < 2 {
            return Err(ConfigError::MaxLeaders(self.max_leaders.get()));
        }
        Ok(Config {
            partition: PartitionConfig {
                c_max: self.c_max,
                p_samp: self.p_samp,
                max_leaders: self.max_leaders,
                fanout: self.fanout,
                max_depth: self.max_depth,
                seed: self.seed.unwrap_or_else(rand::random),
            },
            leaf_k: self.leaf_k,
            prune: PruneConfig {
                degree: self.degree,
                alpha: self.alpha,
                max_occlusion_size: self.max_occlusion_size,
            },
        })
    }
}

/// Build one adjacency list for each row of `data`.
///
/// [`Rows`] is the dataset of a matrix of one [`Element`] type. The element
/// type lives behind `data`, and the architecture and metric behind the
/// kernels, so this function and the three steps compile once.
///
/// The build runs as Rayon jobs in the thread pool of the caller: the pool of
/// an enclosing `ThreadPool::install`, or else the global pool.
///
/// The graph links dataset points only; the caller selects start points and
/// serializes the index.
pub fn build_graph(data: &dyn Dataset, config: &Config) -> ANNResult<Vec<AdjacencyList<u32>>> {
    if data.nrows() == 0 {
        return Err(DatasetError::NoPoints.into());
    }
    if data.dimensions() == 0 {
        return Err(DatasetError::NoDimensions.into());
    }
    if data.nrows() > u32::MAX as usize {
        return Err(DatasetError::TooManyPoints(data.nrows()).into());
    }
    let kernels = kernels::select(data.metric());
    let start = Instant::now();
    let leaves = partitioning::partition(data, &*kernels, &config.partition)?;
    tracked_debug!(
        "PiPNN partition: {} leaves in {:.2?}",
        leaves.len(),
        start.elapsed()
    );
    // The leaf step takes the leaves by value, so their memory is free before
    // final pruning starts.
    let start = Instant::now();
    let candidates =
        leaf_build::build_leaf_candidates(data, &*kernels, leaves, config.leaf_k.get())?;
    tracked_debug!("PiPNN leaf build in {:.2?}", start.elapsed());
    // Final pruning cuts each candidate list in place.
    let start = Instant::now();
    let graph = finalization::prune_overfull(candidates, data, &config.prune);
    tracked_debug!("PiPNN final pruning in {:.2?}", start.elapsed());
    Ok(graph)
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
    use diskann_utils::{
        io::read_bin,
        sampling::medoid::ComputeMedoid,
        test_data_root,
        views::rowmajor::{self, Matrix},
    };
    use diskann_vector::DistanceFunction;
    use diskann_vector::distance::Metric;
    use half::f16;
    use std::fs::File;
    use test_support::nz;

    const SIFT: &str = "sift/siftsmall_learn_256pts.fbin";
    const SIFT_NORMALIZED: &str = "sift/siftsmall_learn_256pts_normalized.fbin";

    /// Every point is a leader (`p_samp = 1.0`), so leader sampling picks the
    /// same set for any seed. Alpha one prunes to degree 2 under L2.
    fn partition_policy() -> Builder {
        Config::builder(nz(2))
            .c_max(nz(4))
            .p_samp(1.0)
            .max_leaders(nz(1000))
            .fanout(vec![nz(2)])
            .max_depth(nz(10))
            .leaf_k(nz(1))
            .seed(1290)
            .alpha(1.0)
    }

    fn load_sift(file: &str) -> rowmajor::Owned<f32> {
        read_bin::<f32>(&mut File::open(test_data_root().join(file)).unwrap()).unwrap()
    }

    /// Degree 16 with the default alpha. PiPNN does not read the build search
    /// list size.
    fn sift_graph(metric: Metric) -> config::Config {
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

    /// Degree 16 with the default alpha, like [`sift_graph`]. 256 x 0.05 = 12.8
    /// rounds up to 13 leaders, so the fanout-10 level runs and the leaves overlap.
    fn sift_pipnn_config() -> Config {
        Config::builder(nz(16))
            .c_max(nz(64))
            .p_samp(0.05)
            .max_leaders(nz(1000))
            .fanout(vec![nz(10), nz(3)])
            .max_depth(nz(10))
            .leaf_k(nz(2))
            .seed(1290)
            .build()
            .unwrap()
    }

    /// Load `data` and its PiPNN graph into the test provider. One start node
    /// points to the medoid, as the disk path starts at the medoid; the provider
    /// leaves start nodes out of search results.
    fn searchable_index(
        data: rowmajor::Ref<'_, f32>,
        graph: config::Config,
        metric: Metric,
        adjacency: Vec<AdjacencyList<u32>>,
    ) -> DiskANNIndex<test_provider::Provider> {
        let medoid = f32::compute_medoid(data);
        let medoid_id = data.rows().position(|row| row == medoid).unwrap();
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
            .rows()
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
        data: rowmajor::Ref<'_, f32>,
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

        for (query_id, query) in data.rows().enumerate() {
            let mut expected: Vec<f32> = data
                .rows()
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

    /// Build the graph of `build_data` under `metric`, index the `f32` rows of
    /// `reference` with it, and assert the exact top 10 of every reference row.
    fn assert_build_finds_exact_top_10<T: Element>(
        build_data: rowmajor::Ref<'_, T>,
        reference: rowmajor::Ref<'_, f32>,
        metric: Metric,
        case: &str,
    ) {
        let config = sift_pipnn_config();

        let adjacency = build_graph(&Rows::new(build_data, metric).unwrap(), &config).unwrap();

        let index = searchable_index(reference, sift_graph(metric), metric, adjacency);
        assert_exact_top_10(&index, reference, metric, case);
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
            let error = partition_policy().p_samp(p_samp).build().unwrap_err();

            assert!(
                matches!(error, ConfigError::SamplingFraction(value) if value.to_bits() == p_samp.to_bits()),
                "p_samp {p_samp}: {error}"
            );
        }
    }

    #[test]
    fn sampling_fraction_endpoints_are_accepted() {
        // The smallest positive fraction and the inclusive upper bound are valid.
        let smallest_positive = f64::from_bits(1);
        for p_samp in [smallest_positive, 1.0] {
            let config = partition_policy().p_samp(p_samp).build();

            assert!(config.is_ok(), "p_samp {p_samp}: {config:?}");
        }
    }

    #[test]
    fn a_given_seed_is_kept_and_a_missing_seed_is_drawn() {
        let degree = nz(2);
        let seed = 1290;

        let given = Config::builder(degree).seed(seed).build().unwrap();
        let drawn = Config::builder(degree).build().unwrap();
        let drawn_again = Config::builder(degree).build().unwrap();

        assert_eq!(given.seed(), seed);
        assert_ne!(drawn.seed(), drawn_again.seed());
    }

    #[test]
    fn an_empty_fanout_is_rejected() {
        // An empty fanout list has no first level.
        let actual = partition_policy().fanout(vec![]).build();

        assert_eq!(actual, Err(ConfigError::EmptyFanout));
    }

    #[test]
    fn a_leader_cap_below_two_is_rejected() {
        // A split needs two leaders.
        let actual = partition_policy().max_leaders(nz(1)).build();

        assert_eq!(actual, Err(ConfigError::MaxLeaders(1)));
    }

    #[test]
    fn a_max_depth_below_the_fanout_levels_is_rejected() {
        // Two fanout levels cannot fit within one split level.
        let actual = partition_policy()
            .fanout(vec![nz(10), nz(3)])
            .max_depth(nz(1))
            .build();

        assert_eq!(
            actual,
            Err(ConfigError::MaxDepth {
                max_depth: 1,
                fanout_levels: 2
            })
        );
    }

    #[test]
    fn a_builder_from_a_graph_copies_its_prune_policy() {
        // Values that differ from the PiPNN defaults, so a default would fail.
        let degree = 7;
        let alpha = 1.5;
        let max_occlusion_size = 300;
        let l_build = 64;
        let graph = config::Builder::new_with(
            degree,
            config::MaxDegree::same(),
            l_build,
            Metric::L2.into(),
            |b| {
                b.alpha(alpha).max_occlusion_size(max_occlusion_size);
            },
        )
        .build()
        .unwrap();

        let config = Builder::from_graph(&graph).build().unwrap();

        assert_eq!(config.degree(), nz(degree));
        assert_eq!(config.alpha(), alpha);
        assert_eq!(
            config.max_occlusion_size(),
            NonZeroU16::new(max_occlusion_size as u16).unwrap()
        );
    }

    #[test]
    fn an_empty_dataset_axis_is_rejected_before_building() {
        // Each shape has one empty axis; validation must reject it before a kernel runs.
        let config = partition_policy().build().unwrap();

        for (rows, columns, expected) in [
            (0, 2, DatasetError::NoPoints),
            (2, 0, DatasetError::NoDimensions),
        ] {
            let data = rowmajor::Ref::try_from_data(&[] as &[f32], rows, columns).unwrap();
            let rows = Rows::new(data, Metric::L2).unwrap();

            let error = build_graph(&rows, &config).unwrap_err();

            assert_eq!(
                error.downcast_ref::<DatasetError>(),
                Some(&expected),
                "{error}"
            );
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

            assert_build_finds_exact_top_10(
                data.as_view(),
                data.as_view(),
                metric,
                &format!("{metric:?}"),
            );
        }
    }

    #[test]
    fn every_point_finds_its_exact_top_10_for_each_vector_type() {
        // The f32 rows run in the metric test. These bounds make f16 and u8
        // conversion exact, so every build must find the same distances against
        // the original f32 rows.
        let data = load_sift(SIFT);
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

        assert_build_finds_exact_top_10(half.as_view(), data.as_view(), metric, "f16");
        assert_build_finds_exact_top_10(unsigned.as_view(), data.as_view(), metric, "u8");
        assert_build_finds_exact_top_10(signed.as_view(), shifted.as_view(), metric, "i8");
    }
}

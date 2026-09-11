/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Deterministic overlapping partition construction for PiPNN.
//!
//! A leader is a sampled point that acts as the center of one child partition.
//! A point can join several leaders, so child partitions can overlap.
//!
//! This module owns recursive splitting, leader sampling, row gathering, and
//! assignment scatter. The partition kernel owns GEMM, norm preparation, and
//! local ranking. An `ObjectPool` supplies reusable scratch to Rayon workers.
//!
//! A configured level assigns each point to `fanout[level]` leaders. A deeper
//! level assigns each point to one leader. Each replica uses a different
//! deterministic seed.

use std::collections::HashSet;

use crate::{ANNError, ANNResult, utils::VectorRepr};
use diskann_utils::{
    object_pool::{AsPooled, ObjectPool},
    views::{MatrixView, MutMatrixView},
};
use rand::{SeedableRng, prelude::IndexedRandom};
use rayon::prelude::*;

use super::{
    PiPNNConfig,
    partition_kernel::{PartitionKernelWorkspace, UNASSIGNED_LEADER, assign_leaders},
    partition_metric::PartitionMetric,
    simd::PiPNNSIMDSchema,
};

// These constants control internal batching and deterministic seed generation.
const PARTITION_SEED: u64 = 1_000;
const REPLICA_SEED_STEP: u64 = 7_919;
const LEADER_CAP: usize = 1_000;
const ASSIGNMENT_CACHE_TARGET_BYTES: usize = 524_288;
const MIN_ASSIGNMENT_STRIPE_POINTS: usize = 32;
const MAX_ASSIGNMENT_STRIPE_POINTS: usize = 1_024;
const PARALLEL_SCATTER_MIN_POINTS: usize = 100_000;
const MAX_PARTITION_ITERATIONS: usize = 30;

/// Error from partition shape checks or recursion progress.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum PartitionError {
    #[error("{buffer} shape {rows} x {cols} overflows usize")]
    ShapeOverflow {
        buffer: &'static str,
        rows: usize,
        cols: usize,
    },
    #[error(
        "partition stopped after {limit} iterations with an oversized cluster of size \
         {size} at level {level}"
    )]
    IterationLimit {
        size: usize,
        level: usize,
        limit: usize,
    },
}

struct PendingPartition {
    point_ids: Vec<u32>,
    level: usize,
    seed: u64,
}

struct PartitionSplit {
    pending: Vec<PendingPartition>,
    leaves: Vec<Vec<u32>>,
}

#[derive(Default)]
struct StripeBuffers {
    point_values: Vec<f32>,
    kernel_workspace: PartitionKernelWorkspace,
}

impl AsPooled<()> for StripeBuffers {
    fn create(_: ()) -> Self {
        Self::default()
    }

    fn modify(&mut self, _: ()) {
        // Keep the largest allocation across leases. `assign_point_stripe` defines the
        // active prefix before each read.
    }
}

/// Reusable buffers for point-to-leader assignment.
///
/// `ObjectPool` locks only when it gives or receives a lease. Numerical work
/// holds the lease, not the pool lock.
type StripeBufferPool = ObjectPool<StripeBuffers>;

/// Build overlapping bounded leaves for all configured replicas.
///
/// Each split samples partition centers and assigns every cluster point to its
/// nearest centers. A cluster above `c_max` is split again. A level without a
/// configured fanout assigns each point to one center. Non-rankable distances
/// do not produce assignments.
pub(super) fn partition<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    config: &PiPNNConfig,
) -> ANNResult<Vec<Vec<u32>>>
where
    A: PiPNNSIMDSchema,
    M: PartitionMetric,
    T: VectorRepr + Send + Sync,
{
    let mut leaves = Vec::new();
    let stripe_buffers = StripeBufferPool::new((), 0, None);
    for replica in 0..config.replicas {
        let seed = replica_seed(replica);
        let mut replica_leaves =
            partition_replica::<A, M, T>(arch, data, config, seed, &stripe_buffers)?;
        leaves.append(&mut replica_leaves);
    }
    Ok(leaves)
}

/// Partition one replica until each leaf has at most `c_max` points.
///
/// The function processes one work queue per recursion level. It merges leaves
/// smaller than `c_min` after the queue becomes empty.
fn partition_replica<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    config: &PiPNNConfig,
    seed: u64,
    stripe_buffers: &StripeBufferPool,
) -> ANNResult<Vec<Vec<u32>>>
where
    A: PiPNNSIMDSchema,
    M: PartitionMetric,
    T: VectorRepr + Send + Sync,
{
    let initial_point_ids = (0..data.nrows() as u32).collect();
    if data.nrows() <= config.c_max {
        return Ok(vec![initial_point_ids]);
    }

    let mut leaves = Vec::new();
    let mut pending = vec![PendingPartition {
        point_ids: initial_point_ids,
        level: 0,
        seed,
    }];

    for _ in 0..MAX_PARTITION_ITERATIONS {
        if pending.is_empty() {
            return Ok(merge_undersized_leaves(leaves, config.c_min, config.c_max));
        }

        // Indexed parallel collection preserves parent-partition order.
        #[allow(clippy::disallowed_methods)]
        let splits: ANNResult<Vec<_>> = pending
            .into_par_iter()
            .map(|partition| {
                split_partition::<A, M, T>(arch, data, config, partition, stripe_buffers)
            })
            .collect();

        let mut next_level = Vec::new();
        for mut split in splits? {
            next_level.append(&mut split.pending);
            leaves.append(&mut split.leaves);
        }
        pending = next_level;
    }

    let Some(largest) = pending
        .iter()
        .max_by_key(|partition| partition.point_ids.len())
    else {
        return Ok(merge_undersized_leaves(leaves, config.c_min, config.c_max));
    };
    Err(ANNError::new(PartitionError::IterationLimit {
        size: largest.point_ids.len(),
        level: largest.level,
        limit: MAX_PARTITION_ITERATIONS,
    }))
}

/// Split one oversized cluster into child partitions.
///
/// The function samples center points, assigns the cluster points, and returns
/// bounded leaves separately from child clusters that need another split.
fn split_partition<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    config: &PiPNNConfig,
    partition: PendingPartition,
    stripe_buffers: &StripeBufferPool,
) -> ANNResult<PartitionSplit>
where
    A: PiPNNSIMDSchema,
    M: PartitionMetric,
    T: VectorRepr + Send + Sync,
{
    let split_seed = mix_seed(partition.seed, partition.point_ids.len() as u64);
    let fanout = config.fanout.get(partition.level).copied().unwrap_or(1);
    let leaders = sample_leaders(&partition.point_ids, config.p_samp, split_seed);
    let clusters = assign_to_leaders::<A, M, T>(
        arch,
        data,
        &partition.point_ids,
        &leaders,
        fanout,
        stripe_buffers,
    )?;

    let mut pending = Vec::new();
    let mut leaves = Vec::new();
    for cluster in clusters {
        if cluster.is_empty() {
            continue;
        }
        if cluster.len() <= config.c_max {
            leaves.push(cluster);
        } else {
            pending.push(PendingPartition {
                point_ids: cluster,
                level: partition.level + 1,
                seed: split_seed,
            });
        }
    }
    Ok(PartitionSplit { pending, leaves })
}

/// Sample point IDs that act as centers for one partition split.
fn sample_leaders(points: &[u32], sampling_fraction: f64, seed: u64) -> Vec<u32> {
    let count = sampled_leader_count(points.len(), sampling_fraction);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    points.choose_multiple(&mut rng, count).copied().collect()
}

/// Return the number of centers to sample from one cluster.
///
/// Splitting requires at least two points. The validated sampling fraction is in
/// `(0, 1]`; round up its point count and keep between two and `LEADER_CAP` centers.
fn sampled_leader_count(points: usize, sampling_fraction: f64) -> usize {
    ((points as f64 * sampling_fraction).ceil() as usize).clamp(2, LEADER_CAP)
}

fn replica_seed(replica: usize) -> u64 {
    PARTITION_SEED.wrapping_add((replica as u64).wrapping_mul(REPLICA_SEED_STEP))
}

// This LCG derives child seeds. Wrapping arithmetic gives the same mapping in
// debug and release builds on all supported platforms.
fn mix_seed(seed: u64, salt: u64) -> u64 {
    seed.wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(salt)
}

/// Assign each cluster point to its nearest sampled partition centers.
///
/// The function gathers center vectors once and evaluates points in bounded
/// stripes. The assignment matrix keeps point order. Scatter preserves this order
/// inside each child partition, which makes recursive sampling deterministic.
fn assign_to_leaders<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    point_ids: &[u32],
    leader_ids: &[u32],
    fanout: usize,
    stripe_buffers: &StripeBufferPool,
) -> ANNResult<Vec<Vec<u32>>>
where
    A: PiPNNSIMDSchema,
    M: PartitionMetric,
    T: VectorRepr + Send + Sync,
{
    let dimension_count = data.ncols();
    let leader_values_len = checked_area("leader data", leader_ids.len(), dimension_count)?;
    let mut leader_values = vec![0.0f32; leader_values_len];
    gather_vectors(data, leader_ids, &mut leader_values)?;

    let leader_matrix =
        MatrixView::try_from(leader_values.as_slice(), leader_ids.len(), dimension_count)
            .map_err(|error| ANNError::new(error.as_static()))?;
    let leaders = M::create_leaders(leader_matrix);
    let leader_count = M::leader_count(&leaders);

    let fanout = fanout.min(leader_count);
    let assignment_len = checked_area("partition assignments", point_ids.len(), fanout)?;
    let mut assignments = vec![0u32; assignment_len];
    let stripe_points = assignment_stripe_point_count(leader_count);
    let stripe_assignment_count = stripe_points * fanout;
    let stripe_count = point_ids.len().div_ceil(stripe_points);
    let worker_stripe_count = stripe_count.div_ceil(rayon::current_num_threads());
    let worker_point_count = checked_area("assignment worker", worker_stripe_count, stripe_points)?;
    let worker_assignment_count = checked_area("assignment worker", worker_point_count, fanout)?;

    // Each worker chunk reuses one buffer lease for all its stripes.
    // `build_graph` runs this operation in the pool from the build context.
    #[allow(clippy::disallowed_methods)]
    assignments
        .par_chunks_mut(worker_assignment_count)
        .enumerate()
        .try_for_each(|(worker, worker_assignments)| {
            let mut buffers = stripe_buffers.get_ref(());
            let worker_first = worker * worker_point_count;
            for (stripe, stripe_assignments) in worker_assignments
                .chunks_mut(stripe_assignment_count)
                .enumerate()
            {
                let first_point = worker_first + stripe * stripe_points;
                let stripe_point_count = stripe_assignments.len() / fanout;
                assign_point_stripe::<A, M, T>(
                    arch,
                    data,
                    &point_ids[first_point..first_point + stripe_point_count],
                    &leaders,
                    fanout,
                    &mut buffers,
                    stripe_assignments,
                )?;
            }
            Ok::<(), ANNError>(())
        })?;

    Ok(scatter_assignments(
        point_ids,
        &assignments,
        fanout,
        leader_count,
    ))
}

/// Assign one point stripe to sampled partition centers.
///
/// The function gathers point IDs into a packed `f32` matrix. The partition
/// kernel owns ranking-distance construction and ranking. This function writes the
/// returned leader-column IDs for partition scatter.
#[inline]
fn assign_point_stripe<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    point_ids: &[u32],
    leaders: &M::Leaders<'_>,
    fanout: usize,
    buffers: &mut StripeBuffers,
    assignments: &mut [u32],
) -> ANNResult<()>
where
    A: PiPNNSIMDSchema,
    M: PartitionMetric,
    T: VectorRepr,
{
    let point_count = point_ids.len();
    let dimensions = data.ncols();
    let point_values_len = checked_area("point stripe", point_count, dimensions)?;
    // Keep each buffer at its largest length. Every operation uses an explicit
    // active prefix.
    if buffers.point_values.len() < point_values_len {
        buffers.point_values.resize(point_values_len, 0.0);
    }
    let StripeBuffers {
        point_values,
        kernel_workspace,
    } = buffers;
    let mut points = MutMatrixView::try_from(
        &mut point_values[..point_values_len],
        point_count,
        dimensions,
    )
    .map_err(|error| ANNError::new(error.as_static()))?;
    gather_vectors(data, point_ids, points.as_mut_slice())?;
    let output = MutMatrixView::try_from(assignments, point_count, fanout)
        .map_err(|error| ANNError::new(error.as_static()))?;
    assign_leaders::<A, M>(arch, points.as_view(), leaders, output, kernel_workspace)
}

fn gather_vectors<T>(data: MatrixView<'_, T>, indices: &[u32], output: &mut [f32]) -> ANNResult<()>
where
    T: VectorRepr,
{
    for (&index, vector_output) in indices.iter().zip(output.chunks_exact_mut(data.ncols())) {
        super::conversion::as_f32_into(data.row(index as usize), vector_output)
            .map_err(Into::<ANNError>::into)?;
    }
    Ok(())
}

/// Group assigned point IDs by child partition.
///
/// Both the serial and parallel paths preserve point order inside each child.
/// This order is required for deterministic recursive sampling.
fn scatter_assignments(
    points: &[u32],
    assignments: &[u32],
    fanout: usize,
    leaders: usize,
) -> Vec<Vec<u32>> {
    if points.len() < PARALLEL_SCATTER_MIN_POINTS {
        return scatter_serial(points, assignments, fanout, leaders);
    }

    let stripe_points = points.len().div_ceil(rayon::current_num_threads());
    let stripe_assignment_count = stripe_points * fanout;
    // Indexed parallel collection preserves stripe order.
    #[allow(clippy::disallowed_methods)]
    let locals: Vec<_> = points
        .par_chunks(stripe_points)
        .zip(assignments.par_chunks(stripe_assignment_count))
        .map(|(points, assignments)| scatter_serial(points, assignments, fanout, leaders))
        .collect();

    let mut sizes = vec![0usize; leaders];
    for local in &locals {
        for (size, cluster) in sizes.iter_mut().zip(local) {
            *size += cluster.len();
        }
    }

    // `build_graph` runs this Rayon operation in the pool from the build context.
    // Each worker creates one independent leader cluster.
    #[allow(clippy::disallowed_methods)]
    sizes
        .into_par_iter()
        .enumerate()
        .map(|(leader, size)| {
            let mut cluster = Vec::with_capacity(size);
            for local in &locals {
                cluster.extend_from_slice(&local[leader]);
            }
            cluster
        })
        .collect()
}

fn scatter_serial(
    points: &[u32],
    assignments: &[u32],
    fanout: usize,
    leaders: usize,
) -> Vec<Vec<u32>> {
    let mut sizes = vec![0usize; leaders];
    for &leader in assignments {
        if leader != UNASSIGNED_LEADER {
            sizes[leader as usize] += 1;
        }
    }
    let mut clusters: Vec<Vec<u32>> = sizes.into_iter().map(Vec::with_capacity).collect();
    for (&point, point_assignments) in points.iter().zip(assignments.chunks_exact(fanout)) {
        for &leader in point_assignments {
            if leader != UNASSIGNED_LEADER {
                clusters[leader as usize].push(point);
            }
        }
    }
    clusters
}

/// Merge leaves smaller than `c_min` without exceeding `c_max`.
///
/// A `HashSet` removes duplicate point IDs across merged leaves. The function
/// sorts each merged result before it returns.
fn merge_undersized_leaves(leaves: Vec<Vec<u32>>, c_min: usize, c_max: usize) -> Vec<Vec<u32>> {
    let mut merged = Vec::with_capacity(leaves.len());
    let mut small_leaves = Vec::new();
    for leaf in leaves {
        if leaf.len() >= c_min {
            merged.push(leaf);
        } else {
            small_leaves.push(leaf);
        }
    }
    if small_leaves.is_empty() {
        return merged;
    }

    let mut small = HashSet::with_capacity(c_max);

    for leaf in small_leaves {
        let combined = small.len() + leaf.len();
        if combined > c_max {
            merged.push(drain_sorted(&mut small));
        }
        small.extend(leaf);
        if small.len() >= c_min {
            merged.push(drain_sorted(&mut small));
        }
    }

    if !small.is_empty() {
        let mut remainder = drain_sorted(&mut small);
        if remainder.len() < c_min
            && let Some(last) = merged.last_mut()
        {
            remainder.retain(|id| !last.contains(id));
            let combined = last.len() + remainder.len();
            if combined <= c_max {
                last.append(&mut remainder);
                last.sort_unstable();
            }
        }
        if !remainder.is_empty() {
            merged.push(remainder);
        }
    }

    merged
}

fn drain_sorted(set: &mut HashSet<u32>) -> Vec<u32> {
    let mut values: Vec<_> = set.drain().collect();
    values.sort_unstable();
    values
}

fn checked_area(buffer: &'static str, rows: usize, cols: usize) -> ANNResult<usize> {
    rows.checked_mul(cols)
        .ok_or_else(|| ANNError::new(PartitionError::ShapeOverflow { buffer, rows, cols }))
}

fn assignment_stripe_point_count(leader_count: usize) -> usize {
    let point_count = ASSIGNMENT_CACHE_TARGET_BYTES / (leader_count * size_of::<f32>());
    let point_count = if point_count.is_power_of_two() {
        point_count
    } else {
        point_count.next_power_of_two() / 2
    };
    point_count.clamp(MIN_ASSIGNMENT_STRIPE_POINTS, MAX_ASSIGNMENT_STRIPE_POINTS)
}

#[cfg(test)]
mod tests {
    use diskann_utils::views::{Matrix, MatrixView};
    use diskann_vector::{Half, distance::Metric};
    use diskann_wide::arch::{self, Target1};
    use rstest::rstest;

    use super::super::{Cosine, CosineNormalized, InnerProduct, L2};
    use super::*;

    struct PartitionCall<'a, T> {
        data: MatrixView<'a, T>,
        config: &'a PiPNNConfig,
    }

    struct DispatchPartition(Metric);

    impl<A, T> Target1<A, ANNResult<Vec<Vec<u32>>>, PartitionCall<'_, T>> for DispatchPartition
    where
        A: PiPNNSIMDSchema,
        T: VectorRepr + Send + Sync,
    {
        fn run(self, arch: A, call: PartitionCall<'_, T>) -> ANNResult<Vec<Vec<u32>>> {
            match self.0 {
                Metric::L2 => partition::<A, L2, T>(arch, call.data, call.config),
                Metric::Cosine => partition::<A, Cosine, T>(arch, call.data, call.config),
                Metric::CosineNormalized => {
                    partition::<A, CosineNormalized, T>(arch, call.data, call.config)
                }
                Metric::InnerProduct => {
                    partition::<A, InnerProduct, T>(arch, call.data, call.config)
                }
            }
        }
    }

    fn partition_with_runtime_metric<T>(
        data: MatrixView<'_, T>,
        config: &PiPNNConfig,
        metric: Metric,
    ) -> ANNResult<Vec<Vec<u32>>>
    where
        T: VectorRepr + Send + Sync,
    {
        arch::dispatch1_no_features(DispatchPartition(metric), PartitionCall { data, config })
    }

    fn partition_config(
        c_min: usize,
        c_max: usize,
        fanout: Vec<usize>,
        replicas: usize,
    ) -> PiPNNConfig {
        PiPNNConfig {
            c_max,
            c_min,
            p_samp: 0.25,
            fanout,
            leaf_k: 1,
            replicas,
        }
    }

    fn unit_circle_points(points: usize) -> Matrix<f32> {
        let coordinates: Vec<f32> = (0..points)
            .flat_map(|point| {
                let angle = std::f32::consts::TAU * point as f32 / points as f32;
                [angle.cos(), angle.sin()]
            })
            .collect();
        Matrix::try_from(coordinates.into_boxed_slice(), points, 2).unwrap()
    }

    fn assert_partition_invariants(
        leaves: &[Vec<u32>],
        points: usize,
        c_max: usize,
        replicas: usize,
    ) {
        let mut occurrences = vec![0usize; points];
        for leaf in leaves {
            assert!(!leaf.is_empty() && leaf.len() <= c_max);
            let unique_ids: HashSet<_> = leaf.iter().copied().collect();
            assert_eq!(unique_ids.len(), leaf.len(), "duplicate ID inside a leaf");
            for &id in leaf {
                assert!((id as usize) < points);
                occurrences[id as usize] += 1;
            }
        }
        assert!(occurrences.iter().all(|&count| count >= replicas));
    }

    mod partition_tests {
        use super::*;

        #[rstest]
        #[case::single_point(1)]
        #[case::below_capacity(7)]
        #[case::at_capacity(8)]
        fn one_leaf_contains_all_points_when_no_split_is_needed(#[case] point_count: usize) {
            // Given
            let data = Matrix::new(0.0f32, point_count, 1);
            let config = partition_config(2, 8, vec![2], 1);
            let expected_leaves = vec![(0..point_count as u32).collect::<Vec<_>>()];

            // When
            let leaves =
                partition_with_runtime_metric(data.as_view(), &config, Metric::L2).unwrap();

            // Then
            assert_eq!(leaves, expected_leaves);
        }

        #[test]
        fn each_replica_emits_its_leaf_when_no_split_is_needed() {
            // Given
            let data = MatrixView::column_vector(&[0.0f32, 1.0]);
            let config = partition_config(1, 2, vec![1], 2);
            let expected_leaves = vec![vec![0, 1], vec![0, 1]];

            // When
            let leaves = partition_with_runtime_metric(data, &config, Metric::L2).unwrap();

            // Then
            assert_eq!(leaves, expected_leaves);
        }

        #[test]
        fn leaf_order_and_point_order_are_independent_of_worker_count() {
            // Given: both runs split the same replicas through several levels.
            let data = unit_circle_points(64);
            let config = partition_config(2, 8, vec![3, 2], 2);
            let single_worker = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .unwrap();
            let three_workers = rayon::ThreadPoolBuilder::new()
                .num_threads(3)
                .build()
                .unwrap();

            // When
            let serial = single_worker.install(|| {
                partition_with_runtime_metric(data.as_view(), &config, Metric::L2).unwrap()
            });
            let parallel = three_workers.install(|| {
                partition_with_runtime_metric(data.as_view(), &config, Metric::L2).unwrap()
            });

            // Then: compare the full order; sorting would hide changes to recursive sampling.
            assert_eq!(parallel, serial);
        }

        #[rstest]
        #[case::l2(Metric::L2)]
        #[case::cosine(Metric::Cosine)]
        #[case::normalized_cosine(Metric::CosineNormalized)]
        #[case::inner_product(Metric::InnerProduct)]
        fn bounded_leaves_preserve_point_coverage(#[case] metric: Metric) {
            // Given: distinct unit vectors allow each metric to make progress.
            let data = unit_circle_points(64);
            let config = partition_config(2, 16, vec![3, 2], 3);

            // When
            let leaves = partition_with_runtime_metric(data.as_view(), &config, metric).unwrap();

            // Then
            assert_partition_invariants(&leaves, 64, 16, 3);
        }

        #[test]
        fn identical_points_return_an_iteration_limit_error() {
            // Given: every point ties for the same first leader, so splitting cannot shrink it.
            let data = Matrix::new(1.0f32, 5, 1);
            let config = partition_config(1, 2, vec![1], 1);

            // When
            let error = partition_with_runtime_metric(data.as_view(), &config, Metric::L2)
                .unwrap_err()
                .downcast::<PartitionError>()
                .unwrap();

            // Then
            assert!(matches!(
                error,
                PartitionError::IterationLimit { size: 5, .. }
            ));
        }

        #[test]
        fn no_leaves_are_emitted_when_every_distance_is_non_rankable() {
            // Given: a split is required, but no point can be assigned to any leader.
            let data = Matrix::new(f32::NAN, 3, 1);
            let config = partition_config(1, 2, vec![1], 1);

            // When
            let leaves =
                partition_with_runtime_metric(data.as_view(), &config, Metric::L2).unwrap();

            // Then
            assert!(leaves.is_empty());
        }
    }

    #[test]
    fn split_assigns_one_leader_after_the_fanout_schedule_ends() {
        // Given: all points are leaders; at level one the only configured fanout is exhausted.
        let data = MatrixView::column_vector(&[0.0f32, 10.0, 20.0, 30.0]);
        let config = PiPNNConfig {
            p_samp: 1.0,
            ..partition_config(1, 2, vec![2], 1)
        };
        let parent = PendingPartition {
            point_ids: vec![0, 1, 2, 3],
            level: 1,
            seed: PARTITION_SEED,
        };
        let expected_leaves = vec![vec![0], vec![1], vec![2], vec![3]];

        // When: each point's unique nearest leader is itself.
        let mut split = split_partition::<_, L2, _>(
            diskann_wide::ARCH,
            data,
            &config,
            parent,
            &StripeBufferPool::new((), 0, None),
        )
        .unwrap();
        split.leaves.sort();

        // Then
        assert!(split.pending.is_empty());
        assert_eq!(split.leaves, expected_leaves);
    }

    mod assign_to_leaders_tests {
        use super::*;

        #[test]
        fn clusters_contain_global_point_ids_in_input_order() {
            // Given: the leader-column order differs from global ID order.
            let data = MatrixView::column_vector(&[0.0f32, 100.0, 4.0, 96.0]);
            let points = [3, 2, 1];
            let leaders = [1, 0];
            // Points 3 and 1 are nearest coordinate 100; point 2 is nearest coordinate 0.
            let expected_clusters = vec![vec![3, 1], vec![2]];

            // When
            let clusters = assign_to_leaders::<_, L2, _>(
                diskann_wide::ARCH,
                data,
                &points,
                &leaders,
                1,
                &StripeBufferPool::new((), 0, None),
            )
            .unwrap();

            // Then
            assert_eq!(clusters, expected_clusters);
        }

        #[test]
        fn fanout_larger_than_the_leader_set_assigns_every_leader_once() {
            // Given
            let data = MatrixView::column_vector(&[0.0f32, 10.0, 4.0]);
            let expected_clusters = vec![vec![2], vec![2]];

            // When
            let clusters = assign_to_leaders::<_, L2, _>(
                diskann_wide::ARCH,
                data,
                &[2],
                &[0, 1],
                3,
                &StripeBufferPool::new((), 0, None),
            )
            .unwrap();

            // Then
            assert_eq!(clusters, expected_clusters);
        }

        #[test]
        fn points_without_rankable_distances_are_omitted() {
            // Given: the NaN point has no leader assignments; the finite point has two.
            let data = MatrixView::column_vector(&[0.0f32, 10.0, f32::NAN, 1.0]);
            let expected_clusters = vec![vec![3], vec![3]];

            // When
            let clusters = assign_to_leaders::<_, L2, _>(
                diskann_wide::ARCH,
                data,
                &[2, 3],
                &[0, 1],
                2,
                &StripeBufferPool::new((), 0, None),
            )
            .unwrap();

            // Then
            assert_eq!(clusters, expected_clusters);
        }

        #[rstest]
        fn partial_final_stripe_preserves_point_assignments(#[values(1, 2)] worker_count: usize) {
            // Given: three stripes, the last containing one point. One worker reuses its lease.
            let point_count = 2 * MAX_ASSIGNMENT_STRIPE_POINTS + 1;
            let data: Vec<f32> = (0..point_count).map(|point| point as f32).collect();
            let points: Vec<u32> = (0..point_count as u32).collect();
            let last_point = point_count as u32 - 1;
            let leaders = [0, last_point];
            // On the line [0, last_point], the midpoint ties and selects leader column 0.
            let midpoint = last_point / 2;
            let expected_clusters = vec![
                (0..=midpoint).collect::<Vec<_>>(),
                (midpoint + 1..=last_point).collect(),
            ];
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(worker_count)
                .build()
                .unwrap();

            // When
            let clusters = pool.install(|| {
                assign_to_leaders::<_, L2, _>(
                    diskann_wide::ARCH,
                    MatrixView::column_vector(&data),
                    &points,
                    &leaders,
                    1,
                    &StripeBufferPool::new((), 0, None),
                )
                .unwrap()
            });

            // Then
            assert_eq!(clusters, expected_clusters);
        }
    }

    #[rstest]
    #[case::f32([-2.0f32, 1.0, 3.0, -4.0, 5.0, 6.0], [5.0, 6.0, -2.0, 1.0, 5.0, 6.0])]
    #[case::f16([-2.0f32, 1.0, 3.0, -4.0, 5.0, 6.0].map(Half::from_f32), [5.0, 6.0, -2.0, 1.0, 5.0, 6.0])]
    #[case::i8([-2i8, 1, 3, -4, 5, 6], [5.0, 6.0, -2.0, 1.0, 5.0, 6.0])]
    #[case::u8([2u8, 1, 3, 4, 5, 6], [5.0, 6.0, 2.0, 1.0, 5.0, 6.0])]
    fn gather_converts_rows_in_requested_id_order<T: VectorRepr>(
        #[case] values: [T; 6],
        #[case] expected: [f32; 6],
    ) {
        // Given: request the final row, then the first row, then repeat the final row.
        let data = MatrixView::try_from(values.as_slice(), 3, 2).unwrap();
        let mut gathered = [f32::NAN; 6];

        // When
        gather_vectors(data, &[2, 0, 2], &mut gathered).unwrap();

        // Then
        assert_eq!(gathered, expected);
    }

    mod merge_undersized_leaves_tests {
        use super::*;

        #[test]
        fn total_below_minimum_is_kept_as_one_leaf() {
            // Given: all available points together still fall below the minimum.
            let leaves = vec![vec![0], vec![1]];
            let expected_leaves = vec![vec![0, 1]];

            // When
            let merged = merge_undersized_leaves(leaves, 3, 4);

            // Then
            assert_eq!(merged, expected_leaves);
        }

        #[test]
        fn leaves_already_at_minimum_keep_their_order() {
            // Given
            let leaves = vec![vec![3, 1, 2], vec![7, 5, 6]];
            let expected_leaves = leaves.clone();

            // When
            let merged = merge_undersized_leaves(leaves, 3, 4);

            // Then
            assert_eq!(merged, expected_leaves);
        }

        #[test]
        fn merged_leaves_contain_sorted_unique_ids() {
            // Given: the shared ID 1 counts only once toward the minimum of three.
            let leaves = vec![vec![3, 1], vec![1, 2]];
            let expected_leaves = vec![vec![1, 2, 3]];

            // When
            let merged = merge_undersized_leaves(leaves, 3, 4);

            // Then
            assert_eq!(merged, expected_leaves);
        }

        #[test]
        fn full_batch_is_emitted_before_another_leaf_would_exceed_capacity() {
            // Given: three pairs cannot fit in a five-point leaf; the first two can.
            let leaves = vec![vec![0, 1], vec![2, 3], vec![4, 5]];
            let expected_leaves = vec![vec![0, 1, 2, 3], vec![4, 5]];

            // When
            let merged = merge_undersized_leaves(leaves, 5, 5);

            // Then
            assert_eq!(merged, expected_leaves);
        }

        #[test]
        fn leaves_that_exactly_fill_capacity_are_combined() {
            // Given
            let leaves = vec![vec![0, 1], vec![2, 3]];
            let expected_leaves = vec![vec![0, 1, 2, 3]];

            // When
            let merged = merge_undersized_leaves(leaves, 4, 4);

            // Then
            assert_eq!(merged, expected_leaves);
        }

        #[test]
        fn overlapping_remainder_fits_after_duplicate_ids_are_removed() {
            // Given: raw lengths total five, but the union has four IDs and fits.
            let leaves = vec![vec![0, 1, 2], vec![2, 3]];
            let expected_leaves = vec![vec![0, 1, 2, 3]];

            // When
            let merged = merge_undersized_leaves(leaves, 3, 4);

            // Then
            assert_eq!(merged, expected_leaves);
        }

        #[test]
        fn new_remainder_ids_form_a_separate_leaf_when_the_last_leaf_is_full() {
            // Given: ID 3 already exists, but the new ID 4 cannot fit in the full leaf.
            let leaves = vec![vec![0, 1, 2, 3], vec![3, 4]];
            let expected_leaves = vec![vec![0, 1, 2, 3], vec![4]];

            // When
            let merged = merge_undersized_leaves(leaves, 3, 4);

            // Then
            assert_eq!(merged, expected_leaves);
        }

        #[test]
        fn entirely_overlapping_remainder_adds_no_leaf() {
            // Given: every remainder ID is already present in the last leaf.
            let leaves = vec![vec![0, 1, 2], vec![1, 2]];
            let expected_leaves = vec![vec![0, 1, 2]];

            // When
            let merged = merge_undersized_leaves(leaves, 3, 4);

            // Then
            assert_eq!(merged, expected_leaves);
        }
    }

    #[rstest]
    #[case::minimum_two_leaders(10, 0.01, 2)]
    #[case::fraction_rounds_up(10, 0.25, 3)]
    #[case::leader_cap(50_000, 1.0, LEADER_CAP)]
    fn sampled_leader_count_respects_sampling_bounds(
        #[case] point_count: usize,
        #[case] sampling_fraction: f64,
        #[case] expected_leader_count: usize,
    ) {
        // When
        let count = sampled_leader_count(point_count, sampling_fraction);

        // Then
        assert_eq!(count, expected_leader_count);
    }

    mod scatter_assignments_tests {
        use super::*;

        #[test]
        fn unassigned_slots_are_skipped_without_reordering_points() {
            // Given: each assignment pair belongs to one point; leader 2 has no points.
            let points = [30, 10, 20];
            let assignments = [1, UNASSIGNED_LEADER, 0, 1, UNASSIGNED_LEADER, 0];
            let expected_clusters = vec![vec![10, 20], vec![30, 10], vec![]];

            // When
            let clusters = scatter_assignments(&points, &assignments, 2, 3);

            // Then
            assert_eq!(clusters, expected_clusters);
        }

        #[rstest]
        fn parallel_stripes_preserve_order_and_skip_unassigned_slots(
            #[values(1, 3)] worker_count: usize,
        ) {
            // Given: three repeating assignments, with a partial final worker chunk.
            let points: Vec<u32> = (0..(PARALLEL_SCATTER_MIN_POINTS as u32 + 1))
                .rev()
                .collect();
            let pattern = [0, UNASSIGNED_LEADER, 1, 0, UNASSIGNED_LEADER, 1];
            let assignments: Vec<u32> =
                pattern.into_iter().cycle().take(2 * points.len()).collect();
            // Leader 0 receives rows 0 and 1 of each triple; leader 1 receives rows 1 and 2.
            let expected_clusters = vec![
                points
                    .iter()
                    .enumerate()
                    .filter(|(row, _)| row % 3 != 2)
                    .map(|(_, &id)| id)
                    .collect::<Vec<_>>(),
                points
                    .iter()
                    .enumerate()
                    .filter(|(row, _)| row % 3 != 0)
                    .map(|(_, &id)| id)
                    .collect(),
                vec![],
            ];
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(worker_count)
                .build()
                .unwrap();

            // When
            let clusters = pool.install(|| scatter_assignments(&points, &assignments, 2, 3));

            // Then
            assert_eq!(clusters, expected_clusters);
        }
    }

    #[test]
    fn overflowing_buffer_shape_returns_an_error() {
        // Given: this shape cannot be represented by an allocation length.
        let rows = usize::MAX;
        let columns = 2;

        // When
        let error = checked_area("partition assignments", rows, columns).unwrap_err();

        // Then
        assert!(matches!(
            error.downcast::<PartitionError>().unwrap(),
            PartitionError::ShapeOverflow { .. }
        ));
    }
}

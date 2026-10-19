/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Randomized ball carving: recursive, overlapping partitioning of the dataset
//! into leaves of at most `c_max` points.
//!
//! One split samples leaders from a cluster and assigns each point to its
//! `fanout` nearest leaders. The points of one leader form a child cluster, so
//! children overlap when `fanout > 1`. A child with at most `c_max` points is a
//! leaf; a larger child is split again at the next level. `fanout[level]` gives
//! the fanout of each level, and levels after the schedule use fanout 1. Each
//! replica repeats the process with a different seed.
//!
//! The partition kernel computes point-to-leader distances and selects the
//! nearest leaders. This module samples leaders, gathers vectors, and groups the
//! assignments into children.
//!
//! Buffer sizes use plain multiplication. The IDs of one split are distinct
//! dataset rows, so a gathered `rows x dimensions` buffer is not larger than the
//! dataset. An assignment buffer holds `points x fanout` IDs. With at most
//! `u32::MAX` points and `LEADER_CAP` leaders, this count fits in a 64-bit
//! `usize`.

use crate::{ANNResult, utils::VectorRepr};
use diskann_utils::{
    object_pool::{AsPooled, ObjectPool},
    views::{Matrix, MatrixView, MutMatrixView},
};
use rand::{Rng, SeedableRng, prelude::IndexedRandom, rngs::StdRng};
use rayon::prelude::*;

use super::{
    PiPNNConfig,
    conversion::gather_as_f32,
    partition_kernel::{PartitionKernelWorkspace, assign_leaders},
    partition_metric::PartitionMetric,
    simd::Simd,
    topk::UNASSIGNED,
};

/// Maximum number of leaders in one split. The PiPNN paper uses the same hard
/// cap ("typically 1000", Section 4.1). A split costs
/// `points x leaders x dimensions` multiply-adds. Without the cap, the first
/// split of 10M points at `p_samp = 0.005` samples 50,000 leaders and does 50
/// times the work.
const LEADER_CAP: usize = 1_000;
/// Number of points in one assignment stripe. A stripe gathers its points as
/// `f32` and computes their distances to all leaders, so it moves
/// `points x (leaders + dimensions) x 4` bytes through the cache.
///
/// Each stripe repeats fixed work, such as one GEMM call over the leader matrix,
/// so 16-point stripes took 1.3 to 2 times as long as the best height. Tall
/// stripes slow down when the stripes of all workers exceed the last-level
/// cache. The L2 cache is not the limit: one worker ran 16 MiB stripes at full
/// speed on a CPU with 512 KiB of L2 per core.
///
/// Sweeps on Zen 3, Zen 4, Emerald Rapids and Ice Lake CPUs covered f32, f16 and
/// u8 data, 128 to 1,536 dimensions, 64 to 1,000 leaders and 1 to 16 workers.
/// 256 points stayed within 11% of the best height, and within 1.4% on average.
/// With 8 leaders and 1,536 dimensions, where the gather dominates, the loss was
/// up to 27%. A rule from the detected cache size did not do better, because a
/// virtual machine reports a last-level cache that other tenants also use.
const ASSIGNMENT_STRIPE_POINTS: usize = 256;
/// Clusters with at least this many points group their points by leader in
/// parallel. Such clusters occur at the first levels, where few splits run at
/// the same time. Serial grouping of 10M points at fanout 10 takes 0.59 s, and
/// parallel grouping takes 0.15 s on 16 threads (AMD EPYC 7763). The serial time
/// does not decrease with more threads, so its share of the build increases on
/// larger machines. Below this size, serial grouping takes a few milliseconds.
const PARALLEL_GROUPING_MIN_POINTS: usize = 100_000;

/// A cluster with more than `c_max` points that waits for its split.
struct Cluster {
    point_ids: Vec<u32>,
    /// Position in the fanout schedule.
    level: usize,
    /// Seed of the random stream of this cluster's split. The parent split
    /// draws it, so every split samples its leaders from its own stream.
    seed: u64,
}

/// The result of one split: leaves, and clusters for the next level.
struct ClusterSplit {
    next_level: Vec<Cluster>,
    leaves: Vec<Vec<u32>>,
}

/// Scratch for one assignment stripe: the gathered `f32` points and the kernel
/// workspace.
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
        // Keep the largest allocation across leases. `assign_point_stripe` sets
        // the active prefix before each read.
    }
}

/// Stripe scratch shared by all splits of a build.
///
/// `ObjectPool` locks only to give or take back a lease. Numerical work holds the
/// lease, not the pool lock.
type StripeBufferPool = ObjectPool<StripeBuffers>;

/// Build the leaves of all replicas.
///
/// A point whose distances to all leaders of a split are not rankable, for
/// example a NaN vector, joins no child of that split.
pub(super) fn partition<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    config: &PiPNNConfig,
) -> ANNResult<Vec<Vec<u32>>>
where
    A: Simd,
    M: PartitionMetric,
    T: VectorRepr + Send + Sync,
{
    let mut leaves = Vec::new();
    let stripe_buffers = StripeBufferPool::new((), 0, None);
    // Replica `r` uses seed `base + r`. `StdRng::seed_from_u64` gives
    // independent streams for adjacent seeds.
    let base_seed = config.seed.unwrap_or_else(rand::random);
    for replica in 0..config.replicas.get() {
        let seed = base_seed.wrapping_add(replica as u64);
        let mut replica_leaves =
            partition_replica::<A, M, T>(arch, data, config, seed, &stripe_buffers)?;
        leaves.append(&mut replica_leaves);
    }
    Ok(leaves)
}

/// Split one replica, one level at a time, until every cluster is a leaf.
///
/// The loop ends. The fanout schedule is finite, and each split sends its
/// clusters one level deeper. After the schedule, fanout is 1, so each point
/// joins one child. A child is then smaller than its parent, or it holds every
/// point and [`split_cluster`] cuts it into leaves.
fn partition_replica<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    config: &PiPNNConfig,
    seed: u64,
    stripe_buffers: &StripeBufferPool,
) -> ANNResult<Vec<Vec<u32>>>
where
    A: Simd,
    M: PartitionMetric,
    T: VectorRepr + Send + Sync,
{
    let initial_point_ids = (0..data.nrows() as u32).collect();
    let c_max = config.c_max.get();
    if data.nrows() <= c_max {
        return Ok(vec![initial_point_ids]);
    }

    let mut leaves = Vec::new();
    let mut pending = vec![Cluster {
        point_ids: initial_point_ids,
        level: 0,
        seed,
    }];
    while !pending.is_empty() {
        // Indexed parallel collection keeps the parent order.
        let splits: ANNResult<Vec<_>> = pending
            .into_par_iter()
            .map(|cluster| split_cluster::<A, M, T>(arch, data, config, cluster, stripe_buffers))
            .collect();

        let mut next_level = Vec::new();
        for mut split in splits? {
            next_level.append(&mut split.next_level);
            leaves.append(&mut split.leaves);
        }
        pending = next_level;
    }
    Ok(leaves)
}

/// Split one cluster into leaves and clusters for the next level.
fn split_cluster<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    config: &PiPNNConfig,
    cluster: Cluster,
    stripe_buffers: &StripeBufferPool,
) -> ANNResult<ClusterSplit>
where
    A: Simd,
    M: PartitionMetric,
    T: VectorRepr + Send + Sync,
{
    let mut rng = StdRng::seed_from_u64(cluster.seed);
    let fanout = config
        .fanout
        .get(cluster.level)
        .map_or(1, |fanout| fanout.get());
    let c_max = config.c_max.get();
    let leader_count = sampled_leader_count(cluster.point_ids.len(), config.p_samp);
    if fanout > 1 && fanout >= leader_count {
        // Every point would join every leader, and each child would repeat this
        // cluster. Split the cluster at the next level instead.
        return Ok(ClusterSplit {
            next_level: vec![Cluster {
                point_ids: cluster.point_ids,
                level: cluster.level + 1,
                seed: rng.random(),
            }],
            leaves: Vec::new(),
        });
    }

    let leaders = sample_leaders(&cluster.point_ids, leader_count, &mut rng);
    let children = assign_to_leaders::<A, M, T>(
        arch,
        data,
        &cluster.point_ids,
        &leaders,
        fanout,
        stripe_buffers,
    )?;

    let mut next_level = Vec::new();
    let mut leaves = Vec::new();
    for child in children {
        if child.is_empty() {
            continue;
        }
        if child.len() <= c_max {
            leaves.push(child);
        } else if fanout == 1 && child.len() == cluster.point_ids.len() {
            // Every point chose this leader, so the split made no progress. This
            // happens when all sampled leaders are copies of one vector: each
            // point is equally near to all of them. Another split can stall the
            // same way, so cut the child into equal leaves in point order. The
            // cut groups points arbitrarily, which can lower recall but keeps
            // the graph valid.
            let leaf_len = child.len().div_ceil(child.len().div_ceil(c_max));
            leaves.extend(child.chunks(leaf_len).map(<[u32]>::to_vec));
        } else {
            next_level.push(Cluster {
                point_ids: child,
                level: cluster.level + 1,
                seed: rng.random(),
            });
        }
    }
    Ok(ClusterSplit { next_level, leaves })
}

/// Sample `count` distinct leaders from the points of a cluster.
fn sample_leaders(points: &[u32], count: usize, rng: &mut StdRng) -> Vec<u32> {
    points.choose_multiple(rng, count).copied().collect()
}

/// Return the number of leaders for a cluster of `points` points.
///
/// A split needs two leaders. `LEADER_CAP` bounds the cost of one split.
fn sampled_leader_count(points: usize, sampling_fraction: f64) -> usize {
    ((points as f64 * sampling_fraction).ceil() as usize).clamp(2, LEADER_CAP)
}

/// Assign each point to its `fanout` nearest leaders and group the points by
/// leader.
///
/// Stripes of points run in parallel. Each child keeps the input order of its
/// points, so the next split samples the same leaders for any thread count.
fn assign_to_leaders<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    point_ids: &[u32],
    leader_ids: &[u32],
    fanout: usize,
    stripe_buffers: &StripeBufferPool,
) -> ANNResult<Vec<Vec<u32>>>
where
    A: Simd,
    M: PartitionMetric,
    T: VectorRepr + Send + Sync,
{
    let mut leader_values = Matrix::new(0.0f32, leader_ids.len(), data.ncols());
    gather_as_f32(data, leader_ids, leader_values.as_mut_slice())?;

    let leaders = M::create_leaders(leader_values.as_view());
    let leader_count = M::leader_count(&leaders);

    let mut assignments = Matrix::new(0u32, point_ids.len(), fanout);
    // This runs in the pool of the build context (see `build_graph`). A lease
    // for each stripe takes one lock, which costs far less than the GEMM and
    // ranking of the stripe.
    assignments
        .par_window_iter_mut(ASSIGNMENT_STRIPE_POINTS)
        .zip(point_ids.par_chunks(ASSIGNMENT_STRIPE_POINTS))
        .try_for_each(|(stripe_assignments, stripe_ids)| {
            assign_point_stripe::<A, M, T>(
                arch,
                data,
                stripe_ids,
                &leaders,
                &mut stripe_buffers.get_ref(()),
                stripe_assignments,
            )
        })?;

    Ok(group_by_leader(
        point_ids,
        assignments.as_view(),
        leader_count,
    ))
}

/// Gather one stripe of points as `f32` and write the leader columns of the
/// nearest leaders of each point to its row of `assignments`, one per column.
#[inline]
fn assign_point_stripe<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    point_ids: &[u32],
    leaders: &M::Leaders<'_>,
    buffers: &mut StripeBuffers,
    assignments: MutMatrixView<'_, u32>,
) -> ANNResult<()>
where
    A: Simd,
    M: PartitionMetric,
    T: VectorRepr,
{
    let point_count = point_ids.len();
    let dimensions = data.ncols();
    let point_values_len = point_count * dimensions;
    // The buffer keeps its largest length. This stripe uses a prefix of it.
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
    )?;
    gather_as_f32(data, point_ids, points.as_mut_slice())?;
    assign_leaders::<A, M>(
        arch,
        points.as_view(),
        leaders,
        assignments,
        kernel_workspace,
    )
}

/// Group the points by assigned leader. Unassigned slots join no child.
///
/// Both paths keep the input order of the points inside each child, because the
/// next split samples its leaders by position.
fn group_by_leader(
    points: &[u32],
    assignments: MatrixView<'_, u32>,
    leaders: usize,
) -> Vec<Vec<u32>> {
    if points.len() < PARALLEL_GROUPING_MIN_POINTS {
        return group_by_leader_serial(points, assignments, leaders);
    }

    let chunk_points = points.len().div_ceil(rayon::current_num_threads());
    // Group each chunk of points on its own. Indexed collection keeps the chunk
    // order.
    let locals: Vec<_> = points
        .par_chunks(chunk_points)
        .zip(assignments.par_window_iter(chunk_points))
        .map(|(points, assignments)| group_by_leader_serial(points, assignments, leaders))
        .collect();

    let mut sizes = vec![0usize; leaders];
    for local in &locals {
        for (size, cluster) in sizes.iter_mut().zip(local) {
            *size += cluster.len();
        }
    }

    // Join the chunk parts of each child in chunk order.
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

fn group_by_leader_serial(
    points: &[u32],
    assignments: MatrixView<'_, u32>,
    leaders: usize,
) -> Vec<Vec<u32>> {
    let mut sizes = vec![0usize; leaders];
    for &leader in assignments.as_slice() {
        if leader != UNASSIGNED {
            sizes[leader as usize] += 1;
        }
    }
    let mut clusters: Vec<Vec<u32>> = sizes.into_iter().map(Vec::with_capacity).collect();
    for (&point, point_assignments) in points.iter().zip(assignments.row_iter()) {
        for &leader in point_assignments {
            if leader != UNASSIGNED {
                clusters[leader as usize].push(point);
            }
        }
    }
    clusters
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::pipnn::{
        L2,
        test_support::{self, dense_points, nz, sorted_members_per_row, thread_pool},
    };
    use diskann_vector::distance::Metric;
    use diskann_wide::ARCH;
    use std::collections::HashSet;

    // Neither leaf order nor point order within a leaf is part of the result.
    // Preserve repeated leaves and IDs so comparison still detects duplicates.
    fn sorted_leaf_memberships(leaves: &[Vec<u32>]) -> Vec<Vec<u32>> {
        let mut leaves = sorted_members_per_row(leaves);
        leaves.sort_unstable();
        leaves
    }

    fn splitting_config() -> PiPNNConfig {
        PiPNNConfig {
            c_max: nz(2),
            p_samp: 1.0,
            fanout: vec![nz(2)],
            leaf_k: nz(1),
            replicas: nz(1),
            seed: Some(1290),
        }
    }

    fn split<M: PartitionMetric>(
        data: MatrixView<'_, f32>,
        config: &PiPNNConfig,
        parent: Cluster,
    ) -> ClusterSplit {
        let buffers = StripeBufferPool::new((), 0, None);
        thread_pool(2)
            .install(|| split_cluster::<_, M, _>(ARCH, data, config, parent, &buffers))
            .unwrap()
    }

    #[test]
    fn sampled_leaders_are_a_repeatable_subset_without_replacement() {
        // (points, sampling fraction, expected leaders)
        for (point_count, fraction, expected_count) in [
            (7, 0.01, 2), // A split needs two leaders.
            (7, 0.31, 3), // 2.17 rounds up.
            (7, 1.0, 7),
            (999, 1.0, 999),
            (1000, 1.0, LEADER_CAP),
            (1001, 1.0, LEADER_CAP),
        ] {
            let case = format!("{point_count} points, fraction {fraction}");
            let ids: Vec<_> = (0..point_count).map(|id| 10 + 3 * id).collect();

            let count = sampled_leader_count(ids.len(), fraction);
            let mut leaders = sample_leaders(&ids, count, &mut StdRng::seed_from_u64(1290));

            assert_eq!(leaders.len(), expected_count, "{case}");
            let distinct: HashSet<_> = leaders.iter().collect();
            assert_eq!(distinct.len(), expected_count, "{case}");
            assert!(leaders.iter().all(|id| ids.contains(id)), "{case}");
            let mut repeated = sample_leaders(&ids, count, &mut StdRng::seed_from_u64(1290));
            leaders.sort_unstable();
            repeated.sort_unstable();
            assert_eq!(leaders, repeated, "{case}");
        }
    }

    #[test]
    fn points_join_their_nearest_leaders() {
        // Leader columns are global IDs [2, 0], at x=10 and x=0.
        // The requested points [5, 4, 1] are at x=8, x=2 and x=100.
        let values = [0.0_f32, 100.0, 10.0, -100.0, 2.0, 8.0];
        let data = MatrixView::try_from(&values[..], 6, 1).unwrap();
        let buffers = StripeBufferPool::new((), 0, None);
        let pool = thread_pool(2);

        for (fanout, expected) in [
            (1, [vec![5, 1], vec![4]]),
            (2, [vec![5, 4, 1], vec![5, 4, 1]]),
        ] {
            let actual = pool
                .install(|| {
                    assign_to_leaders::<_, L2, _>(ARCH, data, &[5, 4, 1], &[2, 0], fanout, &buffers)
                })
                .unwrap();

            assert_eq!(
                sorted_members_per_row(&actual),
                sorted_members_per_row(&expected),
                "fanout {fanout}"
            );
        }
    }

    #[test]
    fn multiple_stripes_and_their_partial_tail_preserve_all_assignments() {
        // Two full stripes, with three points left over.
        // Each dense row selects two of the three centers at 10, 0 and 4.
        // Every row selects 4; its other center is 10 for x=10 and 0 otherwise.
        let point_count = 2 * ASSIGNMENT_STRIPE_POINTS + 3;
        let dimensions = 384;
        let values: Vec<_> = (0..point_count)
            .flat_map(|point| std::iter::repeat_n([0.0_f32, 4.0, 10.0][point % 3], dimensions))
            .collect();
        let data = MatrixView::try_from(values.as_slice(), point_count, dimensions).unwrap();
        let ids: Vec<_> = (0..point_count as u32).rev().collect();
        let expected = [
            ids.iter().copied().filter(|id| id % 3 == 2).collect(),
            ids.iter().copied().filter(|id| id % 3 != 2).collect(),
            ids.clone(),
        ];
        let buffers = StripeBufferPool::new((), 0, None);

        for workers in [1, 3] {
            let actual = thread_pool(workers)
                .install(|| {
                    assign_to_leaders::<_, L2, _>(ARCH, data, &ids, &[2, 0, 1], 2, &buffers)
                })
                .unwrap();

            assert_eq!(
                sorted_members_per_row(&actual),
                sorted_members_per_row(&expected),
                "{workers} workers"
            );
        }
    }

    #[test]
    fn stripes_assign_each_point_to_its_nearest_leaders() {
        // 3,000 points fill 11 stripes and a partial tail on several workers.
        // 300 leaders fill whole SIMD leader groups.
        // Dense fixtures keep the f32 scores exact, so the check needs no
        // tolerance. It accepts any order of tied leaders.
        let (point_count, dimensions, fanout) = (3000, 32, 3);
        let values = dense_points(point_count, dimensions, 1290);
        let data = MatrixView::try_from(values.as_slice(), point_count, dimensions).unwrap();
        let row = |id: u32| &values[id as usize * dimensions..][..dimensions];
        let ids: Vec<u32> = (0..point_count as u32).rev().collect();
        let leaders: Vec<u32> = (0..point_count as u32).step_by(10).collect();
        let buffers = StripeBufferPool::new((), 0, None);

        let clusters = thread_pool(4)
            .install(|| assign_to_leaders::<_, L2, _>(ARCH, data, &ids, &leaders, fanout, &buffers))
            .unwrap();

        let mut chosen = vec![Vec::new(); point_count];
        for (column, cluster) in clusters.iter().enumerate() {
            // `ids` is descending, and grouping keeps the input order.
            assert!(cluster.is_sorted_by(|a, b| a > b), "leader {column}");
            for &id in cluster {
                chosen[id as usize].push(column);
            }
        }
        for (id, columns) in chosen.iter().enumerate() {
            let distance = |column: usize| {
                test_support::distance(Metric::L2, row(id as u32), row(leaders[column]))
            };
            let farthest_chosen = columns
                .iter()
                .map(|&column| distance(column))
                .fold(f64::MIN, f64::max);
            let nearest_other = (0..leaders.len())
                .filter(|column| !columns.contains(column))
                .map(distance)
                .fold(f64::INFINITY, f64::min);
            assert_eq!(columns.len(), fanout, "point {id}");
            assert!(
                farthest_chosen <= nearest_other,
                "point {id}: chosen {columns:?} at {farthest_chosen}, other at {nearest_other}"
            );
        }
    }

    #[test]
    fn reused_stripe_buffers_do_not_carry_points() {
        // Points and leaders lie on a line, so the nearest leaders follow from
        // the coordinates. The first stripe grows the buffer. The second stripe is
        // smaller and reads only a prefix of it.
        let values = [100.0_f32, 0.0, 8.0, 10.0, 2.0];
        let data = MatrixView::try_from(&values[..], 5, 1).unwrap();
        let leader_values = [0.0_f32, 10.0, 100.0];
        let leaders = L2::create_leaders(MatrixView::try_from(&leader_values[..], 3, 1).unwrap());
        let fanout = 2;
        let mut buffers = StripeBuffers::default();

        for ids in [&[4, 2, 0, 3, 1][..], &[2, 1][..]] {
            let mut assignments = Matrix::new(0, ids.len(), fanout);

            assign_point_stripe::<_, L2, _>(
                ARCH,
                data,
                ids,
                &leaders,
                &mut buffers,
                assignments.as_mut_view(),
            )
            .unwrap();

            // The `fanout` leader columns nearest to each point. In one dimension,
            // |x - leader| orders the leaders the same way as squared L2. No two
            // leaders are at the same distance from a point.
            let distance =
                |id: u32, column: u32| (values[id as usize] - leader_values[column as usize]).abs();
            let expected: Vec<Vec<u32>> = ids
                .iter()
                .map(|&id| {
                    let mut columns: Vec<u32> = (0..leader_values.len() as u32).collect();
                    columns.sort_by(|&a, &b| distance(id, a).total_cmp(&distance(id, b)));
                    columns.truncate(fanout);
                    columns
                })
                .collect();
            let actual: Vec<_> = assignments.row_iter().map(<[u32]>::to_vec).collect();
            assert_eq!(actual, expected, "points {ids:?}");
        }
    }

    #[test]
    fn grouping_by_leader_drops_unassigned_slots() {
        let pattern = [
            [2, 0],
            [UNASSIGNED, 1],
            [0, UNASSIGNED],
            [UNASSIGNED, UNASSIGNED],
        ];
        let pool = thread_pool(3);

        // Counts around the parallel threshold cover both grouping paths and
        // parallel chunks of unequal length.
        for point_count in [
            7,
            PARALLEL_GROUPING_MIN_POINTS - 1,
            PARALLEL_GROUPING_MIN_POINTS,
            PARALLEL_GROUPING_MIN_POINTS + 1,
        ] {
            let ids: Vec<_> = (0..point_count).map(|i| 1_000_000 - i as u32).collect();
            let assignments: Vec<_> = (0..point_count).flat_map(|i| pattern[i % 4]).collect();
            let assignments = MatrixView::try_from(assignments.as_slice(), point_count, 2).unwrap();
            let expected = [
                ids.iter().copied().step_by(2).collect(),
                ids.iter().copied().skip(1).step_by(4).collect(),
                ids.iter().copied().step_by(4).collect(),
                vec![],
            ];

            let actual = pool.install(|| group_by_leader(&ids, assignments, 4));

            assert_eq!(
                sorted_members_per_row(&actual),
                sorted_members_per_row(&expected),
                "{point_count} points"
            );
        }
    }

    #[test]
    fn a_split_uses_the_configured_fanout_then_falls_back_to_one_leader() {
        // Every point is a leader, so membership is fixed regardless of output order.
        let values = [0.0_f32, 1.0, 10.0, 11.0];
        let data = MatrixView::try_from(&values[..], 4, 1).unwrap();

        for (level, expected) in [
            (0, vec![vec![0, 1], vec![0, 1], vec![2, 3], vec![2, 3]]),
            (1, vec![vec![0], vec![1], vec![2], vec![3]]),
        ] {
            let parent = Cluster {
                point_ids: vec![0, 1, 2, 3],
                level,
                seed: 1290,
            };

            let actual = split::<L2>(data, &splitting_config(), parent);

            assert_eq!(
                sorted_leaf_memberships(&actual.leaves),
                sorted_leaf_memberships(&expected),
                "level {level}"
            );
            assert!(actual.next_level.is_empty(), "level {level}");
        }
    }

    #[test]
    fn children_above_cmax_are_queued_at_the_next_level() {
        let values = [0.0_f32, 1.0, 10.0, 11.0];
        let data = MatrixView::try_from(&values[..], 4, 1).unwrap();
        let config = PiPNNConfig {
            c_max: nz(1),
            ..splitting_config()
        };
        let parent = Cluster {
            point_ids: vec![0, 1, 2, 3],
            level: 0,
            seed: 1290,
        };

        let actual = split::<L2>(data, &config, parent);

        assert!(actual.leaves.is_empty());
        assert!(actual.next_level.iter().all(|child| child.level == 1));
        let ids: Vec<_> = actual
            .next_level
            .into_iter()
            .map(|child| child.point_ids)
            .collect();
        assert_eq!(
            sorted_leaf_memberships(&ids),
            [vec![0, 1], vec![0, 1], vec![2, 3], vec![2, 3]]
        );
    }

    #[test]
    fn a_fanout_one_split_without_progress_cuts_equal_leaves() {
        // Copies of one vector tie for every leader, so all points choose the
        // same leader and the child equals its parent.
        let values = [1.0_f32; 5];
        let data = MatrixView::try_from(&values[..], 5, 1).unwrap();
        let config = PiPNNConfig {
            fanout: vec![nz(1)],
            ..splitting_config()
        };
        let parent = Cluster {
            point_ids: (0..5).collect(),
            level: 0,
            seed: 1290,
        };

        let actual = split::<L2>(data, &config, parent);

        assert_eq!(
            sorted_leaf_memberships(&actual.leaves),
            [vec![0, 1], vec![2, 3], vec![4]]
        );
        assert!(actual.next_level.is_empty());
    }

    #[test]
    fn a_child_that_repeats_its_parent_at_fanout_two_is_split_again() {
        // Points at x=0, x=10 and x=4 are all leaders. Each point joins itself and
        // x=4, so the child of x=4 holds all points. With fanout above one this
        // is overlap, not a stalled split.
        let values = [0.0_f32, 10.0, 4.0];
        let data = MatrixView::try_from(&values[..], 3, 1).unwrap();
        let parent = Cluster {
            point_ids: vec![0, 1, 2],
            level: 0,
            seed: 1290,
        };

        let actual = split::<L2>(data, &splitting_config(), parent);

        assert_eq!(
            sorted_leaf_memberships(&actual.leaves),
            [vec![0, 2], vec![1]]
        );
        let [child] = &actual.next_level[..] else {
            panic!("expected one pending cluster");
        };
        assert_eq!(
            (child.point_ids.as_slice(), child.level),
            (&[0, 1, 2][..], 1)
        );
    }

    #[test]
    fn a_level_without_more_leaders_than_its_fanout_is_skipped() {
        // Four points at p_samp 0.01 give two leaders. Fanout 2 would put every
        // point into both children, so the cluster moves to the next level.
        let values = [0.0_f32, 1.0, 10.0, 11.0];
        let data = MatrixView::try_from(&values[..], 4, 1).unwrap();
        let config = PiPNNConfig {
            p_samp: 0.01,
            ..splitting_config()
        };
        let parent = Cluster {
            point_ids: vec![0, 1, 2, 3],
            level: 0,
            seed: 1290,
        };

        let actual = split::<L2>(data, &config, parent);

        assert!(actual.leaves.is_empty());
        let [child] = &actual.next_level[..] else {
            panic!("expected one pending cluster");
        };
        assert_eq!(
            (child.point_ids.as_slice(), child.level),
            (&[0, 1, 2, 3][..], 1)
        );
    }

    #[test]
    fn data_within_cmax_forms_one_complete_leaf_per_replica() {
        let values = [0.0_f32, 1.0, 10.0];
        let data = MatrixView::try_from(&values[..], 3, 1).unwrap();
        let pool = thread_pool(1);

        for c_max in [3, 4] {
            let config = PiPNNConfig {
                c_max: nz(c_max),
                replicas: nz(2),
                ..splitting_config()
            };

            let actual = pool
                .install(|| partition::<_, L2, _>(ARCH, data, &config))
                .unwrap();

            assert_eq!(
                sorted_leaf_memberships(&actual),
                [vec![0, 1, 2], vec![0, 1, 2]],
                "c_max {c_max}"
            );
        }
    }

    #[test]
    fn recursive_partitioning_preserves_leaf_membership_across_worker_counts() {
        let point_count = 129;
        let values = dense_points(point_count, 17, 1290);
        let data = MatrixView::try_from(values.as_slice(), point_count, 17).unwrap();
        let config = PiPNNConfig {
            c_max: nz(16),
            p_samp: 0.2,
            fanout: vec![nz(2), nz(2)],
            replicas: nz(2),
            ..splitting_config()
        };

        let first = thread_pool(1)
            .install(|| partition::<_, L2, _>(ARCH, data, &config))
            .unwrap();
        let second = thread_pool(3)
            .install(|| partition::<_, L2, _>(ARCH, data, &config))
            .unwrap();

        assert_eq!(
            sorted_leaf_memberships(&first),
            sorted_leaf_memberships(&second)
        );
        let mut ids: Vec<_> = first.iter().flatten().copied().collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids, (0..point_count as u32).collect::<Vec<_>>());
        for leaf in first {
            assert!(
                !leaf.is_empty() && leaf.len() <= config.c_max.get(),
                "leaf {leaf:?}"
            );
            assert_eq!(
                leaf.iter().copied().collect::<HashSet<_>>().len(),
                leaf.len(),
                "duplicate IDs in leaf {leaf:?}"
            );
        }
    }

    #[test]
    fn replicas_use_different_seeds() {
        // With one seed for every replica, the second replica would repeat the
        // leaves of the first.
        let point_count = 33;
        let values = dense_points(point_count, 17, 1290);
        let data = MatrixView::try_from(values.as_slice(), point_count, 17).unwrap();
        let config = PiPNNConfig {
            c_max: nz(8),
            p_samp: 0.25,
            fanout: vec![nz(1)],
            ..splitting_config()
        };
        let pool = thread_pool(2);
        let leaves = |replicas| {
            let config = PiPNNConfig {
                replicas: nz(replicas),
                ..config.clone()
            };
            let leaves = pool
                .install(|| partition::<_, L2, _>(ARCH, data, &config))
                .unwrap();
            sorted_leaf_memberships(&leaves)
        };

        let one = leaves(1);
        let two = leaves(2);

        assert!(two.iter().any(|leaf| !one.contains(leaf)));
    }

    #[test]
    fn unrankable_points_join_no_leaf() {
        let config = PiPNNConfig {
            c_max: nz(1),
            fanout: vec![nz(1)],
            ..splitting_config()
        };
        let pool = thread_pool(1);

        for (values, expected) in [
            ([0.0, 3.0, f32::NAN], vec![vec![0], vec![1]]),
            ([f32::NAN; 3], vec![]),
        ] {
            let data = MatrixView::try_from(&values[..], 3, 1).unwrap();

            let actual = pool
                .install(|| partition::<_, L2, _>(ARCH, data, &config))
                .unwrap();

            assert_eq!(
                sorted_leaf_memberships(&actual),
                sorted_leaf_memberships(&expected),
                "values {values:?}"
            );
        }
    }
}

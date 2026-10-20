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
//! A fanout-1 split in which one child gets every point makes no progress, for
//! example when all sampled leaders are copies of one vector. The split cuts
//! that child into equal leaves in point order, and the build logs a warning
//! with the number of points it cut. These leaves group points arbitrarily,
//! which can lower recall.
//!
//! The partition kernel computes point-to-leader distances and selects the
//! nearest leaders. This module samples leaders, gathers vectors, and groups the
//! assignments into children.
//!
//! A split assigns its points in parallel stripes. An assignment stripe is a
//! batch of up to `ASSIGNMENT_STRIPE_POINTS` points that one task gathers as
//! `f32` and ranks against all leaders of the split.
//!
//! Buffer sizes use plain multiplication. The IDs of one split are distinct
//! dataset rows, so a gathered `rows x dimensions` buffer is not larger than the
//! dataset. An assignment buffer holds `points x fanout` IDs. The fanout of a
//! split is below its leader count, which is at most its point count, so with at
//! most `u32::MAX` points this count fits in a 64-bit `usize`.

use crate::{ANNResult, utils::VectorRepr};
use diskann_utils::{
    object_pool::{AsPooled, ObjectPool},
    views::{Matrix, MatrixView, MutMatrixView},
};
use rand::{Rng, SeedableRng, prelude::IndexedRandom, rngs::StdRng};
use rayon::prelude::*;

use super::{
    PiPNNConfig,
    conversion::{f32_dimensions, gather_as_f32},
    partition_kernel::{PartitionKernelWorkspace, assign_leaders},
    partition_metric::PartitionMetric,
    simd::Simd,
    topk::UNASSIGNED,
};

/// Number of points in one assignment stripe. Short stripes repeat the fixed
/// cost of a kernel call, and tall stripes overflow the last-level cache once
/// every worker holds one. The height does not follow the detected cache size,
/// because a virtual machine reports a last-level cache that other tenants
/// share.
const ASSIGNMENT_STRIPE_POINTS: usize = 256;
/// Clusters with at least this many points group their points by leader in
/// parallel. Such clusters occur at the first levels, where few splits run at
/// the same time, so a serial pass would leave most workers idle. Smaller
/// clusters group serially, because the parallel pass costs more than it saves.
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
    /// Number of points that the split cut into leaves in point order.
    cut_points: usize,
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
    let mut cut_points = 0;
    let stripe_buffers = StripeBufferPool::new((), 0, None);
    // Replica `r` uses seed `base + r`. `StdRng::seed_from_u64` gives
    // independent streams for adjacent seeds.
    let base_seed = config.seed.unwrap_or_else(rand::random);
    for replica in 0..config.replicas.get() {
        let seed = base_seed.wrapping_add(replica as u64);
        let (mut replica_leaves, replica_cut_points) =
            partition_replica::<A, M, T>(arch, data, config, seed, &stripe_buffers)?;
        leaves.append(&mut replica_leaves);
        cut_points += replica_cut_points;
    }
    if cut_points > 0 {
        tracing::warn!(
            cut_points,
            "PiPNN cut {cut_points} points into leaves in point order because no split \
             could shrink their clusters; these leaves group points arbitrarily and can \
             lower recall"
        );
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
) -> ANNResult<(Vec<Vec<u32>>, usize)>
where
    A: Simd,
    M: PartitionMetric,
    T: VectorRepr + Send + Sync,
{
    let initial_point_ids = (0..data.nrows() as u32).collect();
    let c_max = config.c_max.get();
    if data.nrows() <= c_max {
        return Ok((vec![initial_point_ids], 0));
    }

    let mut leaves = Vec::new();
    let mut cut_points = 0;
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
            cut_points += split.cut_points;
        }
        pending = next_level;
    }
    Ok((leaves, cut_points))
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
    let leader_count = sampled_leader_count(
        cluster.point_ids.len(),
        config.p_samp,
        config.max_leaders.get(),
    );
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
            cut_points: 0,
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
    let mut cut_points = 0;
    for child in children {
        if child.is_empty() {
            continue;
        }
        if child.len() <= c_max {
            leaves.push(child);
        } else if fanout == 1 && child.len() == cluster.point_ids.len() {
            // Every point chose this leader, so the split made no progress. With
            // L2 or cosine, each leader is nearest to itself, so this needs all
            // sampled leaders to be copies of one vector. With inner product, a
            // leader with a large norm can also win every point. Another split can
            // stall the same way, so cut the child into equal leaves in point
            // order. The cut groups points arbitrarily, which can lower recall but
            // keeps the graph valid.
            let leaf_len = child.len().div_ceil(child.len().div_ceil(c_max));
            leaves.extend(child.chunks(leaf_len).map(<[u32]>::to_vec));
            cut_points += child.len();
        } else {
            next_level.push(Cluster {
                point_ids: child,
                level: cluster.level + 1,
                seed: rng.random(),
            });
        }
    }
    Ok(ClusterSplit {
        next_level,
        leaves,
        cut_points,
    })
}

/// Sample `count` distinct leaders from the points of a cluster.
fn sample_leaders(points: &[u32], count: usize, rng: &mut StdRng) -> Vec<u32> {
    points.choose_multiple(rng, count).copied().collect()
}

/// Return the number of leaders for a cluster of `points` points.
///
/// A split needs two leaders. `max_leaders` bounds the cost of one split.
fn sampled_leader_count(points: usize, sampling_fraction: f64, max_leaders: usize) -> usize {
    ((points as f64 * sampling_fraction).ceil() as usize).clamp(2, max_leaders)
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
    let dimensions = f32_dimensions(data)?;
    let mut leader_values = Matrix::new(0.0f32, leader_ids.len(), dimensions);
    gather_as_f32(data, leader_ids, leader_values.as_mut_view())?;

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
                dimensions,
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
    dimensions: usize,
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
    gather_as_f32(data, point_ids, points.as_mut_view())?;
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
        test_support::{dense_points, nz, thread_pool},
    };
    use diskann_wide::ARCH;
    use std::collections::HashSet;

    /// Sort each leaf and then the list of leaves. Neither order is part of the
    /// result. Repeated leaves and points stay, so a comparison still catches
    /// duplicates.
    fn sorted_leaves(leaves: &[Vec<u32>]) -> Vec<Vec<u32>> {
        let mut leaves: Vec<_> = leaves
            .iter()
            .map(|leaf| {
                let mut points = leaf.clone();
                points.sort_unstable();
                points
            })
            .collect();
        leaves.sort_unstable();
        leaves
    }

    /// Every point is a leader (`p_samp = 1.0`), so leader sampling picks the
    /// same set for any seed.
    fn splitting_config() -> PiPNNConfig {
        PiPNNConfig {
            c_max: nz(2),
            p_samp: 1.0,
            max_leaders: nz(1000),
            fanout: vec![nz(2)],
            leaf_k: nz(1),
            replicas: nz(1),
            seed: Some(1290),
        }
    }

    /// Group `ids` by their `fanout` nearest `leaders` under L2, on `n_threads` threads.
    fn assign_l2(
        data: MatrixView<'_, f32>,
        ids: &[u32],
        leaders: &[u32],
        fanout: usize,
        n_threads: usize,
    ) -> Vec<Vec<u32>> {
        let initial_buffers = 0;
        let buffers = StripeBufferPool::new((), initial_buffers, None);
        thread_pool(n_threads)
            .install(|| assign_to_leaders::<_, L2, _>(ARCH, data, ids, leaders, fanout, &buffers))
            .unwrap()
    }

    /// Split the cluster of `point_ids` at `level` in one-dimensional L2 points `x`.
    /// Two threads use a fixed sampling seed and an initially empty scratch pool.
    fn split_1d(
        x: &[f32],
        config: &PiPNNConfig,
        point_ids: Vec<u32>,
        level: usize,
    ) -> ClusterSplit {
        let dimensions = 1;
        let n_threads = 2;
        let initial_buffers = 0;
        let data = MatrixView::try_from(x, x.len(), dimensions).unwrap();
        let parent = Cluster {
            point_ids,
            level,
            seed: 1290,
        };
        let buffers = StripeBufferPool::new((), initial_buffers, None);
        thread_pool(n_threads)
            .install(|| split_cluster::<_, L2, _>(ARCH, data, config, parent, &buffers))
            .unwrap()
    }

    /// Partition `data` under L2 on `n_threads` threads.
    fn partition_l2(
        data: MatrixView<'_, f32>,
        config: &PiPNNConfig,
        n_threads: usize,
    ) -> Vec<Vec<u32>> {
        thread_pool(n_threads)
            .install(|| partition::<_, L2, _>(ARCH, data, config))
            .unwrap()
    }

    #[test]
    fn should_partition_successfully() {
        // x[i] is the one-dimensional vector of point i: two pairs far apart.
        let x = [0.0, 1.0, 10.0, 11.0];
        let dimensions = 1;
        let data = MatrixView::try_from(x.as_slice(), x.len(), dimensions).unwrap();
        // c_max = 2 forces one split. Every point is a leader with fanout 2, so
        // both leaders of a pair create the same leaf.
        let config = splitting_config();
        let n_threads = 1;

        let actual = partition_l2(data, &config, n_threads);

        assert_eq!(
            sorted_leaves(&actual),
            [vec![0, 1], vec![0, 1], vec![2, 3], vec![2, 3]]
        );
    }

    #[test]
    fn the_leader_count_rounds_up_and_stays_between_two_and_max_leaders() {
        let max_leaders = 5;

        for (points, sampling_fraction, expected) in [
            // 7 × 0.01 = 0.07 rounds up to 1, and a split needs two leaders.
            (7, 0.01, 2),
            // 7 × 0.31 = 2.17 rounds up to 3.
            (7, 0.31, 3),
            // 5 × 1.0 reaches the cap exactly.
            (5, 1.0, 5),
            // 6 × 1.0 is above the cap.
            (6, 1.0, 5),
        ] {
            assert_eq!(
                sampled_leader_count(points, sampling_fraction, max_leaders),
                expected,
                "{points} points, fraction {sampling_fraction}"
            );
        }
    }

    #[test]
    fn points_join_their_nearest_leaders() {
        // x[i] is the one-dimensional vector of point i.
        // Leader points 2 and 0 are at x = 10 and x = 0. The points to assign
        // are 5, 4 and 1, at x = 8, 2 and 100. Point 3 is not used.
        let x = [0.0, 100.0, 10.0, 0.0, 2.0, 8.0];
        let dimensions = 1;
        let n_threads = 2;
        let data = MatrixView::try_from(x.as_slice(), x.len(), dimensions).unwrap();

        for (fanout, expected) in [
            // Points 5 and 1 are nearer to x = 10, and point 4 to x = 0.
            (1, [vec![5, 1], vec![4]]),
            // With two leaders, every point joins both groups.
            (2, [vec![5, 4, 1], vec![5, 4, 1]]),
        ] {
            let actual = assign_l2(data, &[5, 4, 1], &[2, 0], fanout, n_threads);

            assert_eq!(actual, expected, "fanout {fanout}");
        }
    }

    #[test]
    fn points_in_every_stripe_and_the_tail_join_their_nearest_leaders() {
        // Two full stripes of 256 points and a tail of 3.
        let point_count = 2 * ASSIGNMENT_STRIPE_POINTS + 3;
        let dimensions = 1;
        // Point i is at 0, 4 or 10 for i % 3 = 0, 1 or 2, so a stripe that
        // writes at a wrong offset gives a point the result of another kind.
        let x: Vec<_> = (0..point_count)
            .map(|i| [0.0_f32, 4.0, 10.0][i % 3])
            .collect();
        let data = MatrixView::try_from(x.as_slice(), point_count, dimensions).unwrap();
        // The IDs differ from the row positions, so using a row position as an
        // ID fails.
        let ids: Vec<u32> = (0..point_count as u32).rev().collect();
        // Leader points 2, 0 and 1 are at 10, 0 and 4. With fanout 2, each
        // point joins the leader at 4 and the nearer of the leaders at 0 and 10.
        let leaders = [2, 0, 1];
        let fanout = 2;
        let expected = [
            // The leader at 10 gets the points at 10.
            ids.iter().copied().filter(|id| id % 3 == 2).collect(),
            // The leader at 0 gets the points at 0 and at 4.
            ids.iter().copied().filter(|id| id % 3 != 2).collect(),
            // The leader at 4 gets every point.
            ids.clone(),
        ];

        // With three threads, the three stripes run in parallel.
        for n_threads in [1, 3] {
            let actual = assign_l2(data, &ids, &leaders, fanout, n_threads);

            assert_eq!(actual, expected, "{n_threads} threads");
        }
    }

    #[test]
    fn reused_stripe_buffers_do_not_carry_points() {
        // x[i] is the one-dimensional vector of point i.
        let x = [100.0, 0.0, 8.0, 10.0, 2.0];
        let dimensions = 1;
        let fanout = 2;
        let initial_assignment = 0;
        let data = MatrixView::try_from(x.as_slice(), x.len(), dimensions).unwrap();
        let leader_x = [0.0, 10.0, 100.0];
        let leaders = L2::create_leaders(
            MatrixView::try_from(leader_x.as_slice(), leader_x.len(), dimensions).unwrap(),
        );
        let mut buffers = StripeBuffers::default();

        // The first stripe grows the buffers. The second stripe is smaller and
        // reads only a prefix of them. Each row lists the indices of the two
        // nearest leaders in `leader_x`, nearest first.
        for (ids, expected) in [
            (
                vec![4, 2, 0, 3, 1],
                vec![vec![0, 1], vec![1, 0], vec![2, 1], vec![1, 0], vec![0, 1]],
            ),
            (vec![2, 1], vec![vec![1, 0], vec![0, 1]]),
        ] {
            let mut assignments = Matrix::new(initial_assignment, ids.len(), fanout);

            assign_point_stripe::<_, L2, _>(
                ARCH,
                data,
                &ids,
                dimensions,
                &leaders,
                &mut buffers,
                assignments.as_mut_view(),
            )
            .unwrap();

            let actual: Vec<_> = assignments.row_iter().map(<[u32]>::to_vec).collect();
            assert_eq!(actual, expected, "points {ids:?}");
        }
    }

    #[test]
    fn grouping_by_leader_drops_unassigned_slots() {
        // Point i takes row i % 4. UNASSIGNED marks a slot without a leader.
        let assignment_rows = [
            [2, 0],
            [UNASSIGNED, 1],
            [0, UNASSIGNED],
            [UNASSIGNED, UNASSIGNED],
        ];
        let n_threads = 3;
        let leader_count = 4;
        let fanout = assignment_rows[0].len();
        let pool = thread_pool(n_threads);

        // Counts around the parallel threshold cover both grouping paths and
        // parallel chunks of unequal length.
        for point_count in [
            7,
            PARALLEL_GROUPING_MIN_POINTS - 1,
            PARALLEL_GROUPING_MIN_POINTS,
            PARALLEL_GROUPING_MIN_POINTS + 1,
        ] {
            let ids: Vec<_> = (0..point_count).map(|i| 1_000_000 - i as u32).collect();
            let assignments: Vec<_> = (0..point_count)
                .flat_map(|i| assignment_rows[i % 4])
                .collect();
            let assignments =
                MatrixView::try_from(assignments.as_slice(), point_count, fanout).unwrap();
            let expected = [
                // Rows 0 and 2 hold leader 0.
                ids.iter().copied().step_by(2).collect(),
                // Row 1 holds leader 1.
                ids.iter().copied().skip(1).step_by(4).collect(),
                // Row 0 holds leader 2.
                ids.iter().copied().step_by(4).collect(),
                // No row holds leader 3.
                vec![],
            ];

            let actual = pool.install(|| group_by_leader(&ids, assignments, leader_count));

            assert_eq!(actual, expected, "{point_count} points");
        }
    }

    #[test]
    fn a_split_uses_the_configured_fanout_then_falls_back_to_one_leader() {
        // x[i] is the one-dimensional vector of point i.
        // Every point is a leader. Points 0, 1 and points 2, 3 are pairs.
        let x = [0.0, 1.0, 10.0, 11.0];

        for (level, expected) in [
            // Fanout 2: each point joins itself and the other point of its pair.
            (0, vec![vec![0, 1], vec![0, 1], vec![2, 3], vec![2, 3]]),
            // The fanout list has no level 1, so each point joins only itself.
            (1, vec![vec![0], vec![1], vec![2], vec![3]]),
        ] {
            let actual = split_1d(&x, &splitting_config(), vec![0, 1, 2, 3], level);

            assert_eq!(sorted_leaves(&actual.leaves), expected, "level {level}");
            assert!(actual.next_level.is_empty(), "level {level}");
            assert_eq!(actual.cut_points, 0, "level {level}");
        }
    }

    #[test]
    fn children_above_cmax_are_queued_at_the_next_level() {
        // x[i] is the one-dimensional vector of point i.
        let x = [0.0, 1.0, 10.0, 11.0];
        let level = 0;
        // With c_max = 1, every two-point child moves to level 1.
        let config = PiPNNConfig {
            c_max: nz(1),
            ..splitting_config()
        };

        let actual = split_1d(&x, &config, vec![0, 1, 2, 3], level);

        assert!(actual.leaves.is_empty());
        assert!(actual.next_level.iter().all(|child| child.level == 1));
        let ids: Vec<_> = actual
            .next_level
            .into_iter()
            .map(|child| child.point_ids)
            .collect();
        assert_eq!(
            sorted_leaves(&ids),
            [vec![0, 1], vec![0, 1], vec![2, 3], vec![2, 3]]
        );
    }

    #[test]
    fn a_fanout_one_split_without_progress_cuts_equal_leaves() {
        // x[i] is the one-dimensional vector of point i.
        // Copies of one vector tie for every leader, so all points choose the
        // same leader and the child equals its parent.
        let x = [1.0; 5];
        let level = 0;
        let config = PiPNNConfig {
            fanout: vec![nz(1)],
            ..splitting_config()
        };

        let actual = split_1d(&x, &config, (0..x.len() as u32).collect(), level);

        // c_max = 2 cuts the five points into leaves of two, in point order.
        assert_eq!(
            sorted_leaves(&actual.leaves),
            [vec![0, 1], vec![2, 3], vec![4]]
        );
        assert!(actual.next_level.is_empty());
        assert_eq!(actual.cut_points, 5);
    }

    #[test]
    fn a_child_that_repeats_its_parent_at_fanout_two_is_split_again() {
        // x[i] is the one-dimensional vector of point i.
        // Points at x=0, x=10 and x=4 are all leaders. Each point joins itself and
        // x=4, so the child of x=4 holds all points. With fanout above one this
        // is overlap, not a stalled split.
        let x = [0.0, 10.0, 4.0];
        let level = 0;

        let actual = split_1d(&x, &splitting_config(), vec![0, 1, 2], level);

        assert_eq!(sorted_leaves(&actual.leaves), [vec![0, 2], vec![1]]);
        let [child] = actual.next_level.as_slice() else {
            panic!("expected one pending cluster");
        };
        assert_eq!(child.point_ids, [0, 1, 2]);
        assert_eq!(child.level, 1);
    }

    #[test]
    fn a_level_without_more_leaders_than_its_fanout_is_skipped() {
        // x[i] is the one-dimensional vector of point i.
        // Four points at p_samp 0.01 give two leaders. Fanout 2 would put every
        // point into both children, so the cluster moves to the next level.
        let x = [0.0, 1.0, 10.0, 11.0];
        let level = 0;
        let config = PiPNNConfig {
            p_samp: 0.01,
            ..splitting_config()
        };

        let actual = split_1d(&x, &config, vec![0, 1, 2, 3], level);

        assert!(actual.leaves.is_empty());
        let [child] = actual.next_level.as_slice() else {
            panic!("expected one pending cluster");
        };
        assert_eq!(child.point_ids, [0, 1, 2, 3]);
        assert_eq!(child.level, 1);
    }

    #[test]
    fn data_within_cmax_forms_one_complete_leaf_per_replica() {
        // x[i] is the one-dimensional vector of point i.
        // The dataset fits c_max, so each replica returns it as one leaf.
        let x = [0.0, 1.0, 10.0];
        let dimensions = 1;
        let n_threads = 1;
        let data = MatrixView::try_from(x.as_slice(), x.len(), dimensions).unwrap();

        for c_max in [x.len(), x.len() + 1] {
            let config = PiPNNConfig {
                c_max: nz(c_max),
                replicas: nz(2),
                ..splitting_config()
            };

            let actual = partition_l2(data, &config, n_threads);

            assert_eq!(
                sorted_leaves(&actual),
                [vec![0, 1, 2], vec![0, 1, 2]],
                "c_max {c_max}"
            );
        }
    }

    /// Build 129 dense 17-dimensional points with two fanout levels and two replicas.
    fn recursive_partition_fixture() -> (Matrix<f32>, PiPNNConfig) {
        let point_count = 129;
        let dimensions = 17;
        let seed = 1290;
        let values = dense_points(point_count, dimensions, seed);
        let data = Matrix::try_from(values.into_boxed_slice(), point_count, dimensions).unwrap();
        let config = PiPNNConfig {
            c_max: nz(16),
            p_samp: 0.2,
            fanout: vec![nz(2), nz(2)],
            replicas: nz(2),
            ..splitting_config()
        };
        (data, config)
    }

    #[test]
    fn the_leaves_do_not_depend_on_the_worker_count() {
        // Each child keeps the input order of its points, so the next split
        // samples the same leaders for any thread count.
        let (data, config) = recursive_partition_fixture();
        let serial_n_threads = 1;
        let parallel_n_threads = 3;

        let first = partition_l2(data.as_view(), &config, serial_n_threads);
        let second = partition_l2(data.as_view(), &config, parallel_n_threads);

        assert_eq!(sorted_leaves(&first), sorted_leaves(&second));
    }

    #[test]
    fn every_point_lands_in_a_leaf_of_at_most_c_max_points() {
        // Two fanout levels and two replicas make several levels of splits.
        let (data, config) = recursive_partition_fixture();
        let n_threads = 1;

        let leaves = partition_l2(data.as_view(), &config, n_threads);

        // Every point is in some leaf.
        let mut ids: Vec<_> = leaves.iter().flatten().copied().collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids, (0..data.nrows() as u32).collect::<Vec<_>>());
        // Each leaf holds 1 to c_max points, each point at most once.
        for leaf in leaves {
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
        let dimensions = 17;
        let seed = 1290;
        let n_threads = 2;
        let values = dense_points(point_count, dimensions, seed);
        let data = MatrixView::try_from(values.as_slice(), point_count, dimensions).unwrap();
        let config = PiPNNConfig {
            c_max: nz(8),
            p_samp: 0.25,
            fanout: vec![nz(1)],
            ..splitting_config()
        };
        let sorted_leaves_with = |replicas| {
            let config = PiPNNConfig {
                replicas: nz(replicas),
                ..config.clone()
            };
            let leaves = partition_l2(data, &config, n_threads);
            sorted_leaves(&leaves)
        };

        let one_replica = sorted_leaves_with(1);
        let two_replicas = sorted_leaves_with(2);

        assert!(two_replicas.iter().any(|leaf| !one_replica.contains(leaf)));
    }

    #[test]
    fn unrankable_points_join_no_leaf() {
        // x[i] in each case is the one-dimensional vector of point i.
        // Single-point leaves expose whether an unrankable point joined a child.
        let dimensions = 1;
        let n_threads = 1;
        let config = PiPNNConfig {
            c_max: nz(1),
            fanout: vec![nz(1)],
            ..splitting_config()
        };

        for (x, expected) in [
            // Point 2 is NaN, so no leader ranks it.
            ([0.0, 3.0, f32::NAN], vec![vec![0], vec![1]]),
            // No point can be ranked.
            ([f32::NAN; 3], vec![]),
        ] {
            let data = MatrixView::try_from(x.as_slice(), x.len(), dimensions).unwrap();

            let actual = partition_l2(data, &config, n_threads);

            assert_eq!(sorted_leaves(&actual), expected, "x {x:?}");
        }
    }
}

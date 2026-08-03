/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::views::{Matrix, MatrixView};
use diskann_vector::{distance::Metric, Half};

use super::*;

fn config(c_min: usize, c_max: usize, fanout: Vec<usize>, replicas: usize) -> PiPNNConfig {
    PiPNNConfig {
        c_max,
        c_min,
        p_samp: 0.25,
        fanout,
        k: 1,
        replicas,
    }
}

fn clustered_data(points: usize, dimensions: usize) -> Matrix<f32> {
    Matrix::new(
        diskann_utils::views::Init({
            let mut position = 0usize;
            move || {
                let point = position / dimensions;
                let dimension = position % dimensions;
                position += 1;
                (point / 8) as f32 * 10.0 + dimension as f32 * 0.01 + point as f32 * 0.001
            }
        }),
        points,
        dimensions,
    )
}

fn directional_data(points: usize, dimensions: usize) -> Matrix<f32> {
    Matrix::new(
        diskann_utils::views::Init({
            let mut position = 0usize;
            move || {
                let point = position / dimensions;
                let dimension = position % dimensions;
                position += 1;
                let angle = std::f32::consts::TAU * point as f32 / points as f32;
                match dimension {
                    0 => angle.cos(),
                    1 => angle.sin(),
                    _ => 0.0,
                }
            }
        }),
        points,
        dimensions,
    )
}

fn sorted_memberships(leaves: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let mut memberships: Vec<Vec<u32>> = leaves
        .iter()
        .map(|leaf| {
            let mut ids = leaf.clone();
            ids.sort_unstable();
            ids
        })
        .collect();
    memberships.sort();
    memberships
}

fn assert_valid_partition(leaves: &[Vec<u32>], points: usize, c_max: usize, replicas: usize) {
    assert!(leaves
        .iter()
        .all(|leaf| !leaf.is_empty() && leaf.len() <= c_max));
    let mut counts = vec![0usize; points];
    for leaf in leaves {
        let mut ids = leaf.clone();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), leaf.len(), "duplicate ID inside a leaf");
        for &id in leaf {
            assert!((id as usize) < points);
            counts[id as usize] += 1;
        }
    }
    assert!(counts.iter().all(|&count| count >= replicas));
}

#[test]
fn returns_one_leaf_at_and_below_c_max() {
    for points in [7, 8] {
        let data = clustered_data(points, 3);
        let leaves = partition(data.as_view(), config(2, 8, vec![2], 1), Metric::L2).unwrap();
        assert_eq!(leaves, vec![(0..points as u32).collect::<Vec<_>>()]);
    }
}

#[test]
fn partition_is_fixed_seed_deterministic_and_bounded() {
    let data = clustered_data(96, 8);
    let config = config(4, 16, vec![3, 2], 2);

    let first = partition(data.as_view(), config.clone(), Metric::L2).unwrap();
    let second = partition(data.as_view(), config, Metric::L2).unwrap();

    assert_eq!(sorted_memberships(&first), sorted_memberships(&second));
    assert_valid_partition(&first, 96, 16, 2);
    assert!(first.iter().map(Vec::len).sum::<usize>() > 96 * 2);
}

#[test]
fn recursion_after_fanout_levels_falls_back_to_one() {
    let data = clustered_data(80, 4);
    let leaves = partition(data.as_view(), config(2, 8, vec![2], 1), Metric::L2).unwrap();

    assert_valid_partition(&leaves, 80, 8, 1);
}

#[test]
fn duplicate_points_return_iteration_limit_instead_of_oversized_leaf() {
    let data = Matrix::new(1.0f32, 24, 4);
    let error = partition(data.as_view(), config(2, 4, vec![1], 1), Metric::L2).unwrap_err();
    let error = error.downcast::<PartitionError>().unwrap();

    assert!(matches!(
        error,
        PartitionError::IterationLimit {
            size: 24,
            limit: MAX_PARTITION_ITERATIONS,
            ..
        }
    ));
}

#[test]
fn global_merge_canonicalizes_small_leaf_membership() {
    let leaves = vec![vec![9, 3, 1], vec![3, 2], vec![8]];

    let merged = global_merge_small(leaves, 4, 8).unwrap();

    assert_eq!(merged, vec![vec![1, 2, 3, 8, 9]]);
}

#[test]
fn global_merge_never_overfills_before_reaching_c_min() {
    let leaves = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11]];

    let merged = global_merge_small(leaves, 11, 11).unwrap();

    assert_eq!(
        merged,
        vec![vec![0, 1, 2, 3, 4, 5, 6, 7], vec![8, 9, 10, 11]]
    );
}

#[test]
fn global_merge_fills_exact_capacity_before_flushing() {
    let merged = global_merge_small(vec![vec![0, 1], vec![2, 3]], 4, 4).unwrap();

    assert_eq!(merged, vec![vec![0, 1, 2, 3]]);
}

#[test]
fn replicas_cover_every_point_once_or_more_per_replica() {
    let data = directional_data(72, 5);
    let leaves = partition(
        data.as_view(),
        config(3, 12, vec![3, 2], 3),
        Metric::CosineNormalized,
    )
    .unwrap();

    assert_valid_partition(&leaves, 72, 12, 3);
}

fn assert_partition_conversion_matches_f32<T>(label: &str, convert: impl Fn(u8) -> T)
where
    T: diskann::utils::VectorRepr + Send + Sync,
{
    let points = 64;
    // Partition gathering converts source vectors before GEMM. Exercise conversion
    // tails around 4-, 8-, and 16-element boundaries and a second 16-lane chunk.
    for dimensions in [1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
        let raw: Vec<u8> = (0..points * dimensions)
            .map(|index| {
                let point = index / dimensions;
                let dimension = index % dimensions;
                ((point * 5 + dimension * 7 + point * dimension) % 23) as u8
            })
            .collect();
        let f32_data: Vec<f32> = raw.iter().map(|&value| value as f32).collect();
        let converted: Vec<T> = raw.iter().copied().map(&convert).collect();
        let config = config(2, 16, vec![2, 1], 1);
        let expected = partition(
            MatrixView::try_from(&f32_data, points, dimensions).unwrap(),
            config.clone(),
            Metric::L2,
        )
        .unwrap();
        let actual = partition(
            MatrixView::try_from(&converted, points, dimensions).unwrap(),
            config,
            Metric::L2,
        )
        .unwrap_or_else(|error| panic!("{label} dimensions={dimensions}: {error}"));

        assert_valid_partition(&actual, points, 16, 1);
        assert_eq!(
            sorted_memberships(&actual),
            sorted_memberships(&expected),
            "{label} dimensions={dimensions}"
        );
    }
}

#[test]
fn f16_partition_matches_f32_across_dimension_boundaries() {
    assert_partition_conversion_matches_f32("f16", |value| Half::from_f32(value as f32));
}

#[test]
fn u8_partition_matches_f32_across_dimension_boundaries() {
    assert_partition_conversion_matches_f32("u8", |value| value);
}

#[test]
fn i8_partition_matches_f32_across_dimension_boundaries() {
    // The same translation in every coordinate preserves L2 ordering.
    assert_partition_conversion_matches_f32("i8", |value| value as i8 - 11);
}

#[test]
fn l2_leader_norms_preserve_scalar_reduction_order() {
    fn next(state: &mut u64) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (((*state >> 40) as f32 / 8_388_608.0) - 1.0) * 1_000.0
    }

    // This fixed case sits on opposite sides of the top-1 boundary depending
    // on whether leader norms use the original scalar reduction or a SIMD
    // reassociation. Point/leader dot products still go through the production
    // GEMM; only the setup norm calculation is under test.
    let dimensions = 129;
    let mut state = 0x3a85_f952_c718_6e49;
    let point: Vec<f32> = (0..dimensions).map(|_| next(&mut state)).collect();
    let leader_zero: Vec<f32> = (0..dimensions).map(|_| next(&mut state)).collect();
    let leader_one: Vec<f32> = (0..dimensions).map(|_| next(&mut state)).collect();
    let data: Vec<f32> = leader_zero
        .into_iter()
        .chain(leader_one)
        .chain(point)
        .collect();
    let data = MatrixView::try_from(data.as_slice(), 3, dimensions).unwrap();

    let clusters = assign_to_leaders(
        data,
        &[2],
        &[0, 1],
        1,
        Metric::L2,
        &PartitionKernel::new(Metric::L2),
        &StripeBufferPool::new((), 0, None),
    )
    .unwrap();

    assert_eq!(clusters, [vec![], vec![2]]);
}

#[test]
fn all_metrics_produce_valid_partitions() {
    let data = directional_data(64, 8);
    let config = config(2, 20, vec![2], 1);

    for metric in [
        Metric::L2,
        Metric::Cosine,
        Metric::CosineNormalized,
        Metric::InnerProduct,
    ] {
        let leaves = partition(data.as_view(), config.clone(), metric).unwrap();
        assert_valid_partition(&leaves, 64, 20, 1);
    }
}

#[test]
fn leader_count_is_bounded() {
    assert_eq!(sample_num_leaders(1, 1.0), 1);
    assert_eq!(sample_num_leaders(10, 0.01), 2);
    assert_eq!(sample_num_leaders(50_000, 1.0), LEADER_CAP);
}

#[test]
fn replica_seed_derivation_is_stable_and_distinct() {
    assert_eq!(replica_seed(0), 1_000);
    assert_eq!(replica_seed(1), 8_919);
}

#[test]
fn assignment_stripes_use_power_of_two_point_counts() {
    assert_eq!(assignment_stripe_point_count(1_000), 128);
    assert_eq!(assignment_stripe_point_count(256), 512);
    assert_eq!(
        assignment_stripe_point_count(1),
        MAX_ASSIGNMENT_STRIPE_POINTS
    );
}

#[test]
fn stripe_buffer_pool_reuses_returned_capacity() {
    let pool = StripeBufferPool::new((), 0, None);
    let points = {
        let mut buffers = pool.get_ref(());
        buffers.points.resize(16, 0.0);
        buffers.points.as_ptr()
    };

    let buffers = pool.get_ref(());
    assert_eq!(buffers.points.as_ptr(), points);
    assert_eq!(buffers.points.len(), 16);
}

#[test]
fn leader_assignment_handles_multiple_stripes() {
    let points = 2_048;
    let data: Vec<f32> = (0..points).map(|point| point as f32).collect();
    let data = MatrixView::try_from(data.as_slice(), points, 1).unwrap();
    let point_ids: Vec<u32> = (0..points as u32).collect();

    let clusters = assign_to_leaders(
        data,
        &point_ids,
        &[0, 2_047],
        1,
        Metric::L2,
        &PartitionKernel::new(Metric::L2),
        &StripeBufferPool::new((), 0, None),
    )
    .unwrap();

    assert_eq!(clusters[0], (0..1_024).collect::<Vec<_>>());
    assert_eq!(clusters[1], (1_024..2_048).collect::<Vec<_>>());
}

#[test]
fn parallel_scatter_matches_serial_order() {
    let points: Vec<u32> = (0..PARALLEL_SCATTER_MIN_POINTS as u32).collect();
    let assignments: Vec<u32> = points
        .iter()
        .flat_map(|point| [point % 7, (point + 3) % 7])
        .collect();

    let expected = scatter_serial(&points, &assignments, 2, 7).unwrap();
    let actual = scatter_assignments(&points, &assignments, 2, 7).unwrap();

    assert_eq!(actual, expected);
}

#[test]
fn rejects_empty_dataset() {
    let data = Matrix::<f32>::new(0.0, 0, 4);
    let error = partition(data.as_view(), config(1, 4, vec![1], 1), Metric::L2).unwrap_err();

    assert_eq!(
        error.downcast::<PartitionError>().unwrap(),
        PartitionError::EmptyDataset
    );
}

#[test]
fn rejects_zero_dimensions() {
    let data = Matrix::<f32>::new(0.0, 4, 0);
    let error = partition(data.as_view(), config(1, 4, vec![1], 1), Metric::L2).unwrap_err();

    assert_eq!(
        error.downcast::<PartitionError>().unwrap(),
        PartitionError::EmptyDimensions
    );
}

#[test]
fn rejects_invalid_gather_output_length() {
    let data = Matrix::<f32>::new(0.0, 2, 2);
    let error = gather_vectors(data.as_view(), &[0, 1], &mut [0.0; 3]).unwrap_err();

    assert_eq!(
        error.downcast::<PartitionError>().unwrap(),
        PartitionError::InvalidBufferLength {
            buffer: "gather output",
            expected: 4,
            actual: 3,
        }
    );
}

#[test]
fn rejects_assignment_to_an_unknown_leader() {
    let error = scatter_serial(&[7], &[2], 1, 2).unwrap_err();

    assert_eq!(
        error.downcast::<PartitionError>().unwrap(),
        PartitionError::InvalidBufferLength {
            buffer: "leader assignment",
            expected: 2,
            actual: 3,
        }
    );
}

#[test]
fn rejects_empty_and_oversized_leaves() {
    for (leaves, size) in [(vec![vec![]], 0), (vec![vec![0, 1, 2]], 3)] {
        let error = validate_leaves(&leaves, 2).unwrap_err();
        assert_eq!(
            error.downcast::<PartitionError>().unwrap(),
            PartitionError::InvalidLeaf { size, limit: 2 }
        );
    }
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::views::MatrixView;
use diskann_vector::distance::Metric;
use half::f16;
use std::collections::BTreeSet;

use super::{
    add_symmetric_edges, allocation_error, build_leaf_candidates, DirectCandidates, LeafBuffers,
    LeafBuildError,
};

fn view<T>(data: &[T], rows: usize, columns: usize) -> MatrixView<'_, T> {
    MatrixView::try_from(data, rows, columns).unwrap()
}

fn pool() -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
}

fn build<T>(
    data: MatrixView<'_, T>,
    leaves: &[Vec<u32>],
    k: usize,
    metric: Metric,
) -> Result<Vec<diskann::graph::AdjacencyList<u32>>, LeafBuildError>
where
    T: diskann::utils::VectorRepr + 'static,
{
    pool().install(|| build_leaf_candidates(data, leaves.to_vec(), k, metric))
}

fn rows(graph: Vec<diskann::graph::AdjacencyList<u32>>) -> Vec<Vec<u32>> {
    graph.into_iter().map(Vec::from).collect()
}

fn brute_force_symmetric_l2(data: &[[f32; 2]], k: usize) -> Vec<Vec<u32>> {
    let mut graph = vec![BTreeSet::new(); data.len()];
    for (source, left) in data.iter().enumerate() {
        let mut nearest: Vec<_> = data
            .iter()
            .enumerate()
            .filter(|(target, _)| *target != source)
            .map(|(target, right)| {
                let distance = left
                    .iter()
                    .zip(right)
                    .map(|(x, y)| (x - y) * (x - y))
                    .sum::<f32>();
                (target, distance)
            })
            .collect();
        nearest.sort_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        for &(target, _) in nearest.iter().take(k) {
            graph[source].insert(target as u32);
            graph[target].insert(source as u32);
        }
    }
    graph
        .into_iter()
        .map(|neighbors| neighbors.into_iter().collect())
        .collect()
}

#[test]
fn leaf_adjacency_matches_an_independent_all_pairs_reference() {
    let points = [
        [0.0_f32, 0.0],
        [1.0, 0.2],
        [3.1, 0.5],
        [7.8, 1.4],
        [-2.3, 4.1],
        [6.7, -3.2],
    ];
    let flat: Vec<_> = points.into_iter().flatten().collect();

    let actual = rows(
        build(
            view(&flat, points.len(), 2),
            &[(0..points.len() as u32).collect()],
            2,
            Metric::L2,
        )
        .unwrap(),
    );

    assert_eq!(actual, brute_force_symmetric_l2(&points, 2));
}

#[test]
fn retains_and_deduplicates_candidates_from_overlapping_leaves() {
    let data = [0.0_f32, 1.0, 2.0, 3.0];
    let leaves = vec![vec![0, 1, 2], vec![0, 2, 3], vec![0, 1, 2]];

    let graph = build(view(&data, 4, 1), &leaves, 2, Metric::L2).unwrap();

    assert_eq!(
        rows(graph),
        [vec![1, 2, 3], vec![0, 2], vec![0, 1, 3], vec![0, 2]]
    );
}

#[test]
fn symmetric_knn_can_give_one_point_more_than_two_k_candidates() {
    let dimensions = 9;
    let mut data = vec![0.0_f32; 10 * dimensions];
    for row in 1..10 {
        data[row * dimensions + row - 1] = 1.0;
    }

    let graph = build(
        view(&data, 10, dimensions),
        &[(0..10).collect()],
        1,
        Metric::L2,
    )
    .unwrap();

    assert_eq!(&*graph[0], &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    assert!(graph.iter().enumerate().all(|(source, neighbors)| {
        neighbors.iter().all(|&target| target as usize != source)
            && neighbors
                .iter()
                .all(|&target| graph[target as usize].contains(source as u32))
    }));
}

#[test]
fn global_id_translation_is_independent_of_leaf_order() {
    let data = [0.0_f32, 10.0, 20.0, 30.0, 40.0];
    let leaves = vec![vec![4, 1, 3]];

    let graph = build(view(&data, 5, 1), &leaves, 2, Metric::L2).unwrap();

    assert_eq!(
        rows(graph),
        [vec![], vec![3, 4], vec![], vec![1, 4], vec![1, 3]]
    );
}

fn assert_source_type<T>(data: &[T])
where
    T: diskann::utils::VectorRepr + 'static,
{
    let leaves = vec![vec![0, 1, 2, 3]];
    let graph = build(view(data, 4, 2), &leaves, 1, Metric::L2).unwrap();
    assert_eq!(rows(graph), [vec![1], vec![0, 2], vec![1, 3], vec![2]]);
}

#[test]
fn gathers_every_supported_source_type_without_full_dataset_conversion() {
    assert_source_type(&[0.0_f32, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
    assert_source_type(&[0_i8, 0, 1, 0, 2, 0, 3, 0]);
    assert_source_type(&[0_u8, 0, 1, 0, 2, 0, 3, 0]);
    assert_source_type(&[
        f16::from_f32(0.0),
        f16::from_f32(0.0),
        f16::from_f32(1.0),
        f16::from_f32(0.0),
        f16::from_f32(2.0),
        f16::from_f32(0.0),
        f16::from_f32(3.0),
        f16::from_f32(0.0),
    ]);
}

#[test]
fn all_metrics_produce_symmetric_unique_non_self_candidates() {
    let data = [1.0_f32, 0.0, 0.8, 0.2, 0.0, 1.0, -1.0, 0.0];
    let leaves = vec![vec![0, 1, 2, 3], vec![0, 1, 2, 3]];

    for metric in [
        Metric::L2,
        Metric::Cosine,
        Metric::CosineNormalized,
        Metric::InnerProduct,
    ] {
        let graph = build(view(&data, 4, 2), &leaves, 2, metric).unwrap();
        for (source, neighbors) in graph.iter().enumerate() {
            assert!(neighbors.iter().all(|&target| target as usize != source));
            assert!(neighbors
                .iter()
                .all(|&target| graph[target as usize].contains(source as u32)));
            assert!(neighbors.windows(2).all(|pair| pair[0] < pair[1]));
        }
    }
}

#[test]
fn parallel_leaf_schedule_does_not_change_candidate_order() {
    let data: Vec<f32> = (0..64).map(|value| value as f32).collect();
    let leaves: Vec<Vec<u32>> = (0..32)
        .map(|offset| (0..16).map(|point| (point + offset) % 64).collect())
        .collect();
    let pool = pool();
    pool.install(|| {
        let expected =
            build_leaf_candidates(view(&data, 64, 1), leaves.clone(), 2, Metric::L2).unwrap();
        for _ in 0..8 {
            let actual =
                build_leaf_candidates(view(&data, 64, 1), leaves.clone(), 2, Metric::L2).unwrap();
            assert_eq!(actual, expected);
        }
    });
}

#[test]
fn rejects_invalid_shape_inputs_without_panicking() {
    let data = [0.0_f32, 1.0];
    let no_dimensions = MatrixView::try_from(&data[..0], 2, 0).unwrap();
    assert!(matches!(
        build(no_dimensions, &[], 1, Metric::L2),
        Err(LeafBuildError::EmptyDimensions)
    ));
    assert!(matches!(
        build(view(&data, 2, 1), &[vec![]], 1, Metric::L2),
        Err(LeafBuildError::EmptyLeaf { leaf: 0 })
    ));
    assert!(matches!(
        build(view(&data, 2, 1), &[vec![0, 2]], 1, Metric::L2),
        Err(LeafBuildError::InvalidPointId {
            leaf: 0,
            point: 2,
            points: 2
        })
    ));
    assert!(matches!(
        build(view(&data, 2, 1), &[vec![2]], 1, Metric::L2),
        Err(LeafBuildError::InvalidPointId { point: 2, .. })
    ));
    assert!(matches!(
        build(view(&data, 2, 1), &[vec![0, 2]], 0, Metric::L2),
        Err(LeafBuildError::InvalidPointId { point: 2, .. })
    ));
    assert!(matches!(
        build(view(&data, 2, 1), &[vec![0, 0]], 1, Metric::L2),
        Err(LeafBuildError::DuplicatePointId { leaf: 0, point: 0 })
    ));
    assert!(matches!(
        build(view(&data, 2, 1), &[vec![1, 0, 1]], 1, Metric::L2),
        Err(LeafBuildError::DuplicatePointId { leaf: 0, point: 1 })
    ));
}

#[test]
fn singleton_and_zero_k_leaves_add_no_candidates() {
    let data = [0.0_f32, 1.0, 2.0];
    let singleton = build(
        view(&data, 3, 1),
        &[vec![0], vec![1], vec![2]],
        1,
        Metric::L2,
    )
    .unwrap();
    let zero_k = build(view(&data, 3, 1), &[vec![0, 1, 2]], 0, Metric::L2).unwrap();
    assert!(singleton.iter().chain(&zero_k).all(|row| row.is_empty()));
}

#[test]
fn reuses_worker_buffers_for_smaller_leaves() {
    let mut buffers = LeafBuffers::default();
    buffers.prepare(0, 64, 128, 2).unwrap();
    let points = buffers.points.as_ptr();
    let dots = buffers.dots.as_ptr();
    let nearest = buffers.nearest.as_ptr();

    buffers.prepare(1, 8, 128, 2).unwrap();

    assert_eq!(buffers.points.as_ptr(), points);
    assert_eq!(buffers.dots.as_ptr(), dots);
    assert_eq!(buffers.nearest.as_ptr(), nearest);
    assert_eq!(buffers.points.len(), 64 * 128);
    assert_eq!(buffers.dots.len(), 64 * 64);
    assert_eq!(buffers.nearest.len(), 64 * 2);
}

#[test]
fn reports_shape_overflow_before_allocating() {
    let mut buffers = LeafBuffers::default();
    assert!(matches!(
        buffers.prepare(7, usize::MAX, 2, 1),
        Err(LeafBuildError::ShapeOverflow { leaf: 7, .. })
    ));
}

#[test]
fn rejects_an_invalid_kernel_position() {
    let mut graph = vec![diskann::graph::AdjacencyList::new(); 2];
    let error = add_symmetric_edges(
        &[10, 20],
        1,
        &[
            crate::leaf_kernel::LeafNeighbor::new(9, 1.0),
            crate::leaf_kernel::LeafNeighbor::new(0, 1.0),
        ],
        &mut graph,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        LeafBuildError::InvalidLocalPosition {
            position: 9,
            points: 2
        }
    ));
}

#[test]
fn skips_duplicate_global_ids_without_self_edges() {
    let mut graph = vec![diskann::graph::AdjacencyList::new(); 2];
    add_symmetric_edges(
        &[7, 7],
        1,
        &[
            crate::leaf_kernel::LeafNeighbor::new(1, 0.0),
            crate::leaf_kernel::LeafNeighbor::new(0, 0.0),
        ],
        &mut graph,
    )
    .unwrap();
    assert!(graph.iter().all(|row| row.is_empty()));
}

#[test]
fn poisoned_candidate_rows_return_errors() {
    let candidates = DirectCandidates::new(1).unwrap();
    let _ = std::panic::catch_unwind(|| {
        let _guard = candidates.rows[0].lock().unwrap();
        panic!("poison candidate row");
    });
    assert!(matches!(
        candidates.add_leaf(&[0], &[diskann::graph::AdjacencyList::new()]),
        Err(LeafBuildError::PoisonedCandidateRow { point: 0 })
    ));
    assert!(matches!(
        candidates.into_rows(),
        Err(LeafBuildError::PoisonedCandidateRow { point: 0 })
    ));
}

#[test]
fn allocation_errors_preserve_buffer_context() {
    let mut values = Vec::<u8>::new();
    let source = values.try_reserve(usize::MAX).unwrap_err();
    let error = allocation_error("test", 1, source);
    assert!(matches!(
        error,
        LeafBuildError::Allocation {
            buffer: "test",
            additional: 1,
            ..
        }
    ));
}

#[test]
fn direct_candidate_accumulator_keeps_unique_sorted_rows() {
    let candidates = DirectCandidates::new(2).unwrap();
    candidates
        .add_leaf(
            &[0, 1],
            &[
                diskann::graph::AdjacencyList::from_iter_untrusted([1, 1]),
                diskann::graph::AdjacencyList::from_iter_untrusted([0]),
            ],
        )
        .unwrap();
    assert_eq!(rows(candidates.into_rows().unwrap()), [vec![1], vec![0]]);
}

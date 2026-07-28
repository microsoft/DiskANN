/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::iter;

use diskann_vector::distance::Metric;

use crate::{
    graph::{
        self, AdjacencyList, DiskANNIndex,
        config::PruneKind,
        test::provider::{self as test_provider, Provider, StartPoint},
    },
    provider::NeighborAccessor,
};

struct PruneCase {
    index: DiskANNIndex<Provider>,
    source: u32,
}

struct PruneConfig {
    metric: Metric,
    source: u32,
    degree: usize,
    alpha: f32,
    prune_kind: PruneKind,
    saturate: bool,
    max_occlusion_size: usize,
}

impl PruneCase {
    fn new(
        vectors: Vec<Vec<f32>>,
        candidates: impl IntoIterator<Item = u32>,
        config: PruneConfig,
    ) -> Self {
        let PruneConfig {
            metric,
            source,
            degree,
            alpha,
            prune_kind,
            saturate,
            max_occlusion_size,
        } = config;
        let dimensions = vectors.first().expect("a source vector is required").len();
        assert!(vectors.iter().all(|vector| vector.len() == dimensions));
        assert!((source as usize) < vectors.len());

        let mut source_neighbors = AdjacencyList::new();
        for candidate in candidates {
            source_neighbors.push(candidate);
        }

        let provider_max_degree = source_neighbors.len().max(degree);
        let start_id = vectors.len() as u32;
        let provider_config = test_provider::Config::new(
            metric,
            provider_max_degree,
            StartPoint::new(start_id, vec![0.0; dimensions]),
        )
        .unwrap();
        let points = vectors.into_iter().enumerate().map(|(id, vector)| {
            let neighbors = if id as u32 == source {
                source_neighbors.clone()
            } else {
                AdjacencyList::new()
            };
            (id as u32, vector, neighbors)
        });
        let provider = Provider::new_from(
            provider_config,
            iter::once((start_id, AdjacencyList::new())),
            points,
        )
        .unwrap();

        let config = graph::config::Builder::new_with(
            degree,
            graph::config::MaxDegree::new(provider_max_degree),
            10,
            prune_kind,
            |builder| {
                builder
                    .alpha(alpha)
                    .saturate_after_prune(saturate)
                    .max_occlusion_size(max_occlusion_size);
            },
        )
        .build()
        .unwrap();

        Self {
            index: DiskANNIndex::new(config, provider, None),
            source,
        }
    }

    async fn run(self, strategy: &test_provider::Strategy) -> AdjacencyList<u32> {
        self.index
            .prune_range(
                strategy,
                &test_provider::Context::default(),
                iter::once(self.source),
            )
            .await
            .unwrap();

        let mut neighbors = AdjacencyList::new();
        self.index
            .provider()
            .neighbors()
            .get_neighbors(self.source, &mut neighbors)
            .await
            .unwrap();
        neighbors
    }
}

fn l2_case(
    positions: &[f32],
    candidates: impl IntoIterator<Item = u32>,
    degree: usize,
    alpha: f32,
    saturate: bool,
    max_occlusion_size: usize,
) -> PruneCase {
    PruneCase::new(
        positions.iter().map(|position| vec![*position]).collect(),
        candidates,
        PruneConfig {
            metric: Metric::L2,
            source: 0,
            degree,
            alpha,
            prune_kind: PruneKind::TriangleInequality,
            saturate,
            max_occlusion_size,
        },
    )
}

#[tokio::test(flavor = "current_thread")]
async fn rows_at_or_below_degree_are_unchanged() {
    for (candidates, degree) in [(vec![], 2), (vec![2], 2), (vec![2, 1], 2)] {
        let expected = candidates.clone();
        let actual = l2_case(&[0.0, 1.0, -1.0], candidates, degree, 1.2, false, 10)
            .run(&test_provider::Strategy::new())
            .await;
        assert_eq!(&*actual, expected);
    }
}

#[tokio::test(flavor = "current_thread")]
async fn equal_distances_keep_current_sorted_neighbor_order() {
    let case = PruneCase::new(
        vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ],
        [3, 1, 2],
        PruneConfig {
            metric: Metric::L2,
            source: 0,
            degree: 2,
            alpha: 1.2,
            prune_kind: PruneKind::TriangleInequality,
            saturate: false,
            max_occlusion_size: 10,
        },
    );

    assert_eq!(&*case.run(&test_provider::Strategy::new()).await, &[2, 1]);
}

#[tokio::test(flavor = "current_thread")]
async fn triangle_prune_revisits_candidates_across_alpha_rounds() {
    let once = l2_case(
        &[0.0, 1.0, 8.0, 12.0, 16.0],
        [1, 2, 3, 4],
        3,
        1.2,
        false,
        10,
    )
    .run(&test_provider::Strategy::new())
    .await;
    let multiple = l2_case(
        &[0.0, 1.0, 8.0, 12.0, 16.0],
        [1, 2, 3, 4],
        3,
        1.44,
        false,
        10,
    )
    .run(&test_provider::Strategy::new())
    .await;

    assert_eq!(&*once, &[1, 3]);
    assert_eq!(&*multiple, &[1, 3, 2]);
}

#[tokio::test(flavor = "current_thread")]
async fn inner_product_uses_occluding_prune() {
    let case = PruneCase::new(
        vec![
            vec![1.0, 0.0],
            vec![3.0, 0.0],
            vec![2.0, 0.0],
            vec![1.0, 1.0],
        ],
        [1, 2, 3],
        PruneConfig {
            metric: Metric::InnerProduct,
            source: 0,
            degree: 2,
            alpha: 1.2,
            prune_kind: PruneKind::Occluding,
            saturate: false,
            max_occlusion_size: 10,
        },
    );

    assert_eq!(&*case.run(&test_provider::Strategy::new()).await, &[1, 2]);
}

#[tokio::test(flavor = "current_thread")]
async fn saturation_appends_candidates_in_pool_order() {
    let unsaturated = l2_case(&[0.0, 1.0, 2.0, 3.0, 4.0], 1..=4, 3, 1.2, false, 10)
        .run(&test_provider::Strategy::new())
        .await;
    let saturated = l2_case(&[0.0, 1.0, 2.0, 3.0, 4.0], 1..=4, 3, 1.2, true, 10)
        .run(&test_provider::Strategy::new())
        .await;

    assert_eq!(&*unsaturated, &[1]);
    assert_eq!(&*saturated, &[1, 2, 3]);
}

#[tokio::test(flavor = "current_thread")]
async fn configured_saturation_requires_alpha_above_one() {
    let neighbors = l2_case(&[0.0, 1.0, 2.0, 3.0, 4.0], 1..=4, 3, 1.0, true, 10)
        .run(&test_provider::Strategy::new())
        .await;

    assert_eq!(&*neighbors, &[1]);
}

#[tokio::test(flavor = "current_thread")]
async fn self_and_unavailable_candidates_are_excluded() {
    let case = l2_case(&[0.0, -1.0, 2.0, 3.0], [0, 1, 2, 3], 2, 1.2, false, 10);
    let strategy = test_provider::Strategy::with_transient(true, [2]);

    assert_eq!(&*case.run(&strategy).await, &[1, 3]);
}

#[tokio::test(flavor = "current_thread")]
async fn max_occlusion_size_truncates_to_nearest_candidates() {
    let case = PruneCase::new(
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 2.0],
            vec![-3.0, 0.0],
            vec![0.0, -4.0],
        ],
        [4, 3, 2, 1],
        PruneConfig {
            metric: Metric::L2,
            source: 0,
            degree: 3,
            alpha: 1.2,
            prune_kind: PruneKind::TriangleInequality,
            saturate: false,
            max_occlusion_size: 2,
        },
    );

    assert_eq!(&*case.run(&test_provider::Strategy::new()).await, &[1, 2]);
}

#[tokio::test(flavor = "current_thread")]
async fn maximum_u16_candidate_pool_is_supported() {
    let num_candidates = u16::MAX as usize;
    let vectors = (0..=num_candidates)
        .map(|position| vec![position as f32])
        .collect();
    let case = PruneCase::new(
        vectors,
        1..=u16::MAX as u32,
        PruneConfig {
            metric: Metric::L2,
            source: 0,
            degree: 1,
            alpha: 1.2,
            prune_kind: PruneKind::TriangleInequality,
            saturate: false,
            max_occlusion_size: num_candidates,
        },
    );

    let strategy = test_provider::Strategy::with_transient(true, 1..u16::MAX as u32);
    assert_eq!(&*case.run(&strategy).await, &[u16::MAX as u32]);
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::*;

fn input(metric: Metric, leaders: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dots = (0..2 * leaders)
        .map(|index| (((index * 13 + 7) % 29) as f32 - 14.0) * 0.125)
        .collect();
    let row_scales = if metric == Metric::Cosine {
        vec![0.0, 16.0]
    } else {
        Vec::new()
    };
    let leader_scales = match metric {
        Metric::L2 => (0..leaders).map(|leader| (leader + 1) as f32).collect(),
        Metric::Cosine => (0..leaders)
            .map(|leader| {
                if leader == 0 {
                    0.0
                } else {
                    (leader + 1) as f32
                }
            })
            .collect(),
        Metric::CosineNormalized | Metric::InnerProduct => Vec::new(),
    };
    (dots, row_scales, leader_scales)
}

#[test]
fn scalar_reference_matches_runtime_dispatch() {
    for metric in [
        Metric::L2,
        Metric::Cosine,
        Metric::CosineNormalized,
        Metric::InnerProduct,
    ] {
        for leaders in [7, 17] {
            let (dots, row_scales, leader_scales) = input(metric, leaders);
            let input = PartitionTopK {
                dots: &dots,
                rows: 2,
                leaders,
                row_scales: &row_scales,
                leader_scales: &leader_scales,
                metric,
            };
            for fanout in [1, 2, 6] {
                let mut expected = vec![u32::MAX; input.rows * fanout];
                nearest_leaders(input, fanout, &mut expected).unwrap();

                let mut actual = vec![u32::MAX; input.rows * fanout];
                process_rows_scalar(input, fanout, &mut actual);

                assert_eq!(
                    actual, expected,
                    "{metric:?}, leaders={leaders}, k={fanout}"
                );
            }
        }
    }
}

#[test]
fn scalar_distance_matches_metric_contract() {
    assert_eq!(distance(Metric::L2, 2.0, 99.0, 9.0), 5.0);
    assert_eq!(distance(Metric::CosineNormalized, 0.25, 99.0, 99.0), 0.75);
    assert_eq!(distance(Metric::InnerProduct, 3.0, 99.0, 99.0), -3.0);
    assert_eq!(distance(Metric::Cosine, 4.0, 4.0, 4.0), 0.5);
    assert_eq!(distance(Metric::Cosine, 4.0, 0.0, 4.0), 1.0);
}

#[test]
fn scalar_topk_orders_candidates_and_preserves_ties() {
    let mut top = [(u32::MAX, f32::INFINITY); MAX_PARTITION_FANOUT];
    for (leader, distance) in [(0, 4.0), (1, 1.0), (2, 3.0), (3, 2.0), (4, 1.0)] {
        insert_topk(&mut top, 4, leader, distance);
    }
    insert_topk(&mut top, 4, 5, f32::NAN);

    assert_eq!(top[..4], [(1, 1.0), (4, 1.0), (3, 2.0), (2, 3.0)]);
}

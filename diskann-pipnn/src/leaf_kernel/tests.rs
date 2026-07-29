/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::*;

fn dots(metric: Metric, points: usize) -> Vec<f32> {
    let mut dots = vec![f32::NAN; points * points];
    for row in 0..points {
        dots[row * points + row] = if metric == Metric::Cosine && row == 0 {
            0.0
        } else {
            1.0 + (row % 5) as f32
        };
        for column in 0..row {
            dots[row * points + column] = (((row * 17 + column * 11) % 23) as f32 - 11.0) * 0.03125;
        }
    }
    dots
}

fn norms(input: LeafTopK<'_>) -> Vec<f32> {
    (0..input.points)
        .map(|row| {
            let squared = input.dots[row * input.points + row];
            if input.metric == Metric::Cosine {
                if squared < f32::MIN_POSITIVE {
                    0.0
                } else {
                    squared.sqrt()
                }
            } else {
                squared
            }
        })
        .collect()
}

#[test]
fn scalar_reference_matches_runtime_dispatch() {
    for metric in [
        Metric::L2,
        Metric::Cosine,
        Metric::CosineNormalized,
        Metric::InnerProduct,
    ] {
        for points in [7, 17] {
            let dots = dots(metric, points);
            let input = LeafTopK {
                dots: &dots,
                points,
                metric,
            };
            for k in [1, 2, 3, 4] {
                let mut expected = vec![LeafNeighbor::default(); points * k];
                nearest_leaf_neighbors(input, k, &mut expected, &mut LeafTopKWorkspace::new())
                    .unwrap();

                let mut actual = vec![LeafNeighbor::default(); points * k];
                let mut worst = vec![f32::INFINITY; points];
                let norms = norms(input);
                process_pairs_scalar(input, k, &mut actual, &norms, &mut worst);

                assert_eq!(actual, expected, "{metric:?}, n={points}, k={k}");
            }
        }
    }
}

#[test]
fn scalar_insertion_orders_candidates_and_rejects_nan() {
    let mut output = [LeafNeighbor::default(); 4];
    let mut worst = [f32::INFINITY];

    for (position, distance) in [(0, 4.0), (1, 1.0), (2, 3.0), (3, 2.0), (4, 0.5)] {
        insert_row(&mut output, &mut worst, 4, 0, position, distance);
    }
    insert_row(&mut output, &mut worst, 4, 0, 5, f32::NAN);

    assert_eq!(
        output,
        [
            LeafNeighbor::new(4, 0.5),
            LeafNeighbor::new(1, 1.0),
            LeafNeighbor::new(3, 2.0),
            LeafNeighbor::new(2, 3.0),
        ]
    );
    assert_eq!(worst, [3.0]);
}

#[test]
fn workspace_can_shrink_and_grow_between_calls() {
    let mut workspace = LeafTopKWorkspace::new();
    for points in [17, 7, 17] {
        let dots = dots(Metric::L2, points);
        let mut output = vec![LeafNeighbor::default(); points * 2];
        nearest_leaf_neighbors(
            LeafTopK {
                dots: &dots,
                points,
                metric: Metric::L2,
            },
            2,
            &mut output,
            &mut workspace,
        )
        .unwrap();
        assert!(output.iter().all(|neighbor| neighbor.position != u32::MAX));
    }
}

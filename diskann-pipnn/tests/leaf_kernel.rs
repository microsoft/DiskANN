/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_pipnn::leaf_kernel::{
    nearest_leaf_neighbors, LeafKernelError, LeafNeighbor, LeafTopK, LeafTopKWorkspace,
};
use diskann_vector::distance::Metric;
use std::cmp::Ordering;

const SIMD_BOUNDARY_POINTS: [usize; 9] = [7, 8, 9, 15, 16, 17, 64, 256, 512];
const ZERO_NORM_POSITION: usize = 0;
const DISTINCT_NORM_POSITION: usize = 2;
const NORM_PERIOD: usize = 5;
const ROW_MIXER: usize = 17;
const COLUMN_MIXER: usize = 11;
const MIX_MODULUS: usize = 23;
const MIX_CENTER: f32 = 11.0;
const DOT_SCALE: f32 = 1.0 / 32.0;
const TIED_COLUMNS: [usize; 2] = [1, 2];

fn differential_input(metric: Metric, points: usize) -> Vec<f32> {
    let mut dots = vec![f32::NAN; points * points];
    for row in 0..points {
        dots[row * points + row] = if metric == Metric::Cosine && row == ZERO_NORM_POSITION {
            0.0
        } else if row == DISTINCT_NORM_POSITION {
            2.0
        } else {
            1.0 + (row % NORM_PERIOD) as f32
        };
        for column in 0..row {
            let pair =
                ((row * ROW_MIXER + column * COLUMN_MIXER) % MIX_MODULUS) as f32 - MIX_CENTER;
            dots[row * points + column] = if row == points - 1 && column == 0 {
                f32::NAN
            } else if TIED_COLUMNS.contains(&column) {
                0.5
            } else {
                pair * DOT_SCALE
            };
        }
    }
    dots
}

fn reference(input: LeafTopK<'_>, requested_k: usize) -> Vec<LeafNeighbor> {
    let k = requested_k.min(input.points.saturating_sub(1));
    let mut output = vec![LeafNeighbor::default(); input.points * k];
    if k == 0 {
        return output;
    }

    let norms: Vec<_> = (0..input.points)
        .map(|row| {
            let diagonal = input.dots[row * input.points + row];
            if input.metric == Metric::Cosine {
                if diagonal < f32::MIN_POSITIVE {
                    0.0
                } else {
                    diagonal.sqrt()
                }
            } else {
                diagonal
            }
        })
        .collect();

    for row in 0..input.points {
        let mut candidates = Vec::with_capacity(input.points - 1);
        for position in 0..input.points {
            if position == row {
                continue;
            }
            let (lower_row, lower_column) = if row > position {
                (row, position)
            } else {
                (position, row)
            };
            let dot = input.dots[lower_row * input.points + lower_column];
            let clamp = |distance: f32| {
                if distance < 0.0 {
                    0.0
                } else {
                    distance
                }
            };
            let distance = match input.metric {
                Metric::L2 => clamp(norms[row] + norms[position] - 2.0 * dot),
                Metric::CosineNormalized => clamp(1.0 - dot),
                Metric::InnerProduct => -dot,
                Metric::Cosine => {
                    let denominator = norms[row] * norms[position];
                    let similarity = if denominator == 0.0 {
                        0.0
                    } else {
                        dot / denominator
                    };
                    clamp(1.0 - similarity)
                }
            };
            if distance.partial_cmp(&f32::INFINITY) == Some(Ordering::Less) {
                candidates.push(LeafNeighbor::new(position as u32, distance));
            }
        }
        candidates.sort_by(|left, right| {
            left.distance
                .partial_cmp(&right.distance)
                .expect("NaN distances were filtered")
        });
        let count = candidates.len().min(k);
        output[row * k..row * k + count].copy_from_slice(&candidates[..count]);
    }
    output
}

#[test]
fn dispatch_matches_reference_across_simd_width_boundaries() {
    for metric in [
        Metric::L2,
        Metric::Cosine,
        Metric::CosineNormalized,
        Metric::InnerProduct,
    ] {
        // Straddle the 8- and 16-lane boundaries, then cover production leaf sizes.
        for points in SIMD_BOUNDARY_POINTS {
            let dots = differential_input(metric, points);
            let input = LeafTopK {
                dots: &dots,
                points,
                metric,
            };
            // Covers every specialized insertion arm (1, 2, 3), the first width
            // that falls back to the general bubble-up (4), and a wider row (5).
            for requested_k in [1, 2, 3, 4, 5] {
                let expected = reference(input, requested_k);
                let mut actual = vec![LeafNeighbor::default(); expected.len()];
                let mut workspace = LeafTopKWorkspace::new();
                nearest_leaf_neighbors(input, requested_k, &mut actual, &mut workspace).unwrap();
                assert_eq!(actual, expected, "{metric:?}, n={points}, k={requested_k}");
            }
        }
    }
}

fn run(dots: &[f32], points: usize, k: usize, metric: Metric) -> (usize, Vec<LeafNeighbor>) {
    let actual_k = k.min(points.saturating_sub(1));
    let mut output = vec![LeafNeighbor::default(); points * actual_k];
    let mut workspace = LeafTopKWorkspace::new();
    let returned_k = nearest_leaf_neighbors(
        LeafTopK {
            dots,
            points,
            metric,
        },
        k,
        &mut output,
        &mut workspace,
    )
    .unwrap();
    assert_eq!(returned_k, actual_k);
    (returned_k, output)
}

#[test]
fn l2_scans_only_the_lower_triangle_and_breaks_ties_by_position() {
    #[rustfmt::skip]
    let dots = [
        0.0, 999.0, 999.0, 999.0,
        0.0,   1.0, 999.0, 999.0,
        0.0,   0.0,   1.0, 999.0,
        0.0,   1.0,   1.0,   2.0,
    ];

    let (_, output) = run(&dots, 4, 2, Metric::L2);

    assert_eq!(
        output,
        [
            LeafNeighbor::new(1, 1.0),
            LeafNeighbor::new(2, 1.0),
            LeafNeighbor::new(0, 1.0),
            LeafNeighbor::new(3, 1.0),
            LeafNeighbor::new(0, 1.0),
            LeafNeighbor::new(3, 1.0),
            LeafNeighbor::new(1, 1.0),
            LeafNeighbor::new(2, 1.0),
        ]
    );
}

#[test]
fn supports_every_leaf_metric() {
    #[rustfmt::skip]
    let dots = [
        1.0, 77.0, 77.0,
        0.0,  1.0, 77.0,
       -1.0,  0.5,  1.0,
    ];

    let cases = [
        (Metric::L2, [1, 2, 1]),
        (Metric::Cosine, [1, 2, 1]),
        (Metric::CosineNormalized, [1, 2, 1]),
        (Metric::InnerProduct, [1, 2, 1]),
    ];

    for (metric, expected) in cases {
        let (_, output) = run(&dots, 3, 1, metric);
        let positions: Vec<_> = output.iter().map(|neighbor| neighbor.position).collect();
        assert_eq!(positions, expected, "metric {metric:?}");
    }
}

#[test]
fn cosine_treats_zero_norm_as_zero_similarity() {
    #[rustfmt::skip]
    let dots = [
        0.0, 11.0, 11.0,
        0.0,  1.0, 11.0,
        0.0,  0.0,  1.0,
    ];

    let (_, output) = run(&dots, 3, 2, Metric::Cosine);

    assert_eq!(output[0], LeafNeighbor::new(1, 1.0));
    assert_eq!(output[1], LeafNeighbor::new(2, 1.0));
}

#[test]
fn preserves_pipnn_metric_edge_semantics() {
    #[rustfmt::skip]
    let out_of_range = [
        1.0, 0.0,
        2.0, 1.0,
    ];
    assert_eq!(run(&out_of_range, 2, 1, Metric::L2).1[0].distance, 0.0);
    assert_eq!(
        run(&out_of_range, 2, 1, Metric::CosineNormalized).1[0].distance,
        0.0
    );
    assert_eq!(run(&out_of_range, 2, 1, Metric::Cosine).1[0].distance, 0.0);

    #[rustfmt::skip]
    let opposite = [
         1.0, 0.0,
        -2.0, 1.0,
    ];
    assert_eq!(run(&opposite, 2, 1, Metric::Cosine).1[0].distance, 3.0);

    let subnormal_squared_norm = f32::MIN_POSITIVE / 2.0;
    #[rustfmt::skip]
    let subnormal = [
        subnormal_squared_norm, 0.0,
        1.0,                    1.0,
    ];
    assert_eq!(run(&subnormal, 2, 1, Metric::Cosine).1[0].distance, 1.0);

    let minimum_normal_squared_norm = f32::MIN_POSITIVE;
    #[rustfmt::skip]
    let minimum_normal = [
        minimum_normal_squared_norm,          0.0,
        minimum_normal_squared_norm.sqrt(),   1.0,
    ];
    assert_eq!(
        run(&minimum_normal, 2, 1, Metric::Cosine).1[0].distance,
        0.0
    );
}

#[test]
fn finite_max_distance_fills_the_final_simd_slot() {
    let points = 9;
    let mut dots = vec![0.0; points * points];
    dots[8 * points] = -f32::MAX;

    let (actual_k, output) = run(&dots, points, points - 1, Metric::InnerProduct);

    assert_eq!(actual_k, 8);
    assert_eq!(
        output[8 * actual_k + actual_k - 1],
        LeafNeighbor::new(0, f32::MAX)
    );
}

#[test]
fn every_metric_ignores_nan_pairs() {
    #[rustfmt::skip]
    let dots = [
        1.0,       0.0, 0.0,
        f32::NAN,  1.0, 0.0,
        0.5,       0.25, 1.0,
    ];

    for metric in [
        Metric::L2,
        Metric::Cosine,
        Metric::CosineNormalized,
        Metric::InnerProduct,
    ] {
        let (_, output) = run(&dots, 3, 1, metric);
        assert_eq!(output[0].position, 2, "metric {metric:?}");
        assert_eq!(output[1].position, 2, "metric {metric:?}");
    }
}

#[test]
fn rejects_incomplete_neighbor_rows() {
    #[rustfmt::skip]
    let dots = [
        1.0,      0.0,
        f32::NAN, 1.0,
    ];
    let mut output = [LeafNeighbor::default(); 2];
    let mut workspace = LeafTopKWorkspace::new();

    let error = nearest_leaf_neighbors(
        LeafTopK {
            dots: &dots,
            points: 2,
            metric: Metric::L2,
        },
        1,
        &mut output,
        &mut workspace,
    )
    .unwrap_err();

    assert_eq!(
        error,
        LeafKernelError::InsufficientRankableNeighbors {
            row: 0,
            neighbors: 1,
        }
    );
}

#[test]
fn clamps_k_to_available_non_self_neighbors() {
    #[rustfmt::skip]
    let dots = [
        1.0, 3.0, 3.0,
        0.0, 1.0, 3.0,
        0.0, 0.0, 1.0,
    ];

    let (actual_k, output) = run(&dots, 3, 99, Metric::L2);

    assert_eq!(actual_k, 2);
    assert_eq!(output.len(), 6);
    for (row, neighbors) in output.chunks_exact(actual_k).enumerate() {
        assert!(neighbors
            .iter()
            .all(|neighbor| neighbor.position as usize != row));
    }
}

#[test]
fn accepts_empty_singleton_and_zero_k_inputs() {
    let mut workspace = LeafTopKWorkspace::new();
    let empty = LeafTopK {
        dots: &[],
        points: 0,
        metric: Metric::L2,
    };
    assert_eq!(
        nearest_leaf_neighbors(empty, 2, &mut [], &mut workspace).unwrap(),
        0
    );

    let singleton = LeafTopK {
        dots: &[4.0],
        points: 1,
        metric: Metric::Cosine,
    };
    assert_eq!(
        nearest_leaf_neighbors(singleton, 2, &mut [], &mut workspace).unwrap(),
        0
    );

    let pair = LeafTopK {
        dots: &[1.0, 0.0, 0.0, 1.0],
        points: 2,
        metric: Metric::InnerProduct,
    };
    assert_eq!(
        nearest_leaf_neighbors(pair, 0, &mut [], &mut workspace).unwrap(),
        0
    );
}

#[test]
fn rejects_invalid_shapes_before_dispatch() {
    let mut workspace = LeafTopKWorkspace::new();
    let error = nearest_leaf_neighbors(
        LeafTopK {
            dots: &[0.0; 8],
            points: 3,
            metric: Metric::L2,
        },
        1,
        &mut [LeafNeighbor::default(); 3],
        &mut workspace,
    )
    .unwrap_err();
    assert_eq!(
        error,
        LeafKernelError::InvalidBufferLength {
            buffer: "lower dot-product matrix",
            expected: 9,
            actual: 8,
        }
    );

    let error = nearest_leaf_neighbors(
        LeafTopK {
            dots: &[0.0; 9],
            points: 3,
            metric: Metric::L2,
        },
        2,
        &mut [LeafNeighbor::default(); 5],
        &mut workspace,
    )
    .unwrap_err();
    assert_eq!(
        error,
        LeafKernelError::InvalidBufferLength {
            buffer: "output",
            expected: 6,
            actual: 5,
        }
    );
}

#[test]
fn rejects_shape_overflow_before_reading_buffers() {
    let mut workspace = LeafTopKWorkspace::new();
    let error = nearest_leaf_neighbors(
        LeafTopK {
            dots: &[],
            points: usize::MAX,
            metric: Metric::L2,
        },
        1,
        &mut [],
        &mut workspace,
    )
    .unwrap_err();

    assert_eq!(error, LeafKernelError::TooManyPoints(usize::MAX));
}

#[test]
fn cosine_zero_norm_masks_nan_norm_at_simd_boundaries() {
    for points in [9, 17] {
        let mut dots = vec![0.0; points * points];
        dots[0] = 0.0;
        for row in 1..points {
            dots[row * points + row] = f32::NAN;
        }

        let (_, output) = run(&dots, points, 1, Metric::Cosine);
        for (row, neighbor) in output.iter().enumerate().skip(1) {
            assert_eq!(
                *neighbor,
                LeafNeighbor::new(0, 1.0),
                "n={points}, row={row}"
            );
        }
    }
}

#[cfg(target_pointer_width = "64")]
#[test]
fn accepts_the_largest_representable_point_count_before_shape_validation() {
    let points = u32::MAX as usize;
    let expected = points.checked_mul(points).unwrap();
    let mut workspace = LeafTopKWorkspace::new();

    let error = nearest_leaf_neighbors(
        LeafTopK {
            dots: &[],
            points,
            metric: Metric::InnerProduct,
        },
        0,
        &mut [],
        &mut workspace,
    )
    .unwrap_err();

    assert_eq!(
        error,
        LeafKernelError::InvalidBufferLength {
            buffer: "lower dot-product matrix",
            expected,
            actual: 0,
        }
    );
}

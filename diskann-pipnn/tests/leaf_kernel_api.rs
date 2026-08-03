/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::cmp::Ordering;

use diskann_pipnn::leaf_kernel::{
    leaf_output_len, LeafKernel, LeafKernelError, LeafNeighbor, LeafTopK, LeafTopKWorkspace,
};
use diskann_utils::views::{MatrixView, MutMatrixView};
use diskann_vector::distance::Metric;

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

fn input(dots: &[f32], points: usize) -> LeafTopK<'_> {
    LeafTopK {
        dots: MatrixView::try_from(dots, points, points).unwrap(),
    }
}

fn reference(dots: &[f32], points: usize, requested_k: usize, metric: Metric) -> Vec<LeafNeighbor> {
    let k = requested_k.min(points.saturating_sub(1));
    let mut output = vec![LeafNeighbor::default(); points * k];
    if k == 0 {
        return output;
    }

    let norms: Vec<_> = (0..points)
        .map(|row| {
            let diagonal = dots[row * points + row];
            if metric == Metric::Cosine {
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

    for row in 0..points {
        let mut candidates = Vec::with_capacity(points - 1);
        for position in 0..points {
            if position == row {
                continue;
            }
            let (lower_row, lower_column) = if row > position {
                (row, position)
            } else {
                (position, row)
            };
            let dot = dots[lower_row * points + lower_column];
            let clamp = |distance: f32| if distance < 0.0 { 0.0 } else { distance };
            let distance = match metric {
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

fn run(dots: &[f32], points: usize, k: usize, metric: Metric) -> (usize, Vec<LeafNeighbor>) {
    let actual_k = k.min(points.saturating_sub(1));
    let mut output = vec![LeafNeighbor::default(); points * actual_k];
    let returned_k = LeafKernel::new(metric, k)
        .nearest_neighbors(
            input(dots, points),
            MutMatrixView::try_from(output.as_mut_slice(), points, actual_k).unwrap(),
            &mut LeafTopKWorkspace::new(),
        )
        .unwrap();
    assert_eq!(returned_k, actual_k);
    (returned_k, output)
}

#[test]
fn prepared_dispatch_matches_reference_across_simd_width_boundaries() {
    for metric in [
        Metric::L2,
        Metric::Cosine,
        Metric::CosineNormalized,
        Metric::InnerProduct,
    ] {
        for points in SIMD_BOUNDARY_POINTS {
            let dots = differential_input(metric, points);
            for requested_k in [1, 2, 3, 4, 5] {
                let expected = reference(&dots, points, requested_k, metric);
                let actual = run(&dots, points, requested_k, metric).1;
                assert_eq!(actual, expected, "{metric:?}, n={points}, k={requested_k}");
            }
        }
    }
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

    assert_eq!(
        run(&dots, 4, 2, Metric::L2).1,
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
    for (metric, expected) in [
        (Metric::L2, [1, 2, 1]),
        (Metric::Cosine, [1, 2, 1]),
        (Metric::CosineNormalized, [1, 2, 1]),
        (Metric::InnerProduct, [1, 2, 1]),
    ] {
        let positions: Vec<_> = run(&dots, 3, 1, metric)
            .1
            .iter()
            .map(|neighbor| neighbor.position)
            .collect();
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

    let output = run(&dots, 3, 2, Metric::Cosine).1;
    assert_eq!(output[0], LeafNeighbor::new(1, 1.0));
    assert_eq!(output[1], LeafNeighbor::new(2, 1.0));
}

#[test]
fn preserves_pipnn_metric_edge_semantics() {
    #[rustfmt::skip]
    let out_of_range = [1.0, 0.0, 2.0, 1.0];
    assert_eq!(run(&out_of_range, 2, 1, Metric::L2).1[0].distance, 0.0);
    assert_eq!(
        run(&out_of_range, 2, 1, Metric::CosineNormalized).1[0].distance,
        0.0
    );
    assert_eq!(run(&out_of_range, 2, 1, Metric::Cosine).1[0].distance, 0.0);

    #[rustfmt::skip]
    let opposite = [1.0, 0.0, -2.0, 1.0];
    assert_eq!(run(&opposite, 2, 1, Metric::Cosine).1[0].distance, 3.0);

    let subnormal = [f32::MIN_POSITIVE / 2.0, 0.0, 1.0, 1.0];
    assert_eq!(run(&subnormal, 2, 1, Metric::Cosine).1[0].distance, 1.0);

    let minimum_normal = [f32::MIN_POSITIVE, 0.0, f32::MIN_POSITIVE.sqrt(), 1.0];
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
        let output = run(&dots, 3, 1, metric).1;
        assert_eq!(output[0].position, 2, "metric {metric:?}");
        assert_eq!(output[1].position, 2, "metric {metric:?}");
    }
}

#[test]
fn rejects_incomplete_neighbor_rows() {
    let dots = [1.0, 0.0, f32::NAN, 1.0];
    let mut output = [LeafNeighbor::default(); 2];
    let error = LeafKernel::new(Metric::L2, 1)
        .nearest_neighbors(
            input(&dots, 2),
            MutMatrixView::try_from(&mut output[..], 2, 1).unwrap(),
            &mut LeafTopKWorkspace::new(),
        )
        .unwrap_err();

    assert_eq!(
        error,
        LeafKernelError::InsufficientRankableNeighbors {
            row: 0,
            neighbors: 1
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
    for (row, neighbors) in output.chunks_exact(actual_k).enumerate() {
        assert!(neighbors
            .iter()
            .all(|neighbor| neighbor.position as usize != row));
    }
}

#[test]
fn accepts_empty_singleton_and_zero_k_inputs() {
    for (dots, points, k, metric) in [
        (&[][..], 0, 2, Metric::L2),
        (&[4.0][..], 1, 2, Metric::Cosine),
        (&[1.0, 0.0, 0.0, 1.0][..], 2, 0, Metric::InnerProduct),
    ] {
        assert_eq!(run(dots, points, k, metric).0, 0);
    }
}

#[test]
fn rejects_non_square_input_and_wrong_output_shape() {
    let dots = [0.0; 6];
    let non_square = LeafTopK {
        dots: MatrixView::try_from(&dots[..], 2, 3).unwrap(),
    };
    let mut output = [LeafNeighbor::default(); 2];
    let kernel = LeafKernel::new(Metric::L2, 1);
    assert_eq!(
        kernel.nearest_neighbors(
            non_square,
            MutMatrixView::try_from(&mut output[..], 2, 1).unwrap(),
            &mut LeafTopKWorkspace::new(),
        ),
        Err(LeafKernelError::NonSquareDots { rows: 2, cols: 3 })
    );

    let square = [0.0; 9];
    let mut wrong = [LeafNeighbor::default(); 3];
    assert_eq!(
        LeafKernel::new(Metric::L2, 2).nearest_neighbors(
            input(&square, 3),
            MutMatrixView::try_from(&mut wrong[..], 3, 1).unwrap(),
            &mut LeafTopKWorkspace::new(),
        ),
        Err(LeafKernelError::InvalidOutputShape {
            expected_rows: 3,
            expected_cols: 2,
            actual_rows: 3,
            actual_cols: 1,
        })
    );
}

#[test]
fn cosine_zero_norm_masks_nan_norm_at_simd_boundaries() {
    for points in [9, 17] {
        let mut dots = vec![0.0; points * points];
        for row in 1..points {
            dots[row * points + row] = f32::NAN;
        }

        let output = run(&dots, points, 1, Metric::Cosine).1;
        for (row, neighbor) in output.iter().enumerate().skip(1) {
            assert_eq!(
                *neighbor,
                LeafNeighbor::new(0, 1.0),
                "n={points}, row={row}"
            );
        }
    }
}

#[test]
fn output_length_rejects_unrepresentable_point_count() {
    assert_eq!(
        leaf_output_len(usize::MAX, 1),
        Err(LeafKernelError::TooManyPoints(usize::MAX))
    );
}

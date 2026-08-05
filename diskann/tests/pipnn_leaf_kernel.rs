/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#![cfg(feature = "pipnn")]

use std::cmp::Ordering;

use diskann::graph::pipnn::leaf_kernel::{
    LeafInput, LeafKernel, LeafKernelError, LeafKernelWorkspace, LeafNeighbor, leaf_neighbor_count,
    leaf_output_len,
};
use diskann_utils::views::{MatrixView, MutMatrixView};
use diskann_vector::distance::Metric;

const SIMD_BOUNDARY_POINTS: [usize; 15] = [2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 256, 512];
const ZERO_NORM_POSITION: usize = 0;
const DISTINCT_NORM_POSITION: usize = 2;
const NORM_PERIOD: usize = 5;
const SOURCE_MIXER: usize = 17;
const TARGET_MIXER: usize = 11;
const MIX_MODULUS: usize = 23;
const MIX_CENTER: f32 = 11.0;
const DOT_SCALE: f32 = 1.0 / 32.0;
const TIED_TARGETS: [usize; 2] = [1, 2];

fn differential_dots(metric: Metric, points: usize) -> Vec<f32> {
    let mut dots = vec![f32::NAN; points * points];
    for source in 0..points {
        dots[source * points + source] = if metric == Metric::Cosine && source == ZERO_NORM_POSITION
        {
            0.0
        } else if source == DISTINCT_NORM_POSITION {
            2.0
        } else {
            1.0 + (source % NORM_PERIOD) as f32
        };
        for target in 0..source {
            let pair =
                ((source * SOURCE_MIXER + target * TARGET_MIXER) % MIX_MODULUS) as f32 - MIX_CENTER;
            dots[source * points + target] = if TIED_TARGETS.contains(&target) {
                0.5
            } else {
                pair * DOT_SCALE
            };
        }
    }
    dots
}

fn test_input(dots: &[f32], points: usize) -> LeafInput<'_> {
    LeafInput {
        dots: MatrixView::try_from(dots, points, points).unwrap(),
    }
}

fn brute_force_reference(
    dots: &[f32],
    points: usize,
    requested_k: usize,
    metric: Metric,
) -> Vec<LeafNeighbor> {
    let leaf_k = requested_k.min(points.saturating_sub(1));
    let mut output = vec![LeafNeighbor::default(); points * leaf_k];
    if leaf_k == 0 {
        return output;
    }

    let norms: Vec<_> = (0..points)
        .map(|source| {
            let diagonal = dots[source * points + source];
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

    for source in 0..points {
        let mut candidates = Vec::with_capacity(points - 1);
        for target in 0..points {
            if target == source {
                continue;
            }
            let (lower_source, lower_target) = if source > target {
                (source, target)
            } else {
                (target, source)
            };
            let dot = dots[lower_source * points + lower_target];
            let clamp = |distance: f32| if distance < 0.0 { 0.0 } else { distance };
            let distance = match metric {
                Metric::L2 => clamp(norms[source] + norms[target] - 2.0 * dot),
                Metric::CosineNormalized => clamp(1.0 - dot),
                Metric::InnerProduct => -dot,
                Metric::Cosine => {
                    let denominator = norms[source] * norms[target];
                    let similarity = if denominator == 0.0 {
                        0.0
                    } else {
                        dot / denominator
                    };
                    clamp(1.0 - similarity)
                }
            };
            if distance.partial_cmp(&f32::INFINITY) == Some(Ordering::Less) {
                candidates.push(LeafNeighbor::new(target as u32, distance));
            }
        }
        candidates.sort_by(|left, right| {
            left.distance
                .partial_cmp(&right.distance)
                .expect("NaN distances were filtered")
        });
        let count = candidates.len().min(leaf_k);
        output[source * leaf_k..source * leaf_k + count].copy_from_slice(&candidates[..count]);
    }
    output
}

fn run_kernel(
    dots: &[f32],
    points: usize,
    requested_k: usize,
    metric: Metric,
) -> (usize, Vec<LeafNeighbor>) {
    let leaf_k = leaf_neighbor_count(points, requested_k).unwrap();
    let mut output = vec![LeafNeighbor::default(); points * leaf_k];
    LeafKernel::new(metric)
        .nearest_neighbors(
            test_input(dots, points),
            MutMatrixView::try_from(output.as_mut_slice(), points, leaf_k).unwrap(),
            &mut LeafKernelWorkspace::new(),
        )
        .unwrap();
    (leaf_k, output)
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
            let dots = differential_dots(metric, points);
            for requested_k in [1, 2, 3, 4, 5] {
                let expected = brute_force_reference(&dots, points, requested_k, metric);
                let actual = run_kernel(&dots, points, requested_k, metric).1;
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
        run_kernel(&dots, 4, 2, Metric::L2).1,
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
        let positions: Vec<_> = run_kernel(&dots, 3, 1, metric)
            .1
            .iter()
            .map(|neighbor| neighbor.target)
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

    let output = run_kernel(&dots, 3, 2, Metric::Cosine).1;
    assert_eq!(output[0], LeafNeighbor::new(1, 1.0));
    assert_eq!(output[1], LeafNeighbor::new(2, 1.0));
}

#[test]
fn clamps_negative_distances_and_preserves_cosine_extremes() {
    #[rustfmt::skip]
    let out_of_range = [1.0, 0.0, 2.0, 1.0];
    assert_eq!(
        run_kernel(&out_of_range, 2, 1, Metric::L2).1[0].distance,
        0.0
    );
    assert_eq!(
        run_kernel(&out_of_range, 2, 1, Metric::CosineNormalized).1[0].distance,
        0.0
    );
    assert_eq!(
        run_kernel(&out_of_range, 2, 1, Metric::Cosine).1[0].distance,
        0.0
    );

    #[rustfmt::skip]
    let opposite = [1.0, 0.0, -2.0, 1.0];
    assert_eq!(
        run_kernel(&opposite, 2, 1, Metric::Cosine).1[0].distance,
        3.0
    );

    let subnormal = [f32::MIN_POSITIVE / 2.0, 0.0, 1.0, 1.0];
    assert_eq!(
        run_kernel(&subnormal, 2, 1, Metric::Cosine).1[0].distance,
        1.0
    );

    let minimum_normal = [f32::MIN_POSITIVE, 0.0, f32::MIN_POSITIVE.sqrt(), 1.0];
    assert_eq!(
        run_kernel(&minimum_normal, 2, 1, Metric::Cosine).1[0].distance,
        0.0
    );
}

#[test]
fn finite_max_distance_fills_the_final_simd_slot() {
    let points = 9;
    let mut dots = vec![0.0; points * points];
    dots[8 * points] = -f32::MAX;

    let (leaf_k, output) = run_kernel(&dots, points, points - 1, Metric::InnerProduct);
    assert_eq!(leaf_k, 8);
    assert_eq!(
        output[8 * leaf_k + leaf_k - 1],
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
        let output = run_kernel(&dots, 3, 1, metric).1;
        assert_eq!(output[0].target, 2, "metric {metric:?}");
        assert_eq!(output[1].target, 2, "metric {metric:?}");
    }
}

#[test]
fn rejects_sources_with_too_few_rankable_neighbors() {
    let dots = [1.0, 0.0, f32::NAN, 1.0];
    let mut output = [LeafNeighbor::default(); 2];
    let error = LeafKernel::new(Metric::L2)
        .nearest_neighbors(
            test_input(&dots, 2),
            MutMatrixView::try_from(&mut output[..], 2, 1).unwrap(),
            &mut LeafKernelWorkspace::new(),
        )
        .unwrap_err();

    assert_eq!(
        error,
        LeafKernelError::InsufficientRankableNeighbors {
            source_index: 0,
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
    let (leaf_k, output) = run_kernel(&dots, 3, 99, Metric::L2);

    assert_eq!(leaf_k, 2);
    for (source, neighbors) in output.chunks_exact(leaf_k).enumerate() {
        assert!(
            neighbors
                .iter()
                .all(|neighbor| neighbor.target as usize != source)
        );
    }
}

#[test]
fn accepts_empty_singleton_and_zero_k_inputs() {
    for (dots, points, requested_k, metric) in [
        (&[][..], 0, 2, Metric::L2),
        (&[4.0][..], 1, 2, Metric::Cosine),
        (&[1.0, 0.0, 0.0, 1.0][..], 2, 0, Metric::InnerProduct),
    ] {
        assert_eq!(run_kernel(dots, points, requested_k, metric).0, 0);
    }
}

#[test]
fn rejects_non_square_input_and_invalid_output_dimensions() {
    let dots = [0.0; 6];
    let non_square = LeafInput {
        dots: MatrixView::try_from(&dots[..], 2, 3).unwrap(),
    };
    let mut output = [LeafNeighbor::default(); 2];
    let kernel = LeafKernel::new(Metric::L2);
    assert_eq!(
        kernel.nearest_neighbors(
            non_square,
            MutMatrixView::try_from(&mut output[..], 2, 1).unwrap(),
            &mut LeafKernelWorkspace::new(),
        ),
        Err(LeafKernelError::NonSquareDots { rows: 2, cols: 3 })
    );

    let square = [0.0; 9];
    let mut wrong_rows = [LeafNeighbor::default(); 2];
    assert_eq!(
        kernel.nearest_neighbors(
            test_input(&square, 3),
            MutMatrixView::try_from(&mut wrong_rows[..], 2, 1).unwrap(),
            &mut LeafKernelWorkspace::new(),
        ),
        Err(LeafKernelError::InvalidOutputRows {
            expected: 3,
            actual: 2,
            columns: 1,
        })
    );

    let mut too_many = [LeafNeighbor::default(); 9];
    assert_eq!(
        kernel.nearest_neighbors(
            test_input(&square, 3),
            MutMatrixView::try_from(&mut too_many[..], 3, 3).unwrap(),
            &mut LeafKernelWorkspace::new(),
        ),
        Err(LeafKernelError::InvalidNeighborCount {
            points: 3,
            neighbors: 3,
            maximum: 2,
        })
    );
}

#[test]
fn cosine_zero_norm_masks_nan_norm_at_simd_boundaries() {
    for points in [9, 17] {
        let mut dots = vec![0.0; points * points];
        for source in 1..points {
            dots[source * points + source] = f32::NAN;
        }

        let output = run_kernel(&dots, points, 1, Metric::Cosine).1;
        for (source, neighbor) in output.iter().enumerate().skip(1) {
            assert_eq!(
                *neighbor,
                LeafNeighbor::new(0, 1.0),
                "n={points}, source={source}"
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

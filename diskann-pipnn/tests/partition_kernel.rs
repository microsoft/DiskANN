/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_pipnn::partition_kernel::{
    nearest_leaders, PartitionKernelError, PartitionTopK, MAX_PARTITION_FANOUT,
};
use diskann_vector::distance::Metric;

fn reference(input: PartitionTopK<'_>, fanout: usize) -> Vec<u32> {
    let mut output = vec![u32::MAX; input.rows * fanout];
    for (row_index, (dots, output)) in input
        .dots
        .chunks_exact(input.leaders)
        .zip(output.chunks_exact_mut(fanout))
        .enumerate()
    {
        let row_scale = input.row_scales.get(row_index).copied().unwrap_or(0.0);
        let mut candidates: Vec<_> = dots
            .iter()
            .enumerate()
            .filter_map(|(leader, &dot)| {
                let leader_scale = input.leader_scales.get(leader).copied().unwrap_or(0.0);
                let distance = match input.metric {
                    Metric::L2 => leader_scale - 2.0 * dot,
                    Metric::CosineNormalized => 1.0 - dot,
                    Metric::InnerProduct => -dot,
                    Metric::Cosine => {
                        let denominator = row_scale.sqrt() * leader_scale;
                        1.0 - if denominator > 0.0 {
                            dot / denominator
                        } else {
                            0.0
                        }
                    }
                };
                (distance.partial_cmp(&f32::INFINITY) == Some(std::cmp::Ordering::Less))
                    .then_some((leader as u32, distance))
            })
            .collect();
        candidates.sort_by(|left, right| left.1.partial_cmp(&right.1).unwrap());
        for (destination, (leader, _)) in output.iter_mut().zip(candidates) {
            *destination = leader;
        }
    }
    output
}

fn differential_input(metric: Metric, leaders: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dots = (0..2 * leaders)
        .map(|index| {
            let leader = index % leaders;
            let row = index / leaders;
            let base = ((leader * 13 + row * 7) % 19) as f32 - 9.0;
            if leader == 2 || leader == 3 {
                1.0
            } else if leader + 1 == leaders {
                f32::NAN
            } else {
                base * 0.25
            }
        })
        .collect();
    let row_scales = if metric == Metric::Cosine {
        vec![0.0, 16.0]
    } else {
        Vec::new()
    };
    let leader_scales = match metric {
        Metric::Cosine => (0..leaders)
            .map(|leader| {
                if leader == 1 {
                    0.0
                } else if leader == 2 || leader == 3 {
                    3.0
                } else {
                    1.0 + leader as f32
                }
            })
            .collect(),
        Metric::L2 => (0..leaders)
            .map(|leader| {
                let norm = if leader == 2 || leader == 3 {
                    3.0
                } else {
                    leader as f32 + 1.0
                };
                norm * norm
            })
            .collect(),
        Metric::CosineNormalized | Metric::InnerProduct => Vec::new(),
    };
    (dots, row_scales, leader_scales)
}

#[test]
fn dispatch_matches_reference_across_simd_width_boundaries() {
    for metric in [
        Metric::L2,
        Metric::Cosine,
        Metric::CosineNormalized,
        Metric::InnerProduct,
    ] {
        for leaders in [7, 8, 9, 15, 16, 17] {
            let (dots, row_scales, leader_scales) = differential_input(metric, leaders);
            for fanout in [1, 2, 16] {
                if fanout >= leaders {
                    continue;
                }
                let input = PartitionTopK {
                    dots: &dots,
                    rows: 2,
                    leaders,
                    row_scales: &row_scales,
                    leader_scales: &leader_scales,
                    metric,
                };
                let expected = reference(input, fanout);
                let mut actual = vec![u32::MAX; expected.len()];
                nearest_leaders(input, fanout, &mut actual).unwrap();
                assert_eq!(
                    actual, expected,
                    "{metric:?}, leaders={leaders}, k={fanout}"
                );
            }
        }
    }
}

#[test]
fn l2_keeps_the_first_leader_when_boundary_distances_tie() {
    #[rustfmt::skip]
    let dots = [
        0.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 4.0, 6.0,
    ];
    let leader_squared_norms = [0.0, 1.0, 4.0, 9.0];
    let mut assignments = [u32::MAX; 4];

    let input = PartitionTopK {
        dots: &dots,
        rows: 2,
        leaders: 4,
        row_scales: &[],
        leader_scales: &leader_squared_norms,
        metric: Metric::L2,
    };

    nearest_leaders(input, 2, &mut assignments).unwrap();

    assert_eq!(assignments, [0, 1, 2, 1]);
}

#[test]
fn supports_every_partition_metric() {
    #[rustfmt::skip]
    let dots = [
        1.0, 0.0, -1.0,
        2.0, 6.0, 0.0,
    ];

    let cases = [
        (Metric::L2, &[][..], &[1.0, 4.0, 9.0][..], [0, 1, 1, 0]),
        (
            Metric::Cosine,
            &[1.0, 4.0][..],
            &[1.0, 2.0, 3.0][..],
            [0, 1, 1, 0],
        ),
        (Metric::CosineNormalized, &[][..], &[][..], [0, 1, 1, 0]),
        (Metric::InnerProduct, &[][..], &[][..], [0, 1, 1, 0]),
    ];

    for (metric, row_scales, leader_scales, expected) in cases {
        let mut assignments = [u32::MAX; 4];
        nearest_leaders(
            PartitionTopK {
                dots: &dots,
                rows: 2,
                leaders: 3,
                row_scales,
                leader_scales,
                metric,
            },
            2,
            &mut assignments,
        )
        .unwrap();

        assert_eq!(assignments, expected, "metric {metric:?}");
    }
}

#[test]
fn cosine_treats_a_zero_norm_as_zero_similarity() {
    let mut assignments = [u32::MAX; 2];

    nearest_leaders(
        PartitionTopK {
            dots: &[100.0, -100.0],
            rows: 1,
            leaders: 2,
            row_scales: &[0.0],
            leader_scales: &[1.0, 1.0],
            metric: Metric::Cosine,
        },
        2,
        &mut assignments,
    )
    .unwrap();

    assert_eq!(assignments, [0, 1]);
}

#[test]
fn finite_max_distance_fills_the_final_simd_slot() {
    let mut assignments = [u32::MAX; 8];
    let mut dots = [0.0; 8];
    dots[7] = -f32::MAX;

    nearest_leaders(
        PartitionTopK {
            dots: &dots,
            rows: 1,
            leaders: 8,
            row_scales: &[],
            leader_scales: &[],
            metric: Metric::InnerProduct,
        },
        8,
        &mut assignments,
    )
    .unwrap();

    assert_eq!(assignments, [0, 1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn ignores_nan_distances_without_displacing_finite_leaders() {
    let mut assignments = [u32::MAX; 2];

    nearest_leaders(
        PartitionTopK {
            dots: &[f32::NAN, 3.0, 2.0],
            rows: 1,
            leaders: 3,
            row_scales: &[],
            leader_scales: &[],
            metric: Metric::InnerProduct,
        },
        2,
        &mut assignments,
    )
    .unwrap();

    assert_eq!(assignments, [1, 2]);
}

#[test]
fn rejects_rows_with_too_few_rankable_distances() {
    let error = nearest_leaders(
        PartitionTopK {
            dots: &[f32::NAN, 3.0],
            rows: 1,
            leaders: 2,
            row_scales: &[],
            leader_scales: &[],
            metric: Metric::InnerProduct,
        },
        2,
        &mut [u32::MAX; 2],
    )
    .unwrap_err();

    assert_eq!(
        error,
        PartitionKernelError::InsufficientRankableDistances { row: 0, fanout: 2 }
    );
}

#[test]
fn accepts_empty_rows_and_zero_fanout() {
    nearest_leaders(
        PartitionTopK {
            dots: &[],
            rows: 0,
            leaders: 3,
            row_scales: &[],
            leader_scales: &[],
            metric: Metric::InnerProduct,
        },
        2,
        &mut [],
    )
    .unwrap();

    nearest_leaders(
        PartitionTopK {
            dots: &[1.0, 2.0, 3.0],
            rows: 1,
            leaders: 3,
            row_scales: &[],
            leader_scales: &[],
            metric: Metric::InnerProduct,
        },
        0,
        &mut [],
    )
    .unwrap();

    // `u32::MAX` leaders still have positions representable by `u32`: the
    // largest position is `u32::MAX - 1`. An empty batch lets us exercise the
    // validation boundary without allocating the declared tile.
    nearest_leaders(
        PartitionTopK {
            dots: &[],
            rows: 0,
            leaders: u32::MAX as usize,
            row_scales: &[],
            leader_scales: &[],
            metric: Metric::InnerProduct,
        },
        0,
        &mut [],
    )
    .unwrap();

    #[cfg(target_pointer_width = "64")]
    assert_eq!(
        nearest_leaders(
            PartitionTopK {
                dots: &[],
                rows: 0,
                leaders: u32::MAX as usize + 1,
                row_scales: &[],
                leader_scales: &[],
                metric: Metric::InnerProduct,
            },
            0,
            &mut [],
        ),
        Err(PartitionKernelError::TooManyLeaders(u32::MAX as usize + 1))
    );
}

#[test]
fn rejects_inconsistent_shapes_and_fanout() {
    let base = PartitionTopK {
        dots: &[0.0; 6],
        rows: 2,
        leaders: 3,
        row_scales: &[],
        leader_scales: &[],
        metric: Metric::InnerProduct,
    };

    assert_eq!(
        nearest_leaders(
            PartitionTopK {
                dots: &[0.0; 5],
                ..base
            },
            2,
            &mut [0; 4],
        ),
        Err(PartitionKernelError::InvalidBufferLength {
            buffer: "dot-product tile",
            expected: 6,
            actual: 5,
        })
    );
    assert_eq!(
        nearest_leaders(base, 2, &mut [0; 3]),
        Err(PartitionKernelError::InvalidBufferLength {
            buffer: "output",
            expected: 4,
            actual: 3,
        })
    );
    assert_eq!(
        nearest_leaders(base, MAX_PARTITION_FANOUT + 1, &mut []),
        Err(PartitionKernelError::InvalidFanout {
            fanout: MAX_PARTITION_FANOUT + 1,
            leaders: 3,
            maximum: MAX_PARTITION_FANOUT,
        })
    );

    let one_leader = PartitionTopK {
        dots: &[0.0],
        rows: 1,
        leaders: 1,
        row_scales: &[],
        leader_scales: &[],
        metric: Metric::InnerProduct,
    };
    assert_eq!(
        nearest_leaders(one_leader, 2, &mut []),
        Err(PartitionKernelError::InvalidFanout {
            fanout: 2,
            leaders: 1,
            maximum: MAX_PARTITION_FANOUT,
        })
    );

    let exact_maximum = PartitionTopK {
        dots: &[],
        rows: 0,
        leaders: MAX_PARTITION_FANOUT,
        row_scales: &[],
        leader_scales: &[],
        metric: Metric::InnerProduct,
    };
    nearest_leaders(exact_maximum, MAX_PARTITION_FANOUT, &mut []).unwrap();
}

#[test]
fn rejects_shape_overflow_before_reading_buffers() {
    let error = nearest_leaders(
        PartitionTopK {
            dots: &[],
            rows: usize::MAX,
            leaders: 2,
            row_scales: &[],
            leader_scales: &[],
            metric: Metric::InnerProduct,
        },
        1,
        &mut [],
    )
    .unwrap_err();

    assert_eq!(
        error,
        PartitionKernelError::ShapeOverflow {
            buffer: "dot-product tile",
            rows: usize::MAX,
            cols: 2,
        }
    );
}

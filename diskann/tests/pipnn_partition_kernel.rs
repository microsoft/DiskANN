/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#![cfg(feature = "pipnn")]
#![allow(
    clippy::unwrap_used,
    reason = "deterministic test fixture construction must abort on invalid setup"
)]

use diskann::graph::pipnn::partition_kernel::{
    MAX_PARTITION_FANOUT, PartitionInput, PartitionKernel, PartitionKernelError, PartitionScales,
};
use diskann_utils::views::{MatrixView, MutMatrixView};
use diskann_vector::distance::Metric;

fn test_input<'a>(
    metric: Metric,
    dots: &'a [f32],
    point_count: usize,
    leader_count: usize,
    point_scales: &'a [f32],
    leader_scales: &'a [f32],
) -> PartitionInput<'a> {
    let scales = match metric {
        Metric::L2 => PartitionScales::L2 {
            leader_squared_norms: leader_scales,
        },
        Metric::Cosine => PartitionScales::Cosine {
            point_squared_norms: point_scales,
            leader_norms: leader_scales,
        },
        Metric::CosineNormalized | Metric::InnerProduct => PartitionScales::None,
    };
    PartitionInput {
        dots: MatrixView::try_from(dots, point_count, leader_count).unwrap(),
        scales,
    }
}

fn brute_force_reference(input: PartitionInput<'_>, fanout: usize, metric: Metric) -> Vec<u32> {
    let point_count = input.dots.nrows();
    let leader_count = input.dots.ncols();
    let (point_scales, leader_scales) = match input.scales {
        PartitionScales::L2 {
            leader_squared_norms,
        } => (&[][..], leader_squared_norms),
        PartitionScales::Cosine {
            point_squared_norms,
            leader_norms,
        } => (point_squared_norms, leader_norms),
        PartitionScales::None => (&[][..], &[][..]),
    };
    let mut assignments = vec![u32::MAX; point_count * fanout];
    for (point, (point_dots, point_assignments)) in input
        .dots
        .as_slice()
        .chunks_exact(leader_count)
        .zip(assignments.chunks_exact_mut(fanout))
        .enumerate()
    {
        let point_scale = point_scales.get(point).copied().unwrap_or(0.0);
        let mut candidates: Vec<_> = point_dots
            .iter()
            .enumerate()
            .filter_map(|(leader, &dot)| {
                let leader_scale = leader_scales.get(leader).copied().unwrap_or(0.0);
                let distance = match metric {
                    Metric::L2 => leader_scale - 2.0 * dot,
                    Metric::CosineNormalized => 1.0 - dot,
                    Metric::InnerProduct => -dot,
                    Metric::Cosine => {
                        let point_norm = if point_scale < f32::MIN_POSITIVE {
                            0.0
                        } else {
                            point_scale.sqrt()
                        };
                        1.0 - if point_norm == 0.0 || leader_scale == 0.0 {
                            0.0
                        } else {
                            dot / (point_norm * leader_scale)
                        }
                    }
                };
                (distance.partial_cmp(&f32::INFINITY) == Some(std::cmp::Ordering::Less))
                    .then_some((leader as u32, distance))
            })
            .collect();
        candidates.sort_by(|left, right| left.1.partial_cmp(&right.1).unwrap());
        for (destination, (leader, _)) in point_assignments.iter_mut().zip(candidates) {
            *destination = leader;
        }
    }
    assignments
}

fn differential_data(metric: Metric, leader_count: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dots = (0..2 * leader_count)
        .map(|index| {
            let leader = index % leader_count;
            let point = index / leader_count;
            let base = ((leader * 13 + point * 7) % 19) as f32 - 9.0;
            if leader == 2 || leader == 3 {
                1.0
            } else if leader + 1 == leader_count {
                f32::NAN
            } else {
                base * 0.25
            }
        })
        .collect();
    let point_scales = if metric == Metric::Cosine {
        vec![0.0, 16.0]
    } else {
        Vec::new()
    };
    let leader_scales = match metric {
        Metric::Cosine => (0..leader_count)
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
        Metric::L2 => (0..leader_count)
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
    (dots, point_scales, leader_scales)
}

fn run(
    metric: Metric,
    input: PartitionInput<'_>,
    fanout: usize,
) -> Result<Vec<u32>, PartitionKernelError> {
    let mut output = vec![u32::MAX; input.dots.nrows() * fanout];
    PartitionKernel::new(metric).nearest_leaders(
        input,
        MutMatrixView::try_from(output.as_mut_slice(), input.dots.nrows(), fanout).unwrap(),
    )?;
    Ok(output)
}

#[test]
fn prepared_dispatch_matches_reference_across_simd_width_boundaries() {
    for metric in [
        Metric::L2,
        Metric::Cosine,
        Metric::CosineNormalized,
        Metric::InnerProduct,
    ] {
        for leader_count in [2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
            let (dots, point_scales, leader_scales) = differential_data(metric, leader_count);
            let input = test_input(
                metric,
                &dots,
                2,
                leader_count,
                &point_scales,
                &leader_scales,
            );
            for fanout in [1, 2, 16] {
                if fanout >= leader_count {
                    continue;
                }
                assert_eq!(
                    run(metric, input, fanout).unwrap(),
                    brute_force_reference(input, fanout, metric),
                    "{metric:?}, leaders={leader_count}, k={fanout}"
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
    let norms = [0.0, 1.0, 4.0, 9.0];

    assert_eq!(
        run(
            Metric::L2,
            test_input(Metric::L2, &dots, 2, 4, &[], &norms),
            2
        )
        .unwrap(),
        [0, 1, 2, 1]
    );
}

#[test]
fn supports_every_partition_metric() {
    #[rustfmt::skip]
    let dots = [
        1.0, 0.0, -1.0,
        2.0, 6.0, 0.0,
    ];
    for (metric, point_scales, leader_scales, expected) in [
        (Metric::L2, &[][..], &[1.0, 4.0, 9.0][..], [0, 1, 1, 0]),
        (
            Metric::Cosine,
            &[1.0, 4.0][..],
            &[1.0, 2.0, 3.0][..],
            [0, 1, 1, 0],
        ),
        (Metric::CosineNormalized, &[][..], &[][..], [0, 1, 1, 0]),
        (Metric::InnerProduct, &[][..], &[][..], [0, 1, 1, 0]),
    ] {
        assert_eq!(
            run(
                metric,
                test_input(metric, &dots, 2, 3, point_scales, leader_scales),
                2,
            )
            .unwrap(),
            expected,
            "metric {metric:?}"
        );
    }
}

#[test]
fn cosine_treats_a_zero_norm_as_zero_similarity() {
    assert_eq!(
        run(
            Metric::Cosine,
            test_input(Metric::Cosine, &[100.0, -100.0], 1, 2, &[0.0], &[1.0, 1.0]),
            2,
        )
        .unwrap(),
        [0, 1]
    );
}

#[test]
fn finite_max_distance_fills_the_final_simd_slot() {
    let mut dots = [0.0; 8];
    dots[7] = -f32::MAX;
    assert_eq!(
        run(
            Metric::InnerProduct,
            test_input(Metric::InnerProduct, &dots, 1, 8, &[], &[]),
            8
        )
        .unwrap(),
        [0, 1, 2, 3, 4, 5, 6, 7]
    );
}

#[test]
fn ignores_nan_distances_without_displacing_finite_leaders() {
    assert_eq!(
        run(
            Metric::InnerProduct,
            test_input(Metric::InnerProduct, &[f32::NAN, 3.0, 2.0], 1, 3, &[], &[]),
            2,
        )
        .unwrap(),
        [1, 2]
    );
}

#[test]
fn rejects_points_with_too_few_rankable_leaders() {
    assert_eq!(
        run(
            Metric::InnerProduct,
            test_input(Metric::InnerProduct, &[f32::NAN, 3.0], 1, 2, &[], &[]),
            2,
        ),
        Err(PartitionKernelError::InsufficientRankableLeaders {
            point: 0,
            fanout: 2,
        })
    );
}

#[test]
fn accepts_empty_points_zero_fanout_and_largest_leader_id() {
    run(
        Metric::InnerProduct,
        test_input(Metric::InnerProduct, &[], 0, 3, &[], &[]),
        2,
    )
    .unwrap();
    run(
        Metric::InnerProduct,
        test_input(Metric::InnerProduct, &[1.0, 2.0, 3.0], 1, 3, &[], &[]),
        0,
    )
    .unwrap();
    run(
        Metric::InnerProduct,
        test_input(Metric::InnerProduct, &[], 0, u32::MAX as usize, &[], &[]),
        0,
    )
    .unwrap();

    #[cfg(target_pointer_width = "64")]
    assert_eq!(
        run(
            Metric::InnerProduct,
            test_input(
                Metric::InnerProduct,
                &[],
                0,
                u32::MAX as usize + 1,
                &[],
                &[],
            ),
            0,
        ),
        Err(PartitionKernelError::TooManyLeaders(u32::MAX as usize + 1))
    );
}

#[test]
fn rejects_wrong_output_scales_and_fanout() {
    let dots = [0.0; 6];
    let valid_input = test_input(Metric::InnerProduct, &dots, 2, 3, &[], &[]);
    let mut wrong_output = [u32::MAX; 3];
    assert_eq!(
        PartitionKernel::new(Metric::InnerProduct).nearest_leaders(
            valid_input,
            MutMatrixView::try_from(&mut wrong_output[..], 1, 3).unwrap(),
        ),
        Err(PartitionKernelError::InvalidOutputShape {
            expected_rows: 2,
            actual_rows: 1,
            actual_cols: 3,
        })
    );

    let wrong_scales = PartitionInput {
        dots: MatrixView::try_from(&dots[..], 2, 3).unwrap(),
        scales: PartitionScales::None,
    };
    assert_eq!(
        run(Metric::L2, wrong_scales, 2),
        Err(PartitionKernelError::InvalidScales { expected: "L2" })
    );

    assert_eq!(
        run(Metric::InnerProduct, valid_input, MAX_PARTITION_FANOUT + 1,),
        Err(PartitionKernelError::InvalidFanout {
            fanout: MAX_PARTITION_FANOUT + 1,
            leader_count: 3,
            maximum: MAX_PARTITION_FANOUT,
        })
    );

    let one = [0.0];
    assert_eq!(
        run(
            Metric::InnerProduct,
            test_input(Metric::InnerProduct, &one, 1, 1, &[], &[]),
            2,
        ),
        Err(PartitionKernelError::InvalidFanout {
            fanout: 2,
            leader_count: 1,
            maximum: MAX_PARTITION_FANOUT,
        })
    );
}

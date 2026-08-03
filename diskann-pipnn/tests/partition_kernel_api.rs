/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_pipnn::partition_kernel::{
    PartitionKernel, PartitionKernelError, PartitionScales, PartitionTopK, MAX_PARTITION_FANOUT,
};
use diskann_utils::views::{MatrixView, MutMatrixView};
use diskann_vector::distance::Metric;

fn input<'a>(
    metric: Metric,
    dots: &'a [f32],
    rows: usize,
    leaders: usize,
    row_scales: &'a [f32],
    leader_scales: &'a [f32],
) -> PartitionTopK<'a> {
    let scales = match metric {
        Metric::L2 => PartitionScales::L2 {
            leader_squared_norms: leader_scales,
        },
        Metric::Cosine => PartitionScales::Cosine {
            row_squared_norms: row_scales,
            leader_norms: leader_scales,
        },
        Metric::CosineNormalized | Metric::InnerProduct => PartitionScales::None,
    };
    PartitionTopK {
        dots: MatrixView::try_from(dots, rows, leaders).unwrap(),
        scales,
    }
}

fn reference(input: PartitionTopK<'_>, fanout: usize, metric: Metric) -> Vec<u32> {
    let rows = input.dots.nrows();
    let leaders = input.dots.ncols();
    let (row_scales, leader_scales) = match input.scales {
        PartitionScales::L2 {
            leader_squared_norms,
        } => (&[][..], leader_squared_norms),
        PartitionScales::Cosine {
            row_squared_norms,
            leader_norms,
        } => (row_squared_norms, leader_norms),
        PartitionScales::None => (&[][..], &[][..]),
    };
    let mut output = vec![u32::MAX; rows * fanout];
    for (row, (dots, output)) in input
        .dots
        .as_slice()
        .chunks_exact(leaders)
        .zip(output.chunks_exact_mut(fanout))
        .enumerate()
    {
        let row_scale = row_scales.get(row).copied().unwrap_or(0.0);
        let mut candidates: Vec<_> = dots
            .iter()
            .enumerate()
            .filter_map(|(leader, &dot)| {
                let leader_scale = leader_scales.get(leader).copied().unwrap_or(0.0);
                let distance = match metric {
                    Metric::L2 => leader_scale - 2.0 * dot,
                    Metric::CosineNormalized => 1.0 - dot,
                    Metric::InnerProduct => -dot,
                    Metric::Cosine => {
                        let row_norm = if row_scale < f32::MIN_POSITIVE {
                            0.0
                        } else {
                            row_scale.sqrt()
                        };
                        1.0 - if row_norm == 0.0 || leader_scale == 0.0 {
                            0.0
                        } else {
                            dot / (row_norm * leader_scale)
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

fn run(
    metric: Metric,
    input: PartitionTopK<'_>,
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
        for leaders in [7, 8, 9, 15, 16, 17] {
            let (dots, row_scales, leader_scales) = differential_input(metric, leaders);
            let input = input(metric, &dots, 2, leaders, &row_scales, &leader_scales);
            for fanout in [1, 2, 16] {
                if fanout >= leaders {
                    continue;
                }
                assert_eq!(
                    run(metric, input, fanout).unwrap(),
                    reference(input, fanout, metric),
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
    let norms = [0.0, 1.0, 4.0, 9.0];

    assert_eq!(
        run(Metric::L2, input(Metric::L2, &dots, 2, 4, &[], &norms), 2).unwrap(),
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
    for (metric, rows, leaders, expected) in [
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
            run(metric, input(metric, &dots, 2, 3, rows, leaders), 2).unwrap(),
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
            input(Metric::Cosine, &[100.0, -100.0], 1, 2, &[0.0], &[1.0, 1.0]),
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
            input(Metric::InnerProduct, &dots, 1, 8, &[], &[]),
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
            input(Metric::InnerProduct, &[f32::NAN, 3.0, 2.0], 1, 3, &[], &[]),
            2,
        )
        .unwrap(),
        [1, 2]
    );
}

#[test]
fn rejects_rows_with_too_few_rankable_distances() {
    assert_eq!(
        run(
            Metric::InnerProduct,
            input(Metric::InnerProduct, &[f32::NAN, 3.0], 1, 2, &[], &[]),
            2,
        ),
        Err(PartitionKernelError::InsufficientRankableDistances { row: 0, fanout: 2 })
    );
}

#[test]
fn accepts_empty_rows_zero_fanout_and_largest_leader_id() {
    run(
        Metric::InnerProduct,
        input(Metric::InnerProduct, &[], 0, 3, &[], &[]),
        2,
    )
    .unwrap();
    run(
        Metric::InnerProduct,
        input(Metric::InnerProduct, &[1.0, 2.0, 3.0], 1, 3, &[], &[]),
        0,
    )
    .unwrap();
    run(
        Metric::InnerProduct,
        input(Metric::InnerProduct, &[], 0, u32::MAX as usize, &[], &[]),
        0,
    )
    .unwrap();

    #[cfg(target_pointer_width = "64")]
    assert_eq!(
        run(
            Metric::InnerProduct,
            input(
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
    let valid_input = input(Metric::InnerProduct, &dots, 2, 3, &[], &[]);
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

    let wrong_scales = PartitionTopK {
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
            leaders: 3,
            maximum: MAX_PARTITION_FANOUT,
        })
    );

    let one = [0.0];
    assert_eq!(
        run(
            Metric::InnerProduct,
            input(Metric::InnerProduct, &one, 1, 1, &[], &[]),
            2,
        ),
        Err(PartitionKernelError::InvalidFanout {
            fanout: 2,
            leaders: 1,
            maximum: MAX_PARTITION_FANOUT,
        })
    );
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use rand::{Rng, RngCore};

use super::*;

fn refs(rows: &[Vec<f32>]) -> Vec<PointRef<'_, usize>> {
    rows.iter()
        .enumerate()
        .map(|(id, vector)| PointRef { id, vector })
        .collect()
}

#[test]
fn packed_paired_preserves_centers_sums_and_rng() {
    let mut data_rng = StdRng::seed_from_u64(123);
    for dim in [1, 7, 8, 9, 31, 32, 33, 128, 896] {
        let rows: Vec<Vec<f32>> = (0..37)
            .map(|_| (0..dim).map(|_| data_rng.random_range(-2.0..2.0)).collect())
            .collect();
        let points = refs(&rows);
        for iterations in [0, 1, 5] {
            for seed in [0, 7, 42] {
                let mut reference = vec![0.0; 2 * dim];
                let mut expected_sums = vec![0.0; 2 * dim];
                let mut reference_rng = StdRng::seed_from_u64(seed);
                clustering::fit_two_means(
                    &points,
                    &mut reference,
                    dim,
                    iterations,
                    &mut reference_rng,
                    &mut expected_sums,
                )
                .unwrap();
                for strategy in [
                    Strategy::Separate,
                    Strategy::PairedPoint,
                    Strategy::PackedPaired,
                ] {
                    let mut centers = vec![0.0; 2 * dim];
                    let mut sums = vec![0.0; 2 * dim];
                    let mut rng = StdRng::seed_from_u64(seed);
                    fit_into(
                        &points,
                        &mut centers,
                        dim,
                        iterations,
                        &mut rng,
                        &mut sums,
                        strategy,
                    )
                    .unwrap();
                    assert_eq!(
                        centers.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        reference.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                    );
                    assert_eq!(sums, expected_sums);
                    let mut check_rng = reference_rng.clone();
                    assert_eq!(rng.next_u64(), check_rng.next_u64());
                }
            }
        }
    }
}

#[test]
fn complete_fit_matches_independent_small_integer_means() {
    let rows = vec![vec![-10.0; 3], vec![10.0; 3]];
    for strategy in Strategy::ALL {
        let run = fit(&refs(&rows), 3, 3, 7, strategy).unwrap();
        let mut centers: Vec<_> = run.centers.chunks_exact(3).collect();
        centers.sort_by(|a, b| a[0].total_cmp(&b[0]));
        assert_eq!(centers, [&[-10.0; 3][..], &[10.0; 3][..]]);
        assert!(run.margins.iter().all(|x| *x == 1200.0));
        assert_eq!(
            run.packed_bytes,
            if matches!(strategy, Strategy::PackedPaired | Strategy::Sgemm) {
                24
            } else {
                0
            }
        );
    }
}

#[test]
fn identical_points_keep_ties_and_empty_centers_including_overflow() {
    for dim in [1, 7, 8, 9, 33, 128] {
        for value in [-0.0, 2.0, f32::MAX] {
            let rows = vec![vec![value; dim]; 9];
            for strategy in Strategy::ALL {
                let run = fit(&refs(&rows), dim, 5, 42, strategy).unwrap();
                assert!(run.assignments.iter().all(|&to| to == 0));
                assert!(run.centers.iter().all(|&x| x == value));
            }
        }
    }
}

#[test]
fn invalid_shapes_fail_before_kernel_access() {
    for rows in [
        vec![],
        vec![vec![1.0]],
        vec![vec![], vec![]],
        vec![vec![0.0], vec![0.0, 1.0]],
    ] {
        for strategy in Strategy::ALL {
            assert!(fit(&refs(&rows), 1, 1, 0, strategy).is_err());
        }
    }
}

#[test]
fn finite_cancellation_is_observable_not_hidden_by_a_tolerance() {
    // True distances are 0.0625 and 0.5625. Expanding squared norms loses
    // the deciding low bits in f32. Paired L2 retains them.
    let point = [10000.25f32];
    let points = [PointRef {
        id: 0,
        vector: point.as_slice(),
    }];
    let centers = [10000.0, 10001.0];
    assert_eq!(paired(&point, &centers, 1), 0);
    let mut runner = Runner::new(Strategy::CenterDifference, 1, 1, &[]);
    let mut sums = [0.0; 2];
    let mut counts = [0; 2];
    runner
        .assign_and_sum(&points, &centers, 1, &mut sums, &mut counts)
        .unwrap();
    assert_eq!(
        counts,
        [0, 1],
        "the experimental formula must expose its loss of precision"
    );
}

#[test]
fn dot_product_overflow_falls_back_to_paired_for_finite_inputs() {
    let rows = vec![vec![f32::MAX; 9], vec![-f32::MAX; 9], vec![0.0; 9]];
    let points = refs(&rows);
    let centers = [rows[0].clone(), rows[1].clone()].concat();
    let mut expected_sums = vec![0.0; 18];
    let mut expected_counts = [0; 2];
    for point in &points {
        clustering::sum_assignment(
            point.vector,
            usize::from(paired(point.vector, &centers, 9)),
            9,
            &mut expected_sums,
            &mut expected_counts,
        );
    }
    assert_eq!(expected_counts, [2, 1]);
    let packed: Vec<_> = rows.iter().flatten().copied().collect();
    for strategy in [Strategy::Sgemm, Strategy::CenterDifference] {
        let mut runner = Runner::new(strategy, rows.len(), 9, &packed);
        let mut sums = vec![0.0; 18];
        let mut counts = [0; 2];
        runner
            .assign_and_sum(&points, &centers, 9, &mut sums, &mut counts)
            .unwrap();
        assert_eq!(counts, expected_counts);
        assert_eq!(sums, expected_sums);
    }
}

struct BatchedCases;
impl<A: Architecture> diskann_wide::arch::Target<A, ()> for BatchedCases {
    #[inline]
    fn run(self, arch: A) {
        for dim in [1, 3, 7, 8, 9, 15, 16, 17, 33, 128] {
            for count in [2, 3, 4, 5, 7, 8, 9, 17] {
                let rows: Vec<Vec<f32>> = (0..count)
                    .map(|i| {
                        (0..dim)
                            .map(|d| ((i * 13 + d * 7) % 17) as f32 - 8.0)
                            .collect()
                    })
                    .collect();
                let points = refs(&rows);
                let centers = [vec![-4.0; dim], vec![5.0; dim]].concat();
                let expected: Vec<_> = rows
                    .iter()
                    .map(|row| {
                        let a: f64 = row.iter().map(|&x| (f64::from(x) + 4.0).powi(2)).sum();
                        let b: f64 = row.iter().map(|&x| (f64::from(x) - 5.0).powi(2)).sum();
                        u8::from(b < a)
                    })
                    .collect();
                let mut output = vec![255; count];
                arch.run3_inline(
                    Batched::<4> { dim },
                    points.as_slice(),
                    centers.as_slice(),
                    output.as_mut_slice(),
                );
                assert_eq!(output, expected);
                arch.run3_inline(
                    Batched::<8> { dim },
                    points.as_slice(),
                    centers.as_slice(),
                    output.as_mut_slice(),
                );
                assert_eq!(output, expected);
            }
        }
    }
}

#[test]
fn batched_scalar_matches_integer_oracle() {
    diskann_wide::arch::Scalar.run(BatchedCases);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn batched_x86_matches_integer_oracle() {
    if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
        arch.run(BatchedCases);
    }
    if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked_miri() {
        arch.run(BatchedCases);
    } else {
        assert_ne!(
            std::env::var("WIDE_TEST_MIN_ARCH").as_deref(),
            Ok("x86-64-v4")
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn batched_neon_matches_integer_oracle() {
    if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
        arch.run(BatchedCases);
    }
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use rand::{Rng, RngCore, SeedableRng};

use super::super::two_means::nearest_two;
use super::*;

const STRATEGIES: [Strategy; 2] = [Strategy::PairedPoint, Strategy::TiledExact];

fn refs(rows: &[Vec<f32>]) -> Vec<PointRef<'_, usize>> {
    rows.iter()
        .enumerate()
        .map(|(id, vector)| PointRef { id, vector })
        .collect()
}

#[test]
fn separated_rows_preserve_means_sums_and_rng() {
    let mut data_rng = StdRng::seed_from_u64(123);
    for dim in [1, 7, 8, 9, 31, 32, 33, 128, 896] {
        let rows: Vec<Vec<f32>> = (0..37)
            .map(|i| {
                (0..dim).map(|_| if i % 2 == 0 { -10.0 } else { 10.0 }
                + data_rng.random_range(-1..=1) as f32).collect()
            })
            .collect();
        let points = refs(&rows);
        for iterations in [0, 1, 5, 10] {
            for seed in [0, 7, 42] {
                let mut reference = vec![0.0; 2 * dim];
                let mut expected_sums = vec![0.0; 2 * dim];
                let mut reference_rng = StdRng::seed_from_u64(seed);
                fit_two_means(
                    &points,
                    &mut reference,
                    dim,
                    iterations,
                    &mut reference_rng,
                    &mut expected_sums,
                )
                .unwrap();
                for strategy in STRATEGIES {
                    let mut centers = vec![0.0; 2 * dim];
                    let mut sums = vec![0.0; 2 * dim];
                    let mut rng = StdRng::seed_from_u64(seed);
                    TwoMeans::new(&points, dim, strategy)
                        .unwrap()
                        .fit_into(&mut centers, iterations, &mut rng, &mut sums)
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
    for strategy in STRATEGIES {
        let points = refs(&rows);
        let prepared = TwoMeans::new(&points, 3, strategy).unwrap();
        let mut fitted = vec![0.0; 6];
        prepared
            .fit_into(&mut fitted, 3, &mut StdRng::seed_from_u64(7), &mut [0.0; 6])
            .unwrap();
        let mut centers: Vec<_> = fitted.chunks_exact(3).collect();
        centers.sort_by(|a, b| a[0].total_cmp(&b[0]));
        assert_eq!(centers, [&[-10.0; 3][..], &[10.0; 3][..]]);
        assert_eq!(prepared.packed_bytes(), 0);
    }
}

#[test]
fn exact_assignment_preserves_overflow_and_cancellation_choices() {
    for rows in [
        [vec![f32::MAX], vec![0.0]],
        [vec![10000.25], vec![10000.25]],
    ] {
        let points = refs(&rows);
        let prepared = TwoMeans::new(&points, 1, Strategy::TiledExact).unwrap();
        for centers in [[0.0, 1.0], [10001.0, 10000.0]] {
            let mut out = [99; 2];
            assert_eq!(prepared.assign(&centers, &mut out).unwrap(), 0);
            assert_eq!(
                out,
                rows.each_ref()
                    .map(|row| nearest_two(row, &centers, 1) as u32)
            );
            assert_eq!(prepared.scratch_bytes(), 0);
            assert!(prepared.assign(&centers[..1], &mut out).is_err());
            assert!(prepared.assign(&centers, &mut out[..1]).is_err());
        }
    }
}

#[test]
fn assignment_hook_preserves_empty_center_and_rejects_unwritten_ids() {
    let rows = [vec![-10.0], vec![10.0]];
    let points = refs(&rows);
    let mut seed = StdRng::seed_from_u64(7);
    let first = seed.random_range(0..2);
    let second = (first + seed.random_range(1..2)) % 2;
    let mut centers = [42.0; 2];
    let mut rng = StdRng::seed_from_u64(7);
    let mut calls = 0;
    fit_with_assignment(
        &points,
        &mut centers,
        5,
        &mut rng,
        &mut [0.0; 2],
        |_, out| {
            calls += 1;
            out.fill(0);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(calls, 5);
    assert_eq!(centers, [0.0, rows[second][0]]);
    assert_eq!(seed.next_u64(), rng.next_u64());
    for value in [2, u32::MAX] {
        assert!(
            fit_with_assignment(
                &points,
                &mut centers,
                1,
                &mut rng,
                &mut [0.0; 2],
                |_, out| {
                    out.fill(value);
                    Ok(())
                }
            )
            .is_err()
        );
    }
    assert!(
        fit_with_assignment(
            &points,
            &mut centers,
            1,
            &mut rng,
            &mut [0.0; 2],
            |_, _| Ok(())
        )
        .is_err()
    );
    assert!(
        fit_with_assignment(&points, &mut centers, 1, &mut rng, &mut [0.0; 2], |_, _| {
            Err(ANNError::message("assignment failed"))
        })
        .is_err()
    );
    assert!(
        fit_with_assignment(&points, &mut [0.0], 1, &mut rng, &mut [0.0; 2], |_, _| Ok(
            ()
        ))
        .is_err()
    );
    assert!(
        fit_with_assignment(
            &points[..1],
            &mut centers,
            1,
            &mut rng,
            &mut [0.0; 2],
            |_, _| Ok(())
        )
        .is_err()
    );
}

#[test]
fn identical_points_keep_ties_and_empty_centers_including_overflow() {
    for dim in [1, 7, 8, 9, 33, 128] {
        for value in [-0.0, 2.0, f32::MAX] {
            let rows = vec![vec![value; dim]; 9];
            for strategy in STRATEGIES {
                let points = refs(&rows);
                let mut centers = vec![0.0; 2 * dim];
                TwoMeans::new(&points, dim, strategy)
                    .unwrap()
                    .fit_into(
                        &mut centers,
                        5,
                        &mut StdRng::seed_from_u64(42),
                        &mut vec![0.0; 2 * dim],
                    )
                    .unwrap();
                assert!(
                    points
                        .iter()
                        .all(|point| nearest_two(point.vector, &centers, dim) == 0)
                );
                assert!(centers.iter().all(|&x| x == value));
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
        for strategy in STRATEGIES {
            assert!(TwoMeans::new(&refs(&rows), 1, strategy).is_err());
        }
    }
}

#[test]
fn invalid_scratch_leaves_centers_sums_and_rng_unchanged() {
    let rows = [vec![1.0, 2.0], vec![3.0, 4.0]];
    let points = refs(&rows);
    for strategy in STRATEGIES {
        let prepared = TwoMeans::new(&points, 2, strategy).unwrap();
        for (center_len, sum_len) in [(3, 4), (4, 3), (5, 4), (4, 5)] {
            let mut centers = vec![17.0; center_len];
            let mut sums = vec![19.0; sum_len];
            let mut rng = StdRng::seed_from_u64(7);
            let mut expected = rng.clone();
            assert!(
                prepared
                    .fit_into(&mut centers, 5, &mut rng, &mut sums)
                    .is_err()
            );
            assert!(centers.iter().all(|&value| value == 17.0));
            assert!(sums.iter().all(|&value| value == 19.0));
            assert_eq!(rng.next_u64(), expected.next_u64());
        }
    }
}

#[test]
fn every_lloyd_step_preserves_assignments_counts_sum_bits_centers_and_rng() {
    struct Checked(TwoPointSquaredL2);
    impl TwoMeansIteration<usize> for Checked {
        fn assign_and_sum(
            &mut self,
            points: &[PointRef<'_, usize>],
            centers: &[f32],
            dim: usize,
            sums: &mut [f64],
            counts: &mut [usize; 2],
        ) -> ANNResult<()> {
            let mut expected = vec![0.0; sums.len()];
            let mut expected_counts = [0; 2];
            for_each_tiled(&self.0, points, centers, dim, |row, to| {
                let point = points[row].vector;
                assert_eq!(to, nearest_two(point, centers, dim), "row={row}, dim={dim}");
                sum_assignment(point, to, dim, &mut expected, &mut expected_counts);
            });
            TiledAssignment(&self.0).assign_and_sum(points, centers, dim, sums, counts)?;
            assert_eq!(*counts, expected_counts);
            assert_eq!(
                sums.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            Ok(())
        }
    }
    for dim in [1, 3, 7, 8, 9, 31, 32, 33, 192, 384, 896] {
        for n in [2, 3, 16, 17, 57] {
            for seed in [7, 42, 2026] {
                let mut data_rng = StdRng::seed_from_u64(seed ^ ((n as u64) << 32) ^ dim as u64);
                let rows: Vec<Vec<f32>> = (0..n)
                    .map(|_| (0..dim).map(|_| data_rng.random_range(-1.0..1.0)).collect())
                    .collect();
                let points = refs(&rows);
                let mut a = vec![0.0; 2 * dim];
                let mut b = a.clone();
                let mut sa = vec![0.0; 2 * dim];
                let mut sb = sa.clone();
                let mut ra = StdRng::seed_from_u64(seed);
                let mut rb = ra.clone();
                fit_two_means(&points, &mut a, dim, 5, &mut ra, &mut sa).unwrap();
                fit_two_means_using(
                    &points,
                    &mut b,
                    dim,
                    5,
                    &mut rb,
                    &mut sb,
                    &mut Checked(TwoPointSquaredL2::new()),
                )
                .unwrap();
                assert_eq!(
                    a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
                assert_eq!(sa, sb);
                assert_eq!(ra.next_u64(), rb.next_u64());
            }
        }
    }
}

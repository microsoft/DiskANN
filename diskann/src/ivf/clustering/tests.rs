/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use rand::{Rng, RngCore, SeedableRng};

use super::two_means::{fit_two_means_with, nearest_two};
use super::*;

fn points(rows: &[Vec<f32>]) -> Vec<PointRef<'_, u32>> {
    rows.iter()
        .enumerate()
        .map(|(i, row)| PointRef {
            id: i as u32,
            vector: row,
        })
        .collect()
}

#[test]
fn two_means_preserves_centroid_bits_assignments_and_rng() {
    let mut data_rng = StdRng::seed_from_u64(123);
    for dim in [1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 104, 128, 384, 768] {
        // Separate allocations ensure the kernel does not assume contiguous rows.
        let mut rows: Vec<Vec<f32>> = (0..57)
            .map(|_| {
                (0..dim)
                    .map(|_| data_rng.random_range(-10.0..10.0))
                    .collect()
            })
            .collect();
        for scale in [1.0, 1e18] {
            for row in &mut rows {
                for x in row {
                    *x *= scale;
                }
            }
            let points = points(&rows);
            for seed in [0, 7, 42] {
                for iterations in [0, 1, 5, 10] {
                    let mut rng = StdRng::seed_from_u64(seed);
                    let mut reference_rng = rng.clone();
                    let mut centers = vec![f32::NAN; 2 * dim];
                    let mut reference = centers.clone();
                    let mut sums = vec![f64::NAN; 2 * dim];
                    let mut reference_sums = sums.clone();
                    fit_two_means(&points, &mut centers, dim, iterations, &mut rng, &mut sums)
                        .unwrap();
                    fit_two_means_with(
                        &points,
                        &mut reference,
                        dim,
                        iterations,
                        &mut reference_rng,
                        &mut reference_sums,
                        nearest,
                    )
                    .unwrap();
                    assert_eq!(
                        centers.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        reference.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        "dim={dim}, seed={seed}, iterations={iterations}"
                    );
                    assert_eq!(sums, reference_sums);
                    assert_eq!(rng.next_u64(), reference_rng.next_u64());
                    for point in &points {
                        assert_eq!(
                            nearest_two(point.vector, &centers, dim),
                            nearest(point.vector, &reference, dim)
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn two_means_known_means_and_reused_scratch() {
    let rows = [vec![0.0], vec![2.0], vec![8.0], vec![10.0]];
    let mut sums = vec![f64::NAN; 2];
    for seed in 0..16 {
        let mut centers = [f32::NAN; 2];
        fit_two_means(
            &points(&rows),
            &mut centers,
            1,
            3,
            &mut StdRng::seed_from_u64(seed),
            &mut sums,
        )
        .unwrap();
        centers.sort_by(f32::total_cmp);
        assert_eq!(centers, [1.0, 9.0]);
        assert_eq!(
            rows.iter()
                .map(|x| nearest_two(x, &centers, 1))
                .collect::<Vec<_>>(),
            [0, 0, 1, 1]
        );
    }
}

#[test]
fn two_means_keeps_first_ties_infinity_and_empty_children() {
    for dim in [1, 7, 8, 31, 32, 33, 128] {
        assert_eq!(
            nearest_two(
                &vec![0.0; dim],
                &[vec![1.0; dim], vec![-1.0; dim]].concat(),
                dim
            ),
            0
        );
        assert_eq!(
            nearest_two(
                &vec![0.0; dim],
                &[vec![f32::MAX; dim], vec![-f32::MAX; dim]].concat(),
                dim
            ),
            0
        );
        assert_eq!(nearest_two(&vec![-0.0; dim], &vec![0.0; 2 * dim], dim), 0);
        let rows = vec![vec![f32::MAX; dim]; 3];
        let mut zero = vec![0.0; 2 * dim];
        let mut one = zero.clone();
        let mut sums = vec![f64::NAN; 2 * dim];
        fit_two_means(
            &points(&rows),
            &mut zero,
            dim,
            0,
            &mut StdRng::seed_from_u64(7),
            &mut sums,
        )
        .unwrap();
        fit_two_means(
            &points(&rows),
            &mut one,
            dim,
            1,
            &mut StdRng::seed_from_u64(7),
            &mut sums,
        )
        .unwrap();
        assert_eq!(zero, one);
        assert!(one.iter().all(|&x| x == f32::MAX));
    }
}

#[test]
fn two_means_invalid_work_does_not_consume_rng() {
    for rows in [vec![], vec![vec![1.0]]] {
        let mut rng = StdRng::seed_from_u64(7);
        let mut reference = rng.clone();
        assert!(
            fit_two_means(&points(&rows), &mut [0.0; 2], 1, 5, &mut rng, &mut [0.0; 2]).is_err()
        );
        assert_eq!(rng.next_u64(), reference.next_u64());
    }
    assert!(split::<u32, u32>(&[], 1, 5, &mut StdRng::seed_from_u64(7)).is_err());
    let rows = [vec![], vec![]];
    assert!(
        split(
            &[ListView {
                id: 0u32,
                points: points(&rows)
            }],
            0,
            5,
            &mut StdRng::seed_from_u64(7)
        )
        .is_err()
    );
}

#[test]
fn split_preserves_all_child_centroids_and_membership_order() {
    for dim in [3, 8, 33, 128, 129, 384] {
        let mut rng = StdRng::seed_from_u64(51);
        let rows: Vec<Vec<f32>> = (0..96)
            .map(|_| (0..dim).map(|_| rng.random_range(-2.0..2.0)).collect())
            .collect();
        let points = points(&rows);
        let lists: Vec<_> = points
            .chunks(24)
            .enumerate()
            .map(|(id, points)| ListView {
                id: id as u32,
                points: points.to_vec(),
            })
            .collect();
        let mut reference_rng = rng.clone();
        let children = split(&lists, dim, 5, &mut rng).unwrap();
        let mut centers = vec![0.0; 2 * lists.len() * dim];
        let mut sums = vec![0.0; 2 * dim];
        for (list, centers) in lists.iter().zip(centers.chunks_exact_mut(2 * dim)) {
            fit_two_means_with(
                &list.points,
                centers,
                dim,
                5,
                &mut reference_rng,
                &mut sums,
                nearest,
            )
            .unwrap();
        }
        let mut members = vec![Vec::new(); 2 * lists.len()];
        for point in &points {
            members[nearest(point.vector, &centers, dim)].push(point.id);
        }
        assert_eq!(
            children
                .centroids
                .as_slice()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            centers.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(children.members, members);
        assert_eq!(rng.next_u64(), reference_rng.next_u64());
    }
}

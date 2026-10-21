/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_wide::arch::{Scalar, Target};
use rand::{Rng, SeedableRng, rngs::StdRng};

use super::*;

fn check<A: Architecture>(arch: A, distances: &[f32]) {
    let mut best = Top1::new();
    let (groups, tail) = distances.as_chunks::<LANES>();
    for (group, values) in groups.iter().enumerate() {
        best.offer_group(arch, values, group * LANES);
    }
    best.offer_each(tail, groups.len() * LANES);
    let expected = distances
        .iter()
        .enumerate()
        .min_by(|(i, x), (j, y)| x.total_cmp(y).then(i.cmp(j)))
        .unwrap()
        .0;
    assert_eq!(best.index, expected, "{distances:?}");
}

struct SelectionCases;

impl<A: Architecture> Target<A, ()> for SelectionCases {
    #[inline(always)]
    fn run(self, arch: A) {
        let mut rng = StdRng::seed_from_u64(0x4012_6e57);
        for len in 1..=129 {
            // Arbitrary bit patterns exercise total ordering, not just finite
            // positive distances. Also exercise exact ties across group borders.
            let mut distances: Vec<_> = (0..len).map(|_| f32::from_bits(rng.random())).collect();
            check(arch, &distances);
            distances.sort_by(f32::total_cmp);
            check(arch, &distances);
            distances.reverse();
            check(arch, &distances);
            for value in [
                f32::NEG_INFINITY,
                -0.0,
                0.0,
                f32::INFINITY,
                f32::from_bits(0x7fff_ffff),
                f32::from_bits(0xffff_ffff),
            ] {
                distances.fill(value);
                check(arch, &distances);
            }
        }
        // Every lane must be selectable, including the tail after several groups.
        for position in 0..35 {
            let mut distances = [f32::INFINITY; 35];
            distances[position] = 1.0;
            check(arch, &distances);
            distances[34] = 1.0;
            check(arch, &distances);
            distances[position] = -0.0;
            distances[34] = 0.0;
            check(arch, &distances);
        }
    }
}

#[test]
fn top1_scalar_total_order() {
    Scalar.run(SelectionCases);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn top1_v3_total_order() {
    if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
        arch.run(SelectionCases);
    } else {
        assert!(!matches!(
            std::env::var("WIDE_TEST_MIN_ARCH").as_deref(),
            Ok("all" | "x86-64-v3" | "x86-64-v4")
        ));
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn top1_v4_total_order() {
    if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked_miri() {
        arch.run(SelectionCases);
    } else {
        assert!(!matches!(
            std::env::var("WIDE_TEST_MIN_ARCH").as_deref(),
            Ok("all" | "x86-64-v4")
        ));
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn top1_neon_total_order() {
    if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
        arch.run(SelectionCases);
    } else {
        assert!(!matches!(
            std::env::var("WIDE_TEST_MIN_ARCH").as_deref(),
            Ok("all" | "neon")
        ));
    }
}

#[test]
fn assignment_matches_original_for_row_and_dimension_tails() {
    let mut rng = StdRng::seed_from_u64(71);
    for dim in [1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 104, 128, 384, 768] {
        for count in 1..=35 {
            let mut scratch = vec![f32::NAN; count];
            let point: Vec<_> = (0..dim).map(|_| rng.random_range(-8.0..8.0)).collect();
            let mut centers: Vec<_> = (0..dim * count)
                .map(|_| rng.random_range(-8.0..8.0))
                .collect();
            for scale in [1.0, 1e18] {
                for x in &mut centers {
                    *x *= scale;
                }
                assert_eq!(
                    nearest(&point, &centers, dim, &mut scratch),
                    super::super::tests::nearest(&point, &centers, dim),
                    "dim={dim}, count={count}"
                );
            }
            // An independent exact oracle: a unique center equal to the query
            // wins even after several complete tiles or in an odd final row.
            for winner in 0..count {
                centers.fill(2.0);
                centers[winner * dim..(winner + 1) * dim].fill(1.0);
                assert_eq!(
                    nearest(&vec![1.0; dim], &centers, dim, &mut scratch),
                    winner
                );
            }
            centers.fill(f32::MAX);
            assert_eq!(nearest(&vec![0.0; dim], &centers, dim, &mut scratch), 0);
            centers.fill(1.0);
            assert_eq!(nearest(&vec![0.0; dim], &centers, dim, &mut scratch), 0);
        }
    }
}

#[test]
#[ignore = "manual release-only kernel benchmark; no timing assertion"]
fn assignment_kernel_timing() {
    use std::{hint::black_box, time::Instant};
    type Kernel = fn(&[f32], &[f32], usize, &mut [f32]) -> usize;
    let kernels: [Kernel; 2] = [
        |point, centers, dim, _| super::super::tests::nearest(point, centers, dim),
        nearest,
    ];
    let mut rng = StdRng::seed_from_u64(17);
    for dim in [128, 384, 768] {
        for count in [2, 8, 32, 128] {
            let mut scratch = vec![0.0; count];
            let data: Vec<f32> = (0..256 * dim)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            let centers: Vec<f32> = (0..count * dim)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            // Every timed query is first checked against the old selector.
            for point in data.chunks_exact(dim) {
                assert_eq!(
                    kernels[0](point, &centers, dim, &mut scratch),
                    kernels[1](point, &centers, dim, &mut scratch)
                );
            }
            let mut timings: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
            for round in 0..9 {
                for step in 0..2 {
                    let kernel = (step + round) % 2;
                    let start = Instant::now();
                    for _ in 0..(4096 / count) {
                        for point in data.chunks_exact(dim) {
                            black_box(kernels[kernel](
                                black_box(point),
                                black_box(&centers),
                                dim,
                                &mut scratch,
                            ));
                        }
                    }
                    if round != 0 {
                        timings[kernel].push(start.elapsed().as_secs_f64() * 1000.0);
                    }
                }
            }
            let medians = timings.map(|mut v| {
                v.sort_by(f64::total_cmp);
                (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
            });
            println!("dim={dim} centers={count} reference/optimized_ms={medians:?}");
        }
    }
}

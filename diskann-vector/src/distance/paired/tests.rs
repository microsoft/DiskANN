/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_wide::arch::{Scalar, Target};
use rand::{rngs::StdRng, Rng, SeedableRng};

use super::*;
use crate::PureDistanceFunction;

struct Cases;

impl<A: Architecture> Target<A, ()> for Cases
where
    L2: SIMDSchema<f32, f32, A, Return = f32>,
{
    #[inline(always)]
    fn run(self, arch: A) {
        let mut rng = StdRng::seed_from_u64(0x7ac3_45af);
        // Exercise every full-vector/unroll/tail boundary, with separately offset
        // slices. The sentinel outside each slice must never enter the reduction.
        for dim in (0..=97).chain([104, 128, 384, 768, 1536]) {
            for offset in 0..4 {
                for scale in [1.0, 1e-15, 1e15] {
                    let mut storage = [
                        vec![f32::NAN; dim + 8],
                        vec![f32::NAN; dim + 8],
                        vec![f32::NAN; dim + 8],
                    ];
                    for (row, start) in storage.iter_mut().zip([offset, offset + 1, offset + 2]) {
                        for x in &mut row[start..start + dim] {
                            *x = rng.random_range(-1.0..1.0) * scale;
                        }
                    }
                    let point = &storage[0][offset..offset + dim];
                    let first = &storage[1][offset + 1..offset + 1 + dim];
                    let second = &storage[2][offset + 2..offset + 2 + dim];
                    check(arch, point, first, second);
                    // Nearly tied centers and exact ties preserve the same bits.
                    let mut close = first.to_vec();
                    if let Some(value) = close.last_mut() {
                        *value = value.next_up();
                    }
                    check(arch, point, first, &close);
                    check(arch, point, first, first);
                }
            }
            let point = vec![f32::MAX; dim];
            check(arch, &point, &point, &vec![-f32::MAX; dim]);
            check(arch, &vec![0.0; dim], &point, &vec![-f32::MAX; dim]);
            check(
                arch,
                &vec![-0.0; dim],
                &vec![0.0; dim],
                &vec![f32::from_bits(1); dim],
            );

            // An independent f64 oracle is exact for these small integer inputs.
            let rows: [Vec<f32>; 3] = std::array::from_fn(|_| {
                (0..dim).map(|_| rng.random_range(-8..=8) as f32).collect()
            });
            let got = pair(arch, &rows[0], &rows[1], &rows[2]);
            for (actual, center) in got.into_iter().zip(&rows[1..]) {
                let expected: f64 = rows[0]
                    .iter()
                    .zip(center)
                    .map(|(&x, &c)| (f64::from(x) - f64::from(c)).powi(2))
                    .sum();
                assert_eq!(f64::from(actual), expected, "dimension {dim}");
            }
        }
    }
}

#[inline(always)]
fn check<A: Architecture>(arch: A, point: &[f32], first: &[f32], second: &[f32])
where
    L2: SIMDSchema<f32, f32, A, Return = f32>,
{
    let got = arch.run3_inline(SquaredL2::default(), point, first, second);
    let expected = [
        simd::simd_op(&L2, arch, point, first),
        simd::simd_op(&L2, arch, point, second),
    ];
    assert_eq!(
        got.map(f32::to_bits),
        expected.map(f32::to_bits),
        "dimension {}",
        point.len()
    );
}

#[test]
fn paired_l2_scalar_matches_existing_and_integer_oracle() {
    Scalar.run(Cases);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn paired_l2_v3_matches_existing_and_integer_oracle() {
    if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
        arch.run(Cases);
    } else {
        eprintln!("paired L2: V3 unavailable on this CPU");
        assert!(!matches!(
            std::env::var("WIDE_TEST_MIN_ARCH").as_deref(),
            Ok("all" | "x86-64-v3" | "x86-64-v4")
        ));
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn paired_l2_v4_matches_existing_and_integer_oracle() {
    if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked_miri() {
        arch.run(Cases);
    } else {
        eprintln!("paired L2: V4 unavailable on this CPU");
        assert!(!matches!(
            std::env::var("WIDE_TEST_MIN_ARCH").as_deref(),
            Ok("all" | "x86-64-v4")
        ));
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn paired_l2_neon_matches_existing_and_integer_oracle() {
    if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
        arch.run(Cases);
    } else {
        eprintln!("paired L2: Neon unavailable on this CPU");
        assert!(!matches!(
            std::env::var("WIDE_TEST_MIN_ARCH").as_deref(),
            Ok("all" | "neon")
        ));
    }
}

#[test]
fn paired_l2_entry_and_lengths() {
    assert_eq!(
        SquaredL2::evaluate_pair(&[1.0, 2.0], &[0.0, 0.0], &[3.0, 4.0]),
        [5.0, 8.0]
    );
    let x = [3.5; 37];
    let first = [2.125; 37];
    let second = [1.0; 37];
    let expected: [f32; 2] = [
        SquaredL2::evaluate(&x, &first),
        SquaredL2::evaluate(&x, &second),
    ];
    assert_eq!(
        SquaredL2::evaluate_pair(&x, &first, &second).map(f32::to_bits),
        expected.map(f32::to_bits)
    );
    for lengths in [[1, 0, 1], [1, 1, 0], [0, 1, 0], [0, 0, 1]] {
        let [x, a, b] = lengths.map(|len| vec![0.0; len]);
        assert!(std::panic::catch_unwind(|| SquaredL2::evaluate_pair(&x, &a, &b)).is_err());
    }
}

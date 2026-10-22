/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::*;

fn refs(rows: &[Vec<f32>]) -> Vec<PointRef<'_, usize>> {
    rows.iter()
        .enumerate()
        .map(|(id, vector)| PointRef { id, vector })
        .collect()
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

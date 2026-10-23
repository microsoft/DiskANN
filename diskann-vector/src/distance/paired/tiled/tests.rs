/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::*;
use crate::PureDistanceFunction;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn cases(
    mut evaluate: impl FnMut([&[f32]; 2], &[f32], &[f32]) -> [[f32; 2]; 2],
    mut reference: impl FnMut(&[f32], &[f32]) -> f32,
) {
    let mut rng = StdRng::seed_from_u64(0xeac7_2026);
    for dim in (0..=97).chain([128, 192, 383, 384, 385, 895, 896, 897, 1536]) {
        for offset in 0..4 {
            for scale in [1.0, 1e-20, 1e18, f32::MAX] {
                let storage: [Vec<f32>; 4] = std::array::from_fn(|_| {
                    (0..dim + 8)
                        .map(|_| rng.random_range(-1.0..1.0) * scale)
                        .collect()
                });
                let rows =
                    std::array::from_fn::<_, 4, _>(|i| &storage[i][offset + i..offset + i + dim]);
                for centers in [[rows[2], rows[3]], [rows[2], rows[2]]] {
                    let expected =
                        [rows[0], rows[1]].map(|p| centers.map(|c| reference(p, c).to_bits()));
                    assert_eq!(
                        evaluate([rows[0], rows[1]], centers[0], centers[1])
                            .map(|r| r.map(f32::to_bits)),
                        expected,
                        "dim={dim}, scale={scale}"
                    );
                }
                let close: Vec<_> = rows[2]
                    .iter()
                    .map(|x| {
                        if x.is_sign_negative() {
                            x.next_up()
                        } else {
                            x.next_down()
                        }
                    })
                    .collect();
                let expected = [rows[0], rows[1]]
                    .map(|p| [rows[2], &close].map(|c| reference(p, c).to_bits()));
                assert_eq!(
                    evaluate([rows[0], rows[1]], rows[2], &close).map(|r| r.map(f32::to_bits)),
                    expected
                );
            }
        }
        let zeros = vec![0.0; dim];
        let negative = vec![-0.0; dim];
        let tiny = vec![f32::from_bits(1); dim];
        let huge = vec![f32::MAX; dim];
        let expected =
            [&negative[..], &tiny].map(|p| [&zeros[..], &huge].map(|c| reference(p, c).to_bits()));
        assert_eq!(
            evaluate([&negative, &tiny], &zeros, &huge).map(|r| r.map(f32::to_bits)),
            expected
        );
    }
}

#[test]
fn tiled_entry_matches_compiled_baseline_for_all_tails_and_extremes() {
    let kernel = TwoPointSquaredL2::new();
    cases(
        |p, a, b| kernel.evaluate(p, a, b),
        |x, c| SquaredL2::evaluate(x, c),
    );
    // Exercise the portable branch even on an AVX-512 host.
    cases(fallback, |x, c| SquaredL2::evaluate(x, c));
}

#[cfg(target_arch = "x86_64")]
#[test]
fn tiled_v3_and_v4_match_each_architectures_original_l2() {
    use diskann_wide::arch::x86_64::{V3, V4};
    if let Some(arch) = V3::new_checked() {
        cases(
            |p, a, b| arch.run3(Tile, p, a, b),
            |x, c| arch.run3(SquaredL2::default(), x, c, c)[0],
        );
    } else {
        assert!(!matches!(
            std::env::var("WIDE_TEST_MIN_ARCH").as_deref(),
            Ok("all" | "x86-64-v3" | "x86-64-v4")
        ));
    }
    if let Some(arch) = V4::new_checked_miri() {
        cases(
            |p, a, b| arch.run3(Tile, p, a, b),
            |x, c| arch.run3(SquaredL2::default(), x, c, c)[0],
        );
    } else {
        assert!(!matches!(
            std::env::var("WIDE_TEST_MIN_ARCH").as_deref(),
            Ok("all" | "x86-64-v4")
        ));
    }
}

#[test]
fn tiled_rejects_all_mismatched_shapes() {
    let kernel = TwoPointSquaredL2::default();
    for lengths in [[1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0], [0, 0, 0, 1]] {
        let rows = lengths.map(|len| vec![0.0; len]);
        assert!(std::panic::catch_unwind(|| kernel.evaluate(
            [&rows[0], &rows[1]],
            &rows[2],
            &rows[3]
        ))
        .is_err());
    }
}

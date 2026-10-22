/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::*;

struct IntegerCases;

impl<A: Architecture> diskann_wide::arch::Target<A, ()> for IntegerCases {
    fn run(self, arch: A) {
        for dim in [1, 7, 8, 9, 31, 32, 33, 128, 384, 896] {
            for count in [2, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
                let values: Vec<_> = (0..count * dim)
                    .map(|i| ((i * 13 + i / dim * 7) % 19) as f32 - 9.0)
                    .collect();
                let packed = pack_rows(values.chunks_exact(dim), dim).unwrap();
                let centers = [vec![-4.0; dim], vec![5.0; dim]].concat();
                let expected: Vec<_> = values
                    .chunks_exact(dim)
                    .map(|row| {
                        let a: f64 = row.iter().map(|&x| (f64::from(x) + 4.0).powi(2)).sum();
                        let b: f64 = row.iter().map(|&x| (f64::from(x) - 5.0).powi(2)).sum();
                        u32::from(b < a)
                    })
                    .collect();
                // Poison padding: it can be read, but must not become a real output.
                let mut poisoned = packed.clone();
                if count % BLOCK_SIZE != 0 {
                    let start = (count / BLOCK_SIZE) * BLOCK_SIZE * dim;
                    for column in poisoned.as_mut_slice()[start..].chunks_exact_mut(BLOCK_SIZE) {
                        column[count % BLOCK_SIZE..].fill(f32::NAN);
                    }
                }
                for metric in [Metric::SquaredL2, Metric::DotProduct] {
                    // Two sentinels detect a masked store outside the requested output.
                    let mut output = vec![12345; count + 2];
                    arch.run3_inline(
                        Kernel {
                            metric,
                            norms: [16.0 * dim as f32, 25.0 * dim as f32],
                        },
                        poisoned.as_view(),
                        centers.as_slice(),
                        &mut output[1..count + 1],
                    );
                    assert_eq!(
                        &output[1..count + 1],
                        expected,
                        "N={count}, D={dim}, {metric:?}"
                    );
                    assert_eq!((output[0], output[count + 1]), (12345, 12345));
                }
                let reference = LloydReference::new(values.chunks_exact(dim), dim).unwrap();
                let mut output = vec![INVALID; count];
                reference.assign(&centers, &mut output).unwrap();
                assert_eq!(output, expected);
            }
        }
    }
}

#[test]
fn scalar_matches_independent_integer_oracle_and_masks_tails() {
    diskann_wide::arch::Scalar.run(IntegerCases);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn x86_matches_independent_integer_oracle_and_masks_tails() {
    if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
        arch.run(IntegerCases);
    }
    if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked_miri() {
        arch.run(IntegerCases);
    } else {
        assert_ne!(
            std::env::var("WIDE_TEST_MIN_ARCH").as_deref(),
            Ok("x86-64-v4")
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_matches_independent_integer_oracle_and_masks_tails() {
    if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
        arch.run(IntegerCases);
    } else {
        assert_ne!(std::env::var("WIDE_TEST_MIN_ARCH").as_deref(), Ok("neon"));
    }
}

#[test]
fn pack_preserves_logical_order_for_scattered_rows_and_zeroes_padding() {
    let values: Vec<_> = (0..17 * 9).map(|i| i as f32).collect();
    let contiguous = pack_rows(values.chunks_exact(9), 9).unwrap();
    let mut rows = vec![Vec::new(); 17];
    for i in (0..17).rev() {
        rows[i] = values[i * 9..(i + 1) * 9].to_vec();
    }
    let scattered = pack_rows(rows.iter().map(Vec::as_slice), 9).unwrap();
    assert_eq!(scattered.as_slice(), contiguous.as_slice());
    assert_eq!(scattered.as_slice().len(), 32 * 9);
    for d in 0..9 {
        assert_eq!(scattered.as_slice()[16 * 9 + d * 16], values[16 * 9 + d]);
        assert!(
            scattered.as_slice()[16 * 9 + d * 16 + 1..16 * 9 + (d + 1) * 16]
                .iter()
                .all(|&x| x == 0.0)
        );
    }
}

#[test]
fn shape_and_allocation_overflows_are_rejected_before_access() {
    assert!(matches!(
        pack_rows(std::iter::empty(), 1),
        Err(PackError::Empty)
    ));
    assert!(matches!(
        pack_rows([&[][..]; 2].into_iter(), 0),
        Err(PackError::Empty)
    ));
    assert!(matches!(
        pack_rows([&[1.0][..], &[2.0, 3.0][..]].into_iter(), 1),
        Err(PackError::Dimension {
            row: 1,
            expected: 1,
            actual: 2
        })
    ));
    assert!(matches!(
        pack_rows(std::iter::repeat_n(&[1.0][..], usize::MAX), 1),
        Err(PackError::Overflow(_))
    ));
    assert!(matches!(
        pack_rows([&[1.0][..]; 2].into_iter(), usize::MAX),
        Err(PackError::Overflow(_))
    ));
    // Fits in usize elements, but exceeds isize bytes.
    assert!(matches!(
        pack_rows([&[1.0][..]; 2].into_iter(), isize::MAX as usize / 16),
        Err(PackError::Overflow(_))
    ));
    let packed = pack_rows([&[1.0][..], &[2.0][..]].into_iter(), 1).unwrap();
    for metric in [Metric::SquaredL2, Metric::DotProduct] {
        let mut out = [99; 2];
        assert!(assign(packed.as_view(), &[1.0], metric, &mut out).is_err());
        assert_eq!(out, [99; 2]);
        assert!(assign(packed.as_view(), &[1.0, 2.0], metric, &mut out[..1]).is_err());
        assert_eq!(out, [99; 2]);
    }
}

#[test]
fn ties_signed_zero_subnormals_and_nonfinite_scores_are_explicit() {
    for value in [0.0, -0.0, f32::from_bits(1), 4.0, f32::MAX] {
        let row = [value; 9];
        let packed = pack_rows([row.as_slice(); 17].into_iter(), 9).unwrap();
        let centers = [row, row].concat();
        for metric in [Metric::SquaredL2, Metric::DotProduct] {
            let mut output = vec![77; 17];
            assign(packed.as_view(), &centers, metric, &mut output).unwrap();
            let expected = if matches!(metric, Metric::DotProduct) && value == f32::MAX {
                INVALID
            } else {
                0
            };
            assert_eq!(output, vec![expected; 17]);
        }
    }
    let packed = pack_rows([&[f32::MAX][..], &[0.0][..]].into_iter(), 1).unwrap();
    let mut out = [0; 2];
    assign(packed.as_view(), &[0.0, 1.0], Metric::SquaredL2, &mut out).unwrap();
    assert_eq!(out, [INVALID, 0]);
    let reference = LloydReference::new([&[f32::MAX][..]; 2].into_iter(), 1).unwrap();
    assert!(matches!(
        reference.assign(&[0.0, 1.0], &mut out),
        Err(ReferenceError::InvalidOutput)
    ));
    assert!(matches!(
        reference.assign(&[0.0], &mut out),
        Err(ReferenceError::Shape(_))
    ));
}

#[test]
fn finite_cancellation_is_not_silently_treated_as_exact_l2() {
    let packed = pack_rows([&[10000.25][..]].into_iter(), 1).unwrap();
    // True distances to centers 0/1 are 0.5625 and 0.0625. Swapping the
    // customary example exposes the expanded-score tie's preference for 0.
    let mut direct = [INVALID];
    let mut dot = [INVALID];
    assign(
        packed.as_view(),
        &[10001.0, 10000.0],
        Metric::SquaredL2,
        &mut direct,
    )
    .unwrap();
    assign(
        packed.as_view(),
        &[10001.0, 10000.0],
        Metric::DotProduct,
        &mut dot,
    )
    .unwrap();
    assert_eq!(direct, [1]);
    assert_eq!(dot, [0]);
    assert_ne!(
        dot,
        [INVALID],
        "finite cancellation is not an overflow fallback"
    );
}

#[test]
fn broken_exact_size_iterators_return_an_error() {
    struct WrongLength {
        left: usize,
        declared: usize,
    }
    impl Iterator for WrongLength {
        type Item = &'static [f32];
        fn next(&mut self) -> Option<Self::Item> {
            if self.left == 0 {
                None
            } else {
                self.left -= 1;
                Some(&[1.0])
            }
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            (self.declared, Some(self.declared))
        }
    }
    impl ExactSizeIterator for WrongLength {}
    for left in [1, 3] {
        assert!(matches!(
            pack_rows(WrongLength { left, declared: 2 }, 1),
            Err(PackError::RowCount { expected: 2 })
        ));
    }
}

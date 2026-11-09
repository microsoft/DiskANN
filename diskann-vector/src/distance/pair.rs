/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Paired f32 L2 distances with the original per-distance arithmetic.

#[cfg(target_arch = "x86_64")]
use diskann_wide::SIMDVector;
use diskann_wide::{arch::Scalar, Architecture, ARCH};

use super::{simd, SquaredL2};

impl SquaredL2 {
    /// Compute the squared L2 distances from one point to two centers.
    ///
    /// Shares point loads on x86 SIMD targets while retaining
    /// [`PureDistanceFunction::evaluate`](crate::PureDistanceFunction::evaluate)'s
    /// accumulation and reduction order. Finite inputs produce bitwise-identical
    /// distances on the same architecture and floating-point environment.
    ///
    /// Scalar and Neon use the existing single-distance implementation.
    ///
    /// # Panics
    ///
    /// Panics if any center has a different length from `point`.
    #[inline(always)]
    pub fn evaluate_pair(point: &[f32], centers: [&[f32]; 2]) -> [f32; 2] {
        for center in centers {
            if point.len() != center.len() {
                length_error(point.len(), center.len());
            }
        }
        ARCH.pair_l2(point, centers)
    }
}

#[cold]
#[inline(never)]
#[expect(clippy::panic)]
fn length_error(point: usize, center: usize) -> ! {
    panic!(
        "lengths must be equal, instead got: xlen = {}, ylen = {}",
        point, center
    )
}

/// Resolve at compile time to the same architecture used by `SquaredL2::evaluate`.
/// In particular, do not silently select runtime AVX2 when the original distance
/// was compiled for Scalar: the latter has a different accumulation order.
trait PairL2Architecture: Architecture {
    fn pair_l2(self, point: &[f32], centers: [&[f32]; 2]) -> [f32; 2];
}

impl PairL2Architecture for Scalar {
    #[inline(always)]
    fn pair_l2(self, point: &[f32], centers: [&[f32]; 2]) -> [f32; 2] {
        centers.map(|center| simd::simd_op(&simd::L2, self, point, center))
    }
}

#[cfg(target_arch = "aarch64")]
impl PairL2Architecture for diskann_wide::arch::aarch64::Neon {
    #[inline(always)]
    fn pair_l2(self, point: &[f32], centers: [&[f32]; 2]) -> [f32; 2] {
        centers.map(|center| simd::simd_op(&simd::L2, self, point, center))
    }
}

#[cfg(target_arch = "x86_64")]
impl PairL2Architecture for diskann_wide::arch::x86_64::V3 {
    #[inline(always)]
    fn pair_l2(self, point: &[f32], centers: [&[f32]; 2]) -> [f32; 2] {
        pair_4x1(self, point, centers)
    }
}

#[cfg(target_arch = "x86_64")]
impl PairL2Architecture for diskann_wide::arch::x86_64::V4 {
    #[inline(always)]
    fn pair_l2(self, point: &[f32], centers: [&[f32]; 2]) -> [f32; 2] {
        pair_4x1(self, point, centers)
    }
}

/// The independent columns retain `Strategy4x1`'s exact operation graph. The only
/// change to the arithmetic loops is interleaving columns and reusing point loads.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn pair_4x1<A>(arch: A, point: &[f32], centers: [&[f32]; 2]) -> [f32; 2]
where
    A: Architecture,
    simd::L2: simd::SIMDSchema<f32, f32, A, Main = simd::Strategy4x1, Return = f32>,
{
    use simd::SIMDSchema;

    let schema = simd::L2;
    let width = <simd::L2 as SIMDSchema<f32, f32, A>>::get_simd_width();
    let len = point.len();
    let trip_count = len / (4 * width);
    let epilogues = (len - trip_count * 4 * width) / width;
    let px = point.as_ptr();
    let py = centers.map(|center| center.as_ptr());

    let mut s0 = [schema.init(arch); 2];
    let mut s1 = s0;
    let mut s2 = s0;
    let mut s3 = s0;

    for block in 0..trip_count {
        let offset = block * 4 * width;
        // SAFETY: All centers were length-checked by `evaluate_pair`; these four
        // complete SIMD vectors lie within [0, trip_count * 4 * width).
        unsafe {
            accumulate_pair(arch, px, &py, offset, &mut s0);
            accumulate_pair(arch, px, &py, offset + width, &mut s1);
            accumulate_pair(arch, px, &py, offset + 2 * width, &mut s2);
            accumulate_pair(arch, px, &py, offset + 3 * width, &mut s3);
        }
    }

    let offset = trip_count * 4 * width;
    // The full-width epilogues update separate accumulators, exactly as in
    // Strategy4x1. They must not be appended to the combined accumulator.
    if epilogues >= 1 {
        // SAFETY: At least one full SIMD vector remains in every checked slice.
        unsafe { accumulate_pair(arch, px, &py, offset, &mut s0) };
    }
    if epilogues >= 2 {
        // SAFETY: At least two full SIMD vectors remain in every checked slice.
        unsafe { accumulate_pair(arch, px, &py, offset + width, &mut s1) };
    }
    if epilogues >= 3 {
        // SAFETY: At least three full SIMD vectors remain in every checked slice.
        unsafe { accumulate_pair(arch, px, &py, offset + 2 * width, &mut s2) };
    }

    let remainder = len % width;
    std::array::from_fn(|column| {
        let mut sum = schema.combine(
            schema.combine(s0[column], s1[column]),
            schema.combine(s2[column], s3[column]),
        );
        if remainder != 0 {
            let offset = len - remainder;
            // SAFETY: The checked slices contain exactly `remainder` items from
            // this offset. L2's own epilogue performs fault-suppressing loads and
            // preserves the original partial-vector accumulation semantics.
            sum = unsafe {
                schema.epilogue(arch, px.add(offset), py[column].add(offset), remainder, sum)
            };
        }
        schema.reduce(sum)
    })
}

/// Load the point once and apply the original L2 accumulation independently to
/// each of the two centers, allowing the compiler to unroll the center loop.
///
/// # Safety
///
/// `point.add(offset)` and every `centers[i].add(offset)` must be valid for a
/// complete SIMD-width read of f32 values.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn accumulate_pair<A>(
    arch: A,
    point: *const f32,
    centers: &[*const f32; 2],
    offset: usize,
    accumulators: &mut [<simd::L2 as simd::SIMDSchema<f32, f32, A>>::Accumulator; 2],
) where
    A: Architecture,
    simd::L2: simd::SIMDSchema<f32, f32, A>,
{
    use simd::SIMDSchema;

    // SAFETY: Guaranteed by this function's caller.
    let point =
        unsafe { <simd::L2 as SIMDSchema<f32, f32, A>>::Left::load_simd(arch, point.add(offset)) };
    for column in 0..2 {
        // SAFETY: Guaranteed by this function's caller for every center.
        let center = unsafe {
            <simd::L2 as SIMDSchema<f32, f32, A>>::Right::load_simd(
                arch,
                centers[column].add(offset),
            )
        };
        accumulators[column] = simd::L2.accumulate(point, center, accumulators[column]);
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::*;
    use crate::PureDistanceFunction;

    #[test]
    fn pair_matches_single_distance_bits_at_all_tail_boundaries() {
        let mut rng = StdRng::seed_from_u64(0x760d_848b_0e50_0c6c);
        for dim in (0..=129).chain([
            255, 256, 257, 383, 511, 512, 513, 767, 768, 769, 895, 896, 897, 1024,
        ]) {
            let point: Vec<f32> = (0..dim + 3).map(|_| rng.random_range(-5.0..5.0)).collect();
            let storage: [Vec<f32>; 2] = std::array::from_fn(|i| {
                (0..dim + i + 1)
                    .map(|_| rng.random_range(-5.0..5.0))
                    .collect()
            });
            let centers: [&[f32]; 2] = std::array::from_fn(|i| &storage[i][i + 1..]);
            let point = &point[3..];
            for (actual, center) in SquaredL2::evaluate_pair(point, centers)
                .into_iter()
                .zip(centers)
            {
                let expected: f32 = SquaredL2::evaluate(point, center);
                assert_eq!(actual.to_bits(), expected.to_bits(), "dim={dim}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "lengths must be equal")]
    fn pair_rejects_a_short_later_center() {
        let point = [1.0f32; 33];
        SquaredL2::evaluate_pair(&point, [&point[..], &point[..32]]);
    }
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_wide::{
    arch::{Target1, Target2},
    Architecture,
};

use crate::{
    distance::{implementations::L1NormFunctor, InnerProduct},
    Half, MathematicalValue, Norm,
};

/// Evaluate the square of the L2 norm of the argument.
///
/// # Implementation
///
/// The implementations behind this method use a naive approach to computing the norm.
/// This is faster but less accurate than more precise methods.
#[derive(Debug, Clone, Copy)]
pub struct FastL2NormSquared;

impl<T, To> Norm<T, To> for FastL2NormSquared
where
    Self: Target1<diskann_wide::arch::Current, To, T>,
    T: Copy,
    To: Copy,
{
    #[inline]
    fn evaluate(&self, x: T) -> To {
        // As an implementation note: if the implementation of `InnerProduct` is inlined
        // into the callsite, then LLVM recognizes that the two ranges overlap and optimizes
        // out half the loads.
        //
        // This means we don't need to reimplement *all* the different unrolling strategies.
        //
        // The down-side is that perhaps the best unrolling strategy is slightly different
        // for norm calculations. It is at least a start though.
        self.run(diskann_wide::ARCH, x)
    }
}

impl<A, T, To> Target1<A, To, T> for FastL2NormSquared
where
    A: Architecture,
    InnerProduct: Target2<A, MathematicalValue<To>, T, T>,
    T: Copy,
    To: Copy,
{
    #[inline(always)]
    fn run(self, arch: A, x: T) -> To {
        (InnerProduct {}).run(arch, x, x).into_inner()
    }
}

/// Evaluate the L2 norm of the argument.
///
/// # Implementation
///
/// The implementations behind this method use a naive approach to computing the norm.
/// This is faster but less accurate than more precise methods.
#[derive(Debug, Clone, Copy)]
pub struct FastL2Norm;

impl<T> Norm<T, f32> for FastL2Norm
where
    Self: Target1<diskann_wide::arch::Current, f32, T>,
{
    #[inline]
    fn evaluate(&self, x: T) -> f32 {
        self.run(diskann_wide::ARCH, x)
    }
}

impl<A, T> Target1<A, f32, T> for FastL2Norm
where
    A: Architecture,
    FastL2NormSquared: Target1<A, f32, T>,
    T: Copy,
{
    #[inline(always)]
    fn run(self, arch: A, x: T) -> f32 {
        (FastL2NormSquared).run(arch, x).sqrt()
    }
}

/// Evaluate the L1 norm of the argument.
///
/// # Implementation
///
/// This implementation uses the SIMD-optimized L1Norm from `distance::simd`.
///
/// ==================================================================================================
/// NOTE: L1Norm IS A LOGICAL UNARY OPERATION
/// --------------------------------------------------------------------------------------------------
/// Although wired through the generic binary 'SIMDSchema'/'simd_op' infrastructure (which expects
/// two input slices of equal length), 'L1Norm' conceptually computes: sum_i |x_i|
/// The right-hand operand is completely ignored and exists ONLY to satisfy the shared execution
/// machinery (loop tiling, epilogue handling, etc.).
/// ==================================================================================================
#[derive(Debug, Clone, Copy)]
pub struct L1Norm;

impl<T> Norm<T, f32> for L1Norm
where
    Self: Target1<diskann_wide::arch::Current, f32, T>,
{
    #[inline]
    fn evaluate(&self, x: T) -> f32 {
        self.run(diskann_wide::ARCH, x)
    }
}

impl<A, T, To> Target1<A, To, T> for L1Norm
where
    A: Architecture,
    L1NormFunctor: Target2<A, To, T, T>,
    T: Copy,
    To: Copy,
{
    #[inline(always)]
    fn run(self, arch: A, x: T) -> To {
        (L1NormFunctor {}).run(arch, x, x)
    }
}

/// Evaluate the LInf norm of the argument.
///
/// # Implementation
///
/// Closed implementation:
///  Supported input types: f32, Half.
///  f32 path: simple scalar loop using abs and max.
///  Half path: widens each element with 'diskann_wide::cast_f16_to_f32' then applies abs
///    and max.
///
/// # Performance
///
/// The Half widening (cast_f16_to_f32) is per element and does not auto-vectorize well,
/// so LInfNorm on large Half slices may become a throughput bottleneck compared to
/// an explicit SIMD reduction (e.g. loading f16x8, converting once, then doing
/// lane-wise abs & max in f32).
/// Callers with large Half inputs should be aware of the potential bottleneck.
///
/// Current behavior is correct but potentially slower than expected for large Half slices.
#[derive(Debug, Clone, Copy)]
pub struct LInfNorm;

impl Norm<&[f32], f32> for LInfNorm {
    #[inline]
    fn evaluate(&self, x: &[f32]) -> f32 {
        self.run(diskann_wide::ARCH, x)
    }
}

impl Norm<&[Half], f32> for LInfNorm {
    #[inline]
    fn evaluate(&self, x: &[Half]) -> f32 {
        self.run(diskann_wide::ARCH, x)
    }
}

impl<A> Target1<A, f32, &[f32]> for LInfNorm
where
    A: Architecture,
{
    #[inline(always)]
    fn run(self, _: A, x: &[f32]) -> f32 {
        let mut m = 0.0f32;
        for &v in x {
            m = m.max(v.abs());
        }
        m
    }
}

impl<A> Target1<A, f32, &[Half]> for LInfNorm
where
    A: Architecture,
{
    #[inline(always)]
    fn run(self, _: A, x: &[Half]) -> f32 {
        let mut m = 0.0f32;
        for &v in x {
            m = m.max(diskann_wide::cast_f16_to_f32(v).abs());
        }
        m
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use rand::{
        distr::{Distribution, StandardUniform, Uniform},
        rngs::StdRng,
        SeedableRng,
    };

    use super::*;
    use crate::Half;

    trait ReferenceL2NormSquared {
        fn reference_l2_norm_squared(self) -> f32;
    }

    impl ReferenceL2NormSquared for &[f32] {
        fn reference_l2_norm_squared(self) -> f32 {
            self.iter().map(|x| x * x).sum()
        }
    }
    impl ReferenceL2NormSquared for &[Half] {
        fn reference_l2_norm_squared(self) -> f32 {
            self.iter()
                .map(|x| {
                    let x = x.to_f32();
                    x * x
                })
                .sum()
        }
    }
    impl ReferenceL2NormSquared for &[i8] {
        fn reference_l2_norm_squared(self) -> f32 {
            self.iter()
                .map(|x| {
                    let x: i32 = (*x).into();
                    x * x
                })
                .sum::<i32>() as f32
        }
    }
    impl ReferenceL2NormSquared for &[u8] {
        fn reference_l2_norm_squared(self) -> f32 {
            self.iter()
                .map(|x| {
                    let x: i32 = (*x).into();
                    x * x
                })
                .sum::<i32>() as f32
        }
    }

    // For testing the fast L2 norm, we are less concerned about numerical accuracy and more
    // that the right sequence of operations are being performed.
    //
    // To that end, try to keep the inpout distribution "nice" to avoid dealing with rounding
    // issues.
    fn test_fast_l2_norm<T>(generator: &mut dyn FnMut(&mut [T]), max_dim: usize, num_trials: usize)
    where
        T: Copy + Default + std::fmt::Debug,
        for<'a> &'a [T]: ReferenceL2NormSquared,
        FastL2NormSquared: for<'a> Norm<&'a [T], f32>,
        FastL2Norm: for<'a> Norm<&'a [T], f32>,
    {
        for dim in 0..max_dim {
            let mut v = vec![T::default(); dim];
            for _ in 0..num_trials {
                // Generate the test case.
                generator(&mut v);
                let reference = v.reference_l2_norm_squared();
                let fast = (FastL2NormSquared).evaluate(&*v);

                // We should keep the distribution nice enough that this is exact.
                assert_eq!(reference, fast, "failed on dim {} with input: {:?}", dim, v);

                let norm = (FastL2Norm).evaluate(&*v);
                assert_eq!(
                    norm,
                    fast.sqrt(),
                    "failed on dim {} with input: {:?}",
                    dim,
                    v
                );
            }
        }
    }

    const MAX_DIM: usize = 256;
    cfg_if::cfg_if! {
        if #[cfg(miri)] {
            const NUM_TRIALS: usize = 1;
        } else {
            const NUM_TRIALS: usize = 16;
        }
    }

    #[test]
    fn test_fast_l2_norm_f32() {
        let mut rng = StdRng::seed_from_u64(0x4033f5b85e3513f3);
        let distribution = Uniform::<i64>::new(-16, 16).unwrap();
        let mut generator = |v: &mut [f32]| {
            v.iter_mut().for_each(|v| {
                *v = distribution.sample(&mut rng) as f32;
            });
        };
        test_fast_l2_norm(&mut generator, MAX_DIM, NUM_TRIALS);
    }

    #[test]
    fn test_fast_l2_norm_f16() {
        let mut rng = StdRng::seed_from_u64(0xfb0cf009aaa309f8);
        let distribution = Uniform::<i64>::new(-16, 16).unwrap();
        let mut generator = |v: &mut [Half]| {
            v.iter_mut().for_each(|v| {
                *v = Half::from_f32(distribution.sample(&mut rng) as f32);
            });
        };
        test_fast_l2_norm(&mut generator, MAX_DIM, NUM_TRIALS);
    }

    #[test]
    fn test_fast_l2_norm_u8() {
        let mut rng = StdRng::seed_from_u64(0xa119d2f91656ae35);
        let distribution = StandardUniform {};
        let mut generator = |v: &mut [u8]| {
            v.iter_mut().for_each(|v| {
                *v = distribution.sample(&mut rng);
            });
        };
        test_fast_l2_norm(&mut generator, MAX_DIM, NUM_TRIALS);
    }

    #[test]
    fn test_fast_l2_norm_i8() {
        let mut rng = StdRng::seed_from_u64(0x9d96fbf7c321886d);
        let distribution = StandardUniform {};
        let mut generator = |v: &mut [i8]| {
            v.iter_mut().for_each(|v| {
                *v = distribution.sample(&mut rng);
            });
        };
        test_fast_l2_norm(&mut generator, MAX_DIM, NUM_TRIALS);
    }

    #[test]
    fn test_linf_norm_f16() {
        let mut rng = StdRng::seed_from_u64(0xfb0cf009aaa309f8);
        let distribution = Uniform::<i64>::new(-16, 16).unwrap();
        let mut generator = |v: &mut [Half]| {
            v.iter_mut().for_each(|v| {
                *v = Half::from_f32(distribution.sample(&mut rng) as f32);
            });
        };

        for dim in 0..MAX_DIM {
            let mut dst = vec![Half::default(); dim];
            for _ in 0..NUM_TRIALS {
                generator(&mut dst);
                let got = (LInfNorm).evaluate(&*dst);
                let expected = dst
                    .iter()
                    .map(|v| diskann_wide::cast_f16_to_f32(*v).abs())
                    .fold(0.0f32, f32::max);

                assert_eq!(
                    got, expected,
                    "LInf(f16) expected {}, got {} - dim {}",
                    expected, got, dim
                );
            }
        }
    }

    #[test]
    fn test_linf_norm_f32() {
        let mut rng = StdRng::seed_from_u64(0x4033f5b85e3513f3);
        let distribution = Uniform::<i64>::new(-16, 16).unwrap();
        let mut generator = |v: &mut [f32]| {
            v.iter_mut().for_each(|v| {
                *v = distribution.sample(&mut rng) as f32;
            });
        };

        for dim in 0..MAX_DIM {
            let mut dst = vec![f32::default(); dim];
            for _ in 0..NUM_TRIALS {
                generator(&mut dst);
                let got = (LInfNorm).evaluate(&*dst);
                let expected = dst.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

                assert_eq!(
                    got, expected,
                    "LInf(f32) expected {}, got {} - dim {}",
                    expected, got, dim
                );
            }
        }
    }
}

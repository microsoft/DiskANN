/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::Half;

/// Calculate the distance between two vectors, treating them as f64.
pub(crate) fn no_vector_compare_f16_as_f64(a: &[Half], b: &[Half]) -> f64 {
    let mut sum: f64 = 0.0;
    debug_assert_eq!(a.len(), b.len());

    for i in 0..a.len() {
        sum += (a[i].to_f32() as f64 - b[i].to_f32() as f64).powi(2);
    }
    sum
}

/// Calculate the distance between two vectors, treating them as f64.
pub(crate) fn no_vector_compare_f32_as_f64(a: &[f32], b: &[f32]) -> f64 {
    let mut sum: f64 = 0.0;
    debug_assert_eq!(a.len(), b.len());

    for i in 0..a.len() {
        sum += (a[i] as f64 - b[i] as f64).powi(2);
    }
    sum
}

// Traits and implementations for help with random number generation.

/// A trait used to generate random inputs from a seeded Rng.
pub(crate) trait GenerateRandomArguments<T> {
    fn generate<R: Rng>(&self, rng: &mut R, dim: usize) -> Vec<T>;
}

/// Generate a collection of normally distributed f32 numbers.
impl GenerateRandomArguments<f32> for Normal<f32> {
    fn generate<R: Rng>(&self, rng: &mut R, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.sample(rng)).collect()
    }
}

/// Generate a collection of normally distributed f16 numbers.
///
/// This works by generating the distribution using `f32` numbers, then lossily converting
/// to `f16`.
impl GenerateRandomArguments<Half> for Normal<f32> {
    fn generate<R: Rng>(&self, rng: &mut R, dim: usize) -> Vec<Half> {
        (0..dim)
            .map(|_| diskann_wide::cast_f32_to_f16(self.sample(rng)))
            .collect()
    }
}

/// Generate a collection of uniformly distributed i8 inputs.
impl GenerateRandomArguments<i8> for rand::distr::StandardUniform {
    fn generate<R: Rng>(&self, rng: &mut R, dim: usize) -> Vec<i8> {
        (0..dim).map(|_| self.sample(rng)).collect()
    }
}

/// Generate a collection of uniformly distributed u8 inputs.
impl GenerateRandomArguments<u8> for rand::distr::StandardUniform {
    fn generate<R: Rng>(&self, rng: &mut R, dim: usize) -> Vec<u8> {
        (0..dim).map(|_| self.sample(rng)).collect()
    }
}

/// Normalize a slice.
pub(crate) trait Normalize {
    fn normalize(&mut self);
}

impl Normalize for [f32] {
    fn normalize(&mut self) {
        let norm = self.iter().map(|x| (*x) * (*x)).sum::<f32>().sqrt();
        if norm == 0.0 {
            return;
        }

        self.iter_mut().for_each(|x| *x /= norm);
    }
}

impl Normalize for [Half] {
    fn normalize(&mut self) {
        let mut copy: Vec<f32> = self.iter().map(|&i| i.into()).collect();
        copy.normalize();
        for (s, c) in std::iter::zip(self.iter_mut(), copy.iter()) {
            *s = diskann_wide::cast_f32_to_f16(*c);
        }
    }
}

/// A normalizing wrapper for distributions.
#[derive(Debug, Clone)]
pub(crate) struct Normalized<T>(pub(crate) T);

impl GenerateRandomArguments<f32> for Normalized<Normal<f32>> {
    fn generate<R: Rng>(&self, rng: &mut R, dim: usize) -> Vec<f32> {
        let mut v = self.0.generate(rng, dim);
        v.normalize();
        v
    }
}

impl GenerateRandomArguments<Half> for Normalized<Normal<f32>> {
    fn generate<R: Rng>(&self, rng: &mut R, dim: usize) -> Vec<Half> {
        let mut v = self.0.generate(rng, dim);
        v.normalize();
        v
    }
}

/// Generate common corner-case values.
///
/// Corner cases include values at the extreme end of a range, or all zeros, or another
/// uniform inputs that tend to yield pathological values.
pub(crate) trait CornerCases: Sized + Copy {
    fn corner_cases() -> Vec<Self>;
}

/// For `f32`, ensure that we at least cover the all zeros case.
///
/// We don't strongly define the behavior infinite or NaN inputs, nor when partial
/// products become infinite.
///
/// To understand this consider the following vector:
///
/// |           x0 |           x1 |           x2 |            x3 |            x4 |            x5 |
/// |--------------|--------------|--------------|---------------|---------------|---------------|
/// | 1.7014117f38 | 1.7014117f38 | 1.7014117f38 | -1.7014117f38 | -1.7014117f38 | -1.7014117f38 |
///
/// If we sum it as ((((x0 + x1) + x2) + x3) + x4) + x5, the answer is Inf32 since the first
/// three sums overflow and the future subtractions are finite values.
///
/// If instead, we sum (x0 + x1 + x2) + (x3 + x4 + x5), the answer is NaN since we have
/// Inf32 - Inf32.
///
/// Finally, if we sum like (x0 + x3) + (x1 + x4) + (x2 + x5), then the sum is 0.
///
/// So, if intermediate values become infinite, or we are given infinite or NaN values to
/// begin with, we end up with different behavior depending on how we chose to associate
/// the arithmetic. Since floating point reassociation is necessary for vectorization
/// (unless one is willing to permute the vectors before-hand and very carefully plan-out
/// the summation sequence to model in-order summation), we can't provide uniform
/// guarantees on the output of the function for all implementations.
///
/// All this is to say - we can't really generate specific corner-cases for floating point
/// numbers, so we instead pick a few finite values.
impl CornerCases for f32 {
    fn corner_cases() -> Vec<Self> {
        vec![0.0, -5.0, 5.0, 10.0]
    }
}

impl CornerCases for Half {
    fn corner_cases() -> Vec<Self> {
        f32::corner_cases()
            .iter()
            .map(|x| diskann_wide::cast_f32_to_f16(*x))
            .collect()
    }
}

/// For small integers, ensure that we generate inputs at both extremes of the dynamic
/// range, as well as the all zeros case.
impl CornerCases for i8 {
    fn corner_cases() -> Vec<Self> {
        vec![i8::MIN, i8::MAX, 0]
    }
}

/// For small integers, ensure that we generate inputs at both extremes of the dynamic
/// range, as well as the all zeros case.
impl CornerCases for u8 {
    fn corner_cases() -> Vec<Self> {
        vec![u8::MIN, u8::MAX, 0]
    }
}

/// A callback used to check results of a distance computation for a given pair of left
/// and right-hand vector types.
pub(crate) trait DistanceChecker<Left, Right> {
    fn check(&mut self, left: &[Left], right: &[Right]);
}

/// A boxed distance function.
type BoxedFn<'a, Left, Right, To> = Box<dyn FnMut(&[Left], &[Right]) -> To + 'a>;

/// Checker for distance functions.
///
/// For pairs of test argument slices, invokes:
/// ```notest
/// self.compare(self.under_test(left, right), self.reference(left, right))
/// ```
///
/// This struct has a lifetime bound for `FnMut` to enable mutable lambdas.
pub(crate) struct Checker<'a, Left, Right, To = f32> {
    /// The function under test.
    under_test: BoxedFn<'a, Left, Right, To>,
    /// A reference implementation with which to compare.
    reference: BoxedFn<'a, Left, Right, To>,
    /// A comparison function with the under-test value given as the first argument and
    /// the reference value provided as the second argument.
    compare: Box<dyn FnMut(To, To) + 'a>,
}

impl<'a, Left, Right, To> Checker<'a, Left, Right, To> {
    /// Construct a new checker by taking ownership of the provided lambdas.
    pub(crate) fn new<L, R, C>(under_test: L, reference: R, compare: C) -> Self
    where
        L: FnMut(&[Left], &[Right]) -> To + 'a,
        R: FnMut(&[Left], &[Right]) -> To + 'a,
        C: FnMut(To, To) + 'a,
    {
        Self {
            under_test: Box::new(under_test),
            reference: Box::new(reference),
            compare: Box::new(compare),
        }
    }
}

impl<Left, Right, To> DistanceChecker<Left, Right> for Checker<'_, Left, Right, To> {
    fn check(&mut self, left: &[Left], right: &[Right]) {
        (self.compare)(
            (self.under_test)(left, right),
            (self.reference)(left, right),
        );
    }
}

/// An ad-hoc checker for situations that require more involved checking than the typical
/// `Checker`.
pub(crate) struct AdHocChecker<'a, Left, Right>(BoxedFn<'a, Left, Right, ()>);

impl<'a, Left, Right> AdHocChecker<'a, Left, Right> {
    pub(crate) fn new<C>(f: C) -> Self
    where
        C: FnMut(&[Left], &[Right]) + 'a,
    {
        Self(Box::new(f))
    }
}

impl<Left, Right> DistanceChecker<Left, Right> for AdHocChecker<'_, Left, Right> {
    fn check(&mut self, left: &[Left], right: &[Right]) {
        (self.0)(left, right)
    }
}

/// Compare a new distance function implementation with a provided reference implementation.
///
/// Testing will occur in two phases:
///
/// 1. Corner Case Checking.
///
/// For all combinations of `Left` and `Right`-hand corner case values, generate a vector
/// of length `dim` with that value broadcasted to all entries and invoke the checker.
///
/// 2. Fuzz Testing.
///
/// Generate `trials` different random pairs of vectors generated by `left_distr` and
/// `right_distr` and invoke the checker on each pair.
pub(crate) fn test_distance_function<Left, Right, Check, LeftDist, RightDist, R>(
    mut checker: Check,
    left_dist: LeftDist,
    right_dist: RightDist,
    dim: usize,
    trials: usize,
    rng: &mut R,
) where
    Check: DistanceChecker<Left, Right>,
    Left: CornerCases,
    Right: CornerCases,
    LeftDist: GenerateRandomArguments<Left>,
    RightDist: GenerateRandomArguments<Right>,
    R: Rng,
{
    // Check corner cases.
    for vleft in Left::corner_cases() {
        for vright in Right::corner_cases() {
            let left = vec![vleft; dim];
            let right = vec![vright; dim];
            checker.check(&left, &right);
        }
    }

    // Perform fuzz testing.
    for _ in 0..trials {
        let left = left_dist.generate(rng, dim);
        let right = right_dist.generate(rng, dim);
        checker.check(&left, &right);
    }
}

// Tests for your tests.
#[cfg(test)]
mod test_test_utils {

    use rand::{Rng, SeedableRng};

    use super::*;

    /// Generate arguments from the given distribution and invoke the checker function
    /// on each element of the generated vectors.
    fn test_generation_and_check_results<T, Dist, R, Checker>(
        distribution: &Dist,
        rng: &mut R,
        max_dim: usize,
        mut checker: Checker,
    ) where
        R: Rng,
        Dist: GenerateRandomArguments<T>,
        Checker: FnMut(&T),
    {
        for dim in 0..=max_dim {
            let v = distribution.generate(rng, dim);
            assert_eq!(v.len(), dim);
            v.iter().for_each(&mut checker);
        }
    }

    #[test]
    fn test_i8_generation() {
        let mut seen: std::collections::HashSet<i8> = std::collections::HashSet::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x078912AF);
        let distribution = rand::distr::StandardUniform {};
        test_generation_and_check_results(&distribution, &mut rng, 256, |i: &i8| {
            seen.insert(*i);
        });

        // Generating all dimensions from 0 to 256 generates 32,896 distinct integers.
        // The probability of generating all 8-bit unsigned integers is very close to 1.
        assert_eq!(seen.len(), 256);
    }

    #[test]
    fn test_u8_generation() {
        let mut seen: std::collections::HashSet<u8> = std::collections::HashSet::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xdef053c);
        let distribution = rand::distr::StandardUniform {};
        test_generation_and_check_results(&distribution, &mut rng, 256, |i: &u8| {
            seen.insert(*i);
        });

        // Generating all dimensions from 0 to 256 generates 32,896 distinct integers.
        // The probability of generating all 8-bit unsigned integers is very close to 1.
        assert_eq!(seen.len(), 256);
    }

    fn test_float_generation<T>(seed: u64)
    where
        rand_distr::Normal<f32>: GenerateRandomArguments<T>,
        T: Copy + Into<f32>,
    {
        let mut low: f32 = f32::MAX;
        let mut high: f32 = f32::MIN;
        let mut count_inside: u64 = 0;
        let mut total_count: u64 = 0;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mean = 0.0;
        let std = 2.0;
        let distribution = rand_distr::Normal::new(mean, std).unwrap();
        test_generation_and_check_results(&distribution, &mut rng, 256, |x: &T| {
            let x: f32 = (*x).into();
            low = low.min(x);
            high = high.max(x);

            // Record the total count as well as the number of items that appear within
            // one standard deviation of the mean.
            total_count += 1;
            if (x - mean).abs() <= std {
                count_inside += 1;
            }
        });

        // Ensure that the number of samples that fell within one standard deviation is
        // at least 65% (trying to account for some variance in sampling.
        assert!((count_inside as f64) / (total_count as f64) >= 0.65);

        // We should have generated enough samples to generate high and low bounds outside
        // of three standard deviations of the mean.
        assert!(high >= mean + 3.0 * std);
        assert!(low <= mean - 3.0 * std);
    }

    #[test]
    fn test_f32_generation() {
        test_float_generation::<f32>(0x132435);
    }

    #[test]
    fn test_f16_generation() {
        test_float_generation::<Half>(0x978675);
    }

    fn simple_inner_product_f32(x: &[f32], y: &[f32]) -> f32 {
        std::iter::zip(x.iter(), y.iter()).map(|(a, b)| a * b).sum()
    }

    #[test]
    fn test_test_distance_function() {
        let mut under_test_count = 0;
        let mut reference_count = 0;
        let mut check_count = 0;

        let dim = 10;
        let trials = 100;

        // Pass a checker that records the number of calls made and adds 1 to the
        // `under_test` lambda.
        //
        // In the comparer, we ensure that the `+1` value was observed.
        let checker = Checker::<f32, f32, f32>::new(
            |left, right| {
                // Make sure the correct dimensions are provided.
                assert!(left.len() == dim);
                assert!(right.len() == dim);
                under_test_count += 1;
                simple_inner_product_f32(left, right) + 1.0
            },
            |left, right| {
                reference_count += 1;
                simple_inner_product_f32(left, right)
            },
            |a: f32, b: f32| {
                check_count += 1;
                assert_eq!(a, b + 1.0);
            },
        );

        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        test_distance_function(
            checker,
            rand_distr::Normal::new(0.0, 1.0).unwrap(),
            rand_distr::Normal::new(0.0, 1.0).unwrap(),
            dim,
            trials,
            &mut rng,
        );

        // Compute the expected number of outcomes.
        let left_cases = f32::corner_cases().len();
        let right_cases = f32::corner_cases().len();
        let expected_corner_cases = left_cases * right_cases;

        let total_expected = expected_corner_cases + trials;

        assert_eq!(under_test_count, total_expected);
        assert_eq!(reference_count, total_expected);
        assert_eq!(check_count, total_expected);
    }

    /// Panics should propagate.
    #[test]
    #[should_panic]
    fn test_error_propagation() {
        let checker = AdHocChecker::<u8, u8>::new(|_, _| panic!("panic"));
        let mut rng = rand::rngs::StdRng::seed_from_u64(64);
        test_distance_function(
            checker,
            rand::distr::StandardUniform {},
            rand::distr::StandardUniform {},
            5,
            10,
            &mut rng,
        )
    }
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A distribution that generated random floating point numbers drawn from the following:
//!
//! - Normal (generating normal values uniformly)
//! - Subnormal
//! - Zero
//!
//! Within each category, positive and negative values are distributed evenly.
//!
//! Much of the inspiration for this code comes from `proptest::num::f32::Any`, and is
//! largely reimplemented here to remove a dependency on `proptest`.

use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};

// Must be a power of two to avoid bias when reducing a uniformly distributed integer.
const TOTAL_WEIGHT: u64 = 128;
const NORMAL_WEIGHT: u64 = 116;
const SUBNORMAL_WEIGHT: u64 = 6;
const ZERO_WEIGHT: u64 = 6;

const _: () = assert!(
    NORMAL_WEIGHT + SUBNORMAL_WEIGHT + ZERO_WEIGHT == TOTAL_WEIGHT,
    "floating point weights must sum to the total weight"
);

trait Layout {
    type Bits;

    const SIGN_MASK: Self::Bits;
    const EXPONENT_MASK: Self::Bits;

    // A value the describes whether or not a particular exponent is zero according to the
    // IEEE spec.
    const EXPONENT_ZERO: Self::Bits;
    const MANTISSA_MASK: Self::Bits;
}

impl Layout for half::f16 {
    type Bits = u16;

    const SIGN_MASK: u16 = 0x8000;
    const EXPONENT_MASK: u16 = 0x7C00;
    const EXPONENT_ZERO: u16 = 0x3C00;
    const MANTISSA_MASK: u16 = 0x03FF;
}

impl Layout for f32 {
    type Bits = u32;

    const SIGN_MASK: u32 = 0x8000_0000;
    const EXPONENT_MASK: u32 = 0x7F80_0000;
    const EXPONENT_ZERO: u32 = 0x3F80_0000;
    const MANTISSA_MASK: u32 = 0x007F_FFFF;
}

////////////
// Finite //
////////////

/// A distribution for generating finite floating point numbers.
pub struct Finite;

macro_rules! finite {
    ($T:ty, $bits:ty, $twice:ty) => {
        impl Distribution<$T> for Finite {
            /// Generate floating point numbers spread more-or-less uniformly across the
            /// distribution of floating point numbers.
            ///
            /// Note that this is *not* a mathematical uniform distribution over some range.
            /// For normal values, every normal floating point number is generated with equal
            /// probability.
            ///
            /// This function also yields subnormal and zeros with some regularity.
            ///
            /// Positive and negative values are equally likely.
            ///
            /// This function does not generate infinities or NaNs.
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $T {
                // Generate a uniformly distributed integer.
                //
                // This integer is twice as large as what's actually needed to generate
                //
                // * the value that will be used to make the final floating point number
                //   (the lower bits).
                //
                // * a selector for the kind of floating point number we are going to
                //   generate (the upper bits).
                //
                // Generating a number twice as big allows us to perform just a single sample
                // from the random number generator without biasing the result.
                let twice: $twice = StandardUniform {}.sample(rng);

                let mut value = twice as $bits;
                let kind = u64::from(twice >> <$bits>::BITS) % TOTAL_WEIGHT;
                let (mask, allow_edge_exponent, allow_zero_mantissa) = if kind < NORMAL_WEIGHT {
                    // Generate a normal floating point number.
                    //
                    // All digits are fair game, but the exponent cannot be all zeros
                    // (indicating a subnormal number) nor can it be all ones (indicating
                    // infinity/NaN).
                    //
                    // The mantissa is allowed to be all zeros.
                    (<$T>::EXPONENT_MASK | <$T>::MANTISSA_MASK, false, true)
                } else if kind < NORMAL_WEIGHT + SUBNORMAL_WEIGHT {
                    // Generate a subnormal floating point number.
                    //
                    // The exponent must be all zero and the mantissa cannot be zero.
                    (<$T as Layout>::MANTISSA_MASK, true, false)
                } else {
                    // Generate zero.
                    (0, true, true)
                };

                // Preserve the sign bit and mask out all other values that do not belong to
                // the kind of floating point number we are generating.
                value &= <$T as Layout>::SIGN_MASK | mask;
                let exponent = value & <$T as Layout>::EXPONENT_MASK;

                // If the all zeros or all ones pattern is not allowed, set it to 0.
                if !allow_edge_exponent
                    && (exponent == 0 || exponent == <$T as Layout>::EXPONENT_MASK)
                {
                    // Clear the exponent bits and set it to the zero value.
                    value &= !<$T as Layout>::EXPONENT_MASK;
                    value |= <$T as Layout>::EXPONENT_ZERO;
                }

                if !allow_zero_mantissa && (value & <$T as Layout>::MANTISSA_MASK == 0) {
                    // Make the mantissa non-zero.
                    value |= 1;
                }

                // We're done!
                <$T>::from_bits(value)
            }
        }
    };
}

finite!(half::f16, u16, u32);
finite!(f32, u32, u64);

//////////////
// NanHappy //
//////////////

/// A distribution where half of the values generated are from [`Finite`], and the other
/// half are random `NaN` with occastional `+/-infinity` values.
pub struct NanHappy;

macro_rules! nan_happy {
    ($T:ty, $bits:ty) => {
        impl Distribution<$T> for NanHappy {
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $T {
                // Generate a uniformly distributed 32-bit integer
                let is_nan: bool = StandardUniform {}.sample(rng);

                if is_nan {
                    let bits: $bits = StandardUniform {}.sample(rng);
                    <$T>::from_bits(bits | <$T>::EXPONENT_MASK)
                } else {
                    (Finite).sample(rng)
                }
            }
        }
    };
}

nan_happy!(f32, u32);

///////////
// Tests //
///////////

#[cfg(not(miri))]
#[cfg(test)]
mod tests {
    use rand::{SeedableRng, distr::Distribution, rngs::StdRng};

    use super::*;

    #[derive(Debug, Default)]
    struct Kinds {
        normal: u64,
        subnormal: u64,
        zero: u64,
    }

    impl Kinds {
        fn sum(&self) -> u64 {
            self.normal + self.subnormal + self.zero
        }
    }

    #[derive(Debug, Default)]
    struct Counts {
        positive: Kinds,
        negative: Kinds,
    }

    impl Counts {
        fn sum_accross(&self) -> Kinds {
            Kinds {
                normal: self.positive.normal + self.negative.normal,
                subnormal: self.positive.subnormal + self.negative.subnormal,
                zero: self.positive.zero + self.negative.zero,
            }
        }
    }

    trait TestFinite {
        fn test_finite(num_trials: usize, seed: u64) -> Counts;
    }

    impl TestFinite for f32 {
        fn test_finite(num_trials: usize, seed: u64) -> Counts {
            let mut counts = Counts::default();
            let mut rng = StdRng::seed_from_u64(seed);
            for _ in 0..num_trials {
                let v: f32 = (Finite).sample(&mut rng);
                // Do not generate infinity or NaN.
                assert!(v.is_finite());
                if v.is_sign_positive() {
                    if v.is_subnormal() {
                        counts.positive.subnormal += 1;
                    } else if v == 0.0 {
                        counts.positive.zero += 1;
                    } else {
                        counts.positive.normal += 1;
                    }
                } else if v.is_subnormal() {
                    counts.negative.subnormal += 1;
                } else if v == 0.0 {
                    counts.negative.zero += 1;
                } else {
                    counts.negative.normal += 1;
                }
            }
            counts
        }
    }

    impl TestFinite for half::f16 {
        fn test_finite(num_trials: usize, seed: u64) -> Counts {
            let mut counts = Counts::default();
            let mut rng = StdRng::seed_from_u64(seed);

            // Work around the lack of `is_subnormal`.
            fn is_subnormal(x: half::f16) -> bool {
                let bits = x.to_bits();
                (bits & half::f16::EXPONENT_MASK) == 0 && (bits & half::f16::MANTISSA_MASK) != 0
            }

            for _ in 0..num_trials {
                let v: half::f16 = (Finite).sample(&mut rng);
                // Do not generate infinity or NaN.
                assert!(v.is_finite());
                if v.is_sign_positive() {
                    if is_subnormal(v) {
                        counts.positive.subnormal += 1;
                    } else if v == half::f16::default() {
                        counts.positive.zero += 1;
                    } else {
                        counts.positive.normal += 1;
                    }
                } else if is_subnormal(v) {
                    counts.negative.subnormal += 1;
                } else if v == half::f16::default() {
                    counts.negative.zero += 1;
                } else {
                    counts.negative.normal += 1;
                }
            }
            counts
        }
    }

    fn test_end_to_end<T>(seed: u64)
    where
        T: TestFinite,
    {
        let num_trials: u64 = 1_000_000;
        let margin = num_trials / 500;
        let counts = T::test_finite(num_trials as usize, seed);

        let positive_count = counts.positive.sum();
        let negative_count = counts.negative.sum();

        println!("Counts = {:?}", counts);

        assert!(positive_count.abs_diff(num_trials / 2) < margin);
        assert!(negative_count.abs_diff(num_trials / 2) < margin);

        assert!(counts.positive.normal.abs_diff(counts.negative.normal) < margin);
        assert!(
            counts
                .positive
                .subnormal
                .abs_diff(counts.negative.subnormal)
                < margin
        );
        assert!(counts.positive.zero.abs_diff(counts.negative.zero) < margin);

        let kinds = counts.sum_accross();

        assert!(
            kinds
                .normal
                .abs_diff(num_trials * NORMAL_WEIGHT / TOTAL_WEIGHT)
                < margin
        );
        assert!(
            kinds
                .subnormal
                .abs_diff(num_trials * SUBNORMAL_WEIGHT / TOTAL_WEIGHT)
                < margin
        );
        assert!(kinds.zero.abs_diff(num_trials * ZERO_WEIGHT / TOTAL_WEIGHT) < margin);
    }

    #[test]
    fn test_f16_distribution() {
        test_end_to_end::<half::f16>(0xb1e3a2096f17ec6d);
    }
    #[test]
    fn test_f32_distribution() {
        test_end_to_end::<f32>(0x868602b120b17347);
    }
}

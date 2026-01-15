/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use rand::{
    Rng, SeedableRng,
    distr::{Distribution, StandardUniform},
    rngs::StdRng,
    seq::SliceRandom,
};

use super::distribution;

pub(crate) trait ScalarDriver: Default + Copy + 'static {
    /// The distribution used for random sampling.
    type Distribution: Distribution<Self>;

    /// A collection of scalar test values.
    ///
    /// The test driver will ensure that all SIMD lanes see all pair-wise combinations of
    /// test values.
    fn test_values() -> &'static [Self];

    /// Return the test distribution for random testing.
    fn distribution() -> Self::Distribution;
}

/// Implementations
impl ScalarDriver for half::f16 {
    type Distribution = distribution::Finite;

    fn test_values() -> &'static [Self] {
        const VALUES: &[half::f16] = &[
            half::f16::from_f32_const(0.0),
            half::f16::from_f32_const(1.0),
            half::f16::from_f32_const(2.0),
            half::f16::from_f32_const(3.0),
            half::f16::from_f32_const(1000.0),
            half::f16::from_f32_const(-0.0),
            half::f16::from_f32_const(-1.0),
            half::f16::from_f32_const(-2.0),
            half::f16::from_f32_const(-3.0),
            half::f16::from_f32_const(-1000.0),
            // exceptional values
            half::f16::INFINITY,
            half::f16::NEG_INFINITY,
            half::f16::NAN,
        ];
        VALUES
    }

    fn distribution() -> Self::Distribution {
        distribution::Finite
    }
}

impl ScalarDriver for f32 {
    type Distribution = distribution::Finite;

    #[cfg(not(miri))]
    fn test_values() -> &'static [Self] {
        &[
            -0.0,
            -1.0,
            -2.0,
            -3.0,
            -4.0,
            -5.0,
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            -10.0,
            -100.0,
            -1000.0,
            -10_000.0,
            -100_000.0,
            10.0,
            100.0,
            1000.0,
            10_000.0,
            100_000.0,
            // A few random subnormal numbers.
            -2.64697e-40,
            -3.25969e-40,
            -7.653053e-39,
            -9.723335e-39,
            -1.0481945e-38,
            2.64697e-40,
            3.25969e-40,
            7.653053e-39,
            9.723335e-39,
            1.0481945e-38,
            // exceptionsl values
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ]
    }

    #[cfg(miri)]
    fn test_values() -> &'static [Self] {
        &[
            -0.0,
            -1.0,
            0.0,
            1.0,
            // A few random subnormal numbers.
            -2.64697e-40,
            -3.25969e-40,
            -7.653053e-39,
            // exceptionsl values
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ]
    }

    fn distribution() -> Self::Distribution {
        distribution::Finite
    }
}

impl ScalarDriver for bool {
    type Distribution = StandardUniform;
    fn test_values() -> &'static [Self] {
        &[true, false]
    }

    fn distribution() -> Self::Distribution {
        StandardUniform {}
    }
}

macro_rules! unsigned_int_scalar_driver {
    ($T:ident) => {
        impl ScalarDriver for $T {
            type Distribution = StandardUniform;

            #[cfg(not(miri))]
            fn test_values() -> &'static [Self] {
                &[
                     0,  1,  2,  3,  4,  5,  6,  7,  8, 9,
                    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                    30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                    40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                    50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                    60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                    70,
                    $T::MAX, $T::MAX - 1,
                ]
            }
            #[cfg(miri)]
            fn test_values() -> &'static [Self] {
                &[0,  1,  2, $T::MAX, $T::MAX - 1]
            }
            fn distribution() -> Self::Distribution {
                StandardUniform {}
            }
        }
    };
    ($($T:ident,)+) => {
        $(unsigned_int_scalar_driver!($T);)*
    };
}

unsigned_int_scalar_driver!(u8, u16, u32, u64,);

macro_rules! signed_int_scalar_driver {
    ($T:ident) => {
        impl ScalarDriver for $T {
            type Distribution = StandardUniform;
            #[cfg(not(miri))]
            fn test_values() -> &'static [Self] {
                &[
                    -1, -2, -3, -4, -5, -6, -7, -8, -9,-10,
                     0,  1,  2,  3,  4,  5,  6,  7,  8, 9,
                    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                    30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                    40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                    50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                    60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                    70,
                    $T::MAX, $T::MAX - 1, $T::MIN, $T::MIN + 1,
                ]
            }
            #[cfg(miri)]
            fn test_values() -> &'static [Self] {
                &[
                    -1, -2, 0,  1, 2,
                    $T::MAX, $T::MAX - 1, $T::MIN, $T::MIN + 1,
                ]
            }
            fn distribution() -> Self::Distribution {
                StandardUniform {}
            }
        }
    };
    ($($T:ident,)+) => {
        $(signed_int_scalar_driver!($T);)*
    };
}

signed_int_scalar_driver!(i8, i16, i32, i64,);

#[derive(Debug, Clone)]
struct SIMDDriver<T>
where
    T: ScalarDriver,
{
    values: Vec<Vec<T>>,
}

impl<T> SIMDDriver<T>
where
    T: ScalarDriver,
{
    fn new<R>(lane_count: usize, rng: &mut R) -> Self
    where
        R: Rng,
    {
        let mut test_values = T::test_values().to_vec();

        let mut values: Vec<Vec<T>> = (0..test_values.len())
            .map(|_| vec![T::default(); lane_count])
            .collect();

        for lane in 0..lane_count {
            test_values.shuffle(rng);
            for (dst, src) in std::iter::zip(values.iter_mut(), test_values.iter()) {
                dst[lane] = *src;
            }
        }

        Self { values }
    }

    fn iter(&self) -> impl Iterator<Item = &[T]> {
        self.values.iter().map(|i| i.as_slice())
    }
}

cfg_if::cfg_if! {
    if #[cfg(miri)] {
        const NUM_RANDOM_TRIALS: usize = 0;
    } else {
        const NUM_RANDOM_TRIALS: usize = 5000;
    }
}

pub(crate) type Fn1<T> = dyn Fn(&[T]);
pub(crate) type Fn2<T, U> = dyn Fn(&[T], &[U]);
pub(crate) type Fn3<T, U, V> = dyn Fn(&[T], &[U], &[V]);

pub(crate) fn drive_unary<T>(f: &Fn1<T>, lane_count: usize, seed: u64)
where
    T: ScalarDriver,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let driver = SIMDDriver::<T>::new(lane_count, &mut rng);

    // Drive the test values;
    for value in driver.iter() {
        f(value)
    }

    // Drive random values.
    let mut arg: Box<[_]> = (0..lane_count).map(|_| T::default()).collect();
    for _ in 0..NUM_RANDOM_TRIALS {
        arg.iter_mut()
            .for_each(|i| *i = T::distribution().sample(&mut rng));
        f(&arg);
    }
}

pub(crate) fn drive_binary<T, U>(f: &Fn2<T, U>, lane_count: (usize, usize), seed: u64)
where
    T: ScalarDriver,
    U: ScalarDriver,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let driver0 = SIMDDriver::<T>::new(lane_count.0, &mut rng);
    let driver1 = SIMDDriver::<U>::new(lane_count.1, &mut rng);

    // Drive all combinations of test values.
    for a0 in driver0.iter() {
        for a1 in driver1.iter() {
            f(a0, a1)
        }
    }

    // Drive random values.
    let mut a0: Box<[_]> = (0..lane_count.0).map(|_| T::default()).collect();
    let mut a1: Box<[_]> = (0..lane_count.1).map(|_| U::default()).collect();
    for _ in 0..NUM_RANDOM_TRIALS {
        a0.iter_mut()
            .for_each(|i| *i = T::distribution().sample(&mut rng));
        a1.iter_mut()
            .for_each(|i| *i = U::distribution().sample(&mut rng));
        f(&a0, &a1);
    }
}

pub(crate) fn drive_ternary<T, U, V>(f: &Fn3<T, U, V>, lane_count: (usize, usize, usize), seed: u64)
where
    T: ScalarDriver,
    U: ScalarDriver,
    V: ScalarDriver,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let driver0 = SIMDDriver::<T>::new(lane_count.0, &mut rng);
    let driver1 = SIMDDriver::<U>::new(lane_count.1, &mut rng);
    let driver2 = SIMDDriver::<V>::new(lane_count.2, &mut rng);

    // Drive all combinations of test values.
    for a0 in driver0.iter() {
        for a1 in driver1.iter() {
            for a2 in driver2.iter() {
                f(a0, a1, a2)
            }
        }
    }

    // Drive random values.
    let mut a0: Box<[_]> = (0..lane_count.0).map(|_| T::default()).collect();
    let mut a1: Box<[_]> = (0..lane_count.1).map(|_| U::default()).collect();
    let mut a2: Box<[_]> = (0..lane_count.2).map(|_| V::default()).collect();
    for _ in 0..NUM_RANDOM_TRIALS {
        a0.iter_mut()
            .for_each(|i| *i = T::distribution().sample(&mut rng));
        a1.iter_mut()
            .for_each(|i| *i = U::distribution().sample(&mut rng));
        a2.iter_mut()
            .for_each(|i| *i = V::distribution().sample(&mut rng));
        f(&a0, &a1, &a2);
    }
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::views::{Init, Matrix};
use half::f16;
use rand::{Rng, distr::Distribution};

///////////////////////
// panic_message_for //
///////////////////////

pub(super) fn panic_message_for<F>(f: F) -> String
where
    F: FnOnce() + std::panic::UnwindSafe,
{
    match std::panic::catch_unwind(f) {
        Ok(()) => panic!("closure did not panic when it was expected"),
        Err(e) => match e.downcast::<String>() {
            Ok(message) => *message,
            Err(e) => std::panic::resume_unwind(e),
        },
    }
}

/////////////////////
// assert_contains //
/////////////////////

macro_rules! assert_contains {
    ($msg:expr, $expected:literal $(,)?) => {
        let msg = $msg;
        assert!(
            msg.contains($expected),
            "message \"{}\" did not contain \"{}\"",
            msg,
            $expected,
        );
    };
}

pub(super) use assert_contains;

////////////////////////
// Test Distributions //
////////////////////////

#[derive(Debug, Clone, Copy)]
pub(super) struct TestDistr;

impl TestDistr {
    pub(super) fn matrix<T>(nrows: usize, ncols: usize, rng: &mut impl rand::Rng) -> Matrix<T>
    where
        Self: Distribution<T>,
    {
        Matrix::new(Init(|| (Self).sample(rng)), nrows, ncols)
    }
}

impl Distribution<f32> for TestDistr {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f32 {
        f32::from(rng.random_range(-16i16..=16)) * 0.25
    }
}

impl Distribution<f16> for TestDistr {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f16 {
        f16::from_f32(<Self as Distribution<f32>>::sample(self, rng))
    }
}

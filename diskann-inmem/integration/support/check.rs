/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # Baseline Checking
//!
//! The [`Regression`](diskann_benchmark_runner::benchmark::Regression) provides a means
//! of performing before/after comparisons against previously generated results. However,
//! presentation of these results is largely left to the devices of the implementors.
//!
//! This module provides a means of aggregating all match failures (if any) and presenting
//! all failures as a single unit.

use diskann_benchmark_runner::report::{Check, Report};

/// Perform a baseline check on `self` and a previously saved result.
pub(crate) trait CheckMatch {
    fn check_match(&self, previous: &Self) -> Report;
}

macro_rules! check_all_fields {
    ($self:expr, $prev:expr, { $($field:ident),+ $(,)? } $(,)?) => {{
        use diskann_benchmark_runner as dbr;

        let Self { $($field),+ } = $self;
        let mut builder = dbr::report::Report::builder();
        $(
            builder.push(
                stringify!($field),
                <_ as $crate::support::check::CheckMatch>::check_match(
                    $field,
                    &$prev.$field,
                )
            );
        )+
        builder
    }};
}

pub(crate) use check_all_fields;

macro_rules! check_match_impl {
    ($T:ty) => {
        impl CheckMatch for $T {
            fn check_match(
                &self,
                previous: &Self,
            ) -> Report {
                Check::equal(previous.clone(), self.clone()).into()
            }
        }
    };
    ($($Ts:ty),+ $(,)?) => {
        $(check_match_impl!($Ts);)+
    }
}

check_match_impl!(
    bool, u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64
);

impl<T> CheckMatch for [T]
where
    T: CheckMatch,
{
    fn check_match(&self, previous: &[T]) -> Report {
        if self.len() != previous.len() {
            return Check::error(format_args!(
                "number of results is different - old = {}, new = {}",
                previous.len(),
                self.len(),
            ))
            .into();
        }

        let mut builder = Report::builder();
        for (i, (got, expected)) in std::iter::zip(self.iter(), previous.iter()).enumerate() {
            builder.push(i, got.check_match(expected));
        }

        builder.finish()
    }
}

impl<T> CheckMatch for Vec<T>
where
    T: CheckMatch,
{
    fn check_match(&self, previous: &Vec<T>) -> Report {
        self.as_slice().check_match(previous.as_slice())
    }
}

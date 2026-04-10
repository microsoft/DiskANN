/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
#[error("input slice cannot be empty")]
pub struct CannotBeEmpty;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Percentiles<T> {
    pub mean: f64,
    pub median: f64,
    pub p90: T,
    pub p99: T,
}

pub trait AsF64Lossy: Copy {
    fn as_f64_lossy(self) -> f64;
}

macro_rules! impl_as_f64_lossy {
    ($T:ty) => {
        impl AsF64Lossy for $T {
            fn as_f64_lossy(self) -> f64 {
                self as f64
            }
        }
    };
    ($($T:ty),* $(,)?) => {
        $(impl_as_f64_lossy!($T);)*
    }
}

impl_as_f64_lossy!(i8, i16, i32, i64, u8, u16, u32, u64, usize, f32, f64);

pub fn mean<T>(x: &[T]) -> Result<f64, CannotBeEmpty>
where
    T: AsF64Lossy + std::iter::Sum,
{
    if x.is_empty() {
        return Err(CannotBeEmpty);
    }

    let s: T = x.iter().copied().sum();
    Ok(s.as_f64_lossy() / x.len() as f64)
}

/// Find the maximum value of the sequence of `f64`.
pub fn max_f64(x: &[f64]) -> Result<f64, CannotBeEmpty> {
    x.iter().copied().reduce(f64::max).ok_or(CannotBeEmpty)
}

/// Return the mean, median, 90th and 99th percentile of the input vector.
///
/// NOTE: This is implemented by sorting the input slice.
pub fn compute_percentiles<T>(x: &mut [T]) -> Result<Percentiles<T>, CannotBeEmpty>
where
    T: std::cmp::Ord + std::ops::Add<Output = T> + AsF64Lossy + std::iter::Sum,
{
    let mean = mean(x)?;

    x.sort_unstable();

    let len = x.len();
    let half = len / 2;
    let median = if len % 2 == 1 {
        x[half].as_f64_lossy()
    } else {
        (x[half - 1].as_f64_lossy() + x[half].as_f64_lossy()) / 2.0
    };

    let p90 = x[((9 * len) / 10).min(len - 1)];
    let p99 = x[((99 * len) / 100).min(len - 1)];

    Ok(Percentiles {
        mean,
        median,
        p90,
        p99,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let empty: &[f32] = &[];
        assert!(matches!(mean(empty).unwrap_err(), CannotBeEmpty));

        let input = [
            -2.049918,
            0.12130953,
            -0.17400686,
            0.7511493,
            0.26361275,
            1.2661924,
            1.023522,
            -2.8727458,
            -1.0132318,
            0.531649,
            -0.8730961,
            1.0494779,
            1.8957608,
            0.45292637,
            0.1296239,
            0.06079646,
            -1.3347862,
            0.122092366,
            -0.82615733,
            1.3791777,
            1.5189241,
            -0.8614088,
            -0.62131107,
            -2.0626633,
            -0.49564686,
        ];

        let r = mean(&input).unwrap();
        assert!((r - -0.10475030175999998).abs() <= 3.0e-17);
    }

    #[test]
    fn test_max() {
        assert_eq!(max_f64(&[1.0, -1.0, f64::NEG_INFINITY]).unwrap(), 1.0);
        assert!(matches!(max_f64(&[]).unwrap_err(), CannotBeEmpty));
        assert_eq!(
            max_f64(&[1.0, -1.0, f64::NAN, f64::NEG_INFINITY]).unwrap(),
            1.0
        );
    }

    #[test]
    fn test_compute_percentils() {
        // Size 0
        {
            let empty: &mut [u64] = &mut [];
            assert!(matches!(
                compute_percentiles(empty).unwrap_err(),
                CannotBeEmpty
            ));
        }

        // Size 1
        {
            let v: &mut [u64] = &mut [10];
            let p = compute_percentiles(v).unwrap();
            let e = Percentiles {
                mean: 10.0,
                median: 10.0,
                p90: 10,
                p99: 10,
            };
            assert_eq!(p, e);
        }

        // Size 2
        {
            let v: &mut [u64] = &mut [2, 1];
            let p = compute_percentiles(v).unwrap();
            let e = Percentiles {
                mean: 1.5,
                median: 1.5,
                p90: 2,
                p99: 2,
            };
            assert_eq!(p, e);
        }

        // Size 3
        {
            let v: &mut [u64] = &mut [2, 1, 3];
            let p = compute_percentiles(v).unwrap();
            let e = Percentiles {
                mean: 2.0,
                median: 2.0,
                p90: 3,
                p99: 3,
            };
            assert_eq!(p, e);
        }

        // Size 9
        {
            let v: &mut [u64] = &mut [2, 1, 3, 4, 9, 6, 7, 5, 8];
            let p = compute_percentiles(v).unwrap();
            let e = Percentiles {
                mean: 5.0,
                median: 5.0,
                p90: 9,
                p99: 9,
            };
            assert_eq!(p, e);
        }

        // Size 10
        {
            let v: &mut [u64] = &mut [2, 10, 1, 3, 4, 9, 6, 7, 5, 8];
            let p = compute_percentiles(v).unwrap();
            let e = Percentiles {
                mean: 5.5,
                median: 5.5,
                p90: 10,
                p99: 10,
            };
            assert_eq!(p, e);
        }

        // Size 10
        {
            let v: &mut [u64] = &mut [2, 10, 1, 3, 4, 9, 6, 11, 7, 5, 8];
            let p = compute_percentiles(v).unwrap();
            let e = Percentiles {
                mean: 6.0,
                median: 6.0,
                p90: 10,
                p99: 11,
            };
            assert_eq!(p, e);
        }
    }
}

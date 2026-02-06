/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use serde::{Deserialize, Serialize};

use super::percentiles::AsF64Lossy;

/// A unit of time representing microseconds.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(transparent)]
pub struct MicroSeconds(u64);

impl MicroSeconds {
    /// Construct a new instance of self over a raw unit of micro-seconds.
    pub fn new(micros: u64) -> Self {
        Self(micros)
    }

    /// Return `self` as seconds.
    pub fn as_seconds(self) -> f64 {
        (self.0 as f64) / 1_000_000.0
    }

    /// Return `self` as microseconds.
    pub fn as_micros(self) -> u64 {
        self.0
    }

    /// Return `self` as microseconds but converted to `f64`.
    pub fn as_f64(self) -> f64 {
        self.0 as f64
    }
}

impl From<std::time::Duration> for MicroSeconds {
    fn from(value: std::time::Duration) -> Self {
        Self::new(value.as_micros() as u64)
    }
}

impl std::fmt::Display for MicroSeconds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}us", self.as_micros())
    }
}

impl std::ops::Add for MicroSeconds {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl std::iter::Sum for MicroSeconds {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let sum: u64 = iter.map(|i| i.0).sum();
        Self(sum)
    }
}

impl AsF64Lossy for MicroSeconds {
    fn as_f64_lossy(self) -> f64 {
        self.as_f64()
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! timed {
    ($($exprs:tt)*) => {{
        let start = ::std::time::Instant::now();
        let result = $($exprs)*;
        let elapsed: $crate::utils::MicroSeconds = start.elapsed().into();
        (elapsed, result)
    }}
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_microseconds() {
        let x = MicroSeconds::new(1_000_001);
        assert_eq!(x.as_micros(), 1_000_001);
        assert_eq!(x.as_f64(), 1_000_001.0f64);
        assert_eq!(x.as_f64_lossy(), 1_000_001.0f64);
        assert_eq!(x.as_seconds(), 1.000001f64);

        assert_eq!(x.to_string(), "1000001us");

        // Add
        assert_eq!(
            MicroSeconds::new(2) + MicroSeconds::new(3),
            MicroSeconds::new(5)
        );

        // Sum
        let x = [
            MicroSeconds::new(1),
            MicroSeconds::new(2),
            MicroSeconds::new(3),
        ];
        let s: MicroSeconds = x.into_iter().sum();
        assert_eq!(s, MicroSeconds::new(6));

        // From Duration
        let x = std::time::Duration::from_micros(12345);
        let y: MicroSeconds = x.into();
        assert_eq!(y, MicroSeconds::new(12345));
    }

    #[test]
    fn test_microseconds_serde() {
        let x: MicroSeconds = serde_json::from_str("15243").unwrap();
        assert_eq!(x, MicroSeconds::new(15243));

        let s = serde_json::to_string(&x).unwrap();
        assert_eq!(s, "15243");
    }
}

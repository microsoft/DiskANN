/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Number utilities for enforcing deserialization constraints and computing relative errors.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

/// Compute the relative change from `before` to `after`.
///
/// This helper is intentionally opinionated for benchmark-style metrics:
///
/// - `before` must be finite and strictly positive.
/// - `after` must be finite and non-negative.
///
/// In other words, this computes:
/// ```text
/// (after - before) / before
/// ```
///
/// Negative values indicate improvements while positive values indicate regressions.
pub fn relative_change(before: f64, after: f64) -> Result<f64, RelativeChangeError> {
    if !before.is_finite() {
        return Err(RelativeChangeError::NonFiniteBefore);
    }
    if before <= 0.0 {
        return Err(RelativeChangeError::NonPositiveBefore);
    }

    let after = NonNegativeFinite::new(after).map_err(RelativeChangeError::InvalidAfter)?;
    let after = after.get();

    let change = (after - before) / before;
    if !change.is_finite() {
        return Err(RelativeChangeError::NonFiniteComputedChange);
    }

    Ok(change)
}

/// Error returned when attempting to compute a relative change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum RelativeChangeError {
    #[error("expected \"before\" to be a finite number")]
    NonFiniteBefore,
    #[error("expected \"before\" to be greater than zero")]
    NonPositiveBefore,
    #[error("invalid \"after\" value: {0}")]
    InvalidAfter(InvalidNonNegativeFinite),
    #[error("computed relative change is not finite")]
    NonFiniteComputedChange,
}

/// A finite floating-point value that is greater than or equal to zero.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct NonNegativeFinite(f64);

impl NonNegativeFinite {
    /// Attempt to construct `Self` from `value`.
    pub const fn new(value: f64) -> Result<Self, InvalidNonNegativeFinite> {
        if !value.is_finite() {
            Err(InvalidNonNegativeFinite::NonFinite)
        } else if value < 0.0 {
            Err(InvalidNonNegativeFinite::Negative)
        } else if value == 0.0 {
            Ok(Self(0.0))
        } else {
            Ok(Self(value))
        }
    }

    /// Return the underlying floating-point value.
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl std::fmt::Display for NonNegativeFinite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl TryFrom<f64> for NonNegativeFinite {
    type Error = InvalidNonNegativeFinite;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<NonNegativeFinite> for f64 {
    fn from(value: NonNegativeFinite) -> Self {
        value.get()
    }
}

impl Serialize for NonNegativeFinite {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_f64(self.0)
    }
}

impl<'de> Deserialize<'de> for NonNegativeFinite {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = f64::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

/// Error returned when attempting to construct a [`NonNegativeFinite`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum InvalidNonNegativeFinite {
    #[error("expected a finite number")]
    NonFinite,
    #[error("expected a non-negative number")]
    Negative,
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let to_non_negative =
            |x: f64| -> Result<NonNegativeFinite, InvalidNonNegativeFinite> { x.try_into() };
        let to_f64 = |x: NonNegativeFinite| -> f64 { x.into() };

        assert_eq!(NonNegativeFinite::new(0.0).unwrap().get(), 0.0);
        assert_eq!(NonNegativeFinite::new(-0.0).unwrap().get(), 0.0);
        assert_eq!(NonNegativeFinite::new(0.25).unwrap().get(), 0.25);

        assert_eq!(to_f64(to_non_negative(0.0).unwrap()), 0.0);
        assert_eq!(to_f64(to_non_negative(-0.0).unwrap()), 0.0);
        assert_eq!(to_f64(to_non_negative(0.25).unwrap()), 0.25);

        assert_eq!(to_non_negative(0.25).unwrap().to_string(), 0.25.to_string());

        assert_eq!(
            NonNegativeFinite::new(-1.0).unwrap_err(),
            InvalidNonNegativeFinite::Negative
        );
        assert_eq!(
            to_non_negative(-1.0).unwrap_err(),
            InvalidNonNegativeFinite::Negative
        );

        assert_eq!(
            NonNegativeFinite::new(f64::INFINITY).unwrap_err(),
            InvalidNonNegativeFinite::NonFinite
        );
        assert_eq!(
            NonNegativeFinite::new(f64::NEG_INFINITY).unwrap_err(),
            InvalidNonNegativeFinite::NonFinite
        );
        assert_eq!(
            NonNegativeFinite::new(f64::NAN).unwrap_err(),
            InvalidNonNegativeFinite::NonFinite
        );
    }

    #[test]
    fn test_serde() {
        let value: NonNegativeFinite = serde_json::from_str("0.1").unwrap();
        assert_eq!(value.get(), 0.1);

        let serialized = serde_json::to_string(&value).unwrap();
        assert_eq!(serialized, "0.1");

        let err = serde_json::from_str::<NonNegativeFinite>("-0.5").unwrap_err();
        assert!(err.to_string().contains("expected a non-negative number"));
    }

    #[test]
    fn test_relative_change() {
        assert_eq!(relative_change(10.0, 10.0).unwrap(), 0.0);
        assert_eq!(relative_change(10.0, 12.5).unwrap(), 0.25);
        assert_eq!(relative_change(10.0, 8.0).unwrap(), -0.2);
        assert_eq!(relative_change(10.0, -0.0).unwrap(), -1.0);

        assert_eq!(
            relative_change(0.0, 1.0).unwrap_err(),
            RelativeChangeError::NonPositiveBefore
        );
        assert_eq!(
            relative_change(-1.0, 1.0).unwrap_err(),
            RelativeChangeError::NonPositiveBefore
        );
        assert_eq!(
            relative_change(f64::NAN, 1.0).unwrap_err(),
            RelativeChangeError::NonFiniteBefore
        );
        assert_eq!(
            relative_change(f64::INFINITY, 1.0).unwrap_err(),
            RelativeChangeError::NonFiniteBefore
        );
        assert_eq!(
            relative_change(1.0, -1.0).unwrap_err(),
            RelativeChangeError::InvalidAfter(InvalidNonNegativeFinite::Negative)
        );
        assert_eq!(
            relative_change(1.0, f64::NAN).unwrap_err(),
            RelativeChangeError::InvalidAfter(InvalidNonNegativeFinite::NonFinite)
        );
        assert_eq!(
            relative_change(f64::MIN_POSITIVE, f64::MAX).unwrap_err(),
            RelativeChangeError::NonFiniteComputedChange
        );
    }
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Tools to prevent regressions by checking against increases or decreases in values.
//!
//! Since benchmarks can be inherently noisy, these tools allow for some fuzziness in values.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

use crate::utils::num::NonNegativeFinite;

use super::{Check, Status};

/// Perform a check against `old` and `new` with the semantics that "lower is better".
pub trait LowerIsBetter<T> {
    fn lower_is_better(&self, old: T, new: T, spec: FmtSpec) -> Check;
}

/// Perform a check against `old` and `new` with the semantics that "higher is better".
pub trait HigherIsBetter<T> {
    fn higher_is_better(&self, old: T, new: T, spec: FmtSpec) -> Check;
}

macro_rules! implement {
    ($trait:ident, $method:ident, $implementor:ty) => {
        implement!(
            @all,
            $trait,
            $method,
            $implementor,
            { u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64 },
        );
    };
    (@all, $trait:ident, $method:ident, $implementor:ty, { $($types:ty),+ $(,)? } $(,)?) => {
        $(implement!(@instance, $trait, $method, $implementor, $types);)+
    };
    (@instance, $trait:ident, $method:ident, $implementor:ty, $type:ty) => {
        impl $trait<$type> for $implementor {
            fn $method(&self, old: $type, new: $type, spec: FmtSpec) -> Check {
                match new.change_from(&old) {
                    Ok(difference) => Tolerance::from(*self).check(difference, spec),
                    Err(err) => Check::error(&err),
                }
            }
        }
    };
}

macro_rules! higher_is_better {
    ($T:ty) => {
        implement!(HigherIsBetter, higher_is_better, $T);
    };
}

macro_rules! lower_is_better {
    ($T:ty) => {
        implement!(LowerIsBetter, lower_is_better, $T);
    };
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FmtSpec {
    // The precision to use when displaying before and after values.
    precision: Option<u8>,
}

impl FmtSpec {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_precision(mut self, precision: u8) -> Self {
        self.precision = Some(precision);
        self
    }
}

trait FmtWith {
    fn fmt_with(&self, f: &mut dyn std::fmt::Write, spec: &FmtSpec) -> std::fmt::Result;
}

struct AsFmtWith<'a, T>(&'a T, &'a FmtSpec);

impl<T> std::fmt::Display for AsFmtWith<'_, T>
where
    T: FmtWith,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt_with(f, self.1)
    }
}

impl FmtWith for f64 {
    fn fmt_with(&self, f: &mut dyn std::fmt::Write, spec: &FmtSpec) -> std::fmt::Result {
        let FmtSpec { precision } = spec;
        match precision {
            Some(p) => {
                let p = usize::from(*p);
                write!(f, "{:.p$}", self)
            }
            None => write!(f, "{}", self),
        }
    }
}

macro_rules! fmt_with {
    ($T:ty) => {
        impl FmtWith for $T {
            fn fmt_with(&self, f: &mut dyn std::fmt::Write, spec: &FmtSpec) -> std::fmt::Result {
                let FmtSpec { precision: _ } = spec;
                write!(f, "{}", self)
            }
        }
    };
    ($($Ts:ty),+ $(,)?) => {
        $(fmt_with!($Ts);)+
    }
}

fmt_with!(u64, i64);

////////////
// Checks //
////////////

/// A limit on the relative decrease allowed.
///
/// The limit is a floating point number in `[0, 1]` and a check from `old` to `new` is
/// successful if
/// ```math
/// new >= old * (1 - limit)
/// ```
/// Supports Rust's primitive integer types up to 64 bits, including `usize`/`isize`,
/// and `f32`/`f64`.
///
/// This check requires that `old` and `new` are positive and finite.
///
/// # Precision
///
/// Integer absolute differences are computed exactly. Floating-point differences are
/// computed in `f64` and are subject to floating-point rounding.
///
/// The relative change magnitude (`absolute / old`) is computed by converting both
/// values to `f64` before division.
///
/// If either `absolute` or `old` is an integer that cannot be represented exactly as
/// an `f64`, the conversion can lose precision. The division itself is subject to
/// floating-point rounding.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxRelativeDecrease {
    #[serde(rename = "max-relative-decrease")]
    limit: Fraction,
}

impl MaxRelativeDecrease {
    /// Construct a new [`MaxRelativeDecrease`].
    ///
    /// Returns `None` if `limit` is not in the range `[0, 1]`.
    ///
    /// # Note
    ///
    /// This returns an `Option` to be more easily usable in const contexts.
    pub const fn new(limit: f64) -> Option<Self> {
        match Fraction::new(limit) {
            Some(limit) => Some(Self { limit }),
            None => None,
        }
    }

    const fn limit(&self) -> Fraction {
        self.limit
    }

    /// Return the numeric limit.
    pub const fn as_f64(&self) -> f64 {
        self.limit().value()
    }
}

higher_is_better!(MaxRelativeDecrease);

/// A limit on the relative increase allowed.
///
/// The limit is a floating point number in `[0, f64::MAX]` and a check from `old` to `new`
/// is successful if:
/// ```math
/// new <= old * (1 + limit)
/// ```
/// Supports Rust's primitive integer types up to 64 bits, including `usize`/`isize`,
/// and `f32`/`f64`.
///
/// This check requires that `old` and `new` are positive and finite.
///
/// # Precision
///
/// Integer absolute differences are computed exactly. Floating-point differences are
/// computed in `f64` and are subject to floating-point rounding.
///
/// The relative change magnitude (`absolute / old`) is computed by converting both
/// values to `f64` before division.
///
/// If either `absolute` or `old` is an integer that cannot be represented exactly as
/// an `f64`, the conversion can lose precision. The division itself is subject to
/// floating-point rounding.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxRelativeIncrease {
    #[serde(rename = "max-relative-increase")]
    limit: NonNegativeFinite,
}

impl MaxRelativeIncrease {
    /// Construct a new [`MaxRelativeIncrease`].
    ///
    /// Returns `None` if `limit` is negative, infinite, or NaN.
    ///
    /// # Note
    ///
    /// This returns an `Option` to be unwrappable in a const context.
    pub const fn new(limit: f64) -> Option<Self> {
        match NonNegativeFinite::new(limit) {
            Ok(limit) => Some(Self { limit }),
            Err(_) => None,
        }
    }

    /// Return the numeric limit.
    pub const fn limit(&self) -> NonNegativeFinite {
        self.limit
    }

    /// Return the numeric limit.
    pub const fn as_f64(&self) -> f64 {
        self.limit().get()
    }
}

lower_is_better!(MaxRelativeIncrease);

/// A limit on the absolute decrease allowed.
///
/// The limit is either a `u64` or a floating point number in `[0, f64::MAX]` and a check
/// from `old` to `new` is successful if:
/// ```math
/// new >= (old - limit)
/// ```
/// Supports Rust's primitive integer types up to 64 bits, including `usize`/`isize`,
/// and `f32`/`f64`.
///
/// # Precision
///
/// Integer absolute differences are computed exactly. Floating-point differences are
/// an `f64`, the conversion can lose precision. The division itself is subject to
/// floating-point rounding.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxAbsoluteDecrease {
    #[serde(rename = "max-absolute-decrease")]
    limit: Absolute,
}

impl MaxAbsoluteDecrease {
    /// Construct a new [`MaxAbsoluteDecrease`] from `limit`.
    pub const fn from_u64(limit: u64) -> Self {
        Self::new(Absolute::Int(limit))
    }

    /// Construct a new [`MaxAbsoluteDecrease`] from `limit`.
    ///
    /// Returns `None` if `limit` is not in `[0, f64::MAX]`.
    pub const fn try_from_f64(limit: f64) -> Option<Self> {
        match NonNegativeFinite::new(limit) {
            Ok(limit) => Some(Self::new(Absolute::Float(limit))),
            Err(_) => None,
        }
    }

    /// Return the limit if it is an integer.
    pub const fn as_u64(&self) -> Option<u64> {
        self.limit.as_u64()
    }

    /// Return the limit if it is a float.
    pub const fn as_f64(&self) -> Option<f64> {
        self.limit.as_f64()
    }

    const fn new(limit: Absolute) -> Self {
        Self { limit }
    }

    const fn limit(&self) -> Absolute {
        self.limit
    }
}

higher_is_better!(MaxAbsoluteDecrease);

/// A limit on the absolute increase allowed.
///
/// The limit is either a `u64` or a floating point number in `[0, f64::MAX]` and a check
/// from `old` to `new` is successful if:
/// ```math
/// new <= (old + limit)
/// ```
/// Supports Rust's primitive integer types up to 64 bits, including `usize`/`isize`,
/// and `f32`/`f64`.
///
/// # Precision
///
/// Integer absolute differences are computed exactly. Floating-point differences are
/// an `f64`, the conversion can lose precision. The division itself is subject to
/// floating-point rounding.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxAbsoluteIncrease {
    #[serde(rename = "max-absolute-increase")]
    limit: Absolute,
}

impl MaxAbsoluteIncrease {
    /// Construct a new [`MaxAbsoluteIncrease`] from `limit`.
    pub const fn from_u64(limit: u64) -> Self {
        Self::new(Absolute::Int(limit))
    }

    /// Construct a new [`MaxAbsoluteIncrease`] from `limit`.
    ///
    /// Returns `None` if `limit` is not in `[0, f64::MAX]`.
    pub const fn try_from_f64(limit: f64) -> Option<Self> {
        match NonNegativeFinite::new(limit) {
            Ok(limit) => Some(Self::new(Absolute::Float(limit))),
            Err(_) => None,
        }
    }

    /// Return the limit if it is an integer.
    pub const fn as_u64(&self) -> Option<u64> {
        self.limit.as_u64()
    }

    /// Return the limit if it is a float.
    pub const fn as_f64(&self) -> Option<f64> {
        self.limit.as_f64()
    }

    const fn new(limit: Absolute) -> Self {
        Self { limit }
    }

    const fn limit(&self) -> Absolute {
        self.limit
    }
}

lower_is_better!(MaxAbsoluteIncrease);

////////////////////
// Implementation //
////////////////////

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(untagged)]
enum Tolerance {
    RelativeDecrease(MaxRelativeDecrease),
    AbsoluteDecrease(MaxAbsoluteDecrease),
    RelativeIncrease(MaxRelativeIncrease),
    AbsoluteIncrease(MaxAbsoluteIncrease),
}

impl Tolerance {
    /// Checking rules:
    ///
    /// * Max relative decrease: Succeeds if
    ///   ```math
    ///   new >= old * (1 - limit) --> limit >= -( (new - old) / old )
    ///                                           ------------------- <-- Relative Change
    ///   ```
    ///   Requires:
    ///   - `limit` is in `[0, 1]` (guaranteed by [`Fraction`]).
    ///   - `new` and `old` are both strictly positive and the relative change is finite
    ///     (guaranteed by [`Difference::relative`]).
    ///
    /// * Max absolute decrease: Succeeds if the absolute change is not decreasing or if it
    ///   decreases by no more than `limit`.
    ///
    ///   Requires:
    ///   - `limit` to be non-negative and finite (guaranteed by [`Absolute`]).
    ///
    /// * Max relative increase: Succeeds if
    ///   ```math
    ///   new <= old * (1 + limit) --> limit >= ( (new - old) / old )
    ///                                          ------------------- <-- Relative Change
    ///   ```
    ///   Requires:
    ///   - `limit` is non-negative and finite (guaranteed by [`NonNegativeFinite`]).
    ///   - `new` and `old` are both strictly positive and the relative change is finite
    ///     (guaranteed by [`Difference::relative`]).
    ///
    /// * Max absolute increase: Succeeds if the absolute change is not increasing or if it
    ///   increases by no more than `limit`.
    ///
    ///   Requires:
    ///   - `limit` to be non-negative and finite (guaranteed by [`Absolute`]).
    fn check(self, difference: Difference, spec: FmtSpec) -> Check {
        // Lazily evaluate the relative change, returning an error if it is not well formed.
        macro_rules! check_relative {
            ($difference:ident) => {
                match $difference.relative() {
                    Ok(relative) => relative,
                    Err(err) => return Check::error(&err),
                }
            };
        }

        let (old, new, absolute) = difference.values();
        let (passed, relative) = match self {
            Self::RelativeDecrease(limit) => {
                let relative = check_relative!(difference);
                let passed = limit.as_f64() >= -relative;
                (passed, Some(relative))
            }
            Self::AbsoluteDecrease(limit) => {
                let passed =
                    absolute.direction != Direction::Decrease || !absolute.value.gt(limit.limit());
                (passed, None)
            }
            Self::RelativeIncrease(limit) => {
                let relative = check_relative!(difference);
                let passed = limit.as_f64() >= relative;
                (passed, Some(relative))
            }
            Self::AbsoluteIncrease(limit) => {
                let passed =
                    absolute.direction != Direction::Increase || !absolute.value.gt(limit.limit());
                (passed, None)
            }
        };

        let status = if passed { Status::Pass } else { Status::Fail };
        let comparison = Comparison {
            old,
            new,
            absolute,
            relative,
            tolerance: self,
            status,
            spec,
        };

        Check::new(comparison)
    }
}

impl std::fmt::Display for Tolerance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RelativeDecrease(v) => write!(f, "min {:+}%", -v.as_f64() * 100.0),
            Self::AbsoluteDecrease(v) => write!(f, "min -{}", v.limit()),
            Self::RelativeIncrease(v) => write!(f, "max {:+}%", v.as_f64() * 100.0),
            Self::AbsoluteIncrease(v) => write!(f, "max +{}", v.limit()),
        }
    }
}

impl From<MaxRelativeDecrease> for Tolerance {
    fn from(tol: MaxRelativeDecrease) -> Self {
        Self::RelativeDecrease(tol)
    }
}

impl From<MaxAbsoluteDecrease> for Tolerance {
    fn from(tol: MaxAbsoluteDecrease) -> Self {
        Self::AbsoluteDecrease(tol)
    }
}

impl From<MaxRelativeIncrease> for Tolerance {
    fn from(tol: MaxRelativeIncrease) -> Self {
        Self::RelativeIncrease(tol)
    }
}

impl From<MaxAbsoluteIncrease> for Tolerance {
    fn from(tol: MaxAbsoluteIncrease) -> Self {
        Self::AbsoluteIncrease(tol)
    }
}

//----------//
// Fraction //
//----------//

/// A number in `[0, 1]`.
#[derive(Debug, Clone, Copy)]
struct Fraction(f64);

impl Fraction {
    const fn new(value: f64) -> Option<Self> {
        if value.is_nan() || value < 0.0 || value > 1.0 {
            None
        } else {
            Some(Self(value))
        }
    }

    const fn value(&self) -> f64 {
        self.0
    }
}

impl Serialize for Fraction {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_f64(self.0)
    }
}

impl<'de> Deserialize<'de> for Fraction {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = f64::deserialize(deserializer)?;
        Self::new(value).ok_or_else(|| serde::de::Error::custom("fractions must be in [0, 1]"))
    }
}

//----------//
// Absolute //
//----------//

/// A non-negative number.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(untagged)]
enum Absolute {
    Int(u64),
    Float(NonNegativeFinite),
}

impl Absolute {
    #[cfg(test)]
    const fn try_from_f64(value: f64) -> Option<Self> {
        match NonNegativeFinite::new(value) {
            Ok(v) => Some(Self::Float(v)),
            Err(_) => None,
        }
    }

    const fn as_u64(&self) -> Option<u64> {
        if let Self::Int(v) = *self {
            Some(v)
        } else {
            None
        }
    }

    const fn as_f64(&self) -> Option<f64> {
        if let Self::Float(v) = *self {
            Some(v.get())
        } else {
            None
        }
    }

    /// Return `true` if the value of `self` is strictly greater than the value of `rhs`.
    fn gt(self, rhs: Self) -> bool {
        const _2_TO_THE_64: f64 = 18_446_744_073_709_551_616.0;

        match (self, rhs) {
            (Self::Int(lhs), Self::Int(rhs)) => lhs > rhs,
            (Self::Float(lhs), Self::Float(rhs)) => lhs.get() > rhs.get(),
            (Self::Int(lhs), Self::Float(rhs)) => {
                // We already know that `rhs` is non-negative and finite.
                let rhs = rhs.get();

                // If the floating point number is 2^64 or greater, there's no way `integer`
                // can be greater than it.
                if rhs >= _2_TO_THE_64 {
                    false
                } else {
                    // Using truncation is correct. If the spacing between floating point
                    // numbers is fractional, it won't matter.
                    //
                    // If the spacing between floating point numbers is an integer, then
                    // truncating doesn't change the numeric value (provided we're less
                    // than 2^64, which we've already checked).
                    lhs > rhs as u64
                }
            }
            (Self::Float(lhs), Self::Int(rhs)) => {
                // We already know that `lhs` is non-negative and finite.
                let lhs = lhs.get();

                // If the floating point number is 2^64 or greater, there's no way `integer`
                // can be greater than or equal to it.
                if lhs >= _2_TO_THE_64 {
                    true
                } else {
                    // This bit is a little tricky. We can't throw away the fractional part.
                    // Instead, first compare the `truncated` `lhs` with the integer `rhs`.
                    //
                    // If that's true, we're done. Otherwise if `truncated` and `rhs` are
                    // equal, we need to check that the fractional part is non-zero, which
                    // would make `lhs` larger than `rhs`.
                    let truncated = lhs as u64;
                    (truncated > rhs) | (truncated == rhs && lhs.fract() != 0.0)
                }
            }
        }
    }
}

impl std::fmt::Display for Absolute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int(v) => write!(f, "{}", v),
            Self::Float(v) => write!(f, "{}", v.get()),
        }
    }
}

impl From<u64> for Absolute {
    fn from(value: u64) -> Self {
        Self::Int(value)
    }
}

impl From<NonNegativeFinite> for Absolute {
    fn from(value: NonNegativeFinite) -> Self {
        Self::Float(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
enum Direction {
    /// Closer to negative infinity.
    Decrease,
    /// The same.
    Equal,
    /// Closer to positive infinity.
    Increase,
}

impl Direction {
    fn from_cmp<T>(old: &T, new: &T) -> Self
    where
        T: Ord,
    {
        match new.cmp(old) {
            std::cmp::Ordering::Less => Self::Decrease,
            std::cmp::Ordering::Equal => Self::Equal,
            std::cmp::Ordering::Greater => Self::Increase,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Difference {
    PosInt {
        absolute: u64,
        direction: Direction,
        old: u64,
        new: u64,
    },
    NegInt {
        absolute: u64,
        direction: Direction,
        old: i64,
        new: i64,
    },
    Float {
        absolute: NonNegativeFinite,
        direction: Direction,
        old: f64,
        new: f64,
    },
}

impl Difference {
    /// Return the values of `old`, `new`, and absolute change.
    fn values(&self) -> (Number, Number, AbsoluteChange) {
        match *self {
            Self::PosInt {
                absolute,
                direction,
                old,
                new,
            } => (
                old.into(),
                new.into(),
                AbsoluteChange::new(absolute.into(), direction),
            ),
            Self::NegInt {
                absolute,
                direction,
                old,
                new,
            } => (
                old.into(),
                new.into(),
                AbsoluteChange::new(absolute.into(), direction),
            ),
            Self::Float {
                absolute,
                direction,
                old,
                new,
            } => (
                old.into(),
                new.into(),
                AbsoluteChange::new(absolute.into(), direction),
            ),
        }
    }

    /// Return the relative difference between the old and new values by:
    ///
    /// ```math
    /// (new - old)
    /// -----------
    ///     old
    /// ```
    ///
    /// This will generate an error under the following situations:
    ///
    /// * `old == 0`: This makes the computation ill-defined.
    ///
    /// * `new == 0`: While technically allowed mathematically, accepting `new == 0` in
    ///   checked-in baselines would cause future relative comparisons to fail due to
    ///   the first check.
    ///
    /// * Either `new` or `old` is negative: This messes with the interpretation of the
    ///   relative tolerance checks defined in [`self`] and so are rejected to ensure
    ///   consistent semantics.
    ///
    /// * Any of `new`, `old`, or the result are infinity or NaN.
    fn relative(&self) -> Result<f64, RelativeError> {
        fn check_sign_and_zero<T>(old: T, new: T) -> Result<(), RelativeError>
        where
            T: Default + PartialOrd + Into<Number>,
        {
            if old == T::default() {
                Err(RelativeError::OldZero)
            } else if new == T::default() {
                Err(RelativeError::NewZero)
            } else if old < T::default() {
                Err(RelativeError::OldNegative(old.into()))
            } else if new < T::default() {
                Err(RelativeError::NewNegative(new.into()))
            } else {
                Ok(())
            }
        }

        fn check_finite(old: f64, new: f64) -> Result<(), RelativeError> {
            if !old.is_finite() {
                Err(RelativeError::OldNotFinite(old))
            } else if !new.is_finite() {
                Err(RelativeError::NewNotFinite(new))
            } else {
                Ok(())
            }
        }

        fn relative(absolute: f64, old: f64, direction: Direction) -> Result<f64, RelativeError> {
            let value = absolute / old;
            if !value.is_finite() {
                Err(RelativeError::RelativeNotFinite(value))
            } else {
                match direction {
                    Direction::Decrease => Ok(-value),
                    Direction::Equal | Direction::Increase => Ok(value),
                }
            }
        }

        match *self {
            Self::PosInt {
                absolute,
                direction,
                old,
                new,
            } => {
                check_sign_and_zero(old, new)?;
                relative(absolute as f64, old as f64, direction)
            }
            Self::NegInt {
                absolute,
                direction,
                old,
                new,
            } => {
                check_sign_and_zero(old, new)?;
                relative(absolute as f64, old as f64, direction)
            }
            Self::Float {
                absolute,
                direction,
                old,
                new,
            } => {
                check_finite(old, new)?;
                check_sign_and_zero(old, new)?;
                relative(absolute.get(), old, direction)
            }
        }
    }
}

#[derive(Debug, Error)]
enum RelativeError {
    #[error("old must be > 0 (got 0)")]
    OldZero,
    #[error("new must be > 0 (got 0)")]
    NewZero,
    #[error("old must be > 0 (got {0})")]
    OldNegative(Number),
    #[error("new must be > 0 (got {0})")]
    NewNegative(Number),
    #[error("old must be finite (got {0})")]
    OldNotFinite(f64),
    #[error("new must be finite (got {0})")]
    NewNotFinite(f64),
    #[error("relative change overflowed (got {0})")]
    RelativeNotFinite(f64),
}

trait ChangeFrom {
    type Error: std::fmt::Display;
    fn change_from(&self, old: &Self) -> Result<Difference, Self::Error>;
}

macro_rules! difference {
    (@int $T:ident) => {
        impl ChangeFrom for $T {
            type Error = std::convert::Infallible;
            fn change_from(&self, old: &Self) -> Result<Difference, std::convert::Infallible> {
                let old: i64 = *old as i64;
                let new: i64 = *self as i64;

                Ok(Difference::NegInt {
                    absolute: new.abs_diff(old),
                    direction: Direction::from_cmp(&old, &new),
                    old,
                    new,
                })
            }
        }
    };
    (@uint $T:ident) => {
        impl ChangeFrom for $T {
            type Error = std::convert::Infallible;
            fn change_from(&self, old: &Self) -> Result<Difference, std::convert::Infallible> {
                let old: u64 = *old as u64;
                let new: u64 = *self as u64;

                Ok(Difference::PosInt {
                    absolute: new.abs_diff(old),
                    direction: Direction::from_cmp(&old, &new),
                    old,
                    new,
                })
            }
        }
    };
    (@float $T:ident) => {
        impl ChangeFrom for $T {
            type Error = FloatError;
            fn change_from(&self, old: &Self) -> Result<Difference, FloatError> {
                let old = *old as f64;
                let new = *self as f64;

                if !old.is_finite() {
                    return Err(FloatError::Old(old));
                }
                if !new.is_finite() {
                    return Err(FloatError::New(new));
                }

                let absolute = (new - old);
                let direction = if absolute == 0.0 {
                    Direction::Equal
                } else if absolute > 0.0 {
                    Direction::Increase
                } else {
                    Direction::Decrease
                };

                let absolute = match NonNegativeFinite::new(absolute.abs()) {
                    Ok(absolute) => absolute,
                    Err(_) => return Err(FloatError::Diff(old, new)),
                };

                Ok(Difference::Float {
                    absolute,
                    direction,
                    old,
                    new,
                })
            }
        }
    };
}

#[derive(Debug, Error)]
enum FloatError {
    #[error("old must be finite (got {0})")]
    Old(f64),
    #[error("new must be finite (got {0})")]
    New(f64),
    #[error("absolute change overflowed (old {0}, new {1})")]
    Diff(f64, f64),
}

difference!(@int i8);
difference!(@int i16);
difference!(@int i32);
difference!(@int i64);
difference!(@int isize);

difference!(@uint u8);
difference!(@uint u16);
difference!(@uint u32);
difference!(@uint u64);
difference!(@uint usize);

difference!(@float f32);
difference!(@float f64);

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum Number {
    PosInt(u64),
    NegInt(i64),
    Float(f64),
}

impl std::fmt::Display for Number {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_with(f, &FmtSpec::default())
    }
}

impl FmtWith for Number {
    fn fmt_with(&self, f: &mut dyn std::fmt::Write, spec: &FmtSpec) -> std::fmt::Result {
        match self {
            Self::PosInt(v) => v.fmt_with(f, spec),
            Self::NegInt(v) => v.fmt_with(f, spec),
            Self::Float(v) => v.fmt_with(f, spec),
        }
    }
}

impl From<u64> for Number {
    fn from(value: u64) -> Self {
        Self::PosInt(value)
    }
}

impl From<i64> for Number {
    fn from(value: i64) -> Self {
        Self::NegInt(value)
    }
}

impl From<f64> for Number {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

#[derive(Debug, Serialize)]
struct AbsoluteChange {
    value: Absolute,
    direction: Direction,
}

impl AbsoluteChange {
    fn new(value: Absolute, direction: Direction) -> Self {
        Self { value, direction }
    }
}

impl FmtWith for AbsoluteChange {
    fn fmt_with(&self, f: &mut dyn std::fmt::Write, spec: &FmtSpec) -> std::fmt::Result {
        let sign = match self.direction {
            Direction::Equal | Direction::Increase => "+",
            Direction::Decrease => "-",
        };

        match &self.value {
            Absolute::Int(v) => write!(f, "{}{}", sign, AsFmtWith(v, spec)),
            Absolute::Float(v) => write!(f, "{}{}", sign, AsFmtWith(&v.get(), spec)),
        }
    }
}

impl std::fmt::Display for AbsoluteChange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_with(f, &FmtSpec::default())
    }
}

//------------//
// Comparison //
//------------//

#[derive(Debug, Serialize)]
struct Comparison {
    old: Number,
    new: Number,
    absolute: AbsoluteChange,
    relative: Option<f64>,
    tolerance: Tolerance,
    status: Status,
    #[serde(skip)]
    spec: FmtSpec,
}

impl super::Checkable for Comparison {
    fn status(&self) -> Status {
        self.status
    }

    fn fmt(&self, f: &mut dyn std::fmt::Write, ctx: super::FmtContext) -> std::fmt::Result {
        let spec = &self.spec;
        write!(
            f,
            "{} => {} (",
            AsFmtWith(&self.old, spec),
            AsFmtWith(&self.new, spec),
        )?;

        match self.relative {
            Some(relative) => write!(f, "{:+.3}%", relative * 100.0)?,
            None => self.absolute.fmt_with(f, spec)?,
        }
        write!(f, ")")?;

        if self.status == Status::Fail {
            write!(f, " [{}] {}", self.tolerance, ctx.fail_marker())?;
        }

        Ok(())
    }

    fn serialize(&self, f: &mut dyn FnMut(&dyn erased_serde::Serialize)) {
        f(&self)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::assert_matches;

    // We test the public API using the resulting `Check`.
    //
    // This provides a nice way of ensuring the implementations are working as expected,
    // generating the proper report, and viewing the layout of the reports.
    //
    // Tests begin with such integration style checks - more unit-style tests exercising
    // numeric corner cases come later.

    #[test]
    fn test_max_relative_decrease() {
        use Status::{Fail, Pass};

        // Constructors/Accessors
        assert_eq!(MaxRelativeDecrease::new(-0.0).unwrap().as_f64(), 0.0);
        assert!(MaxRelativeDecrease::new(-1.0).is_none());
        assert!(MaxRelativeDecrease::new(f64::INFINITY).is_none());
        assert!(MaxRelativeDecrease::new(f64::NEG_INFINITY).is_none());
        assert!(MaxRelativeDecrease::new(f64::NAN).is_none());
        assert_eq!(MaxRelativeDecrease::new(1.0).unwrap().as_f64(), 1.0);
        assert!(MaxRelativeDecrease::new(1.1).is_none());

        // Reports
        let tolerance = MaxRelativeDecrease::new(0.25).unwrap();
        let spec = FmtSpec::new().with_precision(2);
        let cases = [
            (8.0, 10.0, Pass, "8.00 => 10.00 (+25.000%)"),
            (8.0, 8.0, Pass, "8.00 => 8.00 (+0.000%)"),
            (8.0, 7.0, Pass, "8.00 => 7.00 (-12.500%)"),
            (8.0, 6.0, Pass, "8.00 => 6.00 (-25.000%)"),
            (8.0, 5.0, Fail, "8.00 => 5.00 (-37.500%) [min -25%] | FAIL"),
            (0.0, 1.0, Fail, "old must be > 0 (got 0) | FAIL"),
            (1.0, 0.0, Fail, "new must be > 0 (got 0) | FAIL"),
            (-1.0, 1.0, Fail, "old must be > 0 (got -1) | FAIL"),
            (1.0, -1.0, Fail, "new must be > 0 (got -1) | FAIL"),
            (
                f64::INFINITY,
                1.0,
                Fail,
                "old must be finite (got inf) | FAIL",
            ),
            (1.0, f64::NAN, Fail, "new must be finite (got NaN) | FAIL"),
            (
                f64::from_bits(1),
                1.0,
                Fail,
                "relative change overflowed (got inf) | FAIL",
            ),
        ];

        for (old, new, status, expected) in cases {
            let check = Check::higher_is_better(&tolerance, old, new, spec);
            assert_eq!(check.status(), status, "old = {old}, new = {new}");
            assert_eq!(check.render(super::super::FmtContext::Tree), expected);

            let report = super::super::Report::higher_is_better(&tolerance, old, new, spec);
            assert_eq!(report.status(), status, "old = {old}, new = {new}");
            assert_eq!(report.to_string(), expected);
        }
    }

    #[test]
    fn test_max_relative_increase() {
        use Status::{Fail, Pass};

        // Constructors/Accessors
        assert_eq!(MaxRelativeIncrease::new(-0.0).unwrap().as_f64(), 0.0);
        assert!(MaxRelativeIncrease::new(-1.0).is_none());
        assert!(MaxRelativeIncrease::new(f64::INFINITY).is_none());
        assert!(MaxRelativeIncrease::new(f64::NEG_INFINITY).is_none());
        assert!(MaxRelativeIncrease::new(f64::NAN).is_none());
        assert_eq!(MaxRelativeIncrease::new(1.0).unwrap().as_f64(), 1.0);
        assert_eq!(MaxRelativeIncrease::new(10.0).unwrap().as_f64(), 10.0);

        // Reports
        let tolerance = MaxRelativeIncrease::new(0.25).unwrap();
        let spec = FmtSpec::new().with_precision(2);
        let cases = [
            (8.0, 6.0, Pass, "8.00 => 6.00 (-25.000%)"),
            (8.0, 8.0, Pass, "8.00 => 8.00 (+0.000%)"),
            (8.0, 9.0, Pass, "8.00 => 9.00 (+12.500%)"),
            (8.0, 10.0, Pass, "8.00 => 10.00 (+25.000%)"),
            (
                8.0,
                11.0,
                Fail,
                "8.00 => 11.00 (+37.500%) [max +25%] | FAIL",
            ),
            (0.0, 1.0, Fail, "old must be > 0 (got 0) | FAIL"),
            (1.0, 0.0, Fail, "new must be > 0 (got 0) | FAIL"),
            (-1.0, 1.0, Fail, "old must be > 0 (got -1) | FAIL"),
            (1.0, -1.0, Fail, "new must be > 0 (got -1) | FAIL"),
            (f64::NAN, 1.0, Fail, "old must be finite (got NaN) | FAIL"),
            (
                1.0,
                f64::INFINITY,
                Fail,
                "new must be finite (got inf) | FAIL",
            ),
            (
                f64::from_bits(1),
                1.0,
                Fail,
                "relative change overflowed (got inf) | FAIL",
            ),
        ];

        for (old, new, status, expected) in cases {
            let check = Check::lower_is_better(&tolerance, old, new, spec);
            assert_eq!(check.status(), status, "old = {old}, new = {new}");
            assert_eq!(check.render(super::super::FmtContext::Tree), expected);

            let report = super::super::Report::lower_is_better(&tolerance, old, new, spec);
            assert_eq!(report.status(), status, "old = {old}, new = {new}");
            assert_eq!(report.to_string(), expected);
        }
    }

    #[test]
    fn test_max_absolute_decrease() {
        use Status::{Fail, Pass};

        // Constructors/Accessors
        assert_eq!(
            MaxAbsoluteDecrease::try_from_f64(-0.0)
                .unwrap()
                .as_f64()
                .unwrap(),
            0.0
        );
        assert!(MaxAbsoluteDecrease::try_from_f64(-1.0).is_none());
        assert!(MaxAbsoluteDecrease::try_from_f64(f64::INFINITY).is_none());
        assert!(MaxAbsoluteDecrease::try_from_f64(f64::NEG_INFINITY).is_none());
        assert!(MaxAbsoluteDecrease::try_from_f64(f64::NAN).is_none());
        assert!(
            MaxAbsoluteDecrease::try_from_f64(1.0)
                .unwrap()
                .as_u64()
                .is_none()
        );

        assert_eq!(MaxAbsoluteDecrease::from_u64(100).as_u64().unwrap(), 100);
        assert!(MaxAbsoluteDecrease::from_u64(100).as_f64().is_none());

        // Reports
        let tolerance = MaxAbsoluteDecrease::try_from_f64(0.5).unwrap();
        let spec = FmtSpec::new().with_precision(2);
        let cases = [
            (10.0, 11.0, Pass, "10.00 => 11.00 (+1.00)"),
            (10.0, 10.0, Pass, "10.00 => 10.00 (+0.00)"),
            (10.0, 9.75, Pass, "10.00 => 9.75 (-0.25)"),
            (10.0, 9.5, Pass, "10.00 => 9.50 (-0.50)"),
            (10.0, 9.25, Fail, "10.00 => 9.25 (-0.75) [min -0.5] | FAIL"),
            (-1.0, -1.5, Pass, "-1.00 => -1.50 (-0.50)"),
            (0.25, -0.25, Pass, "0.25 => -0.25 (-0.50)"),
            (0.0, -0.5, Pass, "0.00 => -0.50 (-0.50)"),
            (0.5, 0.0, Pass, "0.50 => 0.00 (-0.50)"),
            (
                f64::INFINITY,
                1.0,
                Fail,
                "old must be finite (got inf) | FAIL",
            ),
            (1.0, f64::NAN, Fail, "new must be finite (got NaN) | FAIL"),
        ];

        for (old, new, status, expected) in cases {
            let check = Check::higher_is_better(&tolerance, old, new, spec);
            assert_eq!(check.status(), status, "old = {old}, new = {new}");
            assert_eq!(check.render(super::super::FmtContext::Tree), expected);
        }
    }

    #[test]
    fn test_max_absolute_increase() {
        use Status::{Fail, Pass};

        // Constructors/Accessors
        assert_eq!(
            MaxAbsoluteIncrease::try_from_f64(-0.0)
                .unwrap()
                .as_f64()
                .unwrap(),
            0.0
        );
        assert!(MaxAbsoluteIncrease::try_from_f64(-1.0).is_none());
        assert!(MaxAbsoluteIncrease::try_from_f64(f64::INFINITY).is_none());
        assert!(MaxAbsoluteIncrease::try_from_f64(f64::NEG_INFINITY).is_none());
        assert!(MaxAbsoluteIncrease::try_from_f64(f64::NAN).is_none());
        assert!(
            MaxAbsoluteIncrease::try_from_f64(1.0)
                .unwrap()
                .as_u64()
                .is_none()
        );

        assert_eq!(MaxAbsoluteIncrease::from_u64(100).as_u64().unwrap(), 100);
        assert!(MaxAbsoluteIncrease::from_u64(100).as_f64().is_none());

        // Reports
        let tolerance = MaxAbsoluteIncrease::try_from_f64(0.5).unwrap();
        let spec = FmtSpec::new().with_precision(2);
        let cases = [
            (10.0, 9.0, Pass, "10.00 => 9.00 (-1.00)"),
            (10.0, 10.0, Pass, "10.00 => 10.00 (+0.00)"),
            (10.0, 10.25, Pass, "10.00 => 10.25 (+0.25)"),
            (10.0, 10.5, Pass, "10.00 => 10.50 (+0.50)"),
            (
                10.0,
                10.75,
                Fail,
                "10.00 => 10.75 (+0.75) [max +0.5] | FAIL",
            ),
            (-1.0, -0.5, Pass, "-1.00 => -0.50 (+0.50)"),
            (-0.25, 0.25, Pass, "-0.25 => 0.25 (+0.50)"),
            (0.0, 0.5, Pass, "0.00 => 0.50 (+0.50)"),
            (-0.5, 0.0, Pass, "-0.50 => 0.00 (+0.50)"),
            (f64::NAN, 1.0, Fail, "old must be finite (got NaN) | FAIL"),
            (
                1.0,
                f64::INFINITY,
                Fail,
                "new must be finite (got inf) | FAIL",
            ),
        ];

        for (old, new, status, expected) in cases {
            let check = Check::lower_is_better(&tolerance, old, new, spec);
            assert_eq!(check.status(), status, "old = {old}, new = {new}");
            assert_eq!(check.render(super::super::FmtContext::Tree), expected);
        }
    }

    //------------//
    // Unit Tests //
    //------------//

    #[test]
    fn test_absolute_gt() {
        use std::cmp::Ordering::{Equal, Greater, Less};

        let int = Absolute::Int;
        let float = |value| Absolute::try_from_f64(value).unwrap();
        let two_to_53 = 1u64 << 53;
        let two_to_64 = 18_446_744_073_709_551_616.0f64;
        let below_two_to_64 = two_to_64.next_down();

        let cases = [
            (int(0), int(0), Equal),
            (int(0), int(u64::MAX), Less),
            (int(u64::MAX - 1), int(u64::MAX), Less),
            (float(0.0), float(-0.0), Equal),
            (float(0.0), float(f64::from_bits(1)), Less),
            (float(1.0), float(1.0f64.next_up()), Less),
            (int(0), float(-0.0), Equal),
            (int(0), float(f64::from_bits(1)), Less),
            (int(1), float(0.5), Greater),
            (int(3), float(3.0), Equal),
            (int(3), float(3.5), Less),
            (int(4), float(3.5), Greater),
            // Adjacent integers must remain distinct above f64's exact-integer range.
            (int(two_to_53), float(two_to_53 as f64), Equal),
            (int(two_to_53 + 1), float(two_to_53 as f64), Greater),
            (int(two_to_53 + 1), float((two_to_53 + 2) as f64), Less),
            // The last f64 below 2^64 is exactly 2^64 - 2048.
            (int(u64::MAX - 2047), float(below_two_to_64), Equal),
            (int(u64::MAX - 2048), float(below_two_to_64), Less),
            (int(u64::MAX), float(below_two_to_64), Greater),
            (int(u64::MAX), float(two_to_64), Less),
            (int(u64::MAX), float(f64::MAX), Less),
        ];

        for (lhs, rhs, ordering) in cases {
            assert_eq!(lhs.gt(rhs), ordering == Greater, "{lhs:?} > {rhs:?}");
            assert_eq!(rhs.gt(lhs), ordering == Less, "{rhs:?} > {lhs:?}");
        }
    }

    #[test]
    fn test_absolute_display() {
        assert_eq!(Absolute::Int(10).to_string(), "10");
        assert_eq!(Absolute::try_from_f64(10.5).unwrap().to_string(), "10.5");
    }

    #[test]
    fn test_absolute_change() {
        let int = Absolute::Int;
        let float = |value| Absolute::try_from_f64(value).unwrap();

        let spec = FmtSpec::default().with_precision(1);
        let cases = [
            (int(10), Direction::Increase, None, "+10"),
            (int(0), Direction::Equal, None, "+0"),
            // No formatting precision
            (int(10), Direction::Decrease, None, "-10"),
            (float(1.75), Direction::Increase, None, "+1.75"),
            (float(0.0), Direction::Equal, None, "+0"),
            (float(1.75), Direction::Decrease, None, "-1.75"),
            // limit formatting precision
            (float(1.75), Direction::Increase, Some(spec), "+1.8"),
            (float(0.0), Direction::Equal, Some(spec), "+0.0"),
            (float(1.75), Direction::Decrease, Some(spec), "-1.8"),
        ];

        for (value, direction, spec, expected) in cases {
            let change = AbsoluteChange::new(value, direction);
            match spec {
                Some(spec) => assert_eq!(AsFmtWith(&change, &spec).to_string(), expected),
                None => assert_eq!(change.to_string(), expected),
            }
        }
    }

    #[test]
    fn test_difference_integers() {
        for (old, new, magnitude, direction) in [
            (0u64, 0u64, 0, Direction::Equal),
            (0, u64::MAX, u64::MAX, Direction::Increase),
            (u64::MAX, 0, u64::MAX, Direction::Decrease),
            (u64::MAX - 1, u64::MAX, 1, Direction::Increase),
        ] {
            let difference = new.change_from(&old).unwrap();
            let Difference::PosInt {
                absolute,
                direction: got_direction,
                old: got_old,
                new: got_new,
            } = difference
            else {
                panic!("expected unsigned difference, got {difference:?}");
            };
            assert_eq!((got_old, got_new, absolute), (old, new, magnitude));
            assert_eq!(got_direction, direction);
        }

        for (old, new, magnitude, direction) in [
            (-5i64, -5i64, 0, Direction::Equal),
            (-5, -2, 3, Direction::Increase),
            (2, -5, 7, Direction::Decrease),
            (i64::MIN, i64::MAX, u64::MAX, Direction::Increase),
            (i64::MAX, i64::MIN, u64::MAX, Direction::Decrease),
        ] {
            let difference = new.change_from(&old).unwrap();
            let Difference::NegInt {
                absolute,
                direction: got_direction,
                old: got_old,
                new: got_new,
            } = difference
            else {
                panic!("expected signed difference, got {difference:?}");
            };
            assert_eq!((got_old, got_new, absolute), (old, new, magnitude));
            assert_eq!(got_direction, direction);
        }
    }

    #[test]
    fn test_difference_floats() {
        for (old, new, magnitude, direction) in [
            (0.0f64, -0.0f64, 0.0, Direction::Equal),
            (-1.5, 2.0, 3.5, Direction::Increase),
            (2.0, -1.5, 3.5, Direction::Decrease),
            (
                0.0,
                f64::from_bits(1),
                f64::from_bits(1),
                Direction::Increase,
            ),
        ] {
            let difference = new.change_from(&old).unwrap();
            let Difference::Float {
                absolute,
                direction: got_direction,
                old: got_old,
                new: got_new,
            } = difference
            else {
                panic!("expected floating-point difference, got {difference:?}");
            };
            assert_eq!((got_old, got_new, absolute.get()), (old, new, magnitude));
            assert_eq!(got_direction, direction);
        }

        // f32 inputs must be widened before subtraction.
        let difference = f32::MAX.change_from(&-f32::MAX).unwrap();
        let (_, _, absolute) = difference.values();
        assert_eq!(absolute.value.as_f64(), Some(2.0 * f64::from(f32::MAX)));
    }

    #[test]
    fn test_difference_float_errors() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_matches!(1.0f64.change_from(&invalid), Err(FloatError::Old(_)));
            assert_matches!(invalid.change_from(&1.0), Err(FloatError::New(_)));
        }
        for (old, new) in [(-f64::MAX, f64::MAX), (f64::MAX, -f64::MAX)] {
            assert_matches!(new.change_from(&old), Err(FloatError::Diff(..)));
        }
    }

    #[test]
    fn test_difference_relative() {
        for (old, new, expected) in [(8u64, 10u64, 0.25), (8, 6, -0.25), (8, 8, 0.0)] {
            assert_eq!(new.change_from(&old).unwrap().relative().unwrap(), expected);
            assert_eq!(
                (new as i64)
                    .change_from(&(old as i64))
                    .unwrap()
                    .relative()
                    .unwrap(),
                expected
            );
            assert_eq!(
                (new as f64)
                    .change_from(&(old as f64))
                    .unwrap()
                    .relative()
                    .unwrap(),
                expected
            );
        }

        // Compute the difference before converting large integers to f64.
        let old = 1u64 << 53;
        assert_eq!(
            (old + 1).change_from(&old).unwrap().relative().unwrap(),
            1.0 / old as f64
        );
    }

    #[test]
    fn test_difference_relative_errors() {
        assert_matches!(
            1u64.change_from(&0).unwrap().relative(),
            Err(RelativeError::OldZero)
        );
        assert_matches!(
            0u64.change_from(&1).unwrap().relative(),
            Err(RelativeError::NewZero)
        );
        for (old, new) in [(-1i64, 1i64), (-2, -1)] {
            assert_matches!(
                new.change_from(&old).unwrap().relative(),
                Err(RelativeError::OldNegative(_))
            );
        }
        assert_matches!(
            (-1i64).change_from(&1).unwrap().relative(),
            Err(RelativeError::NewNegative(_))
        );

        for zero in [0.0f64, -0.0] {
            assert_matches!(
                1.0f64.change_from(&zero).unwrap().relative(),
                Err(RelativeError::OldZero)
            );
            assert_matches!(
                zero.change_from(&1.0).unwrap().relative(),
                Err(RelativeError::NewZero)
            );
        }
        assert_matches!(
            1.0f64.change_from(&-1.0).unwrap().relative(),
            Err(RelativeError::OldNegative(_))
        );
        assert_matches!(
            (-1.0f64).change_from(&1.0).unwrap().relative(),
            Err(RelativeError::NewNegative(_))
        );
        assert_matches!(
            f64::MAX.change_from(&f64::from_bits(1)).unwrap().relative(),
            Err(RelativeError::RelativeNotFinite(_))
        );
    }

    #[test]
    fn test_comparison_rendering() {
        let spec = FmtSpec::new().with_precision(2);
        let relative = MaxRelativeIncrease::new(0.05).unwrap();
        let absolute = MaxAbsoluteDecrease::try_from_f64(0.5).unwrap();
        let cases = [
            (
                relative.lower_is_better(100.0, 104.0, spec),
                "100.00 => 104.00 (+4.000%)",
                "100.00 => 104.00 (+4.000%)",
            ),
            (
                relative.lower_is_better(100.0, 106.0, spec),
                "100.00 => 106.00 (+6.000%) [max +5%] | FAIL",
                "100.00 => 106.00 (+6.000%) [max +5%] !",
            ),
            (
                absolute.higher_is_better(10.0, 9.75, spec),
                "10.00 => 9.75 (-0.25)",
                "10.00 => 9.75 (-0.25)",
            ),
            (
                absolute.higher_is_better(10.0, 9.0, spec),
                "10.00 => 9.00 (-1.00) [min -0.5] | FAIL",
                "10.00 => 9.00 (-1.00) [min -0.5] !",
            ),
        ];

        for (check, tree, table) in cases {
            assert_eq!(check.render(super::super::FmtContext::Tree), tree);
            assert_eq!(check.render(super::super::FmtContext::Table), table);
        }
    }

    //---------------//
    // Serialization //
    //---------------//

    #[test]
    fn test_fraction_deserialization() {
        for (input, expected) in [
            (serde_json::json!(0), 0.0),
            (serde_json::json!(-0.0), -0.0),
            (serde_json::json!(0.25), 0.25),
            (serde_json::json!(1), 1.0),
        ] {
            let fraction: Fraction = serde_json::from_value(input).unwrap();
            assert_eq!(fraction.value(), expected);
        }

        for value in [-1.0, -f64::from_bits(1), 1.0f64.next_up(), 2.0] {
            let error = serde_json::from_value::<Fraction>(serde_json::json!(value)).unwrap_err();
            assert!(
                error.to_string().contains("fractions must be in [0, 1]"),
                "value = {value}, error = {error}"
            );
        }

        // JSON cannot represent nonfinite numbers, so deserialize them directly.
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let deserializer =
                serde::de::value::F64Deserializer::<serde::de::value::Error>::new(value);
            let error = Fraction::deserialize(deserializer).unwrap_err();
            assert_eq!(error.to_string(), "fractions must be in [0, 1]");
        }
    }

    #[test]
    fn test_tolerance_serialization() {
        let cases = [
            (
                Tolerance::from(MaxRelativeDecrease::new(0.25).unwrap()),
                serde_json::json!({ "max-relative-decrease": 0.25 }),
            ),
            (
                Tolerance::from(MaxRelativeIncrease::new(0.5).unwrap()),
                serde_json::json!({ "max-relative-increase": 0.5 }),
            ),
            (
                Tolerance::from(MaxAbsoluteDecrease::from_u64(u64::MAX)),
                serde_json::json!({ "max-absolute-decrease": u64::MAX }),
            ),
            (
                Tolerance::from(MaxAbsoluteDecrease::try_from_f64(0.5).unwrap()),
                serde_json::json!({ "max-absolute-decrease": 0.5 }),
            ),
            (
                Tolerance::from(MaxAbsoluteIncrease::from_u64(u64::MAX)),
                serde_json::json!({ "max-absolute-increase": u64::MAX }),
            ),
            (
                Tolerance::from(MaxAbsoluteIncrease::try_from_f64(0.5).unwrap()),
                serde_json::json!({ "max-absolute-increase": 0.5 }),
            ),
        ];

        for (tolerance, expected) in cases {
            assert_eq!(serde_json::to_value(tolerance).unwrap(), expected);
        }
    }

    #[test]
    fn test_comparison_serialization() {
        let spec = FmtSpec::new().with_precision(0);
        let cases = [
            (
                MaxRelativeIncrease::new(0.25)
                    .unwrap()
                    .lower_is_better(8.0f64, 10.0, spec),
                serde_json::json!({
                    "old": 8.0,
                    "new": 10.0,
                    "absolute": { "value": 2.0, "direction": "Increase" },
                    "relative": 0.25,
                    "tolerance": { "max-relative-increase": 0.25 },
                    "status": "PASS",
                }),
            ),
            (
                MaxRelativeDecrease::new(0.25)
                    .unwrap()
                    .higher_is_better(8u64, 5, spec),
                serde_json::json!({
                    "old": 8,
                    "new": 5,
                    "absolute": { "value": 3, "direction": "Decrease" },
                    "relative": -0.375,
                    "tolerance": { "max-relative-decrease": 0.25 },
                    "status": "FAIL",
                }),
            ),
            (
                MaxAbsoluteDecrease::from_u64(u64::MAX - 1).higher_is_better(
                    i64::MAX,
                    i64::MIN,
                    spec,
                ),
                serde_json::json!({
                    "old": i64::MAX,
                    "new": i64::MIN,
                    "absolute": { "value": u64::MAX, "direction": "Decrease" },
                    "relative": null,
                    "tolerance": { "max-absolute-decrease": u64::MAX - 1 },
                    "status": "FAIL",
                }),
            ),
            (
                MaxAbsoluteIncrease::try_from_f64(0.5)
                    .unwrap()
                    .lower_is_better(1.25f64, 1.5, spec),
                serde_json::json!({
                    "old": 1.25,
                    "new": 1.5,
                    "absolute": { "value": 0.25, "direction": "Increase" },
                    "relative": null,
                    "tolerance": { "max-absolute-increase": 0.5 },
                    "status": "PASS",
                }),
            ),
        ];

        for (check, expected) in cases {
            assert_eq!(serde_json::to_value(check).unwrap(), expected);
        }
    }
}

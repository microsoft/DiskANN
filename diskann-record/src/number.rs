/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Lossless container for the on-wire numeric kinds (`u64`, `i64`, `f64`).
//!
//! [`Number`] is the value type produced by the manifest deserializer for every JSON
//! number. The conversion accessors (`as_u32`, `as_i64`, etc.) attempt to narrow into a
//! target Rust type and return `None` when the value is out of range or would lose
//! precision; loaders surface this as [`crate::load::error::Kind::NumberOutOfRange`].

#[cfg(feature = "serde")]
use serde::de::{self, Visitor};
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A numeric value carried in a manifest, preserving the kind the writer chose.
///
/// The wire format distinguishes unsigned, signed, and floating-point numbers; the
/// deserializer preserves that distinction by selecting the matching variant. Use the
/// narrowing accessors (e.g. [`Number::as_u32`], [`Number::as_f64`]) to extract a Rust
/// value of the desired type.
#[derive(Debug, Clone, Copy)]
pub enum Number {
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Number {
    /// Returns the string sentinel for a non-finite `f64`, or `None` for any finite or
    /// integer value.
    ///
    /// JSON cannot represent `NaN`/`\pm inf` as numeric literals, so these are encoded as
    /// the strings `"nan"`, `"inf"`, and `"neg_inf"` instead. [`Number::from_sentinel`]
    /// is the inverse.
    pub(crate) fn sentinel(self) -> Option<&'static str> {
        match self {
            Self::F64(v) if v.is_nan() => Some("nan"),
            Self::F64(v) if v == f64::INFINITY => Some("inf"),
            Self::F64(v) if v == f64::NEG_INFINITY => Some("neg_inf"),
            _ => None,
        }
    }

    /// Decodes a non-finite `f64` sentinel produced by [`Number::sentinel`], or `None`
    /// for any other string.
    pub(crate) fn from_sentinel(s: &str) -> Option<Self> {
        match s {
            "nan" => Some(Self::F64(f64::NAN)),
            "inf" => Some(Self::F64(f64::INFINITY)),
            "neg_inf" => Some(Self::F64(f64::NEG_INFINITY)),
            _ => None,
        }
    }
}

#[cfg(feature = "serde")]
impl Serialize for Number {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if let Some(sentinel) = self.sentinel() {
            return serializer.serialize_str(sentinel);
        }
        match *self {
            Self::U64(v) => serializer.serialize_u64(v),
            Self::I64(v) => serializer.serialize_i64(v),
            Self::F64(v) => serializer.serialize_f64(v),
        }
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Number {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct NumberVisitor;

        impl<'de> Visitor<'de> for NumberVisitor {
            type Value = Number;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a number")
            }

            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Number, E> {
                Ok(Number::U64(v))
            }

            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Number, E> {
                Ok(Number::I64(v))
            }

            fn visit_f64<E: de::Error>(self, v: f64) -> Result<Number, E> {
                Ok(Number::F64(v))
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<Number, E> {
                Number::from_sentinel(v).ok_or_else(|| {
                    de::Error::custom(format!("expected a number or numeric sentinel, got {v:?}"))
                })
            }
        }

        deserializer.deserialize_any(NumberVisitor)
    }
}

macro_rules! try_cast {
    ($v:ident :$T:ty => $U:ty) => {{
        let c = $v as $U;
        if c as $T == $v { Some(c) } else { None }
    }};
}

macro_rules! int {
    ($f:ident, $T:ty) => {
        pub fn $f(self) -> Option<$T> {
            match self {
                Self::U64(v) => v.try_into().ok(),
                Self::I64(v) => v.try_into().ok(),
                Self::F64(v) => try_cast!(v:f64 => $T),
            }
        }
    }
}

macro_rules! float {
    ($f:ident, $T:ty) => {
        pub fn $f(self) -> Option<$T> {
            match self {
                Self::U64(v) => try_cast!(v:u64 => $T),
                Self::I64(v) => try_cast!(v:i64 => $T),
                Self::F64(v) => try_cast!(v:f64 => $T),
            }
        }
    }
}

impl Number {
    int!(as_u8, u8);
    int!(as_u16, u16);
    int!(as_u32, u32);
    int!(as_u64, u64);
    int!(as_usize, usize);

    int!(as_i8, i8);
    int!(as_i16, i16);
    int!(as_i32, i32);
    int!(as_i64, i64);
    int!(as_isize, isize);

    float!(as_f32, f32);
    float!(as_f64, f64);
}

macro_rules! from {
    ($T:ty => $variant:ident) => {
        impl From<$T> for Number {
            fn from(v: $T) -> Self {
                Self::$variant(v.into())
            }
        }
    };
    ($($T:ty => $variant:ident),+ $(,)?) => {
        $(from!($T => $variant);)+
    }
}

from!(
    u64 => U64,
    u32 => U64,
    u16 => U64,
    u8 => U64,
    i64 => I64,
    i32 => I64,
    i16 => I64,
    i8 => I64,
    f32 => F64,
    f64 => F64,
);

impl From<usize> for Number {
    fn from(v: usize) -> Self {
        Self::U64(v.try_into().unwrap())
    }
}

macro_rules! try_from {
    ($T:ty => $f:ident) => {
        impl TryFrom<Number> for $T {
            type Error = ();
            fn try_from(number: Number) -> Result<$T, Self::Error> {
                number.$f().ok_or(())
            }
        }
    };
    ($($T:ty => $f:ident),+ $(,)?) => {
        $(try_from!($T => $f);)+
    }
}

try_from!(
    u64 => as_u64,
    u32 => as_u32,
    u16 => as_u16,
    u8 => as_u8,
    usize => as_usize,

    i64 => as_i64,
    i32 => as_i32,
    i16 => as_i16,
    i8 => as_i8,
    isize => as_isize,

    f32 => as_f32,
    f64 => as_f64,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_accessors_range_check() {
        assert_eq!(Number::U64(255).as_u8(), Some(255));
        assert_eq!(Number::U64(256).as_u8(), None);
        assert_eq!(Number::I64(-1).as_u8(), None);
        assert_eq!(Number::I64(-5).as_i8(), Some(-5));
        assert_eq!(Number::I64(-129).as_i8(), None);
    }

    #[test]
    fn float_to_integer_requires_integral_value() {
        assert_eq!(Number::F64(2.0).as_u32(), Some(2));
        assert_eq!(Number::F64(2.5).as_u32(), None);
        assert_eq!(Number::F64(-1.0).as_u32(), None);
    }

    // Regression for the NaN narrowing-accessor bug.
    #[test]
    fn nan_survives_float_accessors() {
        assert_eq!(
            Number::F64(f64::NAN).as_f64().map(f64::is_nan),
            Some(true),
            "as_f64 dropped a NaN"
        );
        assert_eq!(
            Number::F64(f64::NAN).as_f32().map(f32::is_nan),
            Some(true),
            "as_f32 dropped a NaN"
        );
        // Sanity: infinities already work, so this is specific to NaN.
        assert_eq!(Number::F64(f64::INFINITY).as_f64(), Some(f64::INFINITY));
    }

    #[test]
    fn nan_round_trips_through_try_from() {
        let back = f64::try_from(Number::F64(f64::NAN)).expect("NaN must convert back to f64");
        assert!(back.is_nan());
    }

    #[test]
    fn try_from_surfaces_out_of_range() {
        assert!(u8::try_from(Number::U64(300)).is_err());
        assert_eq!(u16::try_from(Number::U64(300)).unwrap(), 300);
        assert!(usize::try_from(Number::I64(-1)).is_err());
    }

    #[cfg(feature = "disk")]
    #[test]
    fn non_finite_floats_round_trip_via_sentinels() {
        for (value, sentinel) in [
            (f64::NAN, "\"nan\""),
            (f64::INFINITY, "\"inf\""),
            (f64::NEG_INFINITY, "\"neg_inf\""),
        ] {
            let json = serde_json::to_string(&Number::F64(value)).unwrap();
            assert_eq!(json, sentinel);

            let back: Number = serde_json::from_str(&json).unwrap();
            match back {
                Number::F64(v) if value.is_nan() => assert!(v.is_nan()),
                Number::F64(v) => assert_eq!(v, value),
                other => panic!("expected F64, got {other:?}"),
            }
        }
    }

    #[cfg(feature = "disk")]
    #[test]
    fn finite_floats_serialize_as_json_numbers() {
        assert_eq!(serde_json::to_string(&Number::F64(1.5)).unwrap(), "1.5");
        assert_eq!(serde_json::to_string(&Number::U64(7)).unwrap(), "7");
        assert_eq!(serde_json::to_string(&Number::I64(-7)).unwrap(), "-7");
    }
}

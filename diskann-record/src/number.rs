/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Debug, Clone, Copy)]
pub enum Number {
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Serialize for Number {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match *self {
            Self::U64(v) => serializer.serialize_u64(v),
            Self::I64(v) => serializer.serialize_i64(v),
            Self::F64(v) => serializer.serialize_f64(v),
        }
    }
}

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

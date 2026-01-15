/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Example types for `dispatcher` module-level documentation.

use crate::{
    dispatcher::{DispatchRule, FailureScore, Map, MatchScore},
    self_map,
};

/// An example type representing Rust primitive type.
#[derive(Debug, Clone, Copy)]
pub enum DataType {
    Float64,
    Float32,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
}

// Make `DataType` a dispatch type.
self_map!(DataType);

/// A type-domain lifting of Rust primitive types.
pub struct Type<T>(std::marker::PhantomData<T>);

/// Make `Type` reflexive to facilitate dispatch.
impl<T: 'static> Map for Type<T> {
    type Type<'a> = Self;
}

macro_rules! type_map {
    ($variant:ident, $T:ty) => {
        impl DispatchRule<DataType> for Type<$T> {
            type Error = std::convert::Infallible;

            fn try_match(from: &DataType) -> Result<MatchScore, FailureScore> {
                match from {
                    DataType::$variant => Ok(MatchScore(0)),
                    _ => Err(FailureScore(u32::MAX)),
                }
            }

            fn convert(from: DataType) -> Result<Self, Self::Error> {
                assert!(matches!(from, DataType::$variant));
                Ok(Self(std::marker::PhantomData))
            }

            fn description(
                f: &mut std::fmt::Formatter<'_>,
                from: Option<&DataType>,
            ) -> std::fmt::Result {
                match from {
                    None => write!(f, "{:?}", DataType::$variant),
                    Some(v) => {
                        if matches!(v, DataType::$variant) {
                            write!(f, "success")
                        } else {
                            write!(f, "expected {:?} but got {:?}", DataType::$variant, v)
                        }
                    }
                }
            }
        }
    };
}

type_map!(Float64, f64);
type_map!(Float32, f32);
type_map!(UInt8, u8);
type_map!(UInt16, u16);
type_map!(UInt32, u32);
type_map!(UInt64, u64);
type_map!(Int8, i8);
type_map!(Int16, i16);
type_map!(Int32, i32);
type_map!(Int64, i64);

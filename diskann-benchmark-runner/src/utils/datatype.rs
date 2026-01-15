/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use half::f16;
use serde::{Deserialize, Serialize};

use crate::dispatcher::{DispatchRule, FailureScore, Map, MatchScore};

/// An enum representation for common DiskANN data types.
///
/// [`DispatchRule]`s are defined for each type here and it's corresponding [`Type`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DataType {
    Float64,
    Float32,
    Float16,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Bool,
}

impl DataType {
    /// Return the string representation of the enum.
    ///
    /// This is more efficient than using `serde` directly.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Float64 => "float64",
            Self::Float32 => "float32",
            Self::Float16 => "float16",
            Self::UInt8 => "uint8",
            Self::UInt16 => "uint16",
            Self::UInt32 => "uint32",
            Self::UInt64 => "uint64",
            Self::Int8 => "int8",
            Self::Int16 => "int16",
            Self::Int32 => "int32",
            Self::Int64 => "int64",
            Self::Bool => "bool",
        }
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Lifting the enum `DataType` into the Rust type domain.
#[derive(Debug, Default, Clone, Copy)]
pub struct Type<T>(std::marker::PhantomData<T>);

/// The `Type` meta variable maps to itself.
impl<T: 'static> Map for Type<T> {
    type Type<'a> = Self;
}

pub const MATCH_FAIL: FailureScore = FailureScore(1000);

macro_rules! dispatch_rule {
    ($type:ty, $var:ident) => {
        impl DispatchRule<DataType> for Type<$type> {
            type Error = std::convert::Infallible;

            fn try_match(from: &DataType) -> Result<MatchScore, FailureScore> {
                match from {
                    DataType::$var => Ok(MatchScore(0)),
                    _ => Err(MATCH_FAIL),
                }
            }

            fn convert(from: DataType) -> Result<Self, Self::Error> {
                assert!(matches!(from, DataType::$var), "invalid dispatch");
                Ok(Self::default())
            }

            fn description(
                f: &mut std::fmt::Formatter<'_>,
                v: Option<&DataType>,
            ) -> std::fmt::Result {
                match v {
                    Some(v) => match Self::try_match(v) {
                        Ok(_) => write!(f, "successful match"),
                        Err(_) => write!(
                            f,
                            "expected \"{}\" but found {:?}",
                            stringify!($var).to_lowercase(),
                            v.as_str()
                        ),
                    },
                    None => write!(f, "{}", stringify!($var).to_lowercase()),
                }
            }
        }

        impl DispatchRule<&DataType> for Type<$type> {
            type Error = std::convert::Infallible;
            fn try_match(from: &&DataType) -> Result<MatchScore, FailureScore> {
                Self::try_match(*from)
            }
            fn convert(from: &DataType) -> Result<Self, Self::Error> {
                Self::convert(*from)
            }
            fn description(
                f: &mut std::fmt::Formatter<'_>,
                v: Option<&&DataType>,
            ) -> std::fmt::Result {
                Self::description(f, v.map(|v| *v))
            }
        }
    };
}

dispatch_rule!(f64, Float64);
dispatch_rule!(f32, Float32);
dispatch_rule!(f16, Float16);
dispatch_rule!(u8, UInt8);
dispatch_rule!(u16, UInt16);
dispatch_rule!(u32, UInt32);
dispatch_rule!(u64, UInt64);
dispatch_rule!(i8, Int8);
dispatch_rule!(i16, Int16);
dispatch_rule!(i32, Int32);
dispatch_rule!(i64, Int64);
dispatch_rule!(bool, Bool);

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use crate::dispatcher::{Description, Why};

    #[test]
    fn test_as_str() {
        let test = |x: DataType| {
            assert_eq!(format!("{}", x), x.as_str());
            assert_eq!(
                x.as_str(),
                serde_json::to_string(&x).unwrap().trim_matches('"')
            );
        };

        test(DataType::Float32);
        test(DataType::Float16);
        test(DataType::UInt8);
        test(DataType::UInt16);
        test(DataType::UInt32);
        test(DataType::UInt64);
        test(DataType::Int8);
        test(DataType::Int16);
        test(DataType::Int32);
        test(DataType::Int64);
        test(DataType::Bool);
    }

    fn test_description<T>(typename: &str)
    where
        Type<T>: DispatchRule<DataType>,
    {
        assert_eq!(
            Description::<DataType, Type<T>>::new().to_string(),
            typename
        );
    }

    fn test_dispatch_fail<T>(datatype: DataType, typename: &str)
    where
        Type<T>: DispatchRule<DataType>,
    {
        assert_eq!(<Type<T>>::try_match(&datatype), Err(MATCH_FAIL));
        assert_eq!(
            Why::<DataType, Type<T>>::new(&datatype).to_string(),
            format!("expected \"{}\" but found \"{}\"", typename, datatype)
        );
    }

    fn test_dispatch_success<T>(datatype: DataType)
    where
        Type<T>: DispatchRule<DataType>,
    {
        assert_eq!(<Type<T>>::try_match(&datatype), Ok(MatchScore(0)));
        assert_eq!(
            Why::<DataType, Type<T>>::new(&datatype).to_string(),
            "successful match",
        );
    }

    macro_rules! type_test {
        ($test:ident, $T:ty, $var:ident, $($fails:ident),* $(,)?) => {
            #[test]
            fn $test() {
                let typename = stringify!($var).to_lowercase();

                test_description::<$T>(&typename);
                test_dispatch_success::<$T>(DataType::$var);
                $(test_dispatch_fail::<$T>(DataType::$fails, &typename);)*
            }
        }
    }

    type_test!(test_f64, f64, Float64, Float16, UInt8);
    type_test!(test_f32, f32, Float32, Float16, UInt8);
    type_test!(test_f16, f16, Float16, UInt8, UInt16);
    type_test!(test_u8, u8, UInt8, UInt16, UInt32);
    type_test!(test_u16, u16, UInt16, UInt32, UInt64);
    type_test!(test_u32, u32, UInt32, UInt64, Int8);
    type_test!(test_u64, u64, UInt64, Int8, Int16);
    type_test!(test_i8, i8, Int8, Int16, Int32);
    type_test!(test_i16, i16, Int16, Int32, Int64);
    type_test!(test_i32, i32, Int32, Int64, Bool);
    type_test!(test_i64, i64, Int64, Bool, Float32);
    type_test!(test_bool, bool, Bool, Float32, Float16);
}

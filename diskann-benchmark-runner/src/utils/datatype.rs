/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use half::f16;
use serde::{Deserialize, Serialize};

/// An enum representation for common DiskANN data types.
///
/// See also: [`AsDataType`].
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

/// Associate a primitive type `T` with a [`DataType`] enum variant.
pub trait AsDataType: 'static {
    /// The [`DataType`] this type is associated with.
    const DATA_TYPE: DataType;

    /// Return `true` only if `data_type == Self::DATA_TYPE`.
    fn is_match(data_type: DataType) -> bool {
        data_type == Self::DATA_TYPE
    }

    /// Return a [`std::fmt::Display`] compatible struct describing the match with `data_type`.
    /// ```
    /// use diskann_benchmark_runner::utils::datatype::{DataType, AsDataType};
    ///
    /// // Matched data type.
    /// let desc = f32::describe(DataType::Float32);
    /// assert!(desc.is_match());
    /// assert_eq!(desc.to_string(), "successful match");
    ///
    /// // Mismatched data type.
    /// let desc = f32::describe(DataType::Float16);
    /// assert!(!desc.is_match());
    /// assert_eq!(desc.to_string(), "expected \"float32\" but found \"float16\"");
    /// ```
    fn describe(data_type: DataType) -> Describe {
        if data_type == Self::DATA_TYPE {
            Describe(DescribeInner::Match)
        } else {
            Describe(DescribeInner::Mismatch {
                expected: Self::DATA_TYPE,
                got: data_type,
            })
        }
    }
}

/// A [`std::fmt::Display`] compatible result for [`AsDataType::describe`].
#[derive(Debug, Clone, Copy)]
pub struct Describe(DescribeInner);

impl Describe {
    /// Return `true` is the data type match was successful.
    pub fn is_match(&self) -> bool {
        matches!(self.0, DescribeInner::Match)
    }
}

impl std::fmt::Display for Describe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Clone, Copy)]
enum DescribeInner {
    Match,
    Mismatch { expected: DataType, got: DataType },
}

impl std::fmt::Display for DescribeInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Match => write!(f, "successful match"),
            Self::Mismatch { expected, got } => {
                write!(f, "expected \"{}\" but found \"{}\"", expected, got)
            }
        }
    }
}

macro_rules! as_data_type {
    ($type:ty, $var:ident) => {
        impl AsDataType for $type {
            const DATA_TYPE: DataType = DataType::$var;
        }
    };
}

as_data_type!(f64, Float64);
as_data_type!(f32, Float32);
as_data_type!(f16, Float16);
as_data_type!(u8, UInt8);
as_data_type!(u16, UInt16);
as_data_type!(u32, UInt32);
as_data_type!(u64, UInt64);
as_data_type!(i8, Int8);
as_data_type!(i16, Int16);
as_data_type!(i32, Int32);
as_data_type!(i64, Int64);
as_data_type!(bool, Bool);

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

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
        T: AsDataType,
    {
        assert_eq!(T::DATA_TYPE.as_str(), typename);
    }

    fn test_dispatch_fail<T>(datatype: DataType, typename: &str)
    where
        T: AsDataType,
    {
        assert!(!T::is_match(datatype));
        assert_eq!(
            T::describe(datatype).to_string(),
            format!("expected \"{}\" but found \"{}\"", typename, datatype)
        );
    }

    fn test_dispatch_success<T>(datatype: DataType)
    where
        T: AsDataType,
    {
        assert!(T::is_match(datatype));
        assert_eq!(T::describe(datatype).to_string(), "successful match",);
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

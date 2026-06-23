/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use half::f16;
use thiserror::Error;

//////////////
// DataType //
//////////////

#[derive(Debug, Clone, Copy)]
pub(crate) enum DataType {
    F32,
    F16,
    U8,
    I8,
}

impl DataType {
    fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::U8 => "u8",
            Self::I8 => "i8",
        }
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

pub(crate) trait AsDataType {
    const DATA_TYPE: DataType;
}

macro_rules! as_data_type {
    ($T:ty, $variant:ident) => {
        impl AsDataType for $T {
            const DATA_TYPE: DataType = DataType::$variant;
        }
    };
}

as_data_type!(f32, F32);
as_data_type!(f16, F16);
as_data_type!(u8, U8);
as_data_type!(i8, I8);

#[derive(Debug, Error)]
#[error("wrong data-type: expected {}, got {}", self.expected, self.got)]
pub(crate) struct WrongDataType {
    expected: DataType,
    got: DataType,
}

impl WrongDataType {
    fn new(expected: DataType, got: DataType) -> Self {
        Self { expected, got }
    }
}

///////////
// Slice //
///////////

#[derive(Debug, Clone, Copy)]
pub(crate) enum Slice<'a> {
    F32(&'a [f32]),
    F16(&'a [f16]),
    U8(&'a [u8]),
    I8(&'a [i8]),
}

impl<'a> Slice<'a> {
    pub(crate) fn data_type(&self) -> DataType {
        match self {
            Self::F32(_) => DataType::F32,
            Self::F16(_) => DataType::F16,
            Self::U8(_) => DataType::U8,
            Self::I8(_) => DataType::I8,
        }
    }

    pub(crate) fn try_cast<T>(self) -> Result<&'a [T], WrongDataType>
    where
        T: FromSlice,
    {
        T::from_slice(self)
    }
}

pub(crate) trait FromSlice: Sized {
    fn from_slice(slice: Slice<'_>) -> Result<&[Self], WrongDataType>;
}

macro_rules! from_slice {
    ($T:ty, $variant:ident) => {
        impl FromSlice for $T {
            fn from_slice(slice: Slice<'_>) -> Result<&[Self], WrongDataType> {
                if let Slice::$variant(s) = slice {
                    Ok(s)
                } else {
                    Err(WrongDataType::new(DataType::$variant, slice.data_type()))
                }
            }
        }
    };
}

from_slice!(f32, F32);
from_slice!(f16, F16);
from_slice!(u8, U8);
from_slice!(i8, I8);

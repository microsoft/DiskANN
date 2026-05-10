/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::str::FromStr;

use clap::ValueEnum;
use pyo3::prelude::*;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
#[pyclass]
pub enum DataType {
    /// 32 bit float.
    Float,

    /// Unsigned 8-bit integer.
    Uint8,

    /// Signed 8-bit integer.
    Int8,
}

#[derive(thiserror::Error, Debug)]
pub enum ParseDataTypeError {
    #[error("Invalid format for DataType: {0}")]
    InvalidFormat(String),
}

impl FromStr for DataType {
    type Err = ParseDataTypeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "float" => Ok(DataType::Float),
            "uint8" => Ok(DataType::Uint8),
            "int8" => Ok(DataType::Int8),
            _ => Err(ParseDataTypeError::InvalidFormat(String::from(s))),
        }
    }
}

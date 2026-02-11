/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug, Deserialize, Serialize)]
pub enum DataType {
    /// 32 bit float.
    Float,

    /// Unsigned 8-bit integer.
    Uint8,

    /// Signed 8-bit integer.
    Int8,

    /// Half precision float.
    Fp16,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum AssociatedDataType {
    /// 32 bit unsigned integer.
    U32,
}

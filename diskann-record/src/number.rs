/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Number {
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
}

impl From<u64> for Number {
    fn from(v: u64) -> Self {
        Self::U64(v)
    }
}

impl From<usize> for Number {
    fn from(v: usize) -> Self {
        Self::U64(v.try_into().unwrap())
    }
}

impl From<u32> for Number {
    fn from(v: u32) -> Self {
        Self::U64(v.into())
    }
}

impl From<u16> for Number {
    fn from(v: u16) -> Self {
        Self::U64(v.into())
    }
}

impl From<u8> for Number {
    fn from(v: u8) -> Self {
        Self::U64(v.into())
    }
}

impl From<i64> for Number {
    fn from(v: i64) -> Self {
        Self::I64(v)
    }
}

impl From<i32> for Number {
    fn from(v: i32) -> Self {
        Self::I64(v.into())
    }
}

impl From<i16> for Number {
    fn from(v: i16) -> Self {
        Self::I64(v.into())
    }
}

impl From<i8> for Number {
    fn from(v: i8) -> Self {
        Self::I64(v.into())
    }
}

impl From<f32> for Number {
    fn from(v: f32) -> Self {
        Self::F32(v.into())
    }
}

impl From<f64> for Number {
    fn from(v: f64) -> Self {
        Self::F64(v.into())
    }
}

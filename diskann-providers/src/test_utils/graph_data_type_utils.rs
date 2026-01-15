/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_vector::Half;

use crate::{common::MinMax8, model::graph::traits::GraphDataType};

/// Graph data with f32 vector and associated data of unit type (empty).
pub struct GraphDataF32VectorUnitData {}

impl GraphDataType for GraphDataF32VectorUnitData {
    type VectorIdType = u32;
    type VectorDataType = f32;
    type AssociatedDataType = ();
}

/// Graph data with f32 vector and associated data of unit type (empty).
pub struct GraphDataMinMaxVectorUnitData {}

impl GraphDataType for GraphDataMinMaxVectorUnitData {
    type VectorIdType = u32;
    type VectorDataType = MinMax8;
    type AssociatedDataType = ();
}

pub struct GraphDataF32VectorU32Data {}

impl GraphDataType for GraphDataF32VectorU32Data {
    type VectorIdType = u32;
    type VectorDataType = f32;
    type AssociatedDataType = u32;
}

/// Graph data with f32 vector and associated data of u8 array.
pub struct GraphDataHalfByteArrayData {}

impl GraphDataType for GraphDataHalfByteArrayData {
    type VectorIdType = u32;
    type VectorDataType = Half;
    type AssociatedDataType = [u8; 22];
}

/// Graph data with I8 vector and associated data of unit type (empty.)
pub struct GraphDataI8VectorUnitData {}

impl GraphDataType for GraphDataI8VectorUnitData {
    type VectorIdType = u32;
    type VectorDataType = i8;
    type AssociatedDataType = ();
}

/// Graph data with U8 vector and associated data of unit type (empty.)
pub struct GraphDataU8VectorUnitData {}

impl GraphDataType for GraphDataU8VectorUnitData {
    type VectorIdType = u32;
    type VectorDataType = u8;
    type AssociatedDataType = ();
}

pub struct GraphDataU8VectorU32AssociatedData {}

impl GraphDataType for GraphDataU8VectorU32AssociatedData {
    type VectorIdType = u32;
    type VectorDataType = u8;
    type AssociatedDataType = u32;
}

pub struct GraphDataF32WithU64IdVectorUnitData {}

impl GraphDataType for GraphDataF32WithU64IdVectorUnitData {
    type VectorIdType = u64;
    type VectorDataType = f32;
    type AssociatedDataType = ();
}

pub struct GraphDataF32WithU64IdVectorU32Data {}

impl GraphDataType for GraphDataF32WithU64IdVectorU32Data {
    type VectorIdType = u64;
    type VectorDataType = f32;
    type AssociatedDataType = u32;
}

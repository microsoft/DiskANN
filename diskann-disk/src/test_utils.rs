/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Test-only concrete implementations of [`GraphDataType`].

use diskann_providers::common::MinMax8;

use crate::data_model::GraphDataType;

/// Graph data with f32 vector and associated data of unit type (empty).
pub struct GraphDataF32VectorUnitData {}

impl GraphDataType for GraphDataF32VectorUnitData {
    type VectorIdType = u32;
    type VectorDataType = f32;
    type AssociatedDataType = ();
}

/// Graph data with MinMax8 vector and associated data of unit type (empty).
pub struct GraphDataMinMaxVectorUnitData {}

impl GraphDataType for GraphDataMinMaxVectorUnitData {
    type VectorIdType = u32;
    type VectorDataType = MinMax8;
    type AssociatedDataType = ();
}

/// Graph data with f32 vector and associated data of u32.
pub struct GraphDataF32VectorU32Data {}

impl GraphDataType for GraphDataF32VectorU32Data {
    type VectorIdType = u32;
    type VectorDataType = f32;
    type AssociatedDataType = u32;
}

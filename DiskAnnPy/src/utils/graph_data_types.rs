/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_providers::model::graph::traits::GraphDataType;

pub struct GraphDataF32Vector {}

impl GraphDataType for GraphDataF32Vector {
    type VectorIdType = u32;
    type VectorDataType = f32;
    type AssociatedDataType = ();
}

pub struct GraphDataF32VectorU32Data {}

impl GraphDataType for GraphDataF32VectorU32Data {
    type VectorIdType = u32;
    type VectorDataType = f32;
    type AssociatedDataType = u32;
}

pub struct GraphDataInt8Vector {}

impl GraphDataType for GraphDataInt8Vector {
    type VectorIdType = u32;
    type VectorDataType = i8;
    type AssociatedDataType = ();
}

pub struct GraphDataInt8VectorU32Data {}

impl GraphDataType for GraphDataInt8VectorU32Data {
    type VectorIdType = u32;
    type VectorDataType = i8;
    type AssociatedDataType = u32;
}

pub struct GraphDataU8Vector {}

impl GraphDataType for GraphDataU8Vector {
    type VectorIdType = u32;
    type VectorDataType = u8;
    type AssociatedDataType = ();
}

pub struct GraphDataU8VectorU32Data {}

impl GraphDataType for GraphDataU8VectorU32Data {
    type VectorIdType = u32;
    type VectorDataType = u8;
    type AssociatedDataType = u32;
}

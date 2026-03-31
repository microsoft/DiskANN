/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_providers::model::graph::traits::GraphDataType;
use diskann_vector::Half;

pub struct GraphDataF32Vector {}

impl GraphDataType for GraphDataF32Vector {
    type VectorIdType = u32;
    type VectorDataType = f32;
    type AssociatedDataType = ();
}

pub struct GraphDataHalfVector {}

impl GraphDataType for GraphDataHalfVector {
    type VectorIdType = u32;
    type VectorDataType = Half;
    type AssociatedDataType = ();
}

pub struct GraphDataInt8Vector {}

impl GraphDataType for GraphDataInt8Vector {
    type VectorIdType = u32;
    type VectorDataType = i8;
    type AssociatedDataType = ();
}

pub struct GraphDataU8Vector {}

impl GraphDataType for GraphDataU8Vector {
    type VectorIdType = u32;
    type VectorDataType = u8;
    type AssociatedDataType = ();
}

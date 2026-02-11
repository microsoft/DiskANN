/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_providers::{common::MinMax8, model::graph::traits::GraphDataType};
use diskann_vector::Half;

pub struct GraphDataF32Vector {}

impl GraphDataType for GraphDataF32Vector {
    type VectorIdType = u32;
    type VectorDataType = f32;
    type AssociatedDataType = ();
}

pub struct GraphDataMinMaxVector {}

impl GraphDataType for GraphDataMinMaxVector {
    type VectorIdType = u32;
    type VectorDataType = MinMax8;
    type AssociatedDataType = ();
}

pub struct GraphDataF32VectorU32Assoc {}

impl GraphDataType for GraphDataF32VectorU32Assoc {
    type VectorIdType = u32;
    type VectorDataType = f32;
    type AssociatedDataType = u32;
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

pub struct GraphDataFloatVectorU32Data {}

impl GraphDataType for GraphDataFloatVectorU32Data {
    type VectorIdType = u32;
    type VectorDataType = f32;
    type AssociatedDataType = u32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn test_graph_data_f32_vector_types() {
        assert_eq!(size_of::<GraphDataF32Vector>(), 0);
        // Verify type associations
        let _id: <GraphDataF32Vector as GraphDataType>::VectorIdType = 0u32;
        let _data: <GraphDataF32Vector as GraphDataType>::VectorDataType = 0.0f32;
        let _assoc: <GraphDataF32Vector as GraphDataType>::AssociatedDataType = ();
    }

    #[test]
    fn test_graph_data_min_max_vector_types() {
        assert_eq!(size_of::<GraphDataMinMaxVector>(), 0);
        let _id: <GraphDataMinMaxVector as GraphDataType>::VectorIdType = 0u32;
    }

    #[test]
    fn test_graph_data_f32_vector_u32_assoc_types() {
        assert_eq!(size_of::<GraphDataF32VectorU32Assoc>(), 0);
        let _id: <GraphDataF32VectorU32Assoc as GraphDataType>::VectorIdType = 0u32;
        let _data: <GraphDataF32VectorU32Assoc as GraphDataType>::VectorDataType = 0.0f32;
        let _assoc: <GraphDataF32VectorU32Assoc as GraphDataType>::AssociatedDataType = 0u32;
    }

    #[test]
    fn test_graph_data_half_vector_types() {
        assert_eq!(size_of::<GraphDataHalfVector>(), 0);
        let _id: <GraphDataHalfVector as GraphDataType>::VectorIdType = 0u32;
    }

    #[test]
    fn test_graph_data_int8_vector_types() {
        assert_eq!(size_of::<GraphDataInt8Vector>(), 0);
        let _id: <GraphDataInt8Vector as GraphDataType>::VectorIdType = 0u32;
        let _data: <GraphDataInt8Vector as GraphDataType>::VectorDataType = 0i8;
        let _assoc: <GraphDataInt8Vector as GraphDataType>::AssociatedDataType = ();
    }

    #[test]
    fn test_graph_data_u8_vector_types() {
        assert_eq!(size_of::<GraphDataU8Vector>(), 0);
        let _id: <GraphDataU8Vector as GraphDataType>::VectorIdType = 0u32;
        let _data: <GraphDataU8Vector as GraphDataType>::VectorDataType = 0u8;
        let _assoc: <GraphDataU8Vector as GraphDataType>::AssociatedDataType = ();
    }

    #[test]
    fn test_graph_data_float_vector_u32_data_types() {
        assert_eq!(size_of::<GraphDataFloatVectorU32Data>(), 0);
        let _id: <GraphDataFloatVectorU32Data as GraphDataType>::VectorIdType = 0u32;
        let _data: <GraphDataFloatVectorU32Data as GraphDataType>::VectorDataType = 0.0f32;
        let _assoc: <GraphDataFloatVectorU32Data as GraphDataType>::AssociatedDataType = 0u32;
    }
}

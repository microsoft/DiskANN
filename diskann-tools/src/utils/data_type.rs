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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_variants() {
        let _float = DataType::Float;
        let _uint8 = DataType::Uint8;
        let _int8 = DataType::Int8;
        let _fp16 = DataType::Fp16;
    }

    #[test]
    fn test_data_type_clone() {
        let dt = DataType::Float;
        let cloned = dt.clone();
        assert_eq!(dt, cloned);
    }

    #[test]
    fn test_data_type_partial_eq() {
        assert_eq!(DataType::Float, DataType::Float);
        assert_ne!(DataType::Float, DataType::Uint8);
    }

    #[test]
    fn test_data_type_partial_ord() {
        assert!(DataType::Float < DataType::Uint8);
        assert!(DataType::Uint8 < DataType::Int8);
        assert!(DataType::Int8 < DataType::Fp16);
    }

    #[test]
    fn test_data_type_debug() {
        assert_eq!(format!("{:?}", DataType::Float), "Float");
        assert_eq!(format!("{:?}", DataType::Uint8), "Uint8");
        assert_eq!(format!("{:?}", DataType::Int8), "Int8");
        assert_eq!(format!("{:?}", DataType::Fp16), "Fp16");
    }

    #[test]
    fn test_data_type_serialize_deserialize() {
        let dt = DataType::Float;
        let serialized = bincode::serialize(&dt).unwrap();
        let deserialized: DataType = bincode::deserialize(&serialized).unwrap();
        assert_eq!(dt, deserialized);

        let dt2 = DataType::Uint8;
        let serialized2 = bincode::serialize(&dt2).unwrap();
        let deserialized2: DataType = bincode::deserialize(&serialized2).unwrap();
        assert_eq!(dt2, deserialized2);
    }

    #[test]
    fn test_associated_data_type_variants() {
        let _u32 = AssociatedDataType::U32;
    }

    #[test]
    fn test_associated_data_type_clone() {
        let adt = AssociatedDataType::U32;
        let cloned = adt.clone();
        assert_eq!(adt, cloned);
    }

    #[test]
    fn test_associated_data_type_debug() {
        assert_eq!(format!("{:?}", AssociatedDataType::U32), "U32");
    }
}

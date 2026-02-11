/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! DiskANN Disk Index Crate
//!
//! This crate provides disk-based indexing capabilities for DiskANN,
//! including builders, providers, and utilities specific to disk storage.

pub mod build;
pub use build::{
    disk_index_build_parameter, filter_parameter, DiskIndexBuildParameters, QuantizationType,
};

pub mod data_model;
pub mod search;
pub mod storage;
pub mod utils;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify that key types are accessible from the root
        let _ = core::any::type_name::<DiskIndexBuildParameters>();
        let _ = core::any::type_name::<QuantizationType>();
    }
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Build-related modules for disk index construction and configuration.
//!
//! This module contains all the components needed for building disk indexes,
//! including builders and configuration parameters.

pub mod builder;
pub mod configuration;

// Re-export key types for convenience
pub use configuration::{
    disk_index_build_parameter, filter_parameter, DiskIndexBuildParameters, QuantizationType,
};

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # BigANN Runbook Support
//!
//! Runbook parsing and execution are provided by `diskann-utils`.
//! This module provides benchmark-specific adaptors over those shared types.
//!
//! Users can also leverage the [`WithData`] adaptor to convert raw index ranges
//! into actual data points for a dataset.

mod withdata;

pub use diskann_utils::streaming::executors::bigann::{
    Args, Delete, FindGroundtruth, Insert, Replace, RunBook, ScanDirectory, Search, Stage,
};
pub use withdata::{DataArgs, WithData};

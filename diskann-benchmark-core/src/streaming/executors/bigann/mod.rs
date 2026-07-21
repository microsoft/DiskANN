/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # BigANN Runbook Support
//!
//! This module provides an executor for processing BigANN-style runbooks.
//! Please refer to the [`RunBook`] documentation for more details.
//!
//! Users can also leverage the [`WithData`] adaptor to convert raw index ranges
//! into actual data points for a dataset.

mod parsing;
mod runbook;
mod validate;
mod withdata;

pub use runbook::{
    Args, Delete, FindGroundtruth, Insert, Replace, RunBook, ScanDirectory, Search, Stage,
};
pub use withdata::{DataArgs, WithData};

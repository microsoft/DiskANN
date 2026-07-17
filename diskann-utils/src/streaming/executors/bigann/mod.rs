/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # BigANN Runbook Support
//!
//! This module provides an executor for processing BigANN-style runbooks.
//! Please refer to the [`RunBook`] documentation for more details.

mod parsing;
mod runbook;
mod validate;

pub use runbook::{
    Args, Delete, FindGroundtruth, Insert, Replace, RunBook, ScanDirectory, Search, Stage,
};

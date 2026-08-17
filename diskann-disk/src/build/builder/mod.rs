/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Disk index builders and related functionality.
pub mod build;

mod inmem_builder;
mod merged_index;
mod quantizer;
mod tokio;

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use tests::disk_index_builder_tests::{IndexBuildFixture, TestParams};

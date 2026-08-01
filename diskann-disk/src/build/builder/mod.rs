/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Disk index builders and related functionality.
pub mod build;
pub mod quantizer;

pub mod tokio;
mod vamana;

#[cfg(test)]
pub(crate) use vamana::tests::disk_index_builder_tests;

#[cfg(test)]
mod tests;

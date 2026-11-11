/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Disk index builders and related functionality.
pub mod build;
pub mod core;
pub mod quantizer;

pub mod inmem_builder;
pub mod tokio;

#[cfg(feature = "pipnn")]
mod pipnn;

#[cfg(test)]
mod tests;

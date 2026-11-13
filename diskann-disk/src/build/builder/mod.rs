/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Disk index builders and related functionality.
pub mod backend;
pub mod build;
pub mod core;

mod inmem_builder;
pub mod tokio;

#[cfg(feature = "pipnn")]
mod pipnn;

#[cfg(test)]
mod tests;

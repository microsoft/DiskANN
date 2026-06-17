/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Disk index builders and related functionality.
pub mod build;
pub mod core;
pub mod quantizer;

pub mod inmem_builder;

#[cfg(test)]
mod tests;

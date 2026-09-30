/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Incrementally maintained IVF index.

pub mod colocation;
pub mod dynamic;
mod grouped;
pub mod index;
mod online;
pub mod update;

#[cfg(test)]
mod test;

pub use index::{ConfigError, DynamicIvfConfig, DynamicIvfIndex, InsertStats};

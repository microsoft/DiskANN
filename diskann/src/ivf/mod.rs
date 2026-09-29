/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Incrementally maintained IVF index.

pub mod dynamic;
pub mod index;

pub use index::{ConfigError, DynamicIvfConfig, DynamicIvfIndex, InsertStats};

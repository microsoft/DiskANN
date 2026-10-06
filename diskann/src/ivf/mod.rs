/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Incrementally maintained IVF index.

pub mod dynamic;
mod index;
mod online;
pub mod update;

pub use index::{Config, ConfigError, IVFIndex, InsertStats};

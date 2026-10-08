/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Incrementally maintained IVF index.

pub mod traits;
mod index;
mod online;
mod split;
pub mod update;

pub use index::{Config, ConfigError, IVFIndex, InsertStats};

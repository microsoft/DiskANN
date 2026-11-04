/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub mod index;
pub mod region;
pub mod traits;

// Splitting fits lists with k-means from `diskann-quantization`.
#[cfg(feature = "ivf")]
pub mod split;

pub use index::{Config, ConfigError, IVFIndex, InsertStats, ReassignCount};

#[cfg(test)]
#[cfg(feature = "ivf")]
mod tests;

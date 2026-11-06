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

#[cfg(any(test, feature = "testing"))]
pub mod test;

pub use index::{Config, ConfigError, IVFIndex, InsertStats, ReassignCount};

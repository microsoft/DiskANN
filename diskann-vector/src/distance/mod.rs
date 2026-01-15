/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#[cfg(test)]
pub mod reference;

pub mod simd;

pub mod implementations;
pub use implementations::{Cosine, CosineNormalized, FullL2, InnerProduct, SquaredL2};

pub mod distance_provider;
pub use distance_provider::{Distance, DistanceProvider};

mod metric;
pub use metric::Metric;

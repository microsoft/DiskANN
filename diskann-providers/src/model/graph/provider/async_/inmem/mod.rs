/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Native (DiskANN built-in) data providers and strategies for async index build and search.

mod provider;
pub use provider::{DefaultProvider, DefaultProviderParameters, SetStartPoints};

// Extensions
mod scalar;
pub use scalar::{SQError, SQStore, WithBits};

#[cfg(not(test))]
mod product;
pub use product::DefaultQuant;

pub mod spherical;

mod full_precision;
pub(super) use full_precision::Rerank;
pub use full_precision::{
    CreateFullPrecision, FullAccessor, FullPrecisionProvider, FullPrecisionStore, GetFullPrecision,
};

#[cfg(test)]
pub mod product;

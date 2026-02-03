/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Native (DiskANN built-in) data providers and strategies for async index build and search.

mod provider;

pub use provider::{DefaultProvider, DefaultProviderParameters, SetStartPoints};

// Extensions
mod scalar;
pub use scalar::{SQStore, WithBits};
pub use diskann_providers::storage::SQError;

#[cfg(not(test))]
mod product;
pub use product::DefaultQuant;

pub mod spherical;

mod full_precision;
pub use full_precision::{
    CreateFullPrecision, FullAccessor, FullPrecisionProvider, FullPrecisionStore,
};
pub use full_precision::{GetFullPrecision, Rerank};

#[cfg(test)]
pub mod product;
#[cfg(test)]
pub(crate) mod test;

// Helper functions for creating inmem indexes
pub mod diskann_async;

// Storage implementations for inmem providers
pub mod storage;

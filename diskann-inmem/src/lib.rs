/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Native (DiskANN built-in) in-memory data providers and strategies for async index build and search.

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
pub use full_precision::{
    CreateFullPrecision, FullAccessor, FullPrecisionProvider, FullPrecisionStore,
};
pub(crate) use full_precision::{GetFullPrecision, Rerank};

#[cfg(test)]
pub mod product;
#[cfg(test)]
pub(crate) mod test;

pub mod diskann_async;

// Re-export `CreateVectorStore` from `diskann-providers` where it is defined.
// The trait is re-exported here for convenience since it is primarily used by the
// in-mem providers.
pub use diskann_providers::model::graph::provider::async_::common::CreateVectorStore;

// Implement `HasStartingPoints` for `DefaultProvider` so the generalized `SaveWith`
// impls in `diskann-providers` work with our provider.
impl<U, V, D, Ctx> diskann_providers::storage::index_storage::HasStartingPoints
    for DefaultProvider<U, V, D, Ctx>
{
    fn starting_points(&self) -> diskann::ANNResult<Vec<u32>> {
        self.starting_points()
    }
}

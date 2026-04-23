/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Native (DiskANN built-in) data providers and strategies for async index build and search.

use diskann::graph::workingset;

mod provider;
pub use provider::{DefaultProvider, DefaultProviderParameters, SetStartPoints};

/// The in-mem providers pass through prune elements straight back to their underlying
/// providers. This is the working-set precursor and is a ZST because ... it doesn't need to
/// do anything!
#[derive(Debug, Clone, Copy)]
pub struct PassThrough;

impl workingset::AsWorkingSet<Self> for PassThrough {
    fn as_working_set(&self, _capacity: usize) -> Self {
        *self
    }
}

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
pub(super) use full_precision::{GetFullPrecision, Rerank};

pub mod multi;

#[cfg(test)]
pub mod product;
#[cfg(test)]
pub(crate) mod test;

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The inmem index for DiskANN.

#![deny(rustdoc::broken_intra_doc_links)]

pub mod num;

mod arch;
mod buffer;
mod counters;
mod epoch;
mod freelist;
mod ids;
mod neighbors;
mod tag;

mod store;

pub mod layers;
pub mod provider;

pub use provider::{Context, Provider, Strategy};

#[cfg(test)]
mod test;

#[cfg(feature = "integration-test")]
#[doc(hidden)]
pub mod integration;

//----------------//
// Internal Tools //
//----------------//

/// A "public" type that can only be constructed by this crate.
///
/// This helps with public traits with internal methods that we don't want users to call.
#[doc(hidden)]
#[derive(Debug)]
pub struct Hidden(());

impl Hidden {
    const fn new() -> Self {
        Self(())
    }
}

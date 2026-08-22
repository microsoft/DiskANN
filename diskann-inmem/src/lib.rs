/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The inmem index for DiskANN.

#![deny(rustdoc::broken_intra_doc_links)]

pub mod num;

pub mod arch;
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

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#![deny(rustdoc::broken_intra_doc_links)]

pub mod num;

mod buffer;
mod counters;
mod epoch;
mod freelist;
mod neighbors;
mod sharded;
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

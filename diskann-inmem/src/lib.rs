/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#![cfg_attr(docsrs, feature(doc_cfg))]

//! The inmem index for DiskANN.

pub mod num;

mod buffer;
mod counters;
mod epoch;
mod freelist;
mod ids;
mod neighbors;
mod prefetch;
mod tag;

pub mod provider;
pub mod repr;
pub mod store;

pub use provider::{Context, Provider, Strategy};

#[cfg(test)]
mod test;

#[cfg(feature = "integration-test")]
#[doc(hidden)]
pub mod integration;

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#![deny(rustdoc::broken_intra_doc_links)]

mod buffer;
mod neighbors;
mod sync;

pub mod num;

pub mod ids;
pub mod layers;
mod store;

pub mod provider;

pub use provider::{Context, Provider, Strategy};

#[cfg(test)]
mod test;

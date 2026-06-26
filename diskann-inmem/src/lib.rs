/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The inmem index for DiskANN.

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

macro_rules! opaque {
    ($T:ty) => {
        impl From<$T> for diskann::ANNError {
            #[track_caller]
            #[cold]
            fn from(err: $T) -> diskann::ANNError {
                diskann::ANNError::opaque(err)
            }
        }
    };
}

pub(crate) use opaque;

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Built-in [`crate::build::Build`] implementations for the [`diskann::graph::DiskANNIndex`].

mod multi;
mod single;

pub use multi::MultiInsert;
pub use single::SingleInsert;

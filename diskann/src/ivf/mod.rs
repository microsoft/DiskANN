/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! IVF index traits and wrappers.
//!
//! Search selects candidate lists, scans them, and returns the best `k` results.
//! The fixed-partition API inserts by appending to one selected list, while
//! [`dynamic`] defines accessor and mutation contracts for incremental partitions.

pub mod dynamic;
pub mod glue;
pub mod index;

pub use glue::{InsertAccessor, InsertStrategy, ListAccessor, SearchAccessor, SearchStrategy};
pub use index::{IvfIndex, SearchStats};

#[cfg(test)]
mod test;

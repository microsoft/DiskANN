/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! IVF index traits and wrapper.
//!
//! Search selects candidate lists, scans them, and returns the best `k` results.
//! Insert selects one list and appends the new point to it.

pub mod glue;
pub mod index;

pub use glue::{InsertAccessor, InsertStrategy, ListAccessor, SearchAccessor, SearchStrategy};
pub use index::{IvfIndex, SearchStats};

#[cfg(test)]
mod test;

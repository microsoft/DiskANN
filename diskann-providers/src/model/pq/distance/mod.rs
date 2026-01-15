/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub mod common;
pub mod cosine;
pub mod dynamic;
pub mod innerproduct;
pub mod l2;

pub mod multi;

// Exports
pub use dynamic::{DistanceComputer, QueryComputer};

#[cfg(test)]
pub(crate) mod test_utils;

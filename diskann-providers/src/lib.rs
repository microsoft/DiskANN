/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A grab bag of miscellaneous indexing-related functionality, including in-memory indexing.
//!
//! This crate is slowly being deprecated, with its functionality redistributed to other crates
//! as appropriate.

#![cfg_attr(
    not(test),
    warn(clippy::panic, clippy::unwrap_used, clippy::expect_used)
)]
#![cfg_attr(test, allow(clippy::unused_io_amount))]
// With the `tokio` backend compiled out, tokio-only test functions and their
// imports/helpers vanish; silence that test-only fallout instead of sprinkling
// `cfg` gates through test code.
#![cfg_attr(all(test, not(feature = "tokio")), allow(unused_imports, dead_code))]

pub mod utils;

// Backend facade for the synchronous index wrappers. One of the runtime
// backends (`tokio`, `compio`) is required; `diskann` emits the actionable
// `compile_error!` when neither is selected.
#[cfg(any(feature = "tokio", feature = "compio"))]
pub(crate) mod runtime;

pub mod model;

pub mod common;

pub mod index;

pub mod storage;

#[cfg(any(test, feature = "testing"))]
pub mod test_utils;

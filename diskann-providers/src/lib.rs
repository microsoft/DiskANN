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

pub mod utils;

pub mod model;

pub mod common;

pub mod index;

pub mod storage;

#[cfg(any(test, feature = "testing"))]
pub mod test_utils;

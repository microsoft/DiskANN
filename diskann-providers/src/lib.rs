/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
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

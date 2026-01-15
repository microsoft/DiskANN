/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub(crate) mod tokio;

mod cache;
pub(crate) use cache::{TestPath, TestRoot, get_or_save_test_results};

pub(crate) mod cmp;

/// A helper macro for testing error messages that will print the full error message for
/// better debugging.
macro_rules! assert_message_contains {
    ($msg:expr, $contains:literal) => {
        assert!($msg.contains($contains), "failed with:\n\n{}", $msg);
    };
}

pub(crate) use assert_message_contains;

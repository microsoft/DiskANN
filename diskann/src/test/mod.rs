/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub(crate) mod tokio;

mod cache;
pub(crate) use cache::{TestPath, TestRoot, get_or_save_test_results};

pub(crate) mod cmp;

#[test]
fn version_works() {
    let version = super::version();
    assert!(!version.is_empty(), "version should not be empty");
}

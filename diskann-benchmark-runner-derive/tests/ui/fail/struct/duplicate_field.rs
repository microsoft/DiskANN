/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Reflect;

// Case 1 - Direct Rename
#[derive(Reflect)]
struct WithDuplicateName {
    value: usize,
    #[serde(rename = "value")]
    foo: usize,
}

// Case 2- Direct rename + rename-all
#[derive(Reflect)]
#[serde(rename_all = "kebab-case")]
struct WithDuplicateNameAndRename {
    #[serde(rename = "foo-bar")]
    baz: usize,
    foo_bar: usize,
}

fn main() {}

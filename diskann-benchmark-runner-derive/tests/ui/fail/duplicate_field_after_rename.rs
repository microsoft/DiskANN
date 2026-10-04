/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Check that we catch an interaction between `rename_all` and `rename`.

use diskann_benchmark_runner::Reflect;

#[derive(Reflect)]
#[serde(rename_all = "kebab-case")]
struct WithDuplicateNameAndRename {
    #[serde(rename = "foo-bar")]
    baz: usize,
    foo_bar: usize,
}

fn main() {}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Reflect;

// Case 1 - Direct Rename
#[derive(Reflect)]
enum WithDuplicateName {
    Foo {
        value: usize,
        #[serde(rename = "value")]
        foo: usize,
    }
}

// Case 2- Direct rename + rename-all
#[derive(Reflect)]
enum WithDuplicateNameAndRename {
    #[serde(rename_all = "kebab-case")]
    Foo {
        #[serde(rename = "foo-bar")]
        baz: usize,
        foo_bar: usize,
    }
}

fn main() {}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Reflect;

#[derive(Reflect)]
#[serde(rename_all = "lowercase")]
enum DuplicateVariant {
    Foo,
    #[serde(rename = "foo")]
    Other,
}

#[derive(Reflect)]
struct DuplicateStructField {
    value: usize,
    #[serde(rename = "value")]
    other: usize,
}

#[derive(Reflect)]
#[serde(rename_all = "kebab-case")]
struct DuplicateStructFieldAfterRenameAll {
    #[serde(rename = "foo-bar")]
    other: usize,
    foo_bar: usize,
}

#[derive(Reflect)]
enum DuplicateVariantField {
    Value {
        value: usize,
        #[serde(rename = "value")]
        other: usize,
    },
}

#[derive(Reflect)]
enum DuplicateVariantFieldAfterRenameAll {
    #[serde(rename_all = "kebab-case")]
    Value {
        #[serde(rename = "foo-bar")]
        other: usize,
        foo_bar: usize,
    },
}

fn main() {}

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
    Baz,
}

fn main() {}

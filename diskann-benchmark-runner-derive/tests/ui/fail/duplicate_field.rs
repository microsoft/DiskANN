/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Reflect;

#[derive(Reflect)]
struct WithDuplicateName {
    value: usize,
    #[serde(rename = "value")]
    foo: usize,
}

fn main() {}

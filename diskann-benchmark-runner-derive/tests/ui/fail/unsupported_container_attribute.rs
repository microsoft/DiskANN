/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Reflect;

#[derive(Reflect)]
#[serde(default)]
struct UnsupportedContainerAttribute {
    value: usize,
}

fn main() {}

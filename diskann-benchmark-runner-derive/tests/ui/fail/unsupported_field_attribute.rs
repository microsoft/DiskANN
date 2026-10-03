/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Reflect;

#[derive(Reflect)]
struct UnsupportedFieldAttribute {
    #[serde(default)]
    value: usize,
}

fn main() {}

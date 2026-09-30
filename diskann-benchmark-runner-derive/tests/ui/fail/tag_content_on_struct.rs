/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Reflect;

#[derive(Reflect)]
#[serde(tag = "kind", content = "value")]
struct AdjacentlyTaggedStruct {
    value: usize,
}

fn main() {}

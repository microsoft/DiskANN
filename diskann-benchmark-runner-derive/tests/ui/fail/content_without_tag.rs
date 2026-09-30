/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Reflect;

#[derive(Reflect)]
#[serde(content = "value")]
enum ContentWithoutTag {
    Unit,
}

fn main() {}

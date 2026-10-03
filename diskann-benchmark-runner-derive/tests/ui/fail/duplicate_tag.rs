/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Reflect;

#[derive(Reflect)]
#[serde(tag = "first", tag = "second")]
enum DuplicateTag {
    Unit,
}

fn main() {}

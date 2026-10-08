/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Reflect;

#[derive(Reflect)]
#[serde(tag = "first", content = "first")]
enum DuplicateTag {
    Unit,
}

fn main() {}

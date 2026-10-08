/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Reflect;

// Case: Duplicate field rename.
#[derive(Reflect)]
struct DuplicateFieldRename {
    #[serde(rename = "first", rename = "second")]
    value: usize,
}

fn main() {}

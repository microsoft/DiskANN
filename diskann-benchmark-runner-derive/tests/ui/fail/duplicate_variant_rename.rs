/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Reflect;

#[derive(Reflect)]
enum DuplicateVariantRename {
    #[serde(rename = "first", rename = "second")]
    Unit,
}

fn main() {}

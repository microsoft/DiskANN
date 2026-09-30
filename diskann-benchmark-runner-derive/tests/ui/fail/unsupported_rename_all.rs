/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Reflect;

#[derive(Reflect)]
#[serde(rename_all = "camelCase")]
struct UnsupportedRenameAll {
    field_name: usize,
}

fn main() {}

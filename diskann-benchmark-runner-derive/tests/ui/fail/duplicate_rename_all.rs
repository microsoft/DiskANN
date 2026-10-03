/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Reflect;

#[derive(Reflect)]
#[serde(rename_all = "snake_case", rename_all = "kebab-case")]
struct DuplicateRenameAll {
    field_name: usize,
}

fn main() {}

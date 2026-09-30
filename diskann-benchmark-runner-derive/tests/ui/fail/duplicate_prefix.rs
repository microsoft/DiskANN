/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Reflect;

#[derive(Reflect)]
#[reflect(prefix = "one::", prefix = "two::")]
struct DuplicatePrefix;

fn main() {}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Reflect;

#[derive(Reflect)]
#[reflect(prefix = "benchmark::", type_name = "Renamed")]
struct PrefixAndTypeName;

fn main() {}

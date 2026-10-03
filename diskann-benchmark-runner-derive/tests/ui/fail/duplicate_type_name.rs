/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Reflect;

#[derive(Reflect)]
#[reflect(type_name = "One", type_name = "Two")]
struct DuplicateTypeName;

fn main() {}

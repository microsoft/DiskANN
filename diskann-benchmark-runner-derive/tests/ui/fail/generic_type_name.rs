/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Reflect;

#[derive(Reflect)]
#[reflect(type_name = "Generic")]
struct Generic<T> {
    value: T,
}

fn main() {}

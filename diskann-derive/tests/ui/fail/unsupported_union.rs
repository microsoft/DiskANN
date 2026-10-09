/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Reflect;

#[derive(Reflect)]
union UnsupportedUnion {
    value: usize,
}

fn main() {}

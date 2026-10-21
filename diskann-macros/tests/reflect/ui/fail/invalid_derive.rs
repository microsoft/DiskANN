/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Reflect;

struct Unreflected(usize);

#[derive(Reflect)]
struct CannotDerive {
    value: Unreflected,
}

fn main() {}

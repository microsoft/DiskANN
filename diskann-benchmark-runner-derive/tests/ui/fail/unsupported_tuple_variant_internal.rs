/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Reflect;

#[derive(Reflect)]
#[serde(tag = "tag")]
enum UnsupportedEmpty {
    Unit,
    // The following is not allowed by `serde` and thus should be rejected by `Reflect`.
    Tuple(),
}

#[derive(Reflect)]
#[serde(tag = "tag")]
enum UnsupportedTwoTuple {
    Unit,
    // The following is not allowed by `serde` and thus should be rejected by `Reflect`.
    Tuple(usize, usize),
}

fn main() {}

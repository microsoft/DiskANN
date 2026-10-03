/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Reflect;

#[derive(Reflect)]
enum UnsupportedVariantAttribute {
    #[serde(skip)]
    Unit,
}

fn main() {}

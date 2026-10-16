/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Reflect;

#[derive(Reflect)]
#[reflect(prefix = "benchmark::")]
#[serde(rename_all = "kebab-case", tag = "kind", content = "value")]
enum Generic<T, const N: usize> {
    Unit,
    #[serde(rename = "tuple")]
    Tuple(T, Vec<T>),
    #[serde(rename_all = "snake_case")]
    Struct {
        #[serde(rename = "renamed")]
        field_name: T,
    },
}

#[derive(Reflect)]
#[reflect(type_name = "CompleteOverride")]
struct Renamed {
    value: usize,
}

fn main() {
    let _ = <Generic<usize, 2> as Reflect>::ty();
    let _ = <Renamed as Reflect>::ty();
}

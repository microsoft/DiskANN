/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Reflect;

#[derive(Reflect)]
#[serde(tag = "kind")]
enum Direct {
    Value { kind: usize },
}

#[derive(Reflect)]
#[serde(tag = "kind")]
enum ExplicitRename {
    Value {
        #[serde(rename = "kind")]
        value: usize,
    }
}

#[derive(Reflect)]
#[serde(tag = "foo-bar")]
enum RenameAll {
    #[serde(rename_all = "kebab-case")]
    Value { foo_bar: usize }
}

fn main() {}

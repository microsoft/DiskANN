/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Reflect;

#[derive(Reflect)]
#[serde(content = "value")]
enum ContentWithoutTag {
    Unit,
}

#[derive(Reflect)]
#[serde(tag = "kind", content = "first", content = "second")]
enum DuplicateContent {
    Unit,
}

#[derive(Reflect)]
#[serde(tag = "first", tag = "second")]
enum DuplicateTag {
    Unit,
}

#[derive(Reflect)]
#[serde(tag = "same", content = "same")]
enum SameTagAndContent {
    Unit,
}

#[derive(Reflect)]
#[serde(tag = "kind")]
struct TagOnStruct {
    value: usize,
}

#[derive(Reflect)]
#[serde(tag = "kind", content = "value")]
struct TagAndContentOnStruct {
    value: usize,
}

#[derive(Reflect)]
#[serde(tag = "tag")]
enum InternallyTaggedEmptyTuple {
    Tuple(),
}

#[derive(Reflect)]
#[serde(tag = "tag")]
enum InternallyTaggedTuple {
    Tuple(usize, usize),
}

#[derive(Reflect)]
#[serde(tag = "kind")]
enum DirectTagConflict {
    Value { kind: usize },
}

#[derive(Reflect)]
#[serde(tag = "kind")]
enum RenamedTagConflict {
    Value {
        #[serde(rename = "kind")]
        value: usize,
    },
}

#[derive(Reflect)]
#[serde(tag = "foo-bar")]
enum RenameAllTagConflict {
    #[serde(rename_all = "kebab-case")]
    Value { foo_bar: usize },
}

fn main() {}

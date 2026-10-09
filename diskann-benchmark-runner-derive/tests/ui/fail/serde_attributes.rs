/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Reflect;

#[derive(Reflect)]
#[serde(rename_all = "snake_case", rename_all = "kebab-case")]
struct DuplicateRenameAll {
    field_name: usize,
}

#[derive(Reflect)]
struct DuplicateFieldRename {
    #[serde(rename = "first", rename = "second")]
    value: usize,
}

#[derive(Reflect)]
enum DuplicateVariantRename {
    #[serde(rename = "first", rename = "second")]
    Unit,
}

#[derive(Reflect)]
#[serde(rename_all = "camelCase")]
struct UnsupportedRenameAll {
    field_name: usize,
}

#[derive(Reflect)]
#[serde(default)]
struct UnsupportedContainerAttribute {
    value: usize,
}

#[derive(Reflect)]
struct UnsupportedFieldAttribute {
    #[serde(default)]
    value: usize,
}

#[derive(Reflect)]
enum UnsupportedVariantAttribute {
    #[serde(skip)]
    Unit,
}

#[derive(Reflect)]
struct RenameUnnamedField(#[serde(rename = "value")] usize);

#[derive(Reflect)]
#[serde(rename_all = "snake_case")]
struct RenameAllTupleStruct(usize);

#[derive(Reflect)]
#[serde(rename_all = "snake_case")]
struct RenameAllUnitStruct;

#[derive(Reflect)]
enum DuplicateVariantRenameAll {
    #[serde(rename_all = "snake_case", rename_all = "snake_case")]
    Value {
        foo: usize,
    }
}

#[derive(Reflect)]
enum RenameAllTupleVariant {
    #[serde(rename_all = "snake_case")]
    Value(usize),
}

#[derive(Reflect)]
enum RenameAllUnitVariant {
    #[serde(rename_all = "snake_case")]
    Value,
}

fn main() {}

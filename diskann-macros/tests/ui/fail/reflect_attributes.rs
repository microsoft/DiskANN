/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Reflect;

#[derive(Reflect)]
#[reflect(prefix = "one::", prefix = "two::")]
struct DuplicatePrefix;

#[derive(Reflect)]
#[reflect(type_name = "One", type_name = "Two")]
struct DuplicateTypeName;

#[derive(Reflect)]
#[reflect(prefix = "benchmark::", type_name = "Renamed")]
struct PrefixAndTypeName;

#[derive(Reflect)]
#[reflect(type_name = "Generic")]
struct GenericTypeName<T> {
    value: T,
}

#[derive(Reflect)]
#[reflect(rename = "Unsupported")]
struct UnsupportedContainerReflectAttribute;

#[derive(Reflect)]
struct UnsupportedFieldReflectAttribute {
    #[reflect(rename = "value")]
    value: usize,
}

#[derive(Reflect)]
struct UnsupportedFieldReflectAttributeEmpty {
    #[reflect]
    value: usize,
}

#[derive(Reflect)]
enum UnsupportedVariantReflectAttribute {
    #[reflect(rename = "value")]
    Value,
}

#[derive(Reflect)]
enum UnsupportedVariantReflectAttributeEmpty {
    #[reflect]
    Value,
}

fn main() {}

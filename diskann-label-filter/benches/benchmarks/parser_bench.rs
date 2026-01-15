/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use criterion::{criterion_group, BenchmarkId, Criterion};
use diskann_label_filter::{eval_query_expr, parse_query_filter};
use rand::{prelude::IndexedRandom, Rng};
use serde_json::json;
/// Benchmark the original parser without evaluation
fn bench_original_parser(c: &mut Criterion) {
    // Simple filters
    let simple_filter = json!({"a": {"$eq": 1}});

    // Complex filter with nested operators
    let complex_filter = json!({"$and": [
        {"a": {"$eq": 1}},
        {"b": {"$gt": 1}},
        {"specs.cpu": {"$eq": "i7"}},
        {"$or": [
            {"c": {"$lt": 2}},
            {"c": {"$gte": 3}}
        ]},
        {"arr": {"$in": [2,3]}},
        {"tags": {"$in": ["b", "x"]}}
    ]});

    let mut group = c.benchmark_group("parser_only");

    // Benchmark simple filter parsing
    group.bench_function("original_simple", |b| {
        b.iter(|| {
            let ast = parse_query_filter(&simple_filter);
            assert!(ast.is_ok());
        })
    });

    // Benchmark complex filter parsing
    group.bench_function("original_complex", |b| {
        b.iter(|| {
            let ast = parse_query_filter(&complex_filter);
            assert!(ast.is_ok());
        })
    });

    group.finish();
}

/// Benchmark comparing all parsers directly
fn bench_parsers_comparison(c: &mut Criterion) {
    let filters = vec![
        // Simple filter
        ("simple", json!({"a": {"$eq": 1}})),
        // Nested field access
        ("nested", json!({"specs.memory.size": {"$eq": 16}})),
        // Array operations
        ("array", json!({"tags": {"$in": ["c", "f", "g"]}})),
        // Logical operators
        (
            "logical",
            json!({"$and": [{"a": {"$eq": 1}}, {"b": {"$eq": 2}}]}),
        ),
        // Complex filter
        (
            "complex",
            json!({"$and": [
                {"a": {"$eq": 1}},
                {"b": {"$gt": 1}},
                {"$or": [
                    {"c": {"$lt": 2}},
                    {"c": {"$gte": 3}}
                ]},
                {"arr": {"$in": [2,3]}},
                {"tags": {"$in": ["b", "x"]}}
            ]}),
        ),
    ];

    let mut group = c.benchmark_group("parsers_comparison");
    for (name, filter) in filters {
        group.bench_with_input(BenchmarkId::new("original", name), &filter, |b, filter| {
            b.iter(|| {
                let ast = parse_query_filter(filter);
                assert!(ast.is_ok());
            })
        });
    }

    group.finish();
}

#[allow(clippy::disallowed_methods)]
fn rng() -> rand::prelude::ThreadRng {
    rand::rng()
}

/// Helper to generate a random label JSON object with various fields
fn generate_random_label() -> serde_json::Value {
    let mut rng = rng();
    let categories = ["laptop", "desktop", "tablet", "phone"];
    let brands = ["Apple", "Dell", "Lenovo", "HP"];
    let cpus = ["i5", "i7", "i9"];
    let memory_types = ["DDR4", "DDR5"];
    let tags = ["a", "b", "c", "d", "e"];

    // Pre-compute random values
    let a_val = rng.random_range(0..100);
    let b_val = rng.random_range(0..100);
    let c_val = rng.random_range(0..100);
    let category_val = *categories.choose(&mut rng).unwrap();
    let price_val = rng.random_range(500.0..2000.0);
    let in_stock_val = rng.random_bool(0.7);
    let brand_val = *brands.choose(&mut rng).unwrap();
    let cpu_val = *cpus.choose(&mut rng).unwrap();
    let memory_size_val = rng.random_range(8..65);
    let memory_type_val = *memory_types.choose(&mut rng).unwrap();
    let tags_val: Vec<_> = tags.choose_multiple(&mut rng, 3).cloned().collect();

    // Construct JSON
    json!({
        "a": a_val,
        "b": b_val,
        "c": c_val,
        "category": category_val,
        "price": price_val,
        "in_stock": in_stock_val,
        "brand": brand_val,
        "specs": {
            "cpu": cpu_val,
            "memory": {
                "size": memory_size_val,
                "type": memory_type_val
            }
        },
        "tags": tags_val
    })
}

/// Helper to generate a random filter JSON object
fn generate_random_filter() -> serde_json::Value {
    let mut rng = rng();
    let fields = [
        "a",
        "b",
        "c",
        "price",
        "in_stock",
        "category",
        "brand",
        "specs.cpu",
        "specs.memory.size",
    ];
    let categories = ["laptop", "desktop", "tablet", "phone"];
    let brands = ["Apple", "Dell", "Lenovo", "HP"];
    let cpus = ["i5", "i7", "i9"];

    let field = *fields.choose(&mut rng).unwrap();

    let value = match field {
        "a" | "b" | "c" | "specs.memory.size" => {
            let num = rng.random_range(0..100);
            json!({"$eq": num})
        }
        "price" => {
            let price = rng.random_range(1000.0..2000.0);
            json!({"$lt": price})
        }
        "in_stock" => {
            let is_in_stock = rng.random_bool(0.5);
            json!({"$eq": is_in_stock})
        }
        "category" => {
            let category = *categories.choose(&mut rng).unwrap();
            json!({"$eq": category})
        }
        "brand" => {
            let brand = *brands.choose(&mut rng).unwrap();
            json!({"$eq": brand})
        }
        "specs.cpu" => {
            let cpu = *cpus.choose(&mut rng).unwrap();
            json!({"$eq": cpu})
        }
        _ => json!({"$eq": 0}),
    };

    json!({ field: value })
}

/// Benchmark using batches of random labels and filters
fn bench_dynamic(c: &mut Criterion) {
    let mut group = c.benchmark_group("parser");

    group.bench_function("dynamic_original", |b| {
        b.iter_batched(
            || (generate_random_label(), generate_random_filter()),
            |(label, filter)| {
                let _ast = parse_query_filter(&filter).unwrap();
                eval_query_expr(&_ast, &label);
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(
    parser_benches,
    bench_original_parser,
    bench_parsers_comparison,
    bench_dynamic
);

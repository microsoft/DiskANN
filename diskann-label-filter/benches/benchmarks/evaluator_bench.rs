/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use criterion::{criterion_group, Criterion};
use diskann_label_filter::{eval_query_expr, parse_query_filter};
use serde_json::json;

/// benchmark focusing on basic field access
fn bench_simple_query(c: &mut Criterion) {
    let label = json!({
        "a": 1,
        "b": 2,
        "c": 3
    });

    let filter = json!({"a": {"$eq": 1}});

    let mut group = c.benchmark_group("simple_query");

    let original_ast = parse_query_filter(&filter).unwrap();
    group.bench_function("original", |b| {
        b.iter(|| {
            assert!(eval_query_expr(&original_ast, &label));
        })
    });

    group.finish();
}

/// Benchmark focusing on nested field access with dot notation
fn bench_nested_query(c: &mut Criterion) {
    let label = json!({
        "specs": {
            "cpu": "i7",
            "memory": {
                "size": 16,
                "type": "DDR4"
            }
        }
    });

    let filter = json!({"specs.memory.size": {"$eq": 16}});

    let mut group = c.benchmark_group("nested_query");

    let original_ast = parse_query_filter(&filter).unwrap();
    group.bench_function("original", |b| {
        b.iter(|| {
            assert!(eval_query_expr(&original_ast, &label));
        })
    });

    group.finish();
}

/// Benchmark focusing on array operations
fn bench_array_query(c: &mut Criterion) {
    let label = json!({
        "tags": ["a", "b", "c", "d", "e"],
        "scores": [85, 90, 95, 100]
    });

    let filter = json!({"$and": [
        {"tags": {"$in": ["c", "f", "g"]}},
        {"scores": {"$in": [85, 100]}}
    ]});

    let mut group = c.benchmark_group("array_query");

    let original_ast = parse_query_filter(&filter).unwrap();
    group.bench_function("original", |b| {
        b.iter(|| {
            assert!(eval_query_expr(&original_ast, &label));
        })
    });

    group.finish();
}

/// Complex benchmark with different types of operations
fn bench_complex_query(c: &mut Criterion) {
    let label = json!({
        "a": 1,
        "b": 2,
        "c": 3,
        "specs": { "cpu": "i7" },
        "arr": [1,2,3],
        "tags": ["a", "b", "c"],
        "flt": 3.5,
        "int": 5,
        "str": "abc"
    });

    let filter = json!({"$and": [
        {"a": {"$eq": 1}},
        {"b": {"$gt": 1}},
        {"specs.cpu": {"$eq": "i7"}},
        {"$or": [
            {"c": {"$lt": 2}},
            {"c": {"$gte": 3}}
        ]},
        {"arr": {"$in": [2,3]}},
        {"tags": {"$in": ["b", "x"]}},
        {"flt": {"$lte": 3.5}},
        {"int": {"$gte": 5}},
        {"str": {"$eq": "abc"}}
    ]});

    let mut group = c.benchmark_group("complex_query");

    let original_ast = parse_query_filter(&filter).unwrap();
    group.bench_function("original", |b| {
        b.iter(|| {
            assert!(eval_query_expr(&original_ast, &label));
        })
    });

    group.finish();
}

criterion_group!(
    evaluator_benches,
    bench_simple_query,
    bench_nested_query,
    bench_array_query,
    bench_complex_query
);

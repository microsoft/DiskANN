# Label Format Parser

A Rust library for parsing and evaluating MongoDB-style query filters against JSON data.

## Features

- Parse MongoDB-style query filters from JSON into an AST (Abstract Syntax Tree)
- Support for logical operators: `$and`, `$or`, `$not`
- Support for comparison operators: `$eq`, `$ne`, `$lt`, `$lte`, `$gt`, `$gte`, `$in`, `$nin`
- Support for dot notation for nested fields (e.g., `specs.cpu`)
- Support for array matching with `$in` and `$nin` operators
- Multiple optimization strategies for performance

## Usage

```rust
use serde_json::json;
use label_format_parser::{parse_query_filter, eval_query_expr};

// Create a JSON label
let label = json!({
    "a": 1,
    "b": 2,
    "specs": { "cpu": "i7" },
    "tags": ["red", "blue", "green"]
});

// Create a filter that matches labels with a=1 AND b>1 AND specs.cpu="i7" AND tags contains "blue"
let filter = json!({
    "$and": [
        {"a": {"$eq": 1}},
        {"b": {"$gt": 1}},
        {"specs.cpu": {"$eq": "i7"}},
        {"tags": {"$in": ["blue"]}}
    ]
});

// Parse the filter into an AST
let ast = parse_query_filter(&filter).unwrap();

// Evaluate the filter against the label
let matches = eval_query_expr(&ast, &label);
assert!(matches);
```

## Running Benchmarks

The project includes a comprehensive benchmarking suite that can be run with:

```bash
cargo bench
```

Benchmarks are organized in modules under the `benches/benchmarks/` directory:
- `parser_bench.rs`: Evaluates the performance of different parsing strategies
- `evaluator_bench.rs`: Compares the query evaluation performance across implementations

For the current use cases, the original implementation offers the best performance. The focused optimization approach is competitive but doesn't offer significant improvements. The pre-compiled paths approach would need further refinement to be viable.



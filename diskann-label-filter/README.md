# Label Filter Lib

A Rust library for parsing and evaluating filters against JSON meta data.

[label-data-format-rfc.md](../docs/rfcs/cy2025/label-data-format-rfc.md)

## Usage

```rust
use serde_json::json;
use diskann_label_filter::{parse_query_filter, eval_query_expr};

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
let ast = match parse_query_filter(&filter) {
    Ok(ast) => ast,
    Err(e) => {
        eprintln!("Failed to parse filter: {}", e);
        return;
    }
};

// Evaluate the filter against the label
let matches = eval_query_expr(&ast, &label);
assert!(matches);
```

### Examples

Parse AST and output it as simple query expression

```
cargo run --example print_query
```


Process and evaluate JSON line formatted files with:

```
cargo run --example jsonl_reader_example
```

Convert old txt based format into json based file

```

converter <base_input_file> <query_input_file> <base_output_file> <query_output_file>

cargo run --example converter ..\tests\data\disk_index_search\data.256.label ..\tests\data\disk_index_search\query.128.label ..\tests\data\disk_index_search\data.256.label.jsonl ..\tests\data\disk_index_search\query.128.label.jsonl
```

## Running Benchmarks

The project includes a comprehensive benchmarking suite that can be run with:

```bash
cargo bench
```

Benchmarks are organized in modules under the `benches/benchmarks/` directory:
- `parser_bench.rs`: Evaluates the performance of parsing
- `evaluator_bench.rs`: Evaluates the query evaluation performance

## Implementation Details

### Architecture Overview

The `label-filter` library is built around three core components:

1. **Abstract Syntax Tree (AST)**: A hierarchical representation of query filters
2. **Parser**: Converts JSON query filters to the AST representation
3. **Evaluator**: Evaluates the AST against JSON labels

### Abstract Syntax Tree (AST)

The AST is defined in `ast.rs` and consists of:

```rust
pub enum ASTExpr {
    And(Vec<ASTExpr>),          // Logical AND of sub-expressions
    Or(Vec<ASTExpr>),           // Logical OR of sub-expressions
    Not(Box<ASTExpr>),          // Logical NOT of a sub-expression
    Compare { field: String, op: CompareOp }, // Field comparison
}
```

The `CompareOp` enum uses type-safe representations for different comparison operators:

```rust
pub enum CompareOp {
    Eq(Value),       // Equal to any JSON value
    Ne(Value),       // Not equal to any JSON value
    Lt(f64),         // Less than (numeric only)
    Lte(f64),        // Less than or equal (numeric only)
    Gt(f64),         // Greater than (numeric only)
    Gte(f64),        // Greater than or equal (numeric only)
    In(Vec<Value>),  // Value is in array
    Nin(Vec<Value>), // Value is not in array
}
```

The type-safe design ensures that each operator only accepts appropriate value types, enforcing correctness at compile time.

### Parser

The parser (`parser.rs`) converts JSON filter specifications into the AST. Key features:

- Support for logical operators (`$and`, `$or`, `$not`)
- Support for comparison operators (`$eq`, `$ne`, `$lt`, `$lte`, `$gt`, `$gte`, `$in`, `$nin`)
- Automatic handling of implicit `$and` for multiple field conditions
- Support for dot notation to access nested fields (`user.profile.age`)
- Enforced nesting depth limit
- Type checking for operators (e.g., numeric operators require numeric values)

### Evaluator

The evaluator (`evaluator.rs`) applies the AST against JSON labels to determine if they match:

- Recursive traversal of the AST
- Type-aware comparison operations
- Support for array field values with `$in` and `$nin` operators

### Visitor Pattern

The library implements the Visitor pattern to enable extensible operations on the AST:

- `ASTVisitor` trait defines the interface for visitors
- `PrintVisitor` implementation converts AST to human-readable format
- Display implementation for easy debugging and logging



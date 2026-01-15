/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_label_filter::parse_query_filter;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example 1: Parse and print a simple query
    // Format follows the RFC Query Expression Format
    let filter_json = json!({
        "category": {"$eq": "electronics"}
    });

    let query = parse_query_filter(&filter_json)?;
    println!("Simple query:");
    println!("{}", query);
    println!();

    // Example 2: Parse and print a complex query
    // Format follows the RFC Query Expression Format with nested operators
    let complex_filter = json!({
        "$and": [
            {
                "price": {"$lt": 100}
            },
            {
                "$or": [
                    {
                        "category": {"$eq": "electronics"}
                    },
                    {
                        "category": {"$eq": "computers"}
                    }
                ]
            },
            {
                "$not": {
                    "discontinued": {"$eq": true}
                }
            }
        ]
    });

    let complex_query = parse_query_filter(&complex_filter)?;
    println!("Complex query:");
    println!("multi line:");
    println!("{}", complex_query);
    println!();

    // Example 3: Custom indentation
    println!("With custom indentation:");
    println!("{}", complex_query.to_string_with_indent("    "));

    // Example 4: Multiple field conditions in a single object (implicit AND)
    let multi_field_filter = json!({
        "category": {"$eq": "laptop"},
        "price": {"$lt": 1500.0},
        "quantity": {"$gte": 1},
        "in_stock": {"$eq": true},
        "rating": {"$gt": 4.5}
    });

    let multi_field_query = parse_query_filter(&multi_field_filter)?;
    println!("\nMulti-field query (implicit AND):");
    println!("{}", multi_field_query);

    // Example 5: Dot notation for nested fields
    let nested_fields_filter = json!({
        "specifications.processor": {"$eq": "Intel i7"},
        "specifications.cores": {"$gte": 6}
    });

    let nested_fields_query = parse_query_filter(&nested_fields_filter)?;
    println!("\nNested fields query:");
    println!("{}", nested_fields_query);

    // Example 6: Array operations with $in operator
    let array_filter = json!({
        "category": {"$in": ["laptop", "desktop", "tablet"]}
    });

    let array_query = parse_query_filter(&array_filter)?;
    println!("\nArray query with $in operator:");
    println!("{}", array_query);

    Ok(())
}

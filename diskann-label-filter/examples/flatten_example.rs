/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_label_filter::parser::format::Document;
use serde_json::json;

fn main() {
    // Create a label with nested metadata
    let label = Document {
        doc_id: 123,
        label: json!({
            "name": "Product X",
            "details": {
                "color": "blue",
                "dimensions": {
                    "width": 10,
                    "height": 20
                }
            },
            "tags": ["electronics", "featured"],
            "price": 99.99,
            "inStock": true
        }),
    };

    // Get flattened key-value pairs with dot notation
    println!("Flattened key-value pairs:");
    let flattened = label.flatten_metadata();
    for (key, value) in &flattened {
        println!("  \"{}\" => {}", key, value);
    }
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fs::File,
    io::{self, Write},
    path::PathBuf,
};

use diskann_label_filter::{
    eval_query_expr, read_and_parse_queries, read_baselabels, read_ground_truth, read_queries,
};

fn create_sample_files() -> io::Result<(PathBuf, PathBuf, PathBuf)> {
    let documents_path = PathBuf::from("sample_documents.jsonl");
    let queries_path = PathBuf::from("sample_queries.jsonl");
    let ground_truth_path = PathBuf::from("sample_ground_truth.jsonl");

    // Create sample label file
    {
        let mut file = File::create(&documents_path)?;
        writeln!(
            file,
            r#"{{"doc_id": 0, "category": "laptop", "brand": "Apple", "price": 1299.99, "quantity": 5, "in_stock": true, "rating": 4.8, "specifications": {{"processor": "Intel i7", "ram": "16GB", "cores": 8, "base_clock": 2.6}}, "tags": ["premium", "gaming"]}}"#
        )?;
        writeln!(
            file,
            r#"{{"doc_id": 1, "category": "desktop", "brand": "Dell", "price": 899.99, "quantity": 12, "in_stock": true, "rating": 4.2, "specifications": {{"processor": "Intel i5", "ram": "8GB", "cores": 4, "base_clock": 3.2}}, "tags": ["budget", "office"]}}"#
        )?;
        writeln!(
            file,
            r#"{{"doc_id": 2, "category": "laptop", "brand": "Lenovo", "price": 1099.99, "quantity": 8, "in_stock": true, "rating": 4.6, "specifications": {{"processor": "AMD Ryzen 7", "ram": "16GB", "cores": 8, "base_clock": 3.0}}, "tags": ["business", "performance"]}}"#
        )?;
        writeln!(
            file,
            r#"{{"doc_id": 3, "category": "desktop", "brand": "HP", "price": 1199.99, "quantity": 3, "in_stock": true, "rating": 4.3, "specifications": {{"processor": "Intel i7", "ram": "32GB", "cores": 8, "base_clock": 3.8}}, "tags": ["gaming", "high-performance"]}}"#
        )?;
        writeln!(
            file,
            r#"{{"doc_id": 4, "category": "laptop", "brand": "Dell", "price": 799.99, "quantity": 15, "in_stock": true, "rating": 4.0, "specifications": {{"processor": "Intel i5", "ram": "8GB", "cores": 4, "base_clock": 2.4}}, "tags": ["budget", "office"]}}"#
        )?;
    }

    // Create sample query file
    {
        let mut file = File::create(&queries_path)?;
        writeln!(
            file,
            r#"{{"query_id": 0, "filter": {{"category": {{"$eq": "laptop"}}, "price": {{"$lt": 1500.0}}, "quantity": {{"$gte": 1}}, "in_stock": {{"$eq": true}}, "rating": {{"$gt": 4.5}}, "specifications.processor": {{"$eq": "Intel i7"}}, "specifications.cores": {{"$gte": 6}}}}}}"#
        )?;
        writeln!(
            file,
            r#"{{"query_id": 1, "filter": {{"category": {{"$eq": "desktop"}}, "price": {{"$lt": 1000}}, "specifications.cores": {{"$eq": 4}}, "tags": {{"$in": ["budget", "office"]}}}}}}"#
        )?;
        writeln!(
            file,
            r#"{{"query_id": 2, "filter": {{"$or": [{{"brand": {{"$eq": "Apple"}}}}, {{"brand": {{"$eq": "Lenovo"}}}}]}}}}"#
        )?;
        writeln!(
            file,
            r#"{{"query_id": 3, "filter": {{"$and": [{{"category": {{"$eq": "laptop"}}}}, {{"price": {{"$gte": 1000}}}}, {{"specifications.ram": {{"$eq": "16GB"}}}}]}}}}"#
        )?;
    }

    // Create sample ground truth file
    {
        let mut file = File::create(&ground_truth_path)?;
        writeln!(file, r#"{{"distance_func": "l2", "query_num": 4}}"#)?;
        writeln!(
            file,
            r#"{{"query_id": 0, "count": 1, "ids": [0], "distances": [0.234]}}"#
        )?;
        writeln!(
            file,
            r#"{{"query_id": 1, "count": 1, "ids": [1], "distances": [0.222]}}"#
        )?;
        writeln!(
            file,
            r#"{{"query_id": 2, "count": 2, "ids": [0, 2], "distances": [0.211, 0.245]}}"#
        )?;
        writeln!(
            file,
            r#"{{"query_id": 3, "count": 2, "ids": [0, 2], "distances": [0.198, 0.205]}}"#
        )?;
    }

    Ok((documents_path, queries_path, ground_truth_path))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Creating sample files...");
    let (labels_path, queries_path, ground_truth_path) = create_sample_files()?;

    // Read and print label data
    println!("\nReading label data...");
    let labels = read_baselabels(&labels_path)?;
    println!("Found {} labels:", labels.len());
    for label in &labels {
        println!("Document ID: {}", label.doc_id);
        println!("  Label: {:?}", label.label);
    }

    // Read and print query data
    println!("\nReading query data...");
    let queries = read_queries(&queries_path)?;
    println!("Found {} queries:", queries.len());
    for query in &queries {
        println!("  Query ID: {}", query.query_id);
        println!("  Filter: {}", query.filter);
    }

    // Read and print ground truth data
    println!("\nReading ground truth data...");
    let (metadata, results) = read_ground_truth(&ground_truth_path)?;
    println!(
        "Ground truth metadata: {} distance function, {} queries",
        metadata.distance_func, metadata.query_num
    );
    for result in &results {
        println!(
            "  Query ID: {}, Count: {}, IDs: {:?}",
            result.query_id, result.count, result.ids
        );
    }

    // Parse queries and evaluate against labels
    println!("\nParsing and evaluating queries...");
    let parsed_queries = read_and_parse_queries(&queries_path)?;

    for (query_id, expr) in &parsed_queries {
        println!("\nResults for Query ID {}:", query_id);
        let mut matching_doc_ids = Vec::new();

        for label in &labels {
            let json_value = serde_json::to_value(label).unwrap();
            if eval_query_expr(expr, &json_value) {
                matching_doc_ids.push(label.doc_id);
            }
        }

        println!("  Matching doc IDs: {:?}", matching_doc_ids);
    }

    // Clean up sample files
    std::fs::remove_file(labels_path)?;
    std::fs::remove_file(queries_path)?;
    std::fs::remove_file(ground_truth_path)?;

    Ok(())
}

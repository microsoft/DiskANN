/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fs::File,
    io::{self, BufRead, BufReader},
    path::Path,
};

use serde_json;

use crate::parser::{
    ast::ASTExpr,
    format::{Document, GroundTruthMetadata, GroundTruthResult, QueryExpression},
    query_parser::parse_query_filter,
};

/// Error type for JSONL file reading operations
#[derive(Debug)]
pub enum JsonlReadError {
    IoError(io::Error),
    JsonError(serde_json::Error),
    ParseError(String),
}

impl From<io::Error> for JsonlReadError {
    fn from(err: io::Error) -> Self {
        JsonlReadError::IoError(err)
    }
}

impl From<serde_json::Error> for JsonlReadError {
    fn from(err: serde_json::Error) -> Self {
        JsonlReadError::JsonError(err)
    }
}

impl std::fmt::Display for JsonlReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JsonlReadError::IoError(e) => write!(f, "IO error: {}", e),
            JsonlReadError::JsonError(e) => write!(f, "JSON parsing error: {}", e),
            JsonlReadError::ParseError(e) => write!(f, "Parse error: {}", e),
        }
    }
}

impl std::error::Error for JsonlReadError {}

/// Read label metadata from a JSONL file.
///
/// Each line in the file is expected to be a JSON object with an `id` field
/// and any number of additional fields as per the RFC.
///
/// # Arguments
///
/// * `path` - Path to the JSONL file
///
/// # Returns
///
/// A vector of `LabelMetadata` objects.
pub fn read_baselabels<P: AsRef<Path>>(path: P) -> Result<Vec<Document>, JsonlReadError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut documents = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        match serde_json::from_str::<Document>(&line) {
            Ok(label) => documents.push(label),
            Err(e) => {
                return Err(JsonlReadError::ParseError(format!(
                    "Error parsing line {}: {}",
                    line_num + 1,
                    e
                )))
            }
        }
    }

    Ok(documents)
}

/// Read query expressions from a JSONL file.
///
/// Each line in the file is expected to be a JSON object with a `query_id` field
/// and a `filter` field containing the query expression as per the RFC.
///
/// # Arguments
///
/// * `path` - Path to the JSONL file
///
/// # Returns
///
/// A vector of `QueryExpression` objects.
pub fn read_queries<P: AsRef<Path>>(path: P) -> Result<Vec<QueryExpression>, JsonlReadError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut queries = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        match serde_json::from_str::<QueryExpression>(&line) {
            Ok(query) => queries.push(query),
            Err(e) => {
                return Err(JsonlReadError::ParseError(format!(
                    "Error parsing line {}: {}",
                    line_num + 1,
                    e
                )))
            }
        }
    }

    Ok(queries)
}

/// Read ground truth results from a JSONL file.
///
/// The first line is expected to be a JSON object with `distance_func` and `query_num` fields.
/// Subsequent lines are expected to be JSON objects with `query_id`, `count`, `ids`, and `distances` fields.
///
/// # Arguments
///
/// * `path` - Path to the JSONL file
///
/// # Returns
///
/// A tuple containing the metadata and a vector of ground truth results.
pub fn read_ground_truth<P: AsRef<Path>>(
    path: P,
) -> Result<(GroundTruthMetadata, Vec<GroundTruthResult>), JsonlReadError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Read metadata from the first line
    let metadata_line = lines
        .next()
        .ok_or_else(|| JsonlReadError::ParseError("Ground truth file is empty".to_string()))??;

    let metadata = serde_json::from_str::<GroundTruthMetadata>(&metadata_line)
        .map_err(|e| JsonlReadError::ParseError(format!("Error parsing metadata: {}", e)))?;

    // Read results from subsequent lines
    let mut results = Vec::new();
    for (line_num, line) in lines.enumerate() {
        let line = line?;
        match serde_json::from_str::<GroundTruthResult>(&line) {
            Ok(result) => results.push(result),
            Err(e) => {
                return Err(JsonlReadError::ParseError(format!(
                    "Error parsing result line {}: {}",
                    line_num + 1,
                    e
                )))
            }
        }
    }

    Ok((metadata, results))
}

/// Read and parse query expressions from a JSONL file into QueryExpr AST.
///
/// This function combines reading the file and parsing the filter expressions.
///
/// # Arguments
///
/// * `path` - Path to the JSONL file
///
/// # Returns
///
/// A vector of tuples containing the query ID and the parsed QueryExpr.
pub fn read_and_parse_queries<P: AsRef<Path>>(
    path: P,
) -> Result<Vec<(usize, ASTExpr)>, JsonlReadError> {
    let queries = read_queries(path)?;

    let mut parsed_queries = Vec::with_capacity(queries.len());
    for query in queries {
        match parse_query_filter(&query.filter) {
            Ok(expr) => parsed_queries.push((query.query_id, expr)),
            Err(err) => {
                return Err(JsonlReadError::ParseError(format!(
                    "Failed to parse filter for query ID {}: {}",
                    query.query_id, err
                )));
            }
        }
    }

    Ok(parsed_queries)
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::Write};

    use tempfile::tempdir;

    use super::*;

    fn create_test_label_file() -> tempfile::TempDir {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_labels.jsonl");

        let mut file = File::create(&file_path).unwrap();

        // Write sample label data
        writeln!(file, r#"{{"doc_id": 0, "category": "laptop", "brand": "Apple", "price": 1299.99, "quantity": 5, "in_stock": true, "rating": 4.8, "specifications": {{"processor": "Intel i7", "ram": "16GB", "cores": 8, "base_clock": 2.6}}, "tags": ["premium", "gaming"]}}"#).unwrap();
        writeln!(file, r#"{{"doc_id": 1, "category": "desktop", "brand": "Dell", "price": 899.99, "quantity": 12, "in_stock": true, "rating": 4.2, "specifications": {{"processor": "Intel i5", "ram": "8GB", "cores": 4, "base_clock": 3.2}}, "tags": ["budget", "office"]}}"#).unwrap();

        dir
    }

    fn create_test_query_file() -> tempfile::TempDir {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_queries.jsonl");

        let mut file = File::create(&file_path).unwrap();

        // Write sample query data (NOTE: $in operator removed, using $eq instead)
        writeln!(file, r#"{{"query_id": 0, "filter": {{"category": {{"$eq": "laptop"}}, "price": {{"$lt": 1500.0}}, "quantity": {{"$gte": 1}}, "in_stock": {{"$eq": true}}, "rating": {{"$gt": 4.5}}, "specifications.processor": {{"$eq": "Intel i7"}}, "specifications.cores": {{"$gte": 6}}, "$and": [{{"brand": {{"$eq": "Apple"}}}}, {{"price": {{"$gte": 1000.0}}}}], "$or": [{{"category": {{"$eq": "laptop"}}}}, {{"rating": {{"$gt": 4.5}}}}]}}}}"#).unwrap();
        writeln!(file, r#"{{"query_id": 1, "filter": {{"category": {{"$eq": "desktop"}}, "price": {{"$lt": 1000}}, "specifications.cores": {{"$eq": 4}}, "brand": {{"$eq": "Dell"}}}}}}"#).unwrap();

        dir
    }

    fn create_test_ground_truth_file() -> tempfile::TempDir {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_ground_truth.jsonl");

        let mut file = File::create(&file_path).unwrap();

        // Write sample ground truth data
        writeln!(file, r#"{{"distance_func": "l2", "query_num": 2}}"#).unwrap();
        writeln!(
            file,
            r#"{{"query_id": 0, "count": 2, "ids": [0, 1], "distances": [0.234, 0.235]}}"#
        )
        .unwrap();
        writeln!(
            file,
            r#"{{"query_id": 1, "count": 1, "ids": [0], "distances": [0.222]}}"#
        )
        .unwrap();

        dir
    }

    #[test]
    fn test_read_labels() {
        let temp_dir = create_test_label_file();
        let file_path = temp_dir.path().join("test_labels.jsonl");

        let labels = read_baselabels(file_path).unwrap();

        assert_eq!(labels.len(), 2);

        let label0 = &labels[0];
        assert_eq!(label0.doc_id, 0);
        assert_eq!(
            label0.label.get("category").unwrap().as_str().unwrap(),
            "laptop"
        );
        assert_eq!(
            label0.label.get("brand").unwrap().as_str().unwrap(),
            "Apple"
        );

        let label1 = &labels[1];
        assert_eq!(label1.doc_id, 1);
        assert_eq!(
            label1.label.get("category").unwrap().as_str().unwrap(),
            "desktop"
        );
    }

    #[test]
    fn test_read_queries() {
        let temp_dir = create_test_query_file();
        let file_path = temp_dir.path().join("test_queries.jsonl");

        let queries = read_queries(file_path).unwrap();

        assert_eq!(queries.len(), 2);

        let query0 = &queries[0];
        assert_eq!(query0.query_id, 0);
        assert!(query0.filter.is_object());

        let query1 = &queries[1];
        assert_eq!(query1.query_id, 1);
        assert!(query1.filter.is_object());
    }

    #[test]
    fn test_read_ground_truth() {
        let temp_dir = create_test_ground_truth_file();
        let file_path = temp_dir.path().join("test_ground_truth.jsonl");

        let (metadata, results) = read_ground_truth(file_path).unwrap();

        assert_eq!(metadata.distance_func, "l2");
        assert_eq!(metadata.query_num, 2);

        assert_eq!(results.len(), 2);

        let result0 = &results[0];
        assert_eq!(result0.query_id, 0);
        assert_eq!(result0.count, 2);
        assert_eq!(result0.ids, vec![0, 1]);
        assert_eq!(result0.distances, vec![0.234, 0.235]);

        let result1 = &results[1];
        assert_eq!(result1.query_id, 1);
        assert_eq!(result1.count, 1);
        assert_eq!(result1.ids, vec![0]);
        assert_eq!(result1.distances, vec![0.222]);
    }

    #[test]
    fn test_read_and_parse_queries() {
        let temp_dir = create_test_query_file();
        let file_path = temp_dir.path().join("test_queries.jsonl");

        let parsed_queries = read_and_parse_queries(file_path).unwrap();

        assert_eq!(parsed_queries.len(), 2);
        // Additional validation of the parsed expressions would be done in the parser tests
    }
}

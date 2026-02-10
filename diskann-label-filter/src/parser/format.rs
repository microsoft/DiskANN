/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Represents a vector metadata label as defined in the RFC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique identifier for the vector/doc
    pub doc_id: usize,

    /// label in raw json format
    #[serde(flatten)]
    pub label: serde_json::Value,

}


/// Represents a query expression as defined in the RFC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryExpression {
    /// Query identifier
    pub query_id: usize,

    /// Filter expression in raw json format
    pub filter: Value,
}

/// Represents a ground truth result as defined in the RFC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruthResult {
    /// Query identifier
    pub query_id: u64,

    /// Number of matching items
    pub count: usize,

    /// IDs of matching items
    pub ids: Vec<u64>,

    /// Distances to matching items
    pub distances: Vec<f32>,
}

/// Represents the metadata for ground truth results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruthMetadata {
    /// Distance function used
    pub distance_func: String,

    /// Number of queries
    pub query_num: usize,
}

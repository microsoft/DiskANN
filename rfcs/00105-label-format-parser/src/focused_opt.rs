/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use serde_json::Value;

// Keep the original AST structure but optimize the evaluation
use crate::{QueryExpr, CompareOp};

/// An optimized get_value_by_path function that uses a temporary vector
/// to avoid repeated string splits
#[inline]
pub fn get_value_by_path<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    if !path.contains('.') {
        // Fast path for simple fields
        if let Some(obj) = value.as_object() {
            return obj.get(path);
        }
        return None;
    }
    
    // Only split for paths with dots
    let segments: Vec<&str> = path.split('.').collect();
    let mut current = value;
    
    for &key in &segments {
        match current {
            Value::Object(map) => {
                current = map.get(key)?;
            }
            _ => return None,
        }
    }
    Some(current)
}

/// More efficient evaluation function that uses better path lookups
#[inline]
pub fn eval_query_expr(expr: &QueryExpr, label: &Value) -> bool {
    match expr {
        QueryExpr::And(subs) => subs.iter().all(|e| eval_query_expr(e, label)),
        QueryExpr::Or(subs) => subs.iter().any(|e| eval_query_expr(e, label)),
        QueryExpr::Not(sub) => !eval_query_expr(sub, label),
        QueryExpr::Compare { field, op, value } => {
            if let Some(field_val) = get_value_by_path(label, field) {
                match op {
                    CompareOp::Eq => field_val == value,
                    CompareOp::Ne => field_val != value,
                    CompareOp::Lt => {
                        if let (Some(a), Some(b)) = (field_val.as_f64(), value.as_f64()) {
                            a < b
                        } else {
                            false
                        }
                    },
                    CompareOp::Lte => {
                        if let (Some(a), Some(b)) = (field_val.as_f64(), value.as_f64()) {
                            a <= b
                        } else {
                            false
                        }
                    },
                    CompareOp::Gt => {
                        if let (Some(a), Some(b)) = (field_val.as_f64(), value.as_f64()) {
                            a > b
                        } else {
                            false
                        }
                    },
                    CompareOp::Gte => {
                        if let (Some(a), Some(b)) = (field_val.as_f64(), value.as_f64()) {
                            a >= b
                        } else {
                            false
                        }
                    },
                    CompareOp::In => {
                        if let Some(arr) = value.as_array() {
                            match field_val {
                                Value::Array(field_arr) => field_arr.iter().any(|v| arr.contains(v)),
                                _ => arr.contains(field_val),
                            }
                        } else {
                            false
                        }
                    },
                    CompareOp::Nin => {
                        if let Some(arr) = value.as_array() {
                            match field_val {
                                Value::Array(field_arr) => field_arr.iter().all(|v| !arr.contains(v)),
                                _ => !arr.contains(field_val),
                            }
                        } else {
                            true
                        }
                    },
                }
            } else {
                false // Field not found
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use crate::parser::parse_query_filter;

    #[test]
    fn test_optimized_get_value_by_path() {
        let label = json!({
            "a": 1,
            "b": {
                "c": 2,
                "d": {
                    "e": 3
                }
            }
        });
        
        // Simple path (no dots)
        assert_eq!(get_value_by_path(&label, "a"), Some(&json!(1)));
        
        // Nested paths
        assert_eq!(get_value_by_path(&label, "b.c"), Some(&json!(2)));
        assert_eq!(get_value_by_path(&label, "b.d.e"), Some(&json!(3)));
        
        // Non-existent paths
        assert_eq!(get_value_by_path(&label, "z"), None);
        assert_eq!(get_value_by_path(&label, "a.z"), None);
        assert_eq!(get_value_by_path(&label, "b.z"), None);
    }

    #[test]
    fn test_optimized_evaluation() {
        let label = json!({
            "a": 1,
            "b": 2,
            "c": 3,
            "arr": [1, 2, 3],
            "tags": ["x", "y", "z"]
        });
        
        // Test simple comparison operators
        let filter = json!({"a": {"$eq": 1}});
        let expr = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&expr, &label));
        
        // Test logical operators
        let filter = json!({"$and": [
            {"a": {"$eq": 1}},
            {"b": {"$eq": 2}}
        ]});
        let expr = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&expr, &label));
        
        let filter = json!({"$or": [
            {"a": {"$eq": 2}},
            {"b": {"$eq": 2}}
        ]});
        let expr = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&expr, &label));
        
        // Test array operations
        let filter = json!({"arr": {"$in": [2, 4]}});
        let expr = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&expr, &label));
        
        let filter = json!({"arr": {"$nin": [4, 5]}});
        let expr = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&expr, &label));
    }

    #[test]
    fn test_complex_filter_evaluation() {
        let label = json!({
            "a": 1,
            "b": 2, 
            "c": 3,
            "nested": {
                "x": 10,
                "y": 20
            },
            "arr": [1, 2, 3]
        });
        
        let filter = json!({"$and": [
            {"a": {"$eq": 1}},
            {"nested.x": {"$gt": 5}},
            {"$or": [
                {"b": {"$lt": 1}},
                {"c": {"$gte": 3}}
            ]},
            {"arr": {"$in": [2]}}
        ]});
        
        let expr = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&expr, &label));
    }
}

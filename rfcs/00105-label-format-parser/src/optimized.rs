/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Optimized implementation with pre-compiled paths and reduced allocations

use serde_json::Value;
use crate::ast::{QueryExpr, CompareOp};

#[derive(Debug, Clone, PartialEq)]
pub struct PathSegments {
    segments: Vec<String>,
}

impl PathSegments {
    fn new(path: &str) -> Self {
        let segments = path.split('.').map(String::from).collect();
        Self { segments }
    }
}

/// Pre-compiled path lookup for improved performance
#[inline]
fn get_value_by_path<'a>(value: &'a Value, path_segments: &PathSegments) -> Option<&'a Value> {
    let mut current = value;
    for key in &path_segments.segments {
        match current {
            Value::Object(map) => {
                current = map.get(key)?;
            }
            _ => return None,
        }
    }
    Some(current)
}

/// Parse with pre-compiled path segments for the optimized version
pub fn parse_query_filter(filter: &Value) -> Option<QueryExpr> {
    if let Some(obj) = filter.as_object() {
        // Logical operators
        if let Some(arr) = obj.get("$and").and_then(|v| v.as_array()) {
            return Some(QueryExpr::And(arr.iter().filter_map(parse_query_filter).collect()));
        }
        if let Some(arr) = obj.get("$or").and_then(|v| v.as_array()) {
            return Some(QueryExpr::Or(arr.iter().filter_map(parse_query_filter).collect()));
        }
        if let Some(sub) = obj.get("$not") {
            return Some(QueryExpr::Not(Box::new(parse_query_filter(sub)?)));
        }
        // If not a logical operator, treat each field as an implicit AND
        let mut subexprs = vec![];
        for (field, cond) in obj.iter() {
            if field.starts_with('$') { continue; }
            if let Some(cond_obj) = cond.as_object() {
                for (op, val) in cond_obj.iter() {
                    let op = match op.as_str() {
                        "$eq" => CompareOp::Eq,
                        "$ne" => CompareOp::Ne,
                        "$lt" => CompareOp::Lt,
                        "$lte" => CompareOp::Lte,
                        "$gt" => CompareOp::Gt,
                        "$gte" => CompareOp::Gte,
                        "$in" => CompareOp::In,
                        "$nin" => CompareOp::Nin,
                        _ => continue,
                    };
                    subexprs.push(QueryExpr::Compare {
                        field: field.clone(),
                        op,
                        value: val.clone(),
                    });
                }
            }
        }
        if !subexprs.is_empty() {
            if subexprs.len() == 1 {
                Some(subexprs.remove(0))
            } else {
                Some(QueryExpr::And(subexprs))
            }
        } else {
            None
        }
    } else {
        None
    }
}

/// More efficient evaluation function that uses better path lookups
#[inline]
pub fn eval_query_expr(expr: &QueryExpr, label: &Value) -> bool {
    match expr {
        QueryExpr::And(subs) => subs.iter().all(|e| eval_query_expr(e, label)),
        QueryExpr::Or(subs) => subs.iter().any(|e| eval_query_expr(e, label)),
        QueryExpr::Not(sub) => !eval_query_expr(sub, label),
        QueryExpr::Compare { field, op, value } => {
            // Create path segments for efficient lookup
            let path_segments = PathSegments::new(field);
            if let Some(field_val) = get_value_by_path(label, &path_segments) {
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

    #[test]
    fn test_path_segments() {
        let path = "a.b.c";
        let segments = PathSegments::new(path);
        assert_eq!(segments.segments, vec!["a", "b", "c"]);

        let path = "simple";
        let segments = PathSegments::new(path);
        assert_eq!(segments.segments, vec!["simple"]);
    }

    #[test]
    fn test_get_value_by_path() {
        let label = json!({
            "a": 1,
            "b": {
                "c": 2,
                "d": {
                    "e": 3
                }
            }
        });
        
        // Simple path
        let segments = PathSegments::new("a");
        assert_eq!(get_value_by_path(&label, &segments), Some(&json!(1)));
        
        // Nested path
        let segments = PathSegments::new("b.c");
        assert_eq!(get_value_by_path(&label, &segments), Some(&json!(2)));
        
        let segments = PathSegments::new("b.d.e");
        assert_eq!(get_value_by_path(&label, &segments), Some(&json!(3)));
        
        // Non-existent path
        let segments = PathSegments::new("x.y.z");
        assert_eq!(get_value_by_path(&label, &segments), None);
    }

    #[test]
    fn test_parse_query_filter() {
        // Test AND
        let filter = json!({"$and": [{"a": {"$eq": 1}}, {"b": {"$eq": 2}}]});
        let ast = parse_query_filter(&filter).unwrap();
        if let QueryExpr::And(exprs) = ast {
            assert_eq!(exprs.len(), 2);
        } else {
            panic!("Expected AND expression");
        }
        
        // Test OR
        let filter = json!({"$or": [{"a": {"$eq": 1}}, {"b": {"$eq": 2}}]});
        let ast = parse_query_filter(&filter).unwrap();
        if let QueryExpr::Or(exprs) = ast {
            assert_eq!(exprs.len(), 2);
        } else {
            panic!("Expected OR expression");
        }
        
        // Test NOT
        let filter = json!({"$not": {"a": {"$eq": 1}}});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(matches!(ast, QueryExpr::Not(_)));
    }

    #[test]
    fn test_eval_query_expr() {
        let label = json!({
            "a": 1,
            "b": 2,
            "c": 3,
            "arr": [1, 2, 3],
            "tags": ["x", "y", "z"]
        });
        
        // Simple equality
        let expr = QueryExpr::Compare {
            field: "a".to_string(),
            op: CompareOp::Eq,
            value: json!(1),
        };
        assert!(eval_query_expr(&expr, &label));
        
        // Comparison operations
        let expr = QueryExpr::Compare {
            field: "b".to_string(),
            op: CompareOp::Lt,
            value: json!(3),
        };
        assert!(eval_query_expr(&expr, &label));
        
        // Array operations
        let expr = QueryExpr::Compare {
            field: "arr".to_string(),
            op: CompareOp::In,
            value: json!([2, 4]),
        };
        assert!(eval_query_expr(&expr, &label));
        
        // Logical operations
        let expr = QueryExpr::And(vec![
            QueryExpr::Compare {
                field: "a".to_string(),
                op: CompareOp::Eq,
                value: json!(1),
            },
            QueryExpr::Compare {
                field: "b".to_string(),
                op: CompareOp::Eq,
                value: json!(2),
            },
        ]);
        assert!(eval_query_expr(&expr, &label));
    }
}

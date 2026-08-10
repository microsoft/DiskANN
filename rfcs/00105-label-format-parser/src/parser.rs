/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Original query filter parser implementation

use serde_json::Value;
use crate::ast::{QueryExpr, CompareOp};

/// Helper to get a nested value from a label using dot notation (e.g., "specs.cpu")
pub fn get_value_by_path<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = value;
    for key in path.split('.') {
        match current {
            Value::Object(map) => {
                current = map.get(key)?;
            }
            _ => return None,
        }
    }
    Some(current)
}

/// Parse a filter object (serde_json::Value) into a QueryExpr AST
/// Supports logical and comparison operators, and implicit AND for multiple fields.
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use crate::ast::{QueryExpr, CompareOp};

    #[test]
    fn test_get_value_by_path() {
        let label = json!({
            "a": 1,
            "specs": { "cpu": "i7" },
            "nested": { "deeper": { "value": 42 } }
        });
        
        assert_eq!(get_value_by_path(&label, "a"), Some(&json!(1)));
        assert_eq!(get_value_by_path(&label, "specs.cpu"), Some(&json!("i7")));
        assert_eq!(get_value_by_path(&label, "nested.deeper.value"), Some(&json!(42)));
        assert_eq!(get_value_by_path(&label, "nonexistent"), None);
        assert_eq!(get_value_by_path(&label, "specs.nonexistent"), None);
    }

    #[test]
    fn test_parse_query_filter() {
        // Test parsing a simple comparison
        let filter = json!({"a": {"$eq": 1}});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(matches!(
            ast,
            QueryExpr::Compare {
                field,
                op: CompareOp::Eq,
                value
            } if field == "a" && value == json!(1)
        ));

        // Test parsing an AND
        let filter = json!({"$and": [{"a": {"$eq": 1}}, {"b": {"$eq": 2}}]});
        let ast = parse_query_filter(&filter).unwrap();
        if let QueryExpr::And(subs) = ast {
            assert_eq!(subs.len(), 2);
        } else {
            panic!("Expected AND, got {:?}", ast);
        }

        // Test parsing an OR
        let filter = json!({"$or": [{"a": {"$eq": 1}}, {"b": {"$eq": 2}}]});
        let ast = parse_query_filter(&filter).unwrap();
        if let QueryExpr::Or(subs) = ast {
            assert_eq!(subs.len(), 2);
        } else {
            panic!("Expected OR, got {:?}", ast);
        }

        // Test parsing a NOT
        let filter = json!({"$not": {"a": {"$eq": 1}}});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(matches!(ast, QueryExpr::Not(_)));
    }

    #[test]
    fn test_implicit_and() {
        // Multiple fields should be treated as an implicit AND
        let filter = json!({"a": {"$eq": 1}, "b": {"$eq": 2}});
        let ast = parse_query_filter(&filter).unwrap();
        if let QueryExpr::And(subs) = ast {
            assert_eq!(subs.len(), 2);
        } else {
            panic!("Expected implicit AND, got {:?}", ast);
        }
    }
}

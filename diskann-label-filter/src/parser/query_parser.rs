/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use serde_json::Value;

use crate::parser::ast::{ASTExpr, CompareOp};

/// Maximum allowed nesting depth for filter expressions
pub const ALLOWED_DEPTH_LIMIT: usize = 2;

/// Error type for query filter parsing
#[derive(Debug, Clone, PartialEq)]
pub enum QueryFilterError {
    /// Nesting level is too deep
    NestingTooDeep { max_depth: usize },
    /// Unsupported logical operator
    UnsupportedLogicalOperator(String),
    /// Unsupported comparison operator
    UnsupportedComparisonOperator(String),
    /// Invalid value type for operator
    InvalidValueType(String, String),
    /// General parsing failure
    ParseFailure(String),
}

impl std::fmt::Display for QueryFilterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NestingTooDeep { max_depth } => {
                write!(f, "Maximum nesting depth of {} exceeded", max_depth)
            }
            Self::UnsupportedLogicalOperator(op) => {
                write!(f, "Unsupported logical operator: {}", op)
            }
            Self::UnsupportedComparisonOperator(op) => {
                write!(f, "Unsupported comparison operator: {}", op)
            }
            Self::InvalidValueType(expected, got) => {
                write!(f, "Invalid value type: expected {}, got {}", expected, got)
            }
            Self::ParseFailure(msg) => write!(f, "Parse failure: {}", msg),
        }
    }
}

impl std::error::Error for QueryFilterError {}

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
/// Limits nesting to a maximum of 2 levels deep.
pub fn parse_query_filter(filter: &Value) -> Result<ASTExpr, QueryFilterError> {
    parse_query_filter_with_depth(filter, 0)
}

/// Internal function that keeps track of the current nesting depth
fn parse_query_filter_with_depth(
    filter: &Value,
    depth: usize,
) -> Result<ASTExpr, QueryFilterError> {
    // Check if we've reached the maximum allowed depth
    if depth > ALLOWED_DEPTH_LIMIT {
        return Err(QueryFilterError::NestingTooDeep {
            max_depth: ALLOWED_DEPTH_LIMIT,
        });
    }
    if let Some(obj) = filter.as_object() {
        // Logical operators
        if let Some(arr) = obj.get("$and").and_then(|v| v.as_array()) {
            let mut exprs = Vec::new();
            for v in arr {
                match parse_query_filter_with_depth(v, depth + 1) {
                    Ok(expr) => exprs.push(expr),
                    Err(e) => return Err(e),
                }
            }
            if exprs.is_empty() {
                return Err(QueryFilterError::ParseFailure(
                    "Empty $and array or no valid expressions".to_string(),
                ));
            }
            return Ok(ASTExpr::And(exprs));
        }

        if let Some(arr) = obj.get("$or").and_then(|v| v.as_array()) {
            let mut exprs = Vec::new();
            for v in arr {
                match parse_query_filter_with_depth(v, depth + 1) {
                    Ok(expr) => exprs.push(expr),
                    Err(e) => return Err(e),
                }
            }
            if exprs.is_empty() {
                return Err(QueryFilterError::ParseFailure(
                    "Empty $or array or no valid expressions".to_string(),
                ));
            }
            return Ok(ASTExpr::Or(exprs));
        }

        if let Some(sub) = obj.get("$not") {
            return match parse_query_filter_with_depth(sub, depth + 1) {
                Ok(expr) => Ok(ASTExpr::Not(Box::new(expr))),
                Err(e) => Err(e),
            };
        }

        // Check for unknown $-prefixed fields that aren't processed above
        for (field, _) in obj.iter() {
            if field.starts_with('$') {
                return Err(QueryFilterError::UnsupportedLogicalOperator(field.clone()));
            }
        } // If not a logical operator, treat each field as an implicit AND
        let mut subexprs = vec![];
        for (field, cond) in obj.iter() {
            if let Some(cond_obj) = cond.as_object() {
                for (op, val) in cond_obj.iter() {
                    let op = match op.as_str() {
                        "$eq" => CompareOp::Eq(val.clone()),
                        "$ne" => CompareOp::Ne(val.clone()),
                        "$lt" => {
                            if let Some(num) = val.as_f64() {
                                CompareOp::Lt(num)
                            } else {
                                return Err(QueryFilterError::InvalidValueType(
                                    "numeric".to_string(),
                                    val.to_string(),
                                ));
                            }
                        }
                        "$lte" => {
                            if let Some(num) = val.as_f64() {
                                CompareOp::Lte(num)
                            } else {
                                return Err(QueryFilterError::InvalidValueType(
                                    "numeric".to_string(),
                                    val.to_string(),
                                ));
                            }
                        }
                        "$gt" => {
                            if let Some(num) = val.as_f64() {
                                CompareOp::Gt(num)
                            } else {
                                return Err(QueryFilterError::InvalidValueType(
                                    "numeric".to_string(),
                                    val.to_string(),
                                ));
                            }
                        }
                        "$gte" => {
                            if let Some(num) = val.as_f64() {
                                CompareOp::Gte(num)
                            } else {
                                return Err(QueryFilterError::InvalidValueType(
                                    "numeric".to_string(),
                                    val.to_string(),
                                ));
                            }
                        }
                        "$in" | "$nin" => {
                            return Err(QueryFilterError::UnsupportedComparisonOperator(op.clone()))
                        }
                        _ => {
                            return Err(QueryFilterError::UnsupportedComparisonOperator(op.clone()))
                        }
                    };
                    subexprs.push(ASTExpr::Compare {
                        field: field.clone(),
                        op,
                    });
                }
            }
        }
        if !subexprs.is_empty() {
            if subexprs.len() == 1 {
                Ok(subexprs.remove(0))
            } else {
                Ok(ASTExpr::And(subexprs))
            }
        } else {
            Err(QueryFilterError::ParseFailure(
                "No valid expressions found".to_string(),
            ))
        }
    } else {
        Err(QueryFilterError::ParseFailure(
            "Not a JSON object".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::parser::ast::{ASTExpr, CompareOp};
    #[test]
    fn test_invalid_numeric_value_type() {
        // Test with string value for numeric comparison operator
        let filter = json!({
            "age": {"$gt": "not-a-number"}
        });

        let result = parse_query_filter(&filter);
        assert!(
            result.is_err(),
            "Expected error for invalid value type, got: {:?}",
            result
        );

        match result {
            Err(QueryFilterError::InvalidValueType(expected, _)) => {
                assert_eq!(expected, "numeric");
            }
            _ => panic!("Expected InvalidValueType error, got: {:?}", result),
        }
    }

    #[test]
    #[ignore] // TODO: $in operator removed - need to redesign test for other operators
    fn test_invalid_array_value_type() {
        // Test with non-array value for $in operator - NO LONGER SUPPORTED
        let filter = json!({
            "category": {"$in": "not-an-array"}
        });

        let result = parse_query_filter(&filter);
        assert!(
            result.is_err(),
            "Expected error for invalid value type, got: {:?}",
            result
        );

        match result {
            Err(QueryFilterError::InvalidValueType(expected, _)) => {
                assert_eq!(expected, "array");
            }
            _ => panic!("Expected InvalidValueType error, got: {:?}", result),
        }
    }

    #[test]
    fn test_empty_logical_operator_array() {
        // Test with empty $and array
        let filter = json!({
            "$and": []
        });

        let result = parse_query_filter(&filter);
        assert!(
            result.is_err(),
            "Expected error for empty logical operator array, got: {:?}",
            result
        );

        match result {
            Err(QueryFilterError::ParseFailure(msg)) => {
                assert!(msg.contains("Empty $and array"));
            }
            _ => panic!("Expected ParseFailure error, got: {:?}", result),
        }
    }

    #[test]
    fn test_malformed_logical_structure() {
        // Test with logical operator as a direct field value, not an array
        let filter = json!({
            "$and": {"field": {"$eq": 1}}
        });

        let result = parse_query_filter(&filter);
        assert!(
            result.is_err(),
            "Expected error for malformed logical structure, got: {:?}",
            result
        );
    }

    #[test]
    fn test_mixed_valid_and_invalid_operators() {
        // Test with both valid and invalid operators in the same filter
        let filter = json!({
            "$and": [
                {"valid_field": {"$eq": 1}},
                {"invalid_field": {"$invalid": 2}}
            ]
        });

        let result = parse_query_filter(&filter);
        assert!(
            result.is_err(),
            "Expected error for invalid operator, got: {:?}",
            result
        );

        match result {
            Err(QueryFilterError::UnsupportedComparisonOperator(op)) => {
                assert_eq!(op, "$invalid");
            }
            _ => panic!(
                "Expected UnsupportedComparisonOperator error, got: {:?}",
                result
            ),
        }
    }
    #[test]
    fn test_get_value_by_path() {
        let label = json!({
            "a": 1,
            "specs": { "cpu": "i7" },
            "nested": { "deeper": { "value": 42 } }
        });

        assert_eq!(get_value_by_path(&label, "a"), Some(&json!(1)));
        assert_eq!(get_value_by_path(&label, "specs.cpu"), Some(&json!("i7")));
        assert_eq!(
            get_value_by_path(&label, "nested.deeper.value"),
            Some(&json!(42))
        );
        assert_eq!(get_value_by_path(&label, "nonexistent"), None);
        assert_eq!(get_value_by_path(&label, "specs.nonexistent"), None);
    }
    #[test]
    fn test_parse_query_filter() {
        // Test parsing a simple comparison
        let filter = json!({"a": {"$eq": 1}});
        let ast = parse_query_filter(&filter).expect("Failed to parse simple comparison");
        assert!(matches!(
            ast,
            ASTExpr::Compare {
                field,
                op: CompareOp::Eq(value)
            } if field == "a" && value == json!(1)
        ));

        // Test parsing an AND
        let filter = json!({"$and": [{"a": {"$eq": 1}}, {"b": {"$eq": 2}}]});
        let ast = parse_query_filter(&filter).expect("Failed to parse AND");
        if let ASTExpr::And(subs) = ast {
            assert_eq!(subs.len(), 2);
        } else {
            panic!("Expected AND, got {:?}", ast);
        }

        // Test parsing an OR
        let filter = json!({"$or": [{"a": {"$eq": 1}}, {"b": {"$eq": 2}}]});
        let ast = parse_query_filter(&filter).expect("Failed to parse OR");
        if let ASTExpr::Or(subs) = ast {
            assert_eq!(subs.len(), 2);
        } else {
            panic!("Expected OR, got {:?}", ast);
        } // Test parsing a NOT
        let filter = json!({"$not": {"a": {"$eq": 1}}});
        let ast = parse_query_filter(&filter).expect("Failed to parse NOT");
        assert!(matches!(ast, ASTExpr::Not(_)));
    }
    #[test]
    fn test_implicit_and() {
        // Multiple fields should be treated as an implicit AND
        let filter = json!({"a": {"$eq": 1}, "b": {"$eq": 2}});
        let ast = parse_query_filter(&filter).expect("Failed to parse implicit AND");
        if let ASTExpr::And(subs) = ast {
            assert_eq!(subs.len(), 2);
        } else {
            panic!("Expected implicit AND, got {:?}", ast);
        }
    }

    #[test]
    #[ignore] // TODO: This test uses $in operator which is no longer supported
    fn test_nested_expressions() {
        // Test deeply nested logical operations
        let filter = json!({
            "$and": [
                {
                    "$or": [
                        {"a": {"$eq": 1}},
                        {"b": {"$gt": 2}}
                    ]
                },
                {
                    "$not": {"c": {"$lt": 3}}
                },
                {
                    "d": {"$in": [4, 5, 6]},
                    "e": {"$eq": "test"}
                }
            ]
        });
        let ast = parse_query_filter(&filter).unwrap();

        // Verify the outer AND structure
        if let ASTExpr::And(and_exprs) = ast {
            assert_eq!(and_exprs.len(), 3);

            // First branch: OR expression
            if let ASTExpr::Or(or_exprs) = &and_exprs[0] {
                assert_eq!(or_exprs.len(), 2);

                // First OR branch: a = 1
                if let ASTExpr::Compare { field, op } = &or_exprs[0] {
                    assert_eq!(field, "a");
                    if let CompareOp::Eq(value) = op {
                        assert_eq!(*value, json!(1));
                    } else {
                        panic!("Expected CompareOp::Eq, got {:?}", op);
                    }
                } else {
                    panic!("Expected Compare, got {:?}", or_exprs[0]);
                }

                // Second OR branch: b > 2
                if let ASTExpr::Compare { field, op } = &or_exprs[1] {
                    assert_eq!(field, "b");
                    if let CompareOp::Gt(value) = op {
                        assert_eq!(*value, 2.0);
                    } else {
                        panic!("Expected CompareOp::Gt, got {:?}", op);
                    }
                } else {
                    panic!("Expected Compare, got {:?}", or_exprs[1]);
                }
            } else {
                panic!("Expected OR, got {:?}", and_exprs[0]);
            }

            // Second branch: NOT expression
            if let ASTExpr::Not(not_expr) = &and_exprs[1] {
                if let ASTExpr::Compare { field, op } = &**not_expr {
                    assert_eq!(field, "c");
                    if let CompareOp::Lt(value) = op {
                        assert_eq!(*value, 3.0);
                    } else {
                        panic!("Expected CompareOp::Lt, got {:?}", op);
                    }
                } else {
                    panic!("Expected Compare inside NOT, got {:?}", not_expr);
                }
            } else {
                panic!("Expected NOT, got {:?}", and_exprs[1]);
            }

            // Third branch: Implicit AND of d IN [4,5,6] and e = "test"
            // COMMENTED OUT: $in operator is no longer supported
            /*
            if let ASTExpr::And(implicit_and) = &and_exprs[2] {
                assert_eq!(implicit_and.len(), 2);

                // First implicit AND branch: d IN [4,5,6]
                if let ASTExpr::Compare { field, op } = &implicit_and[0] {
                    assert_eq!(field, "d");
                    if let CompareOp::In(values) = op {
                        assert_eq!(values, &[json!(4), json!(5), json!(6)]);
                    } else {
                        panic!("Expected CompareOp::In, got {:?}", op);
                    }
                } else {
                    panic!("Expected Compare for 'd', got {:?}", implicit_and[0]);
                }

                // Second implicit AND branch: e = "test"
                if let ASTExpr::Compare { field, op } = &implicit_and[1] {
                    assert_eq!(field, "e");
                    if let CompareOp::Eq(value) = op {
                        assert_eq!(*value, json!("test"));
                    } else {
                        panic!("Expected CompareOp::Eq, got {:?}", op);
                    }
                } else {
                    panic!("Expected Compare for 'e', got {:?}", implicit_and[1]);
                }
            } else {
                panic!("Expected implicit AND, got {:?}", and_exprs[2]);
            }
            */
        } else {
            panic!("Expected AND, got {:?}", ast);
        }
    }
    #[test]
    /// Test maximum nesting depth of 2
    fn test_max_nested_expressions() {
        // Test deeply nested logical operations
        let filter = json!({
            "$and": [
                {
                    "$or": [
                        {"$and": [{"a": {"$eq": 1}},{"c": {"$lt": 3}}]},
                        {"b": {"$gt": 2}}
                    ]
                },
                {
                    "$not": {"c": {"$lt": 3}}
                },
                {
                    "d": {"$in": [4, 5, 6]},
                    "e": {"$eq": "test"}
                }
            ]
        });

        let result = parse_query_filter(&filter);
        assert!(
            result.is_err(),
            "Expected error for depth limit, got: {:?}",
            result
        );

        match result {
            Err(QueryFilterError::NestingTooDeep { max_depth }) => {
                assert_eq!(max_depth, ALLOWED_DEPTH_LIMIT);
            }
            _ => panic!("Expected NestingTooDeep error, got: {:?}", result),
        }
    }
    #[test]
    fn test_unsupport_logcial_opreator() {
        // Test deeply nested logical operations
        let filter = json!({
            "$any": [
                {
                    "$or": [
                        {"a": {"$eq": 1}},
                        {"c": {"$lt": 3}},
                        {"b": {"$gt": 2}}
                    ]
                },
                {
                    "$not": {"c": {"$lt": 3}}
                }
            ]
        });

        let result = parse_query_filter(&filter);
        assert!(
            result.is_err(),
            "Expected error for unsupported logical operator, got: {:?}",
            result
        );

        match result {
            Err(QueryFilterError::UnsupportedLogicalOperator(op)) => {
                assert_eq!(op, "$any");
            }
            _ => panic!(
                "Expected UnsupportedLogicalOperator error, got: {:?}",
                result
            ),
        }
    }
    #[test]
    fn test_unsupport_comparison_opreator() {
        // Test deeply nested logical operations
        let filter = json!({
            "$and": [
                {
                    "$or": [
                        {"a": {"$cmp": 1}},
                        {"c": {"$lt": 3}},
                        {"b": {"$gt": 2}}
                    ]
                },
                {
                    "$not": {"c": {"$lt": 3}}
                }
            ]
        });

        let result = parse_query_filter(&filter);
        assert!(
            result.is_err(),
            "Expected error for unsupported comparison operator, got: {:?}",
            result
        );

        match result {
            Err(QueryFilterError::UnsupportedComparisonOperator(op)) => {
                assert_eq!(op, "$cmp");
            }
            _ => panic!(
                "Expected UnsupportedComparisonOperator error, got: {:?}",
                result
            ),
        }
    }

    #[test]
    fn test_complex_nested_paths() {
        // Test query with nested field paths
        let filter = json!({
            "$and": [
                {"user.profile.age": {"$gte": 18}},
                {"user.settings.preferences.theme": {"$eq": "dark"}}
            ]
        });

        let ast = parse_query_filter(&filter).unwrap();

        if let ASTExpr::And(and_exprs) = ast {
            assert_eq!(and_exprs.len(), 2);

            // First expression: user.profile.age >= 18
            if let ASTExpr::Compare { field, op } = &and_exprs[0] {
                assert_eq!(field, "user.profile.age");
                if let CompareOp::Gte(value) = op {
                    assert_eq!(*value, 18.0);
                } else {
                    panic!("Expected CompareOp::Gte, got {:?}", op);
                }
            } else {
                panic!("Expected Compare, got {:?}", and_exprs[0]);
            }

            // Second expression: user.settings.preferences.theme = "dark"
            if let ASTExpr::Compare { field, op } = &and_exprs[1] {
                assert_eq!(field, "user.settings.preferences.theme");
                if let CompareOp::Eq(value) = op {
                    assert_eq!(*value, json!("dark"));
                } else {
                    panic!("Expected CompareOp::Eq, got {:?}", op);
                }
            } else {
                panic!("Expected Compare, got {:?}", and_exprs[1]);
            }
        } else {
            panic!("Expected AND, got {:?}", ast);
        }
    }

    #[test]
    fn test_multiple_operators_on_same_field() {
        // Test when a field has multiple comparison operators
        let filter = json!({
            "age": {
                "$gte": 18,
                "$lt": 65
            }
        });

        let ast = parse_query_filter(&filter).unwrap();

        if let ASTExpr::And(and_exprs) = ast {
            assert_eq!(and_exprs.len(), 2);

            // Check both comparisons exist (order may vary)
            let has_gte = and_exprs.iter().any(|expr| {
                if let ASTExpr::Compare { field, op } = expr {
                    if field == "age" {
                        if let CompareOp::Gte(value) = op {
                            return *value == 18.0;
                        }
                    }
                }
                false
            });

            let has_lt = and_exprs.iter().any(|expr| {
                if let ASTExpr::Compare { field, op } = expr {
                    if field == "age" {
                        if let CompareOp::Lt(value) = op {
                            return *value == 65.0;
                        }
                    }
                }
                false
            });

            assert!(has_gte, "Missing age >= 18 condition");
            assert!(has_lt, "Missing age < 65 condition");
        } else {
            panic!("Expected AND, got {:?}", ast);
        }
    }

    #[test]
    fn test_nesting_depth_limit() {
        // This query has 3 levels of nesting which exceeds our limit of 2
        let filter = json!({
            "$and": [
                {
                    "$or": [
                        {
                            "$and": [  // This is at depth 3
                                {"a": {"$eq": 1}},
                                {"b": {"$eq": 2}}
                            ]
                        },
                        {"c": {"$eq": 3}}
                    ]
                },
                {"d": {"$eq": 4}}
            ]
        });

        // The parser should return an error for exceeding depth
        let result = parse_query_filter(&filter);
        assert!(
            result.is_err(),
            "Expected error for depth limit, got: {:?}",
            result
        );

        match result {
            Err(QueryFilterError::NestingTooDeep { max_depth }) => {
                assert_eq!(max_depth, ALLOWED_DEPTH_LIMIT);
            }
            _ => panic!("Expected NestingTooDeep error, got: {:?}", result),
        }
    }

    #[test]
    fn test_unsupported_logical_operator() {
        // Test with an unsupported logical operator
        let filter = json!({
            "$invalid": [{"a": {"$eq": 1}}]
        });

        let result = parse_query_filter(&filter);
        assert!(
            result.is_err(),
            "Expected error for unsupported logical operator, got: {:?}",
            result
        );

        match result {
            Err(QueryFilterError::UnsupportedLogicalOperator(op)) => {
                assert_eq!(op, "$invalid");
            }
            _ => panic!(
                "Expected UnsupportedLogicalOperator error, got: {:?}",
                result
            ),
        }
    }

    #[test]
    fn test_unsupported_comparison_operator() {
        // Test with an unsupported comparison operator
        let filter = json!({
            "a": {"$invalid": 1}
        });

        let result = parse_query_filter(&filter);
        assert!(
            result.is_err(),
            "Expected error for unsupported comparison operator, got: {:?}",
            result
        );

        match result {
            Err(QueryFilterError::UnsupportedComparisonOperator(op)) => {
                assert_eq!(op, "$invalid");
            }
            _ => panic!(
                "Expected UnsupportedComparisonOperator error, got: {:?}",
                result
            ),
        }
    }

    #[test]
    fn test_empty_filter() {
        // Test with an empty filter object
        let filter = json!({});

        let result = parse_query_filter(&filter);
        assert!(
            result.is_err(),
            "Expected error for empty filter, got: {:?}",
            result
        );

        match result {
            Err(QueryFilterError::ParseFailure(_)) => {} // We expect some kind of parse failure
            _ => panic!("Expected ParseFailure error, got: {:?}", result),
        }
    }

    #[test]
    fn test_non_object_filter() {
        // Test with a non-object filter
        let filter = json!(123);

        let result = parse_query_filter(&filter);
        assert!(
            result.is_err(),
            "Expected error for non-object filter, got: {:?}",
            result
        );

        match result {
            Err(QueryFilterError::ParseFailure(msg)) => {
                assert!(msg.contains("Not a JSON object"));
            }
            _ => panic!("Expected ParseFailure error, got: {:?}", result),
        }
    }
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use serde_json::Value;

use crate::{
    parser::ast::{ASTExpr, CompareOp},
    parser::query_parser::get_value_by_path,
};

/// Evaluate a QueryExpr AST against a label (serde_json::Value)
/// Returns true if the label matches the filter, false otherwise.
pub fn eval_query_expr(expr: &ASTExpr, label: &Value) -> bool {
    match expr {
        ASTExpr::And(subs) => subs.iter().all(|e| eval_query_expr(e, label)),
        ASTExpr::Or(subs) => subs.iter().any(|e| eval_query_expr(e, label)),
        ASTExpr::Not(sub) => !eval_query_expr(sub, label),
        ASTExpr::Compare { field, op } => {
            if let Some(field_val) = get_value_by_path(label, field) {
                match op {
                    CompareOp::Eq(value) => field_val == value,
                    CompareOp::Ne(value) => field_val != value,
                    CompareOp::Lt(num) => {
                        if let Some(f1) = field_val.as_f64() {
                            f1 < *num
                        } else {
                            false
                        }
                    }
                    CompareOp::Lte(num) => {
                        if let Some(f1) = field_val.as_f64() {
                            f1 <= *num
                        } else {
                            false
                        }
                    }
                    CompareOp::Gt(num) => {
                        if let Some(f1) = field_val.as_f64() {
                            f1 > *num
                        } else {
                            false
                        }
                    }
                    CompareOp::Gte(num) => {
                        if let Some(f1) = field_val.as_f64() {
                            f1 >= *num
                        } else {
                            false
                        }
                    }
                }
            } else {
                false // Field not found
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::parser::query_parser::parse_query_filter;
    #[test]
    fn test_and_or_not_compare() {
        // Test a complex filter with AND, OR, NOT, and dot notation
        let label = json!({
            "a": 1,
            "b": 2,
            "c": 3,
            "specs": { "cpu": "i7" },
            "arr": [1,2,3]
        });
        let filter = json!({"$and": [
            {"a": {"$eq": 1}},
            {"b": {"$gt": 1}},
            {"specs.cpu": {"$eq": "i7"}},
            {"$or": [
                {"c": {"$lt": 2}},
                {"c": {"$gte": 3}}
            ]},
            {"b": {"$eq": 2}}
        ]});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));
    }
    #[test]
    fn test_not() {
        // Test $not operator
        let label = json!({"x": 5});
        let filter = json!({"$not": {"x": {"$eq": 3}}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));
    }

    #[test]
    fn test_compare_fail() {
        // Test a failing comparison
        let label = json!({"foo": 1});
        let filter = json!({"foo": {"$gt": 10}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(!eval_query_expr(&ast, &label));
    }
    #[test]
    #[ignore] // TODO: $in operator removed - need to redesign test
    fn test_in_with_array_field() {
        // $in with array field value - NO LONGER SUPPORTED
        let label = json!({"tags": ["a", "b", "c"]});
        let filter = json!({"tags": {"$in": ["b", "x"]}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));
    }

    #[test]
    #[ignore] // TODO: $nin operator removed - need to redesign test
    fn test_nin_with_array_field() {
        // $nin with array field value - NO LONGER SUPPORTED
        let label = json!({"tags": ["a", "b", "c"]});
        let filter = json!({"tags": {"$nin": ["x", "y"]}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));
    }

    #[test]
    fn test_dot_notation_missing_field() {
        // Dot notation with missing field
        let label = json!({"specs": {}});
        let filter = json!({"specs.cpu": {"$eq": "i7"}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(!eval_query_expr(&ast, &label));
    }

    #[test]
    fn test_multiple_field_and() {
        // Implicit AND for multiple fields
        let label = json!({"a": 1, "b": 2});
        let filter = json!({"a": {"$eq": 1}, "b": {"$eq": 2}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));
    }
    #[test]
    fn test_ne_operator() {
        // $ne operator
        let label = json!({"foo": 5});
        let filter = json!({"foo": {"$ne": 3}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));
    }

    #[test]
    fn test_all_compare_ops() {
        let label = json!({
            "int": 5,
            "flt": 3.5,
            "str": "abc",
            "arr": [1, 2, 3],
        });
        // $eq
        let filter = json!({"int": {"$eq": 5}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));
        // $ne
        let filter = json!({"int": {"$ne": 3}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));
        // $lt
        let filter = json!({"flt": {"$lt": 4.0}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));
        // $lte
        let filter = json!({"flt": {"$lte": 3.5}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));
        // $gt
        let filter = json!({"int": {"$gt": 4}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));
        // $gte
        let filter = json!({"int": {"$gte": 5}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));
        // NOTE: $in and $nin operators removed - using $eq instead
        // Test $eq (scalar)
        let filter = json!({"int": {"$eq": 5}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));
        // Test $ne (scalar)
        let filter = json!({"int": {"$ne": 1}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));
    }
    #[test]
    fn test_all_logical_ops() {
        let label = json!({"a": 1, "b": 2, "c": 3});
        // $and
        let filter = json!({"$and": [
            {"a": {"$eq": 1}},
            {"b": {"$eq": 2}}
        ]});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));
        // $or
        let filter = json!({"$or": [
            {"a": {"$eq": 2}},
            {"b": {"$eq": 2}}
        ]});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));
        // $not
        let filter = json!({"$not": {"c": {"$eq": 4}}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));
        // Nested logical
        let filter = json!({"$and": [
            {"$or": [
                {"a": {"$eq": 2}},
                {"b": {"$eq": 2}}
            ]},
            {"$not": {"c": {"$eq": 4}}}
        ]});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));
    }
    #[test]
    fn test_numeric_comparison_edge_cases() {
        // Test label with various numeric types and non-numeric values
        let label = json!({
            "integer": 10,
            "float": 10.5,
            "zero": 0,
            "negative": -5,
            "string": "10",
            "bool": true,
            "null": null,
            "array": [1, 2, 3],
            "object": {"value": 10}
        });

        // Test integer equality boundary
        let filter = json!({"integer": {"$eq": 10}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));

        // Test float equality
        let filter = json!({"float": {"$eq": 10.5}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));

        // Test integer to float comparison (equal values)
        // Note: In serde_json, integer 10 is not exactly equal to float 10.0 for $eq
        let filter = json!({"integer": {"$eq": 10.0}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(!eval_query_expr(&ast, &label));

        // Test less than with integers
        let filter = json!({"integer": {"$lt": 11}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));

        // Test less than with boundary (should be false)
        let filter = json!({"integer": {"$lt": 10}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(!eval_query_expr(&ast, &label));

        // Test less than or equal with boundary
        let filter = json!({"integer": {"$lte": 10}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));

        // Test greater than with integers
        let filter = json!({"integer": {"$gt": 9}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));

        // Test greater than with boundary (should be false)
        let filter = json!({"integer": {"$gt": 10}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(!eval_query_expr(&ast, &label));

        // Test greater than or equal with boundary
        let filter = json!({"integer": {"$gte": 10}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));

        // Test negative numbers
        let filter = json!({"negative": {"$lt": 0}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));

        // Test zero
        let filter = json!({"zero": {"$eq": 0}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));

        // Test string (should not parse as number, comparison should fail)
        let filter = json!({"string": {"$gt": 5}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(!eval_query_expr(&ast, &label));

        // Test boolean (should not parse as number, comparison should fail)
        let filter = json!({"bool": {"$lt": 5}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(!eval_query_expr(&ast, &label));

        // Test null (should not parse as number, comparison should fail)
        let filter = json!({"null": {"$gte": 0}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(!eval_query_expr(&ast, &label));

        // Test array (should not parse as number, comparison should fail)
        let filter = json!({"array": {"$lte": 5}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(!eval_query_expr(&ast, &label));

        // Test object (should not parse as number, comparison should fail)
        let filter = json!({"object": {"$lt": 15}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(!eval_query_expr(&ast, &label));
    }
    #[test]
    fn test_mixed_type_comparisons() {
        // Test label with various numeric types
        let label = json!({
            "integer": 10,
            "float": 10.5,
            "string_num": "10",
            "string_text": "hello",
            "large_integer": 9007199254740991i64,  // 2^53 - 1 (max safe integer in JS)
            "scientific": 1.23e2,                // 123 in scientific notation
        });

        // Integer vs float comparison
        let filter = json!({"integer": {"$lt": 10.5}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));

        let filter = json!({"float": {"$gt": 10}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));

        // Equality with different representations - for numeric equality, type matters
        // Scientific notation 1.23e2 is stored as a float (123.0), not equal to integer 123
        let filter = json!({"scientific": {"$eq": 123}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(!eval_query_expr(&ast, &label));

        // String numbers don't get converted to numbers
        let filter = json!({"string_num": {"$eq": 10}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(!eval_query_expr(&ast, &label));

        let filter = json!({"string_num": {"$lt": 11}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(!eval_query_expr(&ast, &label));

        // Large integers are preserved correctly
        let filter = json!({"large_integer": {"$eq": 9007199254740991i64}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));

        // Comparing with exact precision boundaries
        let filter = json!({"float": {"$gt": 10.499999}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));

        let filter = json!({"float": {"$lt": 10.500001}});
        let ast = parse_query_filter(&filter).expect("Failed to parse filter");
        assert!(eval_query_expr(&ast, &label));
    }
}

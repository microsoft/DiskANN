/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Evaluator for the query expression AST

use serde_json::Value;
use crate::ast::{QueryExpr, CompareOp};
use crate::parser::get_value_by_path;

/// Evaluate a QueryExpr AST against a label (serde_json::Value)
/// Returns true if the label matches the filter, false otherwise.
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
                    CompareOp::Lt => field_val.as_f64() < value.as_f64(),
                    CompareOp::Lte => field_val.as_f64() <= value.as_f64(),
                    CompareOp::Gt => field_val.as_f64() > value.as_f64(),
                    CompareOp::Gte => field_val.as_f64() >= value.as_f64(),
                    CompareOp::In => {
                        if let Some(arr) = value.as_array() {
                            match field_val {
                                Value::Array(field_arr) => field_arr.iter().any(|v| arr.contains(v)),
                                _ => arr.contains(field_val),
                            }
                        } else {
                            false
                        }
                    }
                    CompareOp::Nin => {
                        if let Some(arr) = value.as_array() {
                            match field_val {
                                Value::Array(field_arr) => field_arr.iter().all(|v| !arr.contains(v)),
                                _ => !arr.contains(field_val),
                            }
                        } else {
                            true
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
    use super::*;
    use serde_json::json;
    use crate::parser::parse_query_filter;

    #[test]
    fn test_and_or_not_compare() {
        // Test a complex filter with AND, OR, NOT, dot notation, and $in
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
            {"arr": {"$in": [2,3]}}
        ]});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&ast, &label));
    }

    #[test]
    fn test_not() {
        // Test $not operator
        let label = json!({"x": 5});
        let filter = json!({"$not": {"x": {"$eq": 3}}});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&ast, &label));
    }

    #[test]
    fn test_compare_fail() {
        // Test a failing comparison
        let label = json!({"foo": 1});
        let filter = json!({"foo": {"$gt": 10}});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(!eval_query_expr(&ast, &label));
    }

    #[test]
    fn test_in_with_array_field() {
        // $in with array field value
        let label = json!({"tags": ["a", "b", "c"]});
        let filter = json!({"tags": {"$in": ["b", "x"]}});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&ast, &label));
    }

    #[test]
    fn test_nin_with_array_field() {
        // $nin with array field value
        let label = json!({"tags": ["a", "b", "c"]});
        let filter = json!({"tags": {"$nin": ["x", "y"]}});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&ast, &label));
    }

    #[test]
    fn test_dot_notation_missing_field() {
        // Dot notation with missing field
        let label = json!({"specs": {}});
        let filter = json!({"specs.cpu": {"$eq": "i7"}});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(!eval_query_expr(&ast, &label));
    }

    #[test]
    fn test_multiple_field_and() {
        // Implicit AND for multiple fields
        let label = json!({"a": 1, "b": 2});
        let filter = json!({"a": {"$eq": 1}, "b": {"$eq": 2}});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&ast, &label));
    }

    #[test]
    fn test_ne_operator() {
        // $ne operator
        let label = json!({"foo": 5});
        let filter = json!({"foo": {"$ne": 3}});
        let ast = parse_query_filter(&filter).unwrap();
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
        let ast = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&ast, &label));
        // $ne
        let filter = json!({"int": {"$ne": 3}});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&ast, &label));
        // $lt
        let filter = json!({"flt": {"$lt": 4.0}});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&ast, &label));
        // $lte
        let filter = json!({"flt": {"$lte": 3.5}});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&ast, &label));
        // $gt
        let filter = json!({"int": {"$gt": 4}});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&ast, &label));
        // $gte
        let filter = json!({"int": {"$gte": 5}});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&ast, &label));
        // $in (scalar)
        let filter = json!({"int": {"$in": [4,5,6]}});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&ast, &label));
        // $nin (scalar)
        let filter = json!({"int": {"$nin": [1,2,3]}});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&ast, &label));
        // $in (array field)
        let filter = json!({"arr": {"$in": [2,4]}});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&ast, &label));
        // $nin (array field)
        let filter = json!({"arr": {"$nin": [4,5]}});
        let ast = parse_query_filter(&filter).unwrap();
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
        let ast = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&ast, &label));
        // $or
        let filter = json!({"$or": [
            {"a": {"$eq": 2}},
            {"b": {"$eq": 2}}
        ]});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&ast, &label));
        // $not
        let filter = json!({"$not": {"c": {"$eq": 4}}});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&ast, &label));
        // Nested logical
        let filter = json!({"$and": [
            {"$or": [
                {"a": {"$eq": 2}},
                {"b": {"$eq": 2}}
            ]},
            {"$not": {"c": {"$eq": 4}}}
        ]});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&ast, &label));
    }
}

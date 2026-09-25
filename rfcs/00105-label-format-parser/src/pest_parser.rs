/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use pest::Parser;
use pest_derive::Parser;
use serde_json::Value;
use crate::{QueryExpr, CompareOp};
use std::str::FromStr;

#[derive(Parser)]
#[grammar = "query_filter.pest"]
struct QueryFilterParser;

/// Convert a string representation of the filter into a QueryExpr AST
pub fn parse_query_filter_str(input: &str) -> Result<QueryExpr, Box<dyn std::error::Error>> {
    use pest::iterators::Pair;
    
    // Parse the input using the grammar
    let pairs = QueryFilterParser::parse(Rule::query, input)?;
    
    // Helper function to parse a rule into a QueryExpr
    fn parse_pair(pair: Pair<Rule>, current_field: Option<&str>) -> Result<QueryExpr, Box<dyn std::error::Error>> {
        match pair.as_rule() {
            Rule::query => {
                // Query is either an object or logical_top
                let inner = pair.into_inner().next().unwrap();
                parse_pair(inner, None)
            }
            Rule::logical_top => {
                // Top-level logical operation
                let inner = pair.into_inner().next().unwrap(); // logical_op_pair
                parse_pair(inner, None)
            }
            Rule::logical_op_pair => {
                let mut inner_pairs = pair.into_inner();
                let op_key = inner_pairs.next().unwrap();
                let op_str = op_key.as_str();
                let value_expr = inner_pairs.next().unwrap();
                
                match op_str {
                    "\"$and\"" => {
                        if value_expr.as_rule() == Rule::array_of_queries {
                            // Process AND with array of queries
                            let mut exprs = Vec::new();
                            for expr_pair in value_expr.into_inner() {
                                exprs.push(parse_pair(expr_pair, None)?);
                            }
                            Ok(QueryExpr::And(exprs))
                        } else {
                            // Single query
                            let expr = parse_pair(value_expr, None)?;
                            Ok(QueryExpr::And(vec![expr]))
                        }
                    },
                    "\"$or\"" => {
                        if value_expr.as_rule() == Rule::array_of_queries {
                            // Process OR with array of queries
                            let mut exprs = Vec::new();
                            for expr_pair in value_expr.into_inner() {
                                exprs.push(parse_pair(expr_pair, None)?);
                            }
                            Ok(QueryExpr::Or(exprs))
                        } else {
                            // Single query
                            let expr = parse_pair(value_expr, None)?;
                            Ok(QueryExpr::Or(vec![expr]))
                        }
                    },
                    "\"$not\"" => {
                        // Process NOT with a single query
                        let expr = parse_pair(value_expr, None)?;
                        Ok(QueryExpr::Not(Box::new(expr)))
                    },
                    _ => Err(format!("Unknown logical operator: {}", op_str).into()),
                }
            }
            Rule::object => {
                // Object with multiple fields (implicit AND)
                let mut exprs = Vec::new();
                for pair in pair.into_inner() {
                    if pair.as_rule() == Rule::field_pair {
                        let mut inner_pairs = pair.into_inner();
                        let field_expr = inner_pairs.next().unwrap();
                        let value_expr = inner_pairs.next().unwrap();
                        
                        // Field is a string, remove quotes
                        let field = field_expr.as_str();
                        let field = field[1..field.len()-1].to_string();
                        
                        match value_expr.as_rule() {
                            Rule::compare_obj => {
                                let sub_expr = parse_pair(value_expr, Some(&field))?;
                                exprs.push(sub_expr);
                            },
                            Rule::value => {
                                // If it's a simple value, treat it as an equality comparison
                                let value = parse_value(value_expr)?;
                                exprs.push(QueryExpr::Compare {
                                    field,
                                    op: CompareOp::Eq,
                                    value,
                                });
                            },
                            _ => return Err(format!("Invalid value expression: {:?}", value_expr.as_rule()).into()),
                        }
                    }
                }
                
                // If there's only one expression, no need for AND
                if exprs.len() == 1 {
                    Ok(exprs.remove(0))
                } else {
                    Ok(QueryExpr::And(exprs))
                }
            }            Rule::logical_expr => {
                let inner = pair.into_inner().next().unwrap();
                
                match inner.as_rule() {
                    Rule::and_expr => {
                        let mut inner_pairs = inner.into_inner();
                        let _ = inner_pairs.next(); // Skip the "$and" string
                        let array = inner_pairs.next().unwrap();
                        
                        let mut exprs = Vec::new();
                        for expr_pair in array.into_inner() {
                            exprs.push(parse_pair(expr_pair, None)?);
                        }
                        Ok(QueryExpr::And(exprs))
                    },
                    Rule::or_expr => {
                        let mut inner_pairs = inner.into_inner();
                        let _ = inner_pairs.next(); // Skip the "$or" string
                        let array = inner_pairs.next().unwrap();
                        
                        let mut exprs = Vec::new();
                        for expr_pair in array.into_inner() {
                            exprs.push(parse_pair(expr_pair, None)?);
                        }
                        Ok(QueryExpr::Or(exprs))
                    },
                    Rule::not_expr => {
                        let mut inner_pairs = inner.into_inner();
                        let _ = inner_pairs.next(); // Skip the "$not" string
                        let expr = parse_pair(inner_pairs.next().unwrap(), None)?;
                        Ok(QueryExpr::Not(Box::new(expr)))
                    },
                    _ => Err(format!("Unknown logical rule: {:?}", inner.as_rule()).into()),
                }
            }            Rule::compare_obj => {
                let mut inner_pairs = pair.into_inner();
                let op_pair = inner_pairs.next().unwrap();
                let value_pair = inner_pairs.next().unwrap();
                
                // Extract the operator name and remove quotes
                let op_str = op_pair.as_str();
                let op_str = op_str[1..op_str.len()-1].to_string();
                
                let value = parse_value(value_pair)?;
                
                let field = current_field.ok_or("Field name missing for comparison")?.to_string();
                
                // Map operator string to enum
                let op = match op_str.as_str() {
                    "$eq" => CompareOp::Eq,
                    "$ne" => CompareOp::Ne,
                    "$lt" => CompareOp::Lt,
                    "$lte" => CompareOp::Lte,
                    "$gt" => CompareOp::Gt,
                    "$gte" => CompareOp::Gte,
                    "$in" => CompareOp::In,
                    "$nin" => CompareOp::Nin,
                    _ => return Err(format!("Unknown comparison operator: {}", op_str).into()),
                };
                
                Ok(QueryExpr::Compare { field, op, value })
            }
            Rule::array_of_queries => {
                // This should be handled by the calling context (logical_op_pair)
                Err("Unexpected standalone array of queries".into())
            }_ => Err(format!("Unexpected rule: {:?}", pair.as_rule()).into()),
        }
    }
    
    // Helper function to parse values (string, number, boolean, array, null)
    fn parse_value(pair: Pair<Rule>) -> Result<Value, Box<dyn std::error::Error>> {
        match pair.as_rule() {
            Rule::value => {
                let inner = pair.into_inner().next().unwrap();
                parse_value(inner)
            },
            Rule::string => {
                let s = pair.as_str();
                // Remove the quotes
                let s = s[1..s.len()-1].to_string();
                Ok(Value::String(s))
            }
            Rule::number => {
                let s = pair.as_str();
                if s.contains('.') {
                    Ok(Value::Number(serde_json::Number::from_f64(f64::from_str(s)?).unwrap()))
                } else {
                    Ok(Value::Number(serde_json::Number::from(i64::from_str(s)?)))
                }
            }
            Rule::boolean => {
                let b = pair.as_str() == "true";
                Ok(Value::Bool(b))
            }
            Rule::array => {
                let mut values = Vec::new();
                for value_pair in pair.into_inner() {
                    values.push(parse_value(value_pair)?);
                }
                Ok(Value::Array(values))
            }
            Rule::null => Ok(Value::Null),
            Rule::object_value => {
                let mut map = serde_json::Map::new();
                let mut pairs = pair.into_inner();
                
                while let Some(key_pair) = pairs.next() {
                    if key_pair.as_rule() == Rule::string {
                        let key = key_pair.as_str();
                        let key = key[1..key.len()-1].to_string();
                        
                        let value_pair = pairs.next().unwrap();
                        let value = parse_value(value_pair)?;
                        map.insert(key, value);
                    }
                }
                
                Ok(Value::Object(map))
            }
            _ => unreachable!("Unexpected rule in value: {:?}", pair.as_rule()),
        }
    }
      // Get the first (and only) query
    let pair = pairs.peek().unwrap();
    parse_pair(pair, None)
}

/// Parse a filter object (from JSON) into a QueryExpr AST using Pest
/// This function directly converts the JSON to our AST structure by parsing the JSON string
pub fn parse_query_filter(filter: &Value) -> Option<QueryExpr> {
    // Convert to a JSON string and parse with Pest
    let filter_str = serde_json::to_string(filter).ok()?;
    parse_query_filter_str(&filter_str).ok()
}

/// Forward to the main crate's evaluation function
/// We no longer need special handling since we fixed the parser
pub use crate::eval_query_expr;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;    #[test]
    fn test_pest_parser_simple() {        
        // Test a simple filter
        let label = json!({"a": 1});
        let filter = json!({"a": {"$eq": 1}});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&ast, &label));
    }
    
    #[test]
    fn test_pest_parser_logical() {        
        // Test AND
        let label = json!({"a": 1, "b": 2});
        let filter = json!({"$and": [{"a": {"$eq": 1}}, {"b": {"$eq": 2}}]});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&ast, &label));
        
        // Test OR
        let label = json!({"a": 1, "b": 3});
        let filter = json!({"$or": [{"a": {"$eq": 2}}, {"b": {"$eq": 3}}]});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&ast, &label));
        
        // Test NOT
        let label = json!({"a": 1});
        let filter = json!({"$not": {"a": {"$eq": 2}}});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&ast, &label));
    }
      #[test]
    fn test_pest_parser_compare_ops() {        
        let label = json!({
            "int": 5,
            "flt": 3.5,
            "arr": [1, 2, 3]
        });
        
        // Test $eq
        let filter = json!({"int": {"$eq": 5}});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&ast, &label));
        
        // Test $ne
        let filter = json!({"int": {"$ne": 6}});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&ast, &label));
        
        // Test $lt
        let filter = json!({"int": {"$lt": 10}});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&ast, &label));
        
        // Test $in
        let filter = json!({"arr": {"$in": [2, 4]}});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&ast, &label));
    }
      #[test]
    fn test_pest_parser_complex() {        
        // Test a complex filter that our grammar can handle
        let label = json!({
            "a": 1,
            "b": 2,
            "c": 3
        });
        
        let filter = json!({"$and": [
            {"a": {"$eq": 1}},
            {"b": {"$eq": 2}}
        ]});
        let ast = parse_query_filter(&filter).unwrap();
        assert!(eval_query_expr(&ast, &label));
    }
    
    #[test]
    fn test_pest_parser_complex_nested() {        
        let label = json!({
            "a": 1,
            "b": 2,
            "c": 3,
            "specs": { "cpu": "i7" },
            "arr": [1, 2, 3],
            "tags": ["a", "b", "c"]
        });
        
        // Test with a more complex filter structure
        let filter = json!({
            "$and": [
                {"a": {"$eq": 1}},
                {"b": {"$gt": 1}},
                {"specs.cpu": {"$eq": "i7"}},
                {"$or": [
                    {"c": {"$lt": 2}},
                    {"c": {"$gte": 3}}
                ]},
                {"arr": {"$in": [2, 3]}},
                {"tags": {"$in": ["b"]}}
            ]
        });
        
        let ast = parse_query_filter(&filter).unwrap_or_else(|| {
            panic!("Failed to parse complex filter with Pest parser");
        });
        
        // This should evaluate to true
        assert!(eval_query_expr(&ast, &label));
        
        // Now test with a filter that should evaluate to false
        let false_filter = json!({
            "$and": [
                {"a": {"$eq": 1}},
                {"b": {"$gt": 1}},
                {"c": {"$lt": 2}} // This should make it false as c=3
            ]
        });
        
        if let Some(false_ast) = parse_query_filter(&false_filter) {
            assert!(!eval_query_expr(&false_ast, &label));
        } else {
            panic!("Failed to parse simple false filter with Pest parser");
        }
    }
      // New tests that only check parsing without evaluation
    
    #[test]
    fn test_parse_simple_eq() {
        let filter = json!({"a": {"$eq": 1}});
        let filter_str = serde_json::to_string(&filter).unwrap();
        
        // Verify Pest can parse this
        let pairs = QueryFilterParser::parse(Rule::query, &filter_str).expect("Failed to parse simple eq filter");
        assert!(pairs.count() > 0);
        
        // Verify we can convert it to AST
        let ast = parse_query_filter_str(&filter_str).expect("Failed to convert to AST");
        
        match ast {
            QueryExpr::Compare { field, op, value } => {
                assert_eq!(field, "a");
                assert!(matches!(op, CompareOp::Eq));
                assert_eq!(value, json!(1));
            },
            _ => panic!("Expected Compare expression, got {:?}", ast),
        }
    }
      #[test]
    fn test_parse_simple_gt() {
        let filter = json!({"score": {"$gt": 10}});
        let filter_str = serde_json::to_string(&filter).unwrap();
        
        // Verify Pest can parse this
        let pairs = QueryFilterParser::parse(Rule::query, &filter_str).expect("Failed to parse simple gt filter");
        assert!(pairs.count() > 0);
        
        // Verify we can convert it to AST
        let ast = parse_query_filter_str(&filter_str).expect("Failed to convert to AST");
        
        match ast {
            QueryExpr::Compare { field, op, value } => {
                assert_eq!(field, "score");
                assert!(matches!(op, CompareOp::Gt));
                assert_eq!(value, json!(10));
            },
            _ => panic!("Expected Compare expression, got {:?}", ast),
        }
    }
    
    #[test]
    fn test_parse_logical_and() {
        let filter = json!({"$and": [
            {"a": {"$eq": 1}},
            {"b": {"$eq": 2}}
        ]});
        let filter_str = serde_json::to_string(&filter).unwrap();
        
        // Verify Pest can parse this
        let pairs = QueryFilterParser::parse(Rule::query, &filter_str).expect("Failed to parse logical AND filter");
        assert!(pairs.count() > 0);
        
        // Verify we can convert it to AST
        let ast = parse_query_filter_str(&filter_str).expect("Failed to convert to AST");
        
        match ast {
            QueryExpr::And(exprs) => {
                assert_eq!(exprs.len(), 2);
                
                match &exprs[0] {
                    QueryExpr::Compare { field, op, value } => {
                        assert_eq!(field, "a");
                        assert!(matches!(op, CompareOp::Eq));
                        assert_eq!(value, &json!(1));
                    },
                    _ => panic!("Expected Compare expression for first AND item"),
                }
                
                match &exprs[1] {
                    QueryExpr::Compare { field, op, value } => {
                        assert_eq!(field, "b");
                        assert!(matches!(op, CompareOp::Eq));
                        assert_eq!(value, &json!(2));
                    },
                    _ => panic!("Expected Compare expression for second AND item"),
                }
            },
            _ => panic!("Expected AND expression, got {:?}", ast),
        }
    }
    
    #[test]
    fn test_parse_logical_or() {
        let filter = json!({"$or": [
            {"a": {"$eq": 1}},
            {"b": {"$eq": 2}}
        ]});
        let filter_str = serde_json::to_string(&filter).unwrap();
        
        // Verify Pest can parse this
        let pairs = QueryFilterParser::parse(Rule::query, &filter_str).expect("Failed to parse logical OR filter");
        assert!(pairs.count() > 0);
        
        // Verify we can convert it to AST
        let ast = parse_query_filter_str(&filter_str).expect("Failed to convert to AST");
        
        match ast {
            QueryExpr::Or(exprs) => {
                assert_eq!(exprs.len(), 2);
            },
            _ => panic!("Expected OR expression, got {:?}", ast),
        }
    }
    
    #[test]
    fn test_parse_logical_not() {
        let filter = json!({"$not": {"a": {"$eq": 1}}});
        let filter_str = serde_json::to_string(&filter).unwrap();
        
        // Verify Pest can parse this
        let pairs = QueryFilterParser::parse(Rule::query, &filter_str).expect("Failed to parse logical NOT filter");
        assert!(pairs.count() > 0);
        
        // Verify we can convert it to AST
        let ast = parse_query_filter_str(&filter_str).expect("Failed to convert to AST");
        
        match ast {
            QueryExpr::Not(expr) => {
                match *expr {
                    QueryExpr::Compare { field, op, .. } => {
                        assert_eq!(field, "a");
                        assert!(matches!(op, CompareOp::Eq));
                    },
                    _ => panic!("Expected Compare expression inside NOT"),
                }
            },
            _ => panic!("Expected NOT expression, got {:?}", ast),
        }
    }
    
    #[test]
    fn test_parse_in_operator() {
        let filter = json!({"tags": {"$in": ["tag1", "tag2", "tag3"]}});
        let filter_str = serde_json::to_string(&filter).unwrap();
        
        // Verify Pest can parse this
        let pairs = QueryFilterParser::parse(Rule::query, &filter_str).expect("Failed to parse $in operator filter");
        assert!(pairs.count() > 0);
        
        // Verify we can convert it to AST
        let ast = parse_query_filter_str(&filter_str).expect("Failed to convert to AST");
        
        match ast {
            QueryExpr::Compare { field, op, value } => {
                assert_eq!(field, "tags");
                assert!(matches!(op, CompareOp::In));
                assert_eq!(value, json!(["tag1", "tag2", "tag3"]));
            },
            _ => panic!("Expected Compare expression with $in, got {:?}", ast),
        }
    }
    
    #[test]
    fn test_parse_dot_notation() {
        let filter = json!({"user.profile.age": {"$gt": 18}});
        let filter_str = serde_json::to_string(&filter).unwrap();
        
        // Verify Pest can parse this
        let pairs = QueryFilterParser::parse(Rule::query, &filter_str).expect("Failed to parse dot notation filter");
        assert!(pairs.count() > 0);
        
        // Verify we can convert it to AST
        let ast = parse_query_filter_str(&filter_str).expect("Failed to convert to AST");
        
        match ast {
            QueryExpr::Compare { field, op, value } => {
                assert_eq!(field, "user.profile.age");
                assert!(matches!(op, CompareOp::Gt));
                assert_eq!(value, json!(18));
            },
            _ => panic!("Expected Compare expression with dot notation, got {:?}", ast),
        }
    }
    
    #[test]
    fn test_parse_complex_nested_structure() {
        let filter = json!({
            "$and": [
                {"a": {"$eq": 1}},
                {"$or": [
                    {"b": {"$lt": 10}},
                    {"c": {"$gt": 20}}
                ]},
                {"$not": {"d": {"$eq": 30}}}
            ]
        });
        let filter_str = serde_json::to_string(&filter).unwrap();
        
        // Verify Pest can parse this
        let pairs = QueryFilterParser::parse(Rule::query, &filter_str).expect("Failed to parse complex nested filter");
        assert!(pairs.count() > 0);
        
        // Verify we can convert it to AST
        let ast = parse_query_filter_str(&filter_str).expect("Failed to convert to AST");
        
        match ast {
            QueryExpr::And(exprs) => {
                assert_eq!(exprs.len(), 3);
                
                // Check second element which should be an OR
                match &exprs[1] {
                    QueryExpr::Or(or_exprs) => {
                        assert_eq!(or_exprs.len(), 2);
                    },
                    _ => panic!("Expected OR expression as second AND element"),
                }
                
                // Check third element which should be a NOT
                match &exprs[2] {
                    QueryExpr::Not(_) => {},
                    _ => panic!("Expected NOT expression as third AND element"),
                }
            },
            _ => panic!("Expected AND expression, got {:?}", ast),
        }
    }
}

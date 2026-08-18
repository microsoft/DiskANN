/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! AST for query filters

use serde_json::Value;

/// AST for query filters
#[derive(Debug, Clone, PartialEq)]
pub enum QueryExpr {
    /// Logical AND: all sub-expressions must be true
    And(Vec<QueryExpr>),
    /// Logical OR: at least one sub-expression must be true
    Or(Vec<QueryExpr>),
    /// Logical NOT: negates the sub-expression
    Not(Box<QueryExpr>),
    /// Comparison on a field (supports dot notation)
    Compare {
        field: String,
        op: CompareOp,
        value: Value,
    },
}

/// Supported comparison operators
#[derive(Debug, Clone, PartialEq)]
pub enum CompareOp {
    Eq,   // $eq
    Ne,   // $ne
    Lt,   // $lt
    Lte,  // $lte
    Gt,   // $gt
    Gte,  // $gte
    In,   // $in
    Nin,  // $nin
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_query_expr_equality() {
        let expr1 = QueryExpr::Compare {
            field: "test".to_string(),
            op: CompareOp::Eq,
            value: json!(1),
        };
        
        let expr2 = QueryExpr::Compare {
            field: "test".to_string(),
            op: CompareOp::Eq,
            value: json!(1),
        };
        
        let expr3 = QueryExpr::Compare {
            field: "test".to_string(),
            op: CompareOp::Ne,
            value: json!(1),
        };
        
        assert_eq!(expr1, expr2);
        assert_ne!(expr1, expr3);
    }
}

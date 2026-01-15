/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::attribute::AttributeType;
use std::fmt;

/// Why have a [`ASTIdExpr`] when we already have an [`ASTExpr`] enum? Turns
/// out that for a subset of filter queries that do not have relational
/// operators, we can get some performance gains if we encode attributes
/// as integers which converts the cost of a field name + value comparison
/// into the cost of just an integer comparison.
/// But using an ASTExpr as-is would require us to perform attribute->id lookups
/// everytime the ASTExpr is evaluated in the context of a point (i.e, check if
/// the point satisfies the filter expr).
/// Therefore, we convert the ASTExpr into an ASTIdExpr by doing the attribute->
/// integer lookup once at the start of the query, saving the cost of multiple
/// lookups.
#[derive(Debug, Clone, PartialEq)]
pub enum ASTIdExpr<T>
where
    T: AttributeType,
{
    /// Logical AND: all sub-expressions must be true
    And(Vec<ASTIdExpr<T>>),
    /// Logical OR: at least one sub-expression must be true
    Or(Vec<ASTIdExpr<T>>),
    /// Logical NOT: negates the sub-expression
    Not(Box<ASTIdExpr<T>>),
    /// Comparison on a field (supports dot notation)
    Terminal(T),
}

pub trait ASTIdExprVisitorMut<T>
where
    T: AttributeType,
{
    type Output;

    /// Visit an AST expression
    fn visit(&mut self, expr: &ASTIdExpr<T>) -> Self::Output {
        match expr {
            ASTIdExpr::And(exprs) => self.visit_and(exprs),
            ASTIdExpr::Or(exprs) => self.visit_or(exprs),
            ASTIdExpr::Not(expr) => self.visit_not(expr),
            ASTIdExpr::Terminal(id) => self.visit_terminal(id),
        }
    }

    /// Visit an AND expression
    fn visit_and(&mut self, exprs: &[ASTIdExpr<T>]) -> Self::Output;

    /// Visit an OR expression
    fn visit_or(&mut self, exprs: &[ASTIdExpr<T>]) -> Self::Output;

    /// Visit a NOT expression
    fn visit_not(&mut self, expr: &ASTIdExpr<T>) -> Self::Output;

    /// Visit a comparison expression
    fn visit_terminal(&mut self, id: &T) -> Self::Output;
}

pub trait ASTIdExprVisitor<T>
where
    T: AttributeType,
{
    type Output;

    /// Visit an AST expression
    fn visit(&self, expr: &ASTIdExpr<T>) -> Self::Output {
        match expr {
            ASTIdExpr::And(exprs) => self.visit_and(exprs),
            ASTIdExpr::Or(exprs) => self.visit_or(exprs),
            ASTIdExpr::Not(expr) => self.visit_not(expr),
            ASTIdExpr::Terminal(id) => self.visit_terminal(id),
        }
    }

    /// Visit an AND expression
    fn visit_and(&self, exprs: &[ASTIdExpr<T>]) -> Self::Output;

    /// Visit an OR expression
    fn visit_or(&self, exprs: &[ASTIdExpr<T>]) -> Self::Output;

    /// Visit a NOT expression
    fn visit_not(&self, expr: &ASTIdExpr<T>) -> Self::Output;

    /// Visit a comparison expression
    fn visit_terminal(&self, id: &T) -> Self::Output;
}

impl<T> ASTIdExpr<T>
where
    T: AttributeType,
{
    /// Accept a mutable visitor and return its output
    pub fn accept_mut<V: ASTIdExprVisitorMut<T>>(&self, visitor: &mut V) -> V::Output {
        visitor.visit(self)
    }

    ///Accept an immutable visitor and return its output.
    pub fn accept<V: ASTIdExprVisitor<T>>(&self, visitor: &V) -> V::Output {
        visitor.visit(self)
    }
}

struct PrintVisitor {}

impl PrintVisitor {
    fn new() -> Self {
        Self {}
    }
}

impl<T> ASTIdExprVisitorMut<T> for PrintVisitor
where
    T: AttributeType + fmt::Display,
{
    type Output = String;

    fn visit_and(&mut self, exprs: &[ASTIdExpr<T>]) -> Self::Output {
        if exprs.is_empty() {
            return "true".to_string();
        }

        let sub_exprs: Vec<String> = exprs.iter().map(|expr| self.visit(expr)).collect();

        if sub_exprs.len() == 1 {
            sub_exprs[0].clone()
        } else {
            format!("({})", sub_exprs.join(" AND "))
        }
    }

    fn visit_or(&mut self, exprs: &[ASTIdExpr<T>]) -> Self::Output {
        if exprs.is_empty() {
            return "false".to_string();
        }

        let sub_exprs: Vec<String> = exprs.iter().map(|expr| self.visit(expr)).collect();

        if sub_exprs.len() == 1 {
            sub_exprs[0].clone()
        } else {
            format!("({})", sub_exprs.join(" OR "))
        }
    }

    fn visit_not(&mut self, expr: &ASTIdExpr<T>) -> Self::Output {
        let inner = self.visit(expr);
        format!("NOT ({})", inner)
    }

    fn visit_terminal(&mut self, id: &T) -> Self::Output {
        format!("label_id:{}", id)
    }
}

impl<T> fmt::Display for ASTIdExpr<T>
where
    T: AttributeType + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut visitor = PrintVisitor::new();
        let result = self.accept_mut(&mut visitor);
        write!(f, "{}", result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_print_visitor_terminal() {
        let expr = ASTIdExpr::Terminal(42u64);
        let result = expr.to_string();
        assert_eq!(result, "label_id:42");
    }

    #[test]
    fn test_print_visitor_single_and() {
        let expr = ASTIdExpr::And(vec![ASTIdExpr::Terminal(5u64)]);
        let result = expr.to_string();
        assert_eq!(result, "label_id:5");
    }

    #[test]
    fn test_print_visitor_multiple_and() {
        let expr = ASTIdExpr::And(vec![
            ASTIdExpr::Terminal(1u64),
            ASTIdExpr::Terminal(2u64),
            ASTIdExpr::Terminal(3u64),
        ]);
        let result = expr.to_string();
        assert_eq!(result, "(label_id:1 AND label_id:2 AND label_id:3)");
    }

    #[test]
    fn test_print_visitor_empty_and() {
        let expr: ASTIdExpr<u64> = ASTIdExpr::And(vec![]);
        let result = expr.to_string();
        assert_eq!(result, "true");
    }

    #[test]
    fn test_print_visitor_single_or() {
        let expr = ASTIdExpr::Or(vec![ASTIdExpr::Terminal(10u64)]);
        let result = expr.to_string();
        assert_eq!(result, "label_id:10");
    }

    #[test]
    fn test_print_visitor_multiple_or() {
        let expr = ASTIdExpr::Or(vec![ASTIdExpr::Terminal(7u64), ASTIdExpr::Terminal(8u64)]);
        let result = expr.to_string();
        assert_eq!(result, "(label_id:7 OR label_id:8)");
    }

    #[test]
    fn test_print_visitor_empty_or() {
        let expr: ASTIdExpr<u64> = ASTIdExpr::Or(vec![]);
        let result = expr.to_string();
        assert_eq!(result, "false");
    }

    #[test]
    fn test_print_visitor_not() {
        let expr = ASTIdExpr::Not(Box::new(ASTIdExpr::Terminal(15u64)));
        let result = expr.to_string();
        assert_eq!(result, "NOT (label_id:15)");
    }

    #[test]
    fn test_print_visitor_nested_and_or() {
        let expr = ASTIdExpr::And(vec![
            ASTIdExpr::Terminal(1u64),
            ASTIdExpr::Or(vec![ASTIdExpr::Terminal(2u64), ASTIdExpr::Terminal(3u64)]),
        ]);
        let result = expr.to_string();
        assert_eq!(result, "(label_id:1 AND (label_id:2 OR label_id:3))");
    }

    #[test]
    fn test_print_visitor_nested_or_and() {
        let expr = ASTIdExpr::Or(vec![
            ASTIdExpr::And(vec![ASTIdExpr::Terminal(4u64), ASTIdExpr::Terminal(5u64)]),
            ASTIdExpr::Terminal(6u64),
        ]);
        let result = expr.to_string();
        assert_eq!(result, "((label_id:4 AND label_id:5) OR label_id:6)");
    }

    #[test]
    fn test_print_visitor_complex_nested() {
        let expr = ASTIdExpr::And(vec![
            ASTIdExpr::Or(vec![ASTIdExpr::Terminal(1u64), ASTIdExpr::Terminal(2u64)]),
            ASTIdExpr::Not(Box::new(ASTIdExpr::And(vec![
                ASTIdExpr::Terminal(3u64),
                ASTIdExpr::Terminal(4u64),
            ]))),
        ]);
        let result = expr.to_string();
        assert_eq!(
            result,
            "((label_id:1 OR label_id:2) AND NOT ((label_id:3 AND label_id:4)))"
        );
    }

    #[test]
    fn test_print_visitor_deeply_nested_not() {
        let expr = ASTIdExpr::Not(Box::new(ASTIdExpr::Not(Box::new(ASTIdExpr::Terminal(
            99u64,
        )))));
        let result = expr.to_string();
        assert_eq!(result, "NOT (NOT (label_id:99))");
    }

    #[test]
    fn test_fmt_display_implementation() {
        let expr = ASTIdExpr::And(vec![
            ASTIdExpr::Terminal(1u64),
            ASTIdExpr::Or(vec![ASTIdExpr::Terminal(2u64), ASTIdExpr::Terminal(3u64)]),
        ]);
        let formatted = format!("{}", expr);
        assert_eq!(formatted, "(label_id:1 AND (label_id:2 OR label_id:3))");
    }

    #[test]
    fn test_fmt_display_single_terminal() {
        let expr = ASTIdExpr::Terminal(42u64);
        let formatted = format!("{}", expr);
        assert_eq!(formatted, "label_id:42");
    }

    #[test]
    fn test_print_visitor_different_attribute_type() {
        // Test with u32 instead of u64
        let expr = ASTIdExpr::Terminal(100u32);
        let result = expr.to_string();
        assert_eq!(result, "label_id:100");
    }

    #[test]
    fn test_print_visitor_large_and_expression() {
        let terminals: Vec<ASTIdExpr<u64>> = (1..=5).map(ASTIdExpr::Terminal).collect();
        let expr = ASTIdExpr::And(terminals);
        let result = expr.to_string();
        assert_eq!(
            result,
            "(label_id:1 AND label_id:2 AND label_id:3 AND label_id:4 AND label_id:5)"
        );
    }

    #[test]
    fn test_print_visitor_large_or_expression() {
        let terminals: Vec<ASTIdExpr<u64>> = (10..=13).map(ASTIdExpr::Terminal).collect();
        let expr = ASTIdExpr::Or(terminals);
        let result = expr.to_string();
        assert_eq!(
            result,
            "(label_id:10 OR label_id:11 OR label_id:12 OR label_id:13)"
        );
    }
}

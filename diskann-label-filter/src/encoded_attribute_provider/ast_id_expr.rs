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
    /// Equality comparison on a fully encoded attribute.
    Eq(T),
    /// Numeric less-than-or-equal comparison on a field.
    Lte(String, f64),
    /// Numeric greater-than-or-equal comparison on a field.
    Gte(String, f64),
    /// Inclusive numeric range comparison on a field.
    Between(String, f64, f64),
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
            ASTIdExpr::Eq(id) => self.visit_eq(id),
            ASTIdExpr::Lte(field, value) => self.visit_lte(field, *value),
            ASTIdExpr::Gte(field, value) => self.visit_gte(field, *value),
            ASTIdExpr::Between(field, min, max) => self.visit_between(field, *min, *max),
        }
    }

    /// Visit an AND expression
    fn visit_and(&mut self, exprs: &[ASTIdExpr<T>]) -> Self::Output;

    /// Visit an OR expression
    fn visit_or(&mut self, exprs: &[ASTIdExpr<T>]) -> Self::Output;

    /// Visit a NOT expression
    fn visit_not(&mut self, expr: &ASTIdExpr<T>) -> Self::Output;

    /// Visit an equality comparison.
    fn visit_eq(&mut self, id: &T) -> Self::Output;

    /// Visit a less-than-or-equal comparison.
    fn visit_lte(&mut self, field: &str, value: f64) -> Self::Output;

    /// Visit a greater-than-or-equal comparison.
    fn visit_gte(&mut self, field: &str, value: f64) -> Self::Output;

    /// Visit an inclusive range comparison.
    fn visit_between(&mut self, field: &str, min: f64, max: f64) -> Self::Output;
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
            ASTIdExpr::Eq(id) => self.visit_eq(id),
            ASTIdExpr::Lte(field, value) => self.visit_lte(field, *value),
            ASTIdExpr::Gte(field, value) => self.visit_gte(field, *value),
            ASTIdExpr::Between(field, min, max) => self.visit_between(field, *min, *max),
        }
    }

    /// Visit an AND expression
    fn visit_and(&self, exprs: &[ASTIdExpr<T>]) -> Self::Output;

    /// Visit an OR expression
    fn visit_or(&self, exprs: &[ASTIdExpr<T>]) -> Self::Output;

    /// Visit a NOT expression
    fn visit_not(&self, expr: &ASTIdExpr<T>) -> Self::Output;

    /// Visit an equality comparison.
    fn visit_eq(&self, id: &T) -> Self::Output;

    /// Visit a less-than-or-equal comparison.
    fn visit_lte(&self, field: &str, value: f64) -> Self::Output;

    /// Visit a greater-than-or-equal comparison.
    fn visit_gte(&self, field: &str, value: f64) -> Self::Output;

    /// Visit an inclusive range comparison.
    fn visit_between(&self, field: &str, min: f64, max: f64) -> Self::Output;
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

    fn visit_eq(&mut self, id: &T) -> Self::Output {
        format!("label_id:{}", id)
    }

    fn visit_lte(&mut self, field: &str, value: f64) -> Self::Output {
        format!("{}<={}", field, value)
    }

    fn visit_gte(&mut self, field: &str, value: f64) -> Self::Output {
        format!("{}>={}", field, value)
    }

    fn visit_between(&mut self, field: &str, min: f64, max: f64) -> Self::Output {
        format!("{} BETWEEN {} AND {}", field, min, max)
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
        let expr = ASTIdExpr::Eq(42u64);
        let result = expr.to_string();
        assert_eq!(result, "label_id:42");
    }

    #[test]
    fn test_print_visitor_single_and() {
        let expr = ASTIdExpr::And(vec![ASTIdExpr::Eq(5u64)]);
        let result = expr.to_string();
        assert_eq!(result, "label_id:5");
    }

    #[test]
    fn test_print_visitor_multiple_and() {
        let expr = ASTIdExpr::And(vec![
            ASTIdExpr::Eq(1u64),
            ASTIdExpr::Eq(2u64),
            ASTIdExpr::Eq(3u64),
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
        let expr = ASTIdExpr::Or(vec![ASTIdExpr::Eq(10u64)]);
        let result = expr.to_string();
        assert_eq!(result, "label_id:10");
    }

    #[test]
    fn test_print_visitor_multiple_or() {
        let expr = ASTIdExpr::Or(vec![ASTIdExpr::Eq(7u64), ASTIdExpr::Eq(8u64)]);
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
        let expr = ASTIdExpr::Not(Box::new(ASTIdExpr::Eq(15u64)));
        let result = expr.to_string();
        assert_eq!(result, "NOT (label_id:15)");
    }

    #[test]
    fn test_print_visitor_nested_and_or() {
        let expr = ASTIdExpr::And(vec![
            ASTIdExpr::Eq(1u64),
            ASTIdExpr::Or(vec![ASTIdExpr::Eq(2u64), ASTIdExpr::Eq(3u64)]),
        ]);
        let result = expr.to_string();
        assert_eq!(result, "(label_id:1 AND (label_id:2 OR label_id:3))");
    }

    #[test]
    fn test_print_visitor_nested_or_and() {
        let expr = ASTIdExpr::Or(vec![
            ASTIdExpr::And(vec![ASTIdExpr::Eq(4u64), ASTIdExpr::Eq(5u64)]),
            ASTIdExpr::Eq(6u64),
        ]);
        let result = expr.to_string();
        assert_eq!(result, "((label_id:4 AND label_id:5) OR label_id:6)");
    }

    #[test]
    fn test_print_visitor_complex_nested() {
        let expr = ASTIdExpr::And(vec![
            ASTIdExpr::Or(vec![ASTIdExpr::Eq(1u64), ASTIdExpr::Eq(2u64)]),
            ASTIdExpr::Not(Box::new(ASTIdExpr::And(vec![
                ASTIdExpr::Eq(3u64),
                ASTIdExpr::Eq(4u64),
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
        let expr = ASTIdExpr::Not(Box::new(ASTIdExpr::Not(Box::new(ASTIdExpr::Eq(
            99u64,
        )))));
        let result = expr.to_string();
        assert_eq!(result, "NOT (NOT (label_id:99))");
    }

    #[test]
    fn test_fmt_display_implementation() {
        let expr = ASTIdExpr::And(vec![
            ASTIdExpr::Eq(1u64),
            ASTIdExpr::Or(vec![ASTIdExpr::Eq(2u64), ASTIdExpr::Eq(3u64)]),
        ]);
        let formatted = format!("{}", expr);
        assert_eq!(formatted, "(label_id:1 AND (label_id:2 OR label_id:3))");
    }

    #[test]
    fn test_fmt_display_single_terminal() {
        let expr = ASTIdExpr::Eq(42u64);
        let formatted = format!("{}", expr);
        assert_eq!(formatted, "label_id:42");
    }

    #[test]
    fn test_print_visitor_different_attribute_type() {
        // Test with u32 instead of u64
        let expr = ASTIdExpr::Eq(100u32);
        let result = expr.to_string();
        assert_eq!(result, "label_id:100");
    }

    #[test]
    fn test_print_visitor_large_and_expression() {
        let terminals: Vec<ASTIdExpr<u64>> = (1..=5).map(ASTIdExpr::Eq).collect();
        let expr = ASTIdExpr::And(terminals);
        let result = expr.to_string();
        assert_eq!(
            result,
            "(label_id:1 AND label_id:2 AND label_id:3 AND label_id:4 AND label_id:5)"
        );
    }

    #[test]
    fn test_print_visitor_large_or_expression() {
        let terminals: Vec<ASTIdExpr<u64>> = (10..=13).map(ASTIdExpr::Eq).collect();
        let expr = ASTIdExpr::Or(terminals);
        let result = expr.to_string();
        assert_eq!(
            result,
            "(label_id:10 OR label_id:11 OR label_id:12 OR label_id:13)"
        );
    }

    #[test]
    fn test_print_visitor_lte() {
        let expr: ASTIdExpr<u64> = ASTIdExpr::Lte("price".to_string(), 300.0);
        assert_eq!(expr.to_string(), "price<=300");
    }

    #[test]
    fn test_print_visitor_gte() {
        let expr: ASTIdExpr<u64> = ASTIdExpr::Gte("price".to_string(), 125.5);
        assert_eq!(expr.to_string(), "price>=125.5");
    }

    #[test]
    fn test_print_visitor_between() {
        let expr: ASTIdExpr<u64> = ASTIdExpr::Between("price".to_string(), 100.0, 200.0);
        assert_eq!(expr.to_string(), "price BETWEEN 100 AND 200");
    }
}

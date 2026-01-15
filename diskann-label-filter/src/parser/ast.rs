/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::fmt;

use serde_json::Value;

/// AST for query filters https://en.wikipedia.org/wiki/Abstract_syntax_tree
#[derive(Debug, Clone, PartialEq)]
pub enum ASTExpr {
    /// Logical AND: all sub-expressions must be true
    And(Vec<ASTExpr>),
    /// Logical OR: at least one sub-expression must be true
    Or(Vec<ASTExpr>),
    /// Logical NOT: negates the sub-expression
    Not(Box<ASTExpr>),
    /// Comparison on a field (supports dot notation)
    Compare { field: String, op: CompareOp },
}

/// Supported comparison operators with type-safe values
#[derive(Debug, Clone, PartialEq)]
pub enum CompareOp {
    /// Equal comparison, can be used with any value type
    Eq(Value), // $eq
    /// Not equal comparison, can be used with any value type
    Ne(Value), // $ne
    /// Less than comparison, only valid for numeric values
    Lt(f64), // $lt
    /// Less than or equal comparison, only valid for numeric values
    Lte(f64), // $lte
    /// Greater than comparison, only valid for numeric values
    Gt(f64), // $gt
    /// Greater than or equal comparison, only valid for numeric values
    Gte(f64), // $gte
}

impl fmt::Display for CompareOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompareOp::Eq(_) => write!(f, "=="),
            CompareOp::Ne(_) => write!(f, "!="),
            CompareOp::Lt(_) => write!(f, "<"),
            CompareOp::Lte(_) => write!(f, "<="),
            CompareOp::Gt(_) => write!(f, ">"),
            CompareOp::Gte(_) => write!(f, ">="),
        }
    }
}

/// Trait for visiting AST expressions
pub trait ASTVisitor {
    type Output;

    /// Visit an AST expression
    fn visit(&mut self, expr: &ASTExpr) -> Self::Output {
        match expr {
            ASTExpr::And(exprs) => self.visit_and(exprs),
            ASTExpr::Or(exprs) => self.visit_or(exprs),
            ASTExpr::Not(expr) => self.visit_not(expr),
            ASTExpr::Compare { field, op } => self.visit_compare(field, op),
        }
    }

    /// Visit an AND expression
    fn visit_and(&mut self, exprs: &[ASTExpr]) -> Self::Output;

    /// Visit an OR expression
    fn visit_or(&mut self, exprs: &[ASTExpr]) -> Self::Output;

    /// Visit a NOT expression
    fn visit_not(&mut self, expr: &ASTExpr) -> Self::Output;

    /// Visit a comparison expression
    fn visit_compare(&mut self, field: &str, op: &CompareOp) -> Self::Output;
}

/// Implementation of the visitor pattern for ASTExpr
impl ASTExpr {
    /// Accept a visitor and return its output
    pub fn accept<V: ASTVisitor>(&self, visitor: &mut V) -> V::Output {
        visitor.visit(self)
    }
}

/// A visitor that converts AST expressions to a human-readable string
pub struct PrintVisitor {
    indent_level: usize,
    indent_str: String,
}

impl Default for PrintVisitor {
    fn default() -> Self {
        Self::new()
    }
}

impl PrintVisitor {
    /// Create a new PrintVisitor with default settings
    pub fn new() -> Self {
        Self {
            indent_level: 0,
            indent_str: " ".to_string(),
        }
    }

    /// Create a new PrintVisitor with custom indentation
    pub fn with_indent(indent_str: &str) -> Self {
        Self {
            indent_level: 0,
            indent_str: indent_str.to_string(),
        }
    }

    fn indent(&self) -> String {
        self.indent_str.repeat(self.indent_level)
    }

    fn value_to_string(value: &Value) -> String {
        match value {
            Value::String(s) => format!("\"{}\"", s.replace('\"', "\\\"")),
            Value::Array(arr) => {
                let items: Vec<String> = arr.iter().map(Self::value_to_string).collect();
                format!("[{}]", items.join(", "))
            }
            _ => value.to_string(),
        }
    }
}

impl ASTVisitor for PrintVisitor {
    type Output = String;

    fn visit_and(&mut self, exprs: &[ASTExpr]) -> Self::Output {
        if exprs.is_empty() {
            return "true".to_string();
        }

        if exprs.len() == 1 {
            return self.visit(&exprs[0]);
        }

        let current_indent = self.indent();
        self.indent_level += 1;

        let inner: Vec<String> = exprs
            .iter()
            .map(|expr| format!("\n{}{}", self.indent(), self.visit(expr)))
            .collect();

        self.indent_level -= 1;

        format!("AND({}\n{})", inner.join(","), current_indent)
    }

    fn visit_or(&mut self, exprs: &[ASTExpr]) -> Self::Output {
        if exprs.is_empty() {
            return "false".to_string();
        }

        if exprs.len() == 1 {
            return self.visit(&exprs[0]);
        }

        let current_indent = self.indent();
        self.indent_level += 1;

        let inner: Vec<String> = exprs
            .iter()
            .map(|expr| format!("\n{}{}", self.indent(), self.visit(expr)))
            .collect();

        self.indent_level -= 1;

        format!("OR({}\n{})", inner.join(","), current_indent)
    }

    fn visit_not(&mut self, expr: &ASTExpr) -> Self::Output {
        format!("NOT({})", self.visit(expr))
    }

    fn visit_compare(&mut self, field: &str, op: &CompareOp) -> Self::Output {
        let value_str = match op {
            CompareOp::Eq(value) => Self::value_to_string(value),
            CompareOp::Ne(value) => Self::value_to_string(value),
            CompareOp::Lt(num) => num.to_string(),
            CompareOp::Lte(num) => num.to_string(),
            CompareOp::Gt(num) => num.to_string(),
            CompareOp::Gte(num) => num.to_string(),
        };

        format!("{}{}{}", field, op, value_str)
    }
}

/// Display implementation for ASTExpr to easily print as string
impl fmt::Display for ASTExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut visitor = PrintVisitor::new();
        write!(f, "{}", self.accept(&mut visitor))
    }
}

/// Extension methods for ASTExpr for custom string representation
impl ASTExpr {
    /// Convert the AST expression to a human-readable string with custom indentation
    pub fn to_string_with_indent(&self, indent: &str) -> String {
        let mut visitor = PrintVisitor::with_indent(indent);
        self.accept(&mut visitor)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_ast_visitor() {
        // Test simple comparison
        let expr = ASTExpr::Compare {
            field: "age".to_string(),
            op: CompareOp::Gt(30.0),
        };

        assert_eq!(expr.to_string(), "age>30");

        // Test AND expression
        let and_expr = ASTExpr::And(vec![
            ASTExpr::Compare {
                field: "age".to_string(),
                op: CompareOp::Gt(30.0),
            },
            ASTExpr::Compare {
                field: "name".to_string(),
                op: CompareOp::Eq(json!("John")),
            },
        ]);

        let expected_and = "AND(\n age>30,\n name==\"John\"\n)";
        assert_eq!(and_expr.to_string(), expected_and);

        // Test OR expression
        let or_expr = ASTExpr::Or(vec![
            ASTExpr::Compare {
                field: "age".to_string(),
                op: CompareOp::Gt(30.0),
            },
            ASTExpr::Compare {
                field: "name".to_string(),
                op: CompareOp::Eq(json!("John")),
            },
        ]);

        let expected_or = "OR(\n age>30,\n name==\"John\"\n)";
        assert_eq!(or_expr.to_string(), expected_or);

        // Test NOT expression
        let not_expr = ASTExpr::Not(Box::new(ASTExpr::Compare {
            field: "age".to_string(),
            op: CompareOp::Lt(18.0),
        }));

        assert_eq!(not_expr.to_string(), "NOT(age<18)");

        // Test nested expressions
        let nested_expr = ASTExpr::And(vec![
            ASTExpr::Or(vec![
                ASTExpr::Compare {
                    field: "age".to_string(),
                    op: CompareOp::Gt(30.0),
                },
                ASTExpr::Compare {
                    field: "age".to_string(),
                    op: CompareOp::Lt(20.0),
                },
            ]),
            ASTExpr::Not(Box::new(ASTExpr::Compare {
                field: "name".to_string(),
                op: CompareOp::Eq(json!("Admin")),
            })),
        ]);

        let expected_nested = "AND(\n OR(\n  age>30,\n  age<20\n ),\n NOT(name==\"Admin\")\n)";
        assert_eq!(nested_expr.to_string(), expected_nested);
    }
}

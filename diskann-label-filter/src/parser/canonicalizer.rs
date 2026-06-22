/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{ASTExpr, CompareOp};

/// A canonicalization rule that can transform an AST subtree.
pub trait CanonicalizationRule {
    /// Returns true if this rule can rewrite `node`.
    fn applies(&self, node: &ASTExpr) -> bool;

    /// Returns the rewritten subtree.
    fn apply(&self, node: &ASTExpr) -> ASTExpr;
}

/// Walks the AST and applies rules in order until no more rules apply.
pub struct AstCanonicalizer {
    rules: Vec<Box<dyn CanonicalizationRule + Send + Sync>>,
}

impl AstCanonicalizer {
    pub fn new(rules: Vec<Box<dyn CanonicalizationRule + Send + Sync>>) -> Self {
        Self { rules }
    }

    /// Canonicalize the input AST in-place and return the canonicalized tree.
    pub fn canonicalize(&self, expr: &ASTExpr) -> ASTExpr {
        let canonical = self.canonicalize_node(expr);
        canonical
    }

    fn canonicalize_node(&self, node: &ASTExpr) -> ASTExpr {
        let mut normalized = match node {
            ASTExpr::And(exprs) => ASTExpr::And(
                exprs
                    .iter()
                    .map(|expr| self.canonicalize_node(expr))
                    .collect(),
            ),
            ASTExpr::Or(exprs) => ASTExpr::Or(
                exprs
                    .iter()
                    .map(|expr| self.canonicalize_node(expr))
                    .collect(),
            ),
            ASTExpr::Not(expr) => ASTExpr::Not(Box::new(self.canonicalize_node(expr))),

            compare @ ASTExpr::Compare { .. } => compare.clone(),
        };

        loop {
            let mut changed = false;
            for rule in &self.rules {
                if rule.applies(&normalized) {
                    normalized = rule.apply(&normalized);
                    changed = true;
                    break;
                }
            }
            if !changed {
                break;
            }
        }

        normalized
    }
}

impl Default for AstCanonicalizer {
    fn default() -> Self {
        Self::new(vec![Box::new(MergeAndRangeToBetweenRule)])
    }
}

/// Rewrites AND nodes that contain matching `field >= min` and `field <= max`
/// terms into a single `field BETWEEN min AND max` term.
pub struct MergeAndRangeToBetweenRule;

impl MergeAndRangeToBetweenRule {
    fn find_mergeable_pair(exprs: &[ASTExpr]) -> Option<(usize, usize, String, f64, f64)> {
        for (i, left) in exprs.iter().enumerate() {
            let (left_field, left_op) = match left {
                ASTExpr::Compare { field, op } => (field.as_str(), op),
                _ => continue,
            };

            for (j, right) in exprs.iter().enumerate().skip(i + 1) {
                let (right_field, right_op) = match right {
                    ASTExpr::Compare { field, op } => (field.as_str(), op),
                    _ => continue,
                };

                if left_field != right_field {
                    continue;
                }

                match (left_op, right_op) {
                    (CompareOp::Gte(min), CompareOp::Lte(max))
                    | (CompareOp::Lte(max), CompareOp::Gte(min)) => {
                        return Some((i, j, left_field.to_string(), *min, *max));
                    }
                    _ => {}
                }
            }
        }

        None
    }
}

impl CanonicalizationRule for MergeAndRangeToBetweenRule {
    fn applies(&self, node: &ASTExpr) -> bool {
        match node {
            ASTExpr::And(exprs) => Self::find_mergeable_pair(exprs).is_some(),
            _ => false,
        }
    }

    fn apply(&self, node: &ASTExpr) -> ASTExpr {
        let ASTExpr::And(exprs) = node else {
            return node.clone();
        };

        let Some((left_idx, right_idx, field, min, max)) = Self::find_mergeable_pair(exprs) else {
            return node.clone();
        };

        let mut rewritten = Vec::with_capacity(exprs.len().saturating_sub(1));
        for (idx, expr) in exprs.iter().enumerate() {
            if idx != left_idx && idx != right_idx {
                rewritten.push(expr.clone());
            }
        }

        rewritten.push(ASTExpr::Compare {
            field,
            op: CompareOp::Between(min, max),
        });

        if rewritten.len() == 1 {
            rewritten.remove(0)
        } else {
            ASTExpr::And(rewritten)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rewrite_simple_and_to_between() {
        let mut expr = ASTExpr::And(vec![
            ASTExpr::Compare {
                field: "price".to_string(),
                op: CompareOp::Gte(10.0),
            },
            ASTExpr::Compare {
                field: "price".to_string(),
                op: CompareOp::Lte(20.0),
            },
        ]);

        let canonical = AstCanonicalizer::default().canonicalize(&mut expr);

        assert!(matches!(
            canonical,
            ASTExpr::Compare {
                field,
                op: CompareOp::Between(min, max)
            } if field == "price" && min == 10.0 && max == 20.0
        ));
    }

    #[test]
    fn test_rewrite_inside_nested_tree() {
        let mut expr = ASTExpr::And(vec![
            ASTExpr::Compare {
                field: "category".to_string(),
                op: CompareOp::Eq(serde_json::json!("book")),
            },
            ASTExpr::And(vec![
                ASTExpr::Compare {
                    field: "price".to_string(),
                    op: CompareOp::Lte(20.0),
                },
                ASTExpr::Compare {
                    field: "price".to_string(),
                    op: CompareOp::Gte(10.0),
                },
            ]),
        ]);

        let canonical = AstCanonicalizer::default().canonicalize(&mut expr);

        match canonical {
            ASTExpr::And(exprs) => {
                assert_eq!(exprs.len(), 2);
                assert!(exprs.iter().any(|expr| {
                    matches!(
                        expr,
                        ASTExpr::Compare {
                            field,
                            op: CompareOp::Between(min, max)
                        } if field == "price" && *min == 10.0 && *max == 20.0
                    )
                }));
            }
            _ => panic!("Expected AND expression"),
        }
    }

    #[test]
    fn test_no_rewrite_for_different_fields() {
        let mut expr = ASTExpr::And(vec![
            ASTExpr::Compare {
                field: "price".to_string(),
                op: CompareOp::Gte(10.0),
            },
            ASTExpr::Compare {
                field: "rating".to_string(),
                op: CompareOp::Lte(20.0),
            },
        ]);

        let canonical = AstCanonicalizer::default().canonicalize(&mut expr);
        assert_eq!(canonical, expr);
    }
}

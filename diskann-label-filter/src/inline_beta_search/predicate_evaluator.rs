/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNResult;

use crate::{
    attribute::AttributeType,
    encoded_attribute_provider::ast_id_expr::{ASTIdExpr, ASTIdExprVisitor},
    set::Set,
};

/// Trait for evaluating a predicate in the context of a point. Essentially, it checks
/// if the labels of a point satisfy the predicate.
///
/// Type parameters:
///     T: AttributeType (u64 in the current implementation)
///     ST: Set type, Set<T>
pub struct PredicateEvaluator<'a, T, ST>
where
    ST: Set<T>,
    T: AttributeType,
{
    labels_of_point: &'a ST,
    _phantom_t: std::marker::PhantomData<T>,
}

impl<'a, T, ST> PredicateEvaluator<'a, T, ST>
where
    ST: Set<T>,
    T: AttributeType,
{
    pub fn new(labels_of_point: &'a ST) -> Self {
        Self {
            labels_of_point,
            _phantom_t: std::marker::PhantomData,
        }
    }
}

impl<'a, T, ST> ASTIdExprVisitor<T> for PredicateEvaluator<'a, T, ST>
where
    ST: Set<T>,
    T: AttributeType,
{
    type Output = ANNResult<bool>;

    /// Visit an AND expression - all sub-expressions must be true
    fn visit_and(&self, exprs: &[ASTIdExpr<T>]) -> Self::Output {
        if exprs.is_empty() {
            return Ok(true); // Empty AND is vacuously true
        }

        for expr in exprs {
            match self.visit(expr) {
                Ok(true) => continue,          // Continue if true
                Ok(false) => return Ok(false), // If any sub-expression is false, AND is false
                Err(e) => return Err(e),       // Propagate error
            }
        }
        Ok(true)
    }

    /// Visit an OR expression - at least one sub-expression must be true
    fn visit_or(&self, exprs: &[ASTIdExpr<T>]) -> Self::Output {
        if exprs.is_empty() {
            return Ok(false); // Empty OR is false
        }

        for expr in exprs {
            match self.visit(expr) {
                Ok(true) => return Ok(true), // If any sub-expression is true, OR is true
                Ok(false) => continue,       // Continue if false
                Err(e) => return Err(e),     // Propagate error
            }
        }
        Ok(false)
    }

    /// Visit a NOT expression - negate the result of the sub-expression
    fn visit_not(&self, expr: &ASTIdExpr<T>) -> Self::Output {
        match self.visit(expr) {
            Ok(result) => Ok(!result),
            Err(e) => Err(e),
        }
    }

    /// Visit a comparison expression - check if the label exists in labels_of_point
    fn visit_terminal(&self, label_id: &T) -> Self::Output {
        self.labels_of_point.contains(label_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoded_attribute_provider::ast_id_expr::ASTIdExpr;
    use roaring::RoaringTreemap;

    fn labels(ids: &[u64]) -> RoaringTreemap {
        let mut s = RoaringTreemap::new();
        for &id in ids {
            s.insert(id);
        }
        s
    }

    #[test]
    fn terminal_checks_membership() {
        let set = labels(&[1, 2, 3]);
        let eval = PredicateEvaluator::new(&set);
        assert!(ASTIdExpr::Terminal(2u64).accept(&eval).unwrap());
        assert!(!ASTIdExpr::Terminal(9u64).accept(&eval).unwrap());
    }

    #[test]
    fn and_requires_all_members() {
        let set = labels(&[1, 2]);
        let eval = PredicateEvaluator::new(&set);
        let all_present =
            ASTIdExpr::And(vec![ASTIdExpr::Terminal(1u64), ASTIdExpr::Terminal(2u64)]);
        assert!(all_present.accept(&eval).unwrap());
        let one_missing =
            ASTIdExpr::And(vec![ASTIdExpr::Terminal(1u64), ASTIdExpr::Terminal(3u64)]);
        assert!(!one_missing.accept(&eval).unwrap());
    }

    #[test]
    fn empty_and_is_vacuously_true() {
        let set = labels(&[]);
        let eval = PredicateEvaluator::new(&set);
        assert!(ASTIdExpr::<u64>::And(vec![]).accept(&eval).unwrap());
    }

    #[test]
    fn or_requires_any_member() {
        let set = labels(&[1]);
        let eval = PredicateEvaluator::new(&set);
        let any_present = ASTIdExpr::Or(vec![ASTIdExpr::Terminal(9u64), ASTIdExpr::Terminal(1u64)]);
        assert!(any_present.accept(&eval).unwrap());
        let none_present =
            ASTIdExpr::Or(vec![ASTIdExpr::Terminal(8u64), ASTIdExpr::Terminal(9u64)]);
        assert!(!none_present.accept(&eval).unwrap());
    }

    #[test]
    fn empty_or_is_false() {
        let set = labels(&[1]);
        let eval = PredicateEvaluator::new(&set);
        assert!(!ASTIdExpr::<u64>::Or(vec![]).accept(&eval).unwrap());
    }

    #[test]
    fn not_negates_inner() {
        let set = labels(&[1]);
        let eval = PredicateEvaluator::new(&set);
        assert!(ASTIdExpr::Not(Box::new(ASTIdExpr::Terminal(9u64)))
            .accept(&eval)
            .unwrap());
        assert!(!ASTIdExpr::Not(Box::new(ASTIdExpr::Terminal(1u64)))
            .accept(&eval)
            .unwrap());
    }

    #[test]
    fn nested_and_or_not_combination() {
        let set = labels(&[1, 2, 3]);
        let eval = PredicateEvaluator::new(&set);
        // (1 AND 2) OR (NOT 5) -> both branches true, overall true
        let expr = ASTIdExpr::Or(vec![
            ASTIdExpr::And(vec![ASTIdExpr::Terminal(1u64), ASTIdExpr::Terminal(2u64)]),
            ASTIdExpr::Not(Box::new(ASTIdExpr::Terminal(5u64))),
        ]);
        assert!(expr.accept(&eval).unwrap());

        // (1 AND 9) AND (NOT 2) -> first branch false, overall false
        let expr2 = ASTIdExpr::And(vec![
            ASTIdExpr::And(vec![ASTIdExpr::Terminal(1u64), ASTIdExpr::Terminal(9u64)]),
            ASTIdExpr::Not(Box::new(ASTIdExpr::Terminal(2u64))),
        ]);
        assert!(!expr2.accept(&eval).unwrap());
    }
}

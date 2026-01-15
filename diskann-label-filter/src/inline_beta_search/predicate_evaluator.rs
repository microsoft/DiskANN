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

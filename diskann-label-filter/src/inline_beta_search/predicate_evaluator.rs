/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::{Arc, RwLock};

use diskann::{ANNError, ANNErrorKind, ANNResult};

use crate::{
    attribute::AttributeValue,
    encoded_attribute_provider::{
        ast_id_expr::{ASTIdExpr, ASTIdExprVisitor},
        attribute_encoder::AttributeEncoder,
    },
    set::Set,
};

/// Trait for evaluating a predicate in the context of a point. Essentially, it checks
/// if the labels of a point satisfy the predicate.
///
/// Type parameters:
///     T: AttributeType (u64 in the current implementation)
///     ST: Set type, Set<T>
pub struct PredicateEvaluator<'a, ST>
where
    ST: Set<u64>,
{
    labels_of_point: &'a ST,
    attribute_map: &'a Arc<RwLock<AttributeEncoder>>,
}

impl<'a, ST> PredicateEvaluator<'a, ST>
where
    ST: Set<u64>,
{
    pub(crate) fn new(
        labels_of_point: &'a ST,
        attribute_map: &'a Arc<RwLock<AttributeEncoder>>,
    ) -> Self {
        Self {
            labels_of_point,
            attribute_map,
        }
    }

    fn matches_numeric_bound(attr_value: &AttributeValue, bound: f64, is_lte: bool) -> bool {
        match attr_value.as_numeric_f64() {
            Some(value) if is_lte => value <= bound,
            Some(value) => value >= bound,
            None => false,
        }
    }

    fn matches_between(attr_value: &AttributeValue, min: f64, max: f64) -> bool {
        match attr_value.as_numeric_f64() {
            Some(value) => value >= min && value <= max,
            None => false,
        }
    }
}

impl<'a, ST> ASTIdExprVisitor<u64> for PredicateEvaluator<'a, ST>
where
    ST: Set<u64>,
{
    type Output = ANNResult<bool>;

    /// Visit an AND expression - all sub-expressions must be true
    fn visit_and(&self, exprs: &[ASTIdExpr<u64>]) -> Self::Output {
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
    fn visit_or(&self, exprs: &[ASTIdExpr<u64>]) -> Self::Output {
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
    fn visit_not(&self, expr: &ASTIdExpr<u64>) -> Self::Output {
        match self.visit(expr) {
            Ok(result) => Ok(!result),
            Err(e) => Err(e),
        }
    }

    fn visit_eq(&self, label_id: &u64) -> Self::Output {
        self.labels_of_point.contains(label_id)
    }

    fn visit_lte(&self, field: &str, value: f64) -> Self::Output {
        let encoder = self.attribute_map.read().map_err(|_| {
            ANNError::message(
                ANNErrorKind::LockPoisonError,
                "Failed to acquire read lock on attribute map",
            )
        })?;

        for attr_id in self.labels_of_point.clone() {
            if let Some(attribute) = encoder.get_by_id(attr_id) {
                if attribute.field_name() == field
                    && Self::matches_numeric_bound(attribute.attr_value(), value, true)
                {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    fn visit_gte(&self, field: &str, value: f64) -> Self::Output {
        let encoder = self.attribute_map.read().map_err(|_| {
            ANNError::message(
                ANNErrorKind::LockPoisonError,
                "Failed to acquire read lock on attribute map",
            )
        })?;

        for attr_id in self.labels_of_point.clone() {
            if let Some(attribute) = encoder.get_by_id(attr_id) {
                if attribute.field_name() == field
                    && Self::matches_numeric_bound(attribute.attr_value(), value, false)
                {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    fn visit_between(&self, field: &str, min: f64, max: f64) -> Self::Output {
        let encoder = self.attribute_map.read().map_err(|_| {
            ANNError::message(
                ANNErrorKind::LockPoisonError,
                "Failed to acquire read lock on attribute map",
            )
        })?;

        for attr_id in self.labels_of_point.clone() {
            if let Some(attribute) = encoder.get_by_id(attr_id) {
                if attribute.field_name() == field
                    && Self::matches_between(attribute.attr_value(), min, max)
                {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }
}

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNResult;

use crate::{
    encoded_attribute_provider::ast_id_expr::{ASTIdExpr, ASTIdExprVisitor},
    set::{Set, SetProvider},
};

/// Evaluates a predicate for a specific point using the inverted index.
///
/// Instead of holding a pre-fetched label set for a point and checking
/// `labels_of_point.contains(attr_id)`, this evaluator holds a reference to
/// the entire inverted index (attr_id → set of doc_ids) and a specific
/// `doc_id`, then checks `inv_index.get(attr_id)?.contains(doc_id)`.
///
/// This is semantically equivalent to `PredicateEvaluator` but avoids
/// needing to look up the forward index for each point.
pub struct InvertedPredicateEvaluator<'a, SP> {
    /// The inverted index mapping attribute_id → set of doc_ids.
    inv_index: &'a SP,
    /// The doc_id to evaluate the predicate against.
    doc_id: u64,
}

impl<'a, SP> InvertedPredicateEvaluator<'a, SP> {
    pub fn new(inv_index: &'a SP, doc_id: u64) -> Self {
        Self { inv_index, doc_id }
    }
}

impl<'a, SP> ASTIdExprVisitor<u64> for InvertedPredicateEvaluator<'a, SP>
where
    SP: SetProvider<u64, u64>,
{
    type Output = ANNResult<bool>;

    /// Visit an AND expression - all sub-expressions must be true
    fn visit_and(&self, exprs: &[ASTIdExpr<u64>]) -> Self::Output {
        if exprs.is_empty() {
            return Ok(true);
        }

        for expr in exprs {
            match self.visit(expr) {
                Ok(true) => continue,
                Ok(false) => return Ok(false),
                Err(e) => return Err(e),
            }
        }
        Ok(true)
    }

    /// Visit an OR expression - at least one sub-expression must be true
    fn visit_or(&self, exprs: &[ASTIdExpr<u64>]) -> Self::Output {
        if exprs.is_empty() {
            return Ok(false);
        }

        for expr in exprs {
            match self.visit(expr) {
                Ok(true) => return Ok(true),
                Ok(false) => continue,
                Err(e) => return Err(e),
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

    /// Visit a terminal: check if doc_id is in the posting list for this attribute.
    fn visit_terminal(&self, attr_id: &u64) -> Self::Output {
        match self.inv_index.get(attr_id)? {
            Some(set) => {
                Ok(set.contains(&self.doc_id)?)
            }
            None => Ok(false),
        }
    }
}

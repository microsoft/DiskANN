/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{traits::posting_list_trait::PostingList, ASTExpr};

/// A trait for evaluating query expressions against an inverted index.
///
/// This trait provides methods to evaluate abstract syntax tree (AST) expressions
/// and determine which documents match the query criteria. It operates on posting
/// lists to efficiently compute query results.
///
/// This trait is independent from `InvertedIndex` and can be implemented separately,
/// allowing for flexible query evaluation strategies.
pub trait QueryEvaluator {
    /// The error type returned by query evaluation operations.
    type Error: std::error::Error + Send + Sync + 'static;

    /// The posting list type used to store document IDs.
    type PostingList: PostingList;

    /// The document ID type used to identify documents.
    type DocId: Copy + Into<usize> + From<usize> + std::fmt::Debug;

    /// Evaluates a query expression and returns the matching documents.
    ///
    /// # Arguments
    ///
    /// * `query_expr` - The query expression to evaluate (e.g., AND, OR, comparison operations).
    ///
    /// # Returns
    ///
    /// Returns a posting list containing all document IDs that match the query,
    /// or an error if the evaluation fails.
    fn evaluate_query(
        &self,
        query_expr: &ASTExpr,
    ) -> std::result::Result<Self::PostingList, Self::Error>;

    /// Checks if a specific document matches a query expression.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The document ID to check.
    /// * `expr` - The query expression to evaluate.
    ///
    /// # Returns
    ///
    /// Returns `true` if the document matches the expression, `false` otherwise,
    /// or an error if the evaluation fails.
    fn is_match(
        &self,
        doc_id: Self::DocId,
        expr: &ASTExpr,
    ) -> std::result::Result<bool, Self::Error> {
        let bs = self.evaluate_query(expr)?;
        Ok(bs.contains(doc_id.into()))
    }

    /// Counts the number of documents that match a query expression.
    ///
    /// # Arguments
    ///
    /// * `expr` - The query expression to evaluate.
    ///
    /// # Returns
    ///
    /// Returns the count of matching documents, or an error if the evaluation fails.
    fn count_matches(&self, expr: &ASTExpr) -> std::result::Result<usize, Self::Error> {
        let bs = self.evaluate_query(expr)?;
        Ok(bs.len())
    }
}

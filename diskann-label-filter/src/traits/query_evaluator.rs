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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::posting_list_trait::{PostingList, RoaringPostingList};
    use crate::CompareOp;

    /// A minimal evaluator that always returns a fixed posting list, used to
    /// exercise the default `is_match` and `count_matches` methods of the trait.
    struct FixedEvaluator {
        list: RoaringPostingList,
    }

    impl QueryEvaluator for FixedEvaluator {
        type Error = <RoaringPostingList as PostingList>::Error;
        type PostingList = RoaringPostingList;
        type DocId = usize;

        fn evaluate_query(
            &self,
            _query_expr: &ASTExpr,
        ) -> std::result::Result<Self::PostingList, Self::Error> {
            Ok(self.list.clone())
        }
    }

    fn sample_expr() -> ASTExpr {
        ASTExpr::Compare {
            field: "field".to_owned(),
            op: CompareOp::Eq(serde_json::json!("value")),
        }
    }

    #[test]
    fn default_is_match_and_count_matches() {
        let mut list = RoaringPostingList::empty();
        list.insert(1);
        list.insert(2);
        list.insert(3);
        let evaluator = FixedEvaluator { list };

        let expr = sample_expr();
        assert!(evaluator.is_match(2, &expr).unwrap());
        assert!(!evaluator.is_match(9, &expr).unwrap());
        assert_eq!(evaluator.count_matches(&expr).unwrap(), 3);
    }
}

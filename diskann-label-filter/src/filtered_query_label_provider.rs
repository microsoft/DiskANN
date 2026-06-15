/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A QueryLabelProvider that evaluates ASTExpr filters using RoaringAttributeStore.
//! This allows filtering during search without wrapping the strategy layer.

use std::sync::{Arc, RwLock};

use diskann::{
    graph::index::{QueryLabelProvider, QueryVisitDecision},
    neighbor::Neighbor,
    ANNResult,
};

use crate::{
    encoded_attribute_provider::{
        attribute_encoder::AttributeEncoder, encoded_filter_expr::EncodedFilterExpr,
        roaring_attribute_store::RoaringAttributeStore,
    },
    inline_beta_search::predicate_evaluator::PredicateEvaluator,
    traits::attribute_accessor::AttributeAccessor,
    traits::attribute_store::AttributeStore,
    ASTExpr,
};

/// A QueryLabelProvider that evaluates an ASTExpr filter against attributes
/// stored in a RoaringAttributeStore.
///
/// This provider evaluates the filter expression for each candidate point during
/// search traversal, enabling filtering without modifying the search strategy.
pub struct FilteredQueryLabelProvider {
    encoded_filter: EncodedFilterExpr,
    attribute_store: Arc<RoaringAttributeStore<u32>>,
    attribute_map: Arc<RwLock<AttributeEncoder>>,
}

impl std::fmt::Debug for FilteredQueryLabelProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilteredQueryLabelProvider")
            .field("filter", &"EncodedFilterExpr")
            .finish()
    }
}

impl FilteredQueryLabelProvider {
    /// Create a new FilteredQueryLabelProvider.
    ///
    /// # Arguments
    ///
    /// * `filter_expr` - The AST expression to evaluate for filtering
    /// * `attribute_store` - The store containing point attributes
    ///
    /// # Returns
    ///
    /// A new provider that will evaluate the filter for each point visited during search.
    pub fn new(
        filter_expr: ASTExpr,
        attribute_store: Arc<RoaringAttributeStore<u32>>,
    ) -> ANNResult<Self> {
        let attribute_map = attribute_store.attribute_map();
        let encoded_filter = EncodedFilterExpr::new(&filter_expr, attribute_map.clone())?;

        Ok(Self {
            encoded_filter,
            attribute_store,
            attribute_map,
        })
    }
}

impl QueryLabelProvider<u32> for FilteredQueryLabelProvider {
    fn is_match(&self, vec_id: u32) -> bool {
        // Get the encoded attributes for this vec_id and evaluate the filter
        match self.attribute_store.attribute_accessor() {
            Ok(mut accessor) => {
                let result = accessor
                    .visit_labels_of_point(vec_id, |_, opt_set| {
                        match opt_set {
                            Some(set) => {
                                let mut evaluator =
                                    PredicateEvaluator::new(&*set, &self.attribute_map);
                                match self.encoded_filter.encoded_filter_expr().accept(&mut evaluator) {
                                    Ok(matched) => matched,
                                    Err(_) => {
                                        // If evaluation fails, reject the point
                                        tracing::warn!(
                                            "Filter evaluation failed for point {}",
                                            vec_id
                                        );
                                        false
                                    }
                                }
                            }
                            None => false, // No attributes = no match
                        }
                    })
                    .unwrap_or(false);
                result
            }
            Err(_) => false, // If we can't get accessor, reject
        }
    }

    fn on_visit(&self, neighbor: Neighbor<u32>) -> QueryVisitDecision<u32> {
        // Use the default behavior: Accept if is_match, else Reject
        if self.is_match(neighbor.id) {
            QueryVisitDecision::Accept(neighbor)
        } else {
            QueryVisitDecision::Reject
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filtered_query_label_provider_creation() {
        let store = Arc::new(RoaringAttributeStore::<u32>::new());

        // This should succeed even with an empty filter expression
        // (actual expression parsing happens elsewhere)
        let _result = FilteredQueryLabelProvider::new(ASTExpr::default(), store);
    }
}

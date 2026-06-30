/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::error::IntoANNResult;
use diskann::graph::glue::{SearchPostProcess, SearchStrategy};
use diskann::neighbor::Neighbor;
use diskann::provider::{DataProvider, HasId};
use diskann::utils::IntoUsize;

use diskann::{ANNError, ANNResult};
use roaring::RoaringTreemap;

use crate::encoded_attribute_provider::{
    document_provider::DocumentProvider, encoded_filter_expr::EncodedFilterExpr,
    roaring_attribute_store::RoaringAttributeStore,
};
use crate::inline_beta_search::encoded_document_accessor::EncodedDocumentAccessor;
use crate::inline_beta_search::predicate_evaluator::PredicateEvaluator;
use crate::query::FilteredQuery;
use crate::traits::attribute_store::AttributeStore;

pub struct InlineBetaStrategy<Strategy> {
    beta: f32,
    inner: Strategy,
}

impl<'q, DP, Strategy, Q>
    SearchStrategy<
        'q,
        DocumentProvider<DP, RoaringAttributeStore<DP::InternalId>>,
        &'q FilteredQuery<Q>,
    > for InlineBetaStrategy<Strategy>
where
    DP: DataProvider,
    DP::InternalId: IntoUsize,
    Strategy: SearchStrategy<'q, DP, &'q Q>,
    Q: Send + Sync + 'q,
{
    type SearchAccessorError = ANNError;
    type SearchAccessor = EncodedDocumentAccessor<Strategy::SearchAccessor>;

    fn search_accessor(
        &'q self,
        provider: &'q DocumentProvider<DP, RoaringAttributeStore<DP::InternalId>>,
        context: &'q DP::Context,
        query: &'q FilteredQuery<Q>,
    ) -> Result<Self::SearchAccessor, Self::SearchAccessorError> {
        let inner_accessor = self
            .inner
            .search_accessor(provider.inner_provider(), context, query.query())
            .into_ann_result()?;
        let attribute_accessor = provider.attribute_store().attribute_accessor()?;
        let attribute_map = provider.attribute_store().attribute_map();

        EncodedDocumentAccessor::new(
            inner_accessor,
            attribute_accessor,
            attribute_map,
            query.filter_expr(),
            self.beta,
        )
    }
}

/// [`DefaultPostProcessor`] delegation for [`InlineBetaStrategy`]. The processor wraps
/// the inner strategy's default processor with [`FilterResults`].
impl<'q, DP, Strategy, Q>
    diskann::graph::glue::DefaultPostProcessor<
        'q,
        DocumentProvider<DP, RoaringAttributeStore<DP::InternalId>>,
        &'q FilteredQuery<Q>,
    > for InlineBetaStrategy<Strategy>
where
    DP: DataProvider,
    DP::InternalId: IntoUsize,
    Strategy: diskann::graph::glue::DefaultPostProcessor<'q, DP, &'q Q>,
    Q: Send + Sync + 'q,
{
    type Processor = FilterResults<Strategy::Processor>;

    fn default_post_processor(&'q self) -> Self::Processor {
        FilterResults {
            inner_post_processor: self.inner.default_post_processor(),
        }
    }
}

pub struct InlineBetaComputer {
    beta_value: f32,
    filter_expr: EncodedFilterExpr,
}

impl InlineBetaComputer {
    pub(crate) fn new(beta_value: f32, filter_expr: EncodedFilterExpr) -> Self {
        Self {
            beta_value,
            filter_expr,
        }
    }

    pub(crate) fn filter_expr(&self) -> &EncodedFilterExpr {
        &self.filter_expr
    }

    pub(crate) fn apply(&self, distance: f32, attributes: &RoaringTreemap) -> f32 {
        let eval = PredicateEvaluator::new(attributes);
        match self.filter_expr.encoded_filter_expr().accept(&eval) {
            Ok(matched) => {
                if matched {
                    distance * self.beta_value
                } else {
                    distance
                }
            }
            Err(_) => {
                // TODO: If predicate evaluation fails, we are taking the approach that we
                // will simply return the score returned by the inner computer, as though no
                // predicate was specified.
                tracing::warn!("Predicate evaluation failed in OnlineBetaComputer application");
                distance
            }
        }
    }
}

pub struct FilterResults<IPP> {
    inner_post_processor: IPP,
}

impl<'q, Q, IA, IPP> SearchPostProcess<EncodedDocumentAccessor<IA>, &'q FilteredQuery<Q>>
    for FilterResults<IPP>
where
    IA: HasId + Send + Sync,
    Q: Send + Sync,
    IPP: SearchPostProcess<IA, &'q Q> + Send + Sync,
{
    type Error = ANNError;

    async fn post_process<I, B>(
        &self,
        accessor: &mut EncodedDocumentAccessor<IA>,
        query: &'q FilteredQuery<Q>,
        candidates: I,
        output: &mut B,
    ) -> Result<usize, Self::Error>
    where
        I: Iterator<Item = diskann::neighbor::Neighbor<IA::Id>> + Send,
        B: diskann::graph::SearchOutputBuffer<<IA as diskann::provider::HasId>::Id> + Send + ?Sized,
    {
        // This is a poor implementation - we should ideally be caching attributes of a point
        // along with neighbor. But that involves changes in the core algo, so I'm leaving it
        // out for now.
        //
        // TODO: Fix for performance.
        let mut filtered_candidates = Vec::<Neighbor<IA::Id>>::new();
        for candidate in candidates {
            accessor.attributes_for(candidate.id, |computer, attributes| -> ANNResult<()> {
                let pe = PredicateEvaluator::new(&*attributes);

                if computer.filter_expr().encoded_filter_expr().accept(&pe)? {
                    filtered_candidates.push(Neighbor::new(candidate.id, candidate.distance));
                }
                Ok(())
            })??;
        }

        // Assuming that the job of the post processor is to only forward the right set of
        // candidates and that there will be a "terminal" post processor that copies data
        // into "output".
        self.inner_post_processor
            .post_process(
                accessor.inner_accessor(),
                query.query(),
                filtered_candidates.into_iter(),
                output,
            )
            .await
            .map_err(|e| e.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attribute::{Attribute, AttributeValue};
    use crate::encoded_attribute_provider::attribute_encoder::AttributeEncoder;
    use crate::parser::ast::{ASTExpr, CompareOp};
    use serde_json::Value;
    use std::sync::{Arc, RwLock};

    /// Build an [`InlineBetaComputer`] whose filter is `category == "electronics"`,
    /// returning the encoded attribute id that a matching point must carry.
    fn build_computer(beta: f32) -> (InlineBetaComputer, u64) {
        let mut encoder = AttributeEncoder::new();
        let attr = Attribute::from_value(
            "category".to_string(),
            AttributeValue::String("electronics".to_string()),
        );
        let id = encoder.insert(&attr);
        let map = Arc::new(RwLock::new(encoder));

        let ast = ASTExpr::Compare {
            field: "category".to_string(),
            op: CompareOp::Eq(Value::String("electronics".to_string())),
        };
        let filter_expr = EncodedFilterExpr::new(&ast, map).unwrap();
        (InlineBetaComputer::new(beta, filter_expr), id)
    }

    #[test]
    fn apply_scales_distance_when_predicate_matches() {
        let (computer, id) = build_computer(0.5);
        let mut attrs = RoaringTreemap::new();
        attrs.insert(id);
        // Predicate matches -> distance is scaled by beta.
        assert_eq!(computer.apply(10.0, &attrs), 5.0);
    }

    #[test]
    fn apply_returns_distance_when_predicate_does_not_match() {
        let (computer, _id) = build_computer(0.5);
        // Point carries no attributes -> predicate fails -> distance unchanged.
        let attrs = RoaringTreemap::new();
        assert_eq!(computer.apply(10.0, &attrs), 10.0);
    }
}

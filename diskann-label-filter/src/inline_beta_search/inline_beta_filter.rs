/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::error::IntoANNResult;
use diskann::graph::glue::{SearchPostProcess, SearchStrategy};
use diskann::neighbor::Neighbor;
use diskann::provider::{Accessor, BuildQueryComputer, DataProvider};

use diskann::ANNError;
use diskann_utils::future::AsyncFriendly;
use diskann_vector::PreprocessedDistanceFunction;
use roaring::RoaringTreemap;

use crate::document::EncodedDocument;
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

impl<Strategy> InlineBetaStrategy<Strategy> {
    /// Create a new InlineBetaStrategy with the given beta value and inner strategy.
    pub fn new(beta: f32, inner: Strategy) -> Self {
        Self { beta, inner }
    }
}

impl<DP, Strategy, Q>
    SearchStrategy<DocumentProvider<DP, RoaringAttributeStore<DP::InternalId>>, FilteredQuery<Q>>
    for InlineBetaStrategy<Strategy>
where
    DP: DataProvider,
    Strategy: SearchStrategy<DP, Q>,
    Q: AsyncFriendly + Clone,
{
    type QueryComputer = InlineBetaComputer<Strategy::QueryComputer>;
    type PostProcessor = FilterResults<Strategy::PostProcessor>;
    type SearchAccessorError = ANNError;
    type SearchAccessor<'a> = EncodedDocumentAccessor<Strategy::SearchAccessor<'a>>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a DocumentProvider<DP, RoaringAttributeStore<DP::InternalId>>,
        context: &'a DP::Context,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        let inner_accessor = self
            .inner
            .search_accessor(provider.inner_provider(), context)
            .into_ann_result()?;
        let attribute_accessor = provider.attribute_store().attribute_accessor()?;
        let attribute_map = provider.attribute_store().attribute_map();

        Ok(EncodedDocumentAccessor::new(
            inner_accessor,
            attribute_accessor,
            attribute_map,
            self.beta,
        ))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        FilterResults {
            inner_post_processor: self.inner.post_processor(),
        }
    }
}

pub struct InlineBetaComputer<Inner> {
    inner_computer: Inner,
    beta_value: f32,
    filter_expr: EncodedFilterExpr,
    is_valid_filter: bool, //optimization to avoid evaluating empty predicates.
}

impl<Inner> InlineBetaComputer<Inner> {
    pub(crate) fn new(
        inner_computer: Inner,
        beta_value: f32,
        filter_expr: EncodedFilterExpr,
        is_valid_filter: bool,
    ) -> Self {
        Self {
            inner_computer,
            beta_value,
            filter_expr,
            is_valid_filter,
        }
    }

    pub(crate) fn filter_expr(&self) -> &EncodedFilterExpr {
        &self.filter_expr
    }

    pub(crate) fn is_valid_filter(&self) -> bool {
        self.is_valid_filter
    }
}

impl<Inner, V> PreprocessedDistanceFunction<EncodedDocument<V, &RoaringTreemap>, f32>
    for InlineBetaComputer<Inner>
where
    Inner: PreprocessedDistanceFunction<V>,
{
    fn evaluate_similarity(&self, changing: EncodedDocument<V, &RoaringTreemap>) -> f32 {
        let (vec, attrs) = changing.destructure();
        let sim = self.inner_computer.evaluate_similarity(vec);
        let pred_eval = PredicateEvaluator::new(attrs);
        if self.is_valid_filter {
            match self
                .filter_expr
                .encoded_filter_expr()
                .as_ref()
                .unwrap()
                .accept(&pred_eval)
            {
                Ok(matched) => {
                    if matched {
                        return sim * self.beta_value;
                    } else {
                        return sim;
                    }
                }
                Err(_) => {
                    //If predicate evaluation fails for any reason, we simply revert
                    //to unfiltered search.
                    tracing::warn!("Predicate evaluation failed");
                    return sim;
                }
            }
        } else {
            //If predicate evaluation fails, we will return the score returned by the
            //inner computer, as though no predicate was specified.
            tracing::warn!(
                "Predicate evaluation failed in OnlineBetaComputer::evaluate_similarity()"
            );
            sim
        }
    }
}

pub struct FilterResults<IPP> {
    inner_post_processor: IPP,
}

impl<Q, IA, IPP> SearchPostProcess<EncodedDocumentAccessor<IA>, FilteredQuery<Q>>
    for FilterResults<IPP>
where
    IA: BuildQueryComputer<Q>,
    Q: Clone + AsyncFriendly,
    IPP: SearchPostProcess<IA, Q> + Send + Sync,
{
    type Error = ANNError;

    async fn post_process<I, B>(
        &self,
        accessor: &mut EncodedDocumentAccessor<IA>,
        query: &FilteredQuery<Q>,
        computer: &InlineBetaComputer<<IA as BuildQueryComputer<Q>>::QueryComputer>,
        candidates: I,
        output: &mut B,
    ) -> Result<usize, Self::Error>
    where
        I: Iterator<Item = diskann::neighbor::Neighbor<IA::Id>> + Send,
        B: diskann::graph::SearchOutputBuffer<<IA as diskann::provider::HasId>::Id> + Send + ?Sized,
    {
        //This is a poor implementation - we should ideally be caching attributes of a point
        //along with neighbor. But that involves changes in the core algo, so I'm leaving it out
        //for now.
        //TODO: Fix for performance.
        let mut filtered_candidates = Vec::<Neighbor<IA::Id>>::new();
        for candidate in candidates {
            let doc = accessor.get_element(candidate.id).await?;
            let pe = PredicateEvaluator::new(doc.attributes());

            if computer.is_valid_filter() {
                if computer
                    .filter_expr()
                    .encoded_filter_expr()
                    .as_ref()
                    .unwrap()
                    .accept(&pe)?
                {
                    filtered_candidates.push(Neighbor::new(candidate.id, candidate.distance));
                }
            }
        }

        //Assuming that the job of the post processor is to only forward the right set of candidates and that
        //there will be a "terminal" post processor that copies data into "output".
        self.inner_post_processor
            .post_process(
                accessor.inner_accessor(),
                query.query(),
                &computer.inner_computer,
                filtered_candidates.into_iter(),
                output,
            )
            .await
            .map_err(|e| e.into())
    }
}

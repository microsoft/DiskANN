/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::error::IntoANNResult;
use diskann::graph::glue::{SearchPostProcess, SearchStrategy};
use diskann::neighbor::Neighbor;
use diskann::provider::{Accessor, BuildQueryComputer, DataProvider};

use diskann::ANNError;
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
    SearchStrategy<
        DocumentProvider<DP, RoaringAttributeStore<DP::InternalId>>,
        FilteredQuery<'_, Q>,
    > for InlineBetaStrategy<Strategy>
where
    DP: DataProvider,
    Strategy: SearchStrategy<DP, Q>,
    Q: Send + Sync + ?Sized,
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
}

impl<Inner> InlineBetaComputer<Inner> {
    pub(crate) fn new(
        inner_computer: Inner,
        beta_value: f32,
        filter_expr: EncodedFilterExpr,
    ) -> Self {
        Self {
            inner_computer,
            beta_value,
            filter_expr,
        }
    }

    pub(crate) fn filter_expr(&self) -> &EncodedFilterExpr {
        &self.filter_expr
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
        if self
            .filter_expr
            .encoded_filter_expr()
            .accept(&pred_eval)
            .expect("Expected predicate evaluation to not error out!")
        {
            sim * self.beta_value
        } else {
            sim
        }
    }
}

pub struct FilterResults<IPP> {
    inner_post_processor: IPP,
}

impl<'a, Q, IA, IPP> SearchPostProcess<EncodedDocumentAccessor<IA>, FilteredQuery<'a, Q>>
    for FilterResults<IPP>
where
    IA: BuildQueryComputer<Q>,
    IPP: SearchPostProcess<IA, Q> + Send + Sync,
    Q: Send + Sync + ?Sized,
{
    type Error = ANNError;

    async fn post_process<I, B>(
        &self,
        accessor: &mut EncodedDocumentAccessor<IA>,
        query: &FilteredQuery<'a, Q>,
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

            if computer.filter_expr().encoded_filter_expr().accept(&pe)? {
                filtered_candidates.push(Neighbor::new(candidate.id, candidate.distance));
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

#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};

    use diskann::{
        graph::{
            glue::{self, SearchPostProcess, SearchStrategy},
            search_output_buffer::IdDistance,
            test::provider::{Config, Context, Provider, StartPoint, Strategy},
        },
        neighbor::Neighbor,
        provider::{BuildQueryComputer, SetElement},
    };
    use diskann_vector::{distance::Metric, PreprocessedDistanceFunction};
    use roaring::RoaringTreemap;
    use serde_json::Value;

    use crate::{
        attribute::{Attribute, AttributeValue},
        document::EncodedDocument,
        encoded_attribute_provider::{
            attribute_encoder::AttributeEncoder, encoded_filter_expr::EncodedFilterExpr,
            roaring_attribute_store::RoaringAttributeStore,
        },
        inline_beta_search::encoded_document_accessor::EncodedDocumentAccessor,
        query::FilteredQuery,
        traits::attribute_store::AttributeStore,
        ASTExpr, CompareOp,
    };

    use super::{FilterResults, InlineBetaComputer};

    // -----------------------------------------------------------------------
    // Stub inner distance computer
    // -----------------------------------------------------------------------

    /// Always returns a fixed constant distance, regardless of the vector value.
    struct ConstComputer(f32);

    impl PreprocessedDistanceFunction<&[f32], f32> for ConstComputer {
        fn evaluate_similarity(&self, _: &[f32]) -> f32 {
            self.0
        }
    }

    // -----------------------------------------------------------------------
    // Helper: build an AttributeEncoder + ASTExpr for `field == value`,
    // returning (attr_map, ast_expr, encoded_id_of_that_attribute).
    // -----------------------------------------------------------------------

    fn setup_encoder_and_filter(
        field: &str,
        value: &str,
    ) -> (Arc<RwLock<AttributeEncoder>>, ASTExpr, u64) {
        let mut encoder = AttributeEncoder::new();
        let attr = Attribute::from_value(field, AttributeValue::String(value.to_owned()));
        let encoded_id = encoder.insert(&attr);
        let attr_map = Arc::new(RwLock::new(encoder));
        let ast_expr = ASTExpr::Compare {
            field: field.to_string(),
            op: CompareOp::Eq(Value::String(value.to_string())),
        };
        (attr_map, ast_expr, encoded_id)
    }

    // -----------------------------------------------------------------------
    // Test 1: when the filter matches, evaluate_similarity returns inner * beta
    // -----------------------------------------------------------------------

    #[test]
    fn test_evaluate_similarity_filter_match_scales_by_beta() {
        let (attr_map, ast_expr, color_red_id) = setup_encoder_and_filter("color", "red");
        let filter_expr = EncodedFilterExpr::new(&ast_expr, attr_map).expect("filter expr");

        let beta = 2.5_f32;
        let inner_dist = 4.0_f32;
        let computer = InlineBetaComputer::new(ConstComputer(inner_dist), beta, filter_expr);

        // Bitmap contains the encoded ID for "color=red" → predicate matches
        let mut matching_map = RoaringTreemap::new();
        matching_map.insert(color_red_id);
        let doc = EncodedDocument::new(&[1.0f32, 0.0][..], &matching_map);

        assert_eq!(
            computer.evaluate_similarity(doc),
            inner_dist * beta,
            "a matched filter should multiply the inner similarity by beta"
        );
    }

    // -----------------------------------------------------------------------
    // Test 2: when the filter does not match, evaluate_similarity is unchanged
    // -----------------------------------------------------------------------

    #[test]
    fn test_evaluate_similarity_no_filter_match_preserves_score() {
        let (attr_map, ast_expr, _) = setup_encoder_and_filter("color", "red");
        let filter_expr = EncodedFilterExpr::new(&ast_expr, attr_map).expect("filter expr");

        let beta = 2.5_f32;
        let inner_dist = 4.0_f32;
        let computer = InlineBetaComputer::new(ConstComputer(inner_dist), beta, filter_expr);

        // Empty bitmap → no attribute matches the predicate
        let empty_map = RoaringTreemap::new();
        let doc = EncodedDocument::new(&[1.0f32, 0.0][..], &empty_map);

        assert_eq!(
            computer.evaluate_similarity(doc),
            inner_dist,
            "an unmatched filter should leave the inner similarity unchanged"
        );
    }

    // -----------------------------------------------------------------------
    // Test 3: post_process forwards only filter-matching candidates to the
    //         inner post processor (and therefore to the output buffer).
    // -----------------------------------------------------------------------

    #[test]
    fn test_post_process_only_passes_matching_candidates_to_inner() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("test tokio runtime");

        // IDs 0 and 1 carry color=red  (should pass the filter)
        // IDs 2 and 3 carry color=blue (should be dropped by the filter)
        let attr_store = RoaringAttributeStore::<u32>::new();
        let red = Attribute::from_value("color", AttributeValue::String("red".to_owned()));
        let blue = Attribute::from_value("color", AttributeValue::String("blue".to_owned()));
        for id in 0u32..2 {
            attr_store
                .set_element(&id, std::slice::from_ref(&red))
                .expect("set red attr");
        }
        for id in 2u32..4 {
            attr_store
                .set_element(&id, std::slice::from_ref(&blue))
                .expect("set blue attr");
        }

        // The attribute_map is shared so EncodedFilterExpr sees the same encodings
        // as those stored by the attribute store.
        let attr_map = attr_store.attribute_map();

        let ast_expr = ASTExpr::Compare {
            field: "color".to_string(),
            op: CompareOp::Eq(Value::String("red".to_string())),
        };
        let filter_expr = EncodedFilterExpr::new(&ast_expr, attr_map.clone()).expect("filter expr");

        // Build the inner vector provider: start point at u32::MAX + 2-D zero vectors for 0..3
        let config = Config::new(Metric::L2, 10, StartPoint::new(u32::MAX, vec![1.0f32, 0.0]))
            .expect("provider config");
        let inner_provider = Provider::new(config);
        let ctx = Context::new();
        rt.block_on(async {
            for id in 0u32..4 {
                inner_provider
                    .set_element(&ctx, &id, &[0.0f32, 0.0] as &[f32])
                    .await
                    .expect("add vector to inner provider");
            }
        });

        // Obtain the inner search accessor and derive an inner computer from it
        let strategy = Strategy::new();
        let inner_accessor = strategy
            .search_accessor(&inner_provider, &ctx)
            .expect("inner accessor");
        let inner_computer = inner_accessor
            .build_query_computer(&[0.0f32, 0.0][..])
            .expect("inner computer");

        // Wrap accessor + attribute store into an EncodedDocumentAccessor
        let attribute_accessor = attr_store.attribute_accessor().expect("attribute accessor");
        let mut doc_accessor =
            EncodedDocumentAccessor::new(inner_accessor, attribute_accessor, attr_map, 2.0);

        let computer = InlineBetaComputer::new(inner_computer, 2.0, filter_expr);

        // Four candidates: 0 and 1 match (red); 2 and 3 do not (blue)
        let candidates = [
            Neighbor::new(0u32, 1.0_f32),
            Neighbor::new(1u32, 2.0_f32),
            Neighbor::new(2u32, 3.0_f32),
            Neighbor::new(3u32, 4.0_f32),
        ];

        let mut ids = [u32::MAX; 4];
        let mut distances = [f32::MAX; 4];
        let mut output = IdDistance::new(&mut ids, &mut distances);

        let query_vec = [0.0f32, 0.0];
        let filter_query = FilteredQuery::new(&query_vec[..], ast_expr);

        // CopyIds simply copies whatever it receives into the output buffer,
        // so the output reflects exactly what FilterResults lets through.
        let count = rt
            .block_on(
                FilterResults {
                    inner_post_processor: glue::CopyIds,
                }
                .post_process(
                    &mut doc_accessor,
                    &filter_query,
                    &computer,
                    candidates.into_iter(),
                    &mut output,
                ),
            )
            .expect("post_process");

        // Only the two red-labeled candidates should have been forwarded
        assert_eq!(count, 2, "exactly 2 of 4 candidates should pass the filter");
        let passed = &ids[..count];
        assert!(
            passed.contains(&0),
            "ID 0 (color=red) should pass the filter"
        );
        assert!(
            passed.contains(&1),
            "ID 1 (color=red) should pass the filter"
        );
    }
}

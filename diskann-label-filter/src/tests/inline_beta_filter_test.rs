/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

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
    inline_beta_search::{
        encoded_document_accessor::EncodedDocumentAccessor,
        inline_beta_filter::{FilterResults, InlineBetaComputer},
    },
    query::FilteredQuery,
    traits::attribute_store::AttributeStore,
    ASTExpr, CompareOp,
};

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
            FilterResults::new(glue::CopyIds).post_process(
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

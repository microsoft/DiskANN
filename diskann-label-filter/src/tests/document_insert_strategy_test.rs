/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

use diskann::{
    graph::{
        glue::{InsertStrategy, PruneStrategy, SearchExt, SearchStrategy},
        test::provider::{Config, Context, Provider, StartPoint, Strategy},
    },
    provider::BuildQueryComputer,
};
use diskann_vector::distance::Metric;

use crate::{
    document::Document,
    encoded_attribute_provider::{
        document_insert_strategy::{
            DocumentInsertStrategy, DocumentPruneStrategy, DocumentSearchAccessor,
        },
        document_provider::DocumentProvider,
        roaring_attribute_store::RoaringAttributeStore,
    },
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a minimal test provider with a single start point and three dimensions.
fn make_test_provider() -> Provider {
    let config = Config::new(
        Metric::L2,
        10,
        StartPoint::new(u32::MAX, vec![1.0f32, 2.0, 0.0]),
    )
    .expect("test provider config should be valid");
    Provider::new(config)
}

fn make_doc_provider(provider: Provider) -> DocumentProvider<Provider, RoaringAttributeStore<u32>> {
    DocumentProvider::new(provider, RoaringAttributeStore::new())
}

/// `search_accessor` successfully creates a `DocumentSearchAccessor` wrapping the
/// inner accessor.
#[test]
fn test_search_accessor_creates_wrapped_accessor() {
    let strategy = DocumentInsertStrategy::new(Strategy::new());
    let provider = make_doc_provider(make_test_provider());
    let context = Context::new();

    let result = <DocumentInsertStrategy<Strategy> as SearchStrategy<
        DocumentProvider<Provider, RoaringAttributeStore<u32>>,
        &Document<'_, [f32]>,
    >>::search_accessor(&strategy, &provider, &context);

    assert!(result.is_ok());
}

#[test]
fn test_insert_search_accessor_creates_wrapped_accessor() {
    let strategy = DocumentInsertStrategy::new(Strategy::new());
    let provider = make_doc_provider(make_test_provider());
    let context = Context::new();

    let result = <DocumentInsertStrategy<Strategy> as InsertStrategy<
        DocumentProvider<Provider, RoaringAttributeStore<u32>>,
        &Document<'_, [f32]>,
    >>::insert_search_accessor(&strategy, &provider, &context);

    assert!(result.is_ok());
}

#[test]
fn test_prune_accessor_delegates_to_inner_provider() {
    let doc_prune_strategy = DocumentPruneStrategy::new(Strategy::new());
    let provider = make_doc_provider(make_test_provider());
    let context = Context::new();

    let result = <DocumentPruneStrategy<Strategy> as PruneStrategy<
        DocumentProvider<Provider, RoaringAttributeStore<u32>>,
    >>::prune_accessor(&doc_prune_strategy, &provider, &context);

    assert!(result.is_ok());
}

#[test]
fn test_build_query_computer_extracts_vector_from_document() {
    let provider = make_test_provider();
    let context = Context::new();
    let strategy_inner = Strategy::new();
    let inner_accessor = strategy_inner
        .search_accessor(&provider, &context)
        .expect("creating search accessor should succeed");
    let doc_accessor = DocumentSearchAccessor::new(inner_accessor);

    let vector = vec![1.0f32, 2.0, 0.0];
    let doc = Document::new(vector.as_slice(), vec![]);

    let result = <DocumentSearchAccessor<_> as BuildQueryComputer<&Document<'_, [f32]>>>::build_query_computer(&doc_accessor, &doc);

    assert!(
        result.is_ok(),
        "build_query_computer should succeed for a valid vector"
    );
}

#[test]
fn test_terminate_early_delegates_to_inner() {
    let provider = make_test_provider();
    let context = Context::new();
    let strategy_inner = Strategy::new();
    let mut inner_accessor = strategy_inner
        .search_accessor(&provider, &context)
        .expect("creating search accessor should succeed");
    let inner_terminate_early = inner_accessor.terminate_early();
    let mut doc_accessor = DocumentSearchAccessor::new(inner_accessor);
    assert_eq!(
        inner_terminate_early,
        doc_accessor.terminate_early(),
        "terminate_early should have same value as inner accessor"
    );
}

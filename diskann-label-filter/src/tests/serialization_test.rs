/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Round-trip tests for `SaveWith` / `LoadWith` on `RoaringAttributeStore` and
//! `DocumentProvider`.

use std::sync::Arc;

use diskann::{
    graph::config::{Builder as ConfigBuilder, MaxDegree},
    provider::{DataProvider, DefaultContext},
    utils::ONE,
};
use diskann_providers::{
    index::{diskann_async, wrapped_async::DiskANNIndex},
    model::{
        configuration::IndexConfiguration,
        graph::provider::async_::{
            common::{FullPrecision, TableBasedDeletes},
            inmem::{self, CreateFullPrecision, DefaultProvider, DefaultProviderParameters},
        },
    },
    storage::{
        AsyncIndexMetadata, LoadWith, SaveWith, StorageReadProvider, VirtualStorageProvider,
    },
    utils::create_rnd_from_seed_in_tests,
};
use diskann_utils::test_data_root;
use diskann_vector::distance::Metric;

use crate::{
    attribute::{Attribute, AttributeValue},
    document::Document,
    encoded_attribute_provider::{
        document_insert_strategy::DocumentInsertStrategy, document_provider::DocumentProvider,
        roaring_attribute_store::RoaringAttributeStore,
    },
    set::traits::SetProvider,
    traits::attribute_store::AttributeStore,
};

/// Verify that an empty store round-trips correctly.
#[test]
fn test_roundtrip_empty_store() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();

    let store: RoaringAttributeStore<u32> = RoaringAttributeStore::new();
    let provider = VirtualStorageProvider::new_memory();
    let prefix = String::from("/index");

    rt.block_on(async {
        store.save_with(&provider, &prefix).await.unwrap();
        let loaded: RoaringAttributeStore<u32> =
            RoaringAttributeStore::load_with(&provider, &prefix)
                .await
                .unwrap();

        let index_arc = loaded.get_index();
        let index = index_arc.read().unwrap();
        assert_eq!(index.count().unwrap(), 0);
    });
}

/// Verify a store with several vectors and attribute types round-trips identically.
#[test]
fn test_roundtrip_with_data() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();

    let store: RoaringAttributeStore<u32> = RoaringAttributeStore::new();

    let attrs_0 = vec![
        Attribute::from_value("category", AttributeValue::String("electronics".into())),
        Attribute::from_value("price", AttributeValue::Real(299.99)),
        Attribute::from_value("available", AttributeValue::Bool(true)),
    ];
    let attrs_1 = vec![
        Attribute::from_value("category", AttributeValue::String("books".into())),
        Attribute::from_value("pages", AttributeValue::Integer(320)),
    ];
    let attrs_2 = vec![
        Attribute::from_value("flag", AttributeValue::Empty),
        Attribute::from_value("category", AttributeValue::String("electronics".into())),
    ];

    store.set_element(&0u32, &attrs_0).unwrap();
    store.set_element(&1u32, &attrs_1).unwrap();
    store.set_element(&2u32, &attrs_2).unwrap();

    let provider = VirtualStorageProvider::new_memory();
    let prefix = String::from("/index");

    rt.block_on(async {
        store.save_with(&provider, &prefix).await.unwrap();
        let loaded: RoaringAttributeStore<u32> =
            RoaringAttributeStore::load_with(&provider, &prefix)
                .await
                .unwrap();

        let orig_index_arc = store.get_index();
        let orig_index = orig_index_arc.read().unwrap();
        let loaded_index_arc = loaded.get_index();
        let loaded_index = loaded_index_arc.read().unwrap();

        assert_eq!(
            orig_index.count().unwrap(),
            loaded_index.count().unwrap(),
            "forward index entry count mismatch"
        );

        for node_id in [0u32, 1u32, 2u32] {
            let orig_set = orig_index
                .get(&node_id)
                .unwrap()
                .expect("original node missing");
            let loaded_set = loaded_index
                .get(&node_id)
                .unwrap()
                .expect("loaded node missing");
            assert_eq!(
                orig_set.len(),
                loaded_set.len(),
                "attribute-id set len mismatch for node {node_id}"
            );
            for id in orig_set.iter() {
                assert!(
                    loaded_set.contains(id),
                    "attribute id {id} missing after round-trip for node {node_id}"
                );
            }
        }

        // Attribute map size should match.
        let orig_attr_arc = store.attribute_map();
        let orig_attr = orig_attr_arc.read().unwrap();
        let loaded_attr_arc = loaded.attribute_map();
        let loaded_attr = loaded_attr_arc.read().unwrap();
        assert_eq!(
            orig_attr.len(),
            loaded_attr.len(),
            "attribute map size mismatch"
        );

        // ensure attribute values are deserialized correctly.
        for attrs in [attrs_0, attrs_1, attrs_2] {
            for attr in &attrs {
                let orig_id = orig_attr.get(&attr).unwrap();
                let loaded_id = loaded_attr.get(&attr).unwrap();
                assert_eq!(
                    orig_id, loaded_id,
                    "attribute ID mismatch for attribute {attr:?}"
                );
            }
        }

        // Inverted index must contain the same node sets for every attribute ID.
        let orig_inv_arc = store.get_inv_index();
        let orig_inv = orig_inv_arc.read().unwrap();
        let loaded_inv_arc = loaded.get_inv_index();
        let loaded_inv = loaded_inv_arc.read().unwrap();
        assert_eq!(
            orig_inv.count().unwrap(),
            loaded_inv.count().unwrap(),
            "inverted index entry count mismatch"
        );
        let mut attr_ids: Vec<u64> = Vec::new();
        orig_attr
            .for_each(|_, id| -> Result<(), ()> {
                attr_ids.push(id);
                Ok(())
            })
            .unwrap();
        for attr_id in attr_ids {
            let orig_nodes = orig_inv
                .get(&attr_id)
                .unwrap()
                .expect("attribute missing from original inverted index");
            let loaded_nodes = loaded_inv
                .get(&attr_id)
                .unwrap()
                .unwrap_or_else(|| panic!("attribute id {attr_id} missing from loaded inverted index"));
            assert_eq!(
                orig_nodes.len(),
                loaded_nodes.len(),
                "inverted index node-set len mismatch for attribute id {attr_id}"
            );
            for node_id in orig_nodes.iter() {
                assert!(
                    loaded_nodes.contains(node_id),
                    "node {node_id} missing from loaded inverted index for attribute id {attr_id}"
                );
            }
        }
    });
}

/// A u64 internal-id store must round-trip correctly and reject a u32 file.
#[test]
fn test_roundtrip_u64_node_ids() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();

    let store: RoaringAttributeStore<u64> = RoaringAttributeStore::new();
    let attrs = vec![Attribute::from_value(
        "kind",
        AttributeValue::String("doc".into()),
    )];
    // Use large IDs that wouldn't fit in a u32.
    store.set_element(&(u32::MAX as u64 + 1), &attrs).unwrap();
    store.set_element(&(u32::MAX as u64 + 2), &attrs).unwrap();

    let provider = VirtualStorageProvider::new_memory();
    let prefix = String::from("/idx64");

    rt.block_on(async {
        store.save_with(&provider, &prefix).await.unwrap();

        // Loading as u64 must succeed.
        let loaded: RoaringAttributeStore<u64> =
            RoaringAttributeStore::load_with(&provider, &prefix)
                .await
                .unwrap();

        {
            let index_arc = loaded.get_index();
            let index = index_arc.read().unwrap();
            assert_eq!(index.count().unwrap(), 2);
        } // index guard dropped before next await
    });
}

/// Loading a u64 label file as a u32 store must be rejected.
#[test]
fn test_load_u64_file_as_u32_fails() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();

    let store: RoaringAttributeStore<u64> = RoaringAttributeStore::new();
    let attrs = vec![Attribute::from_value(
        "kind",
        AttributeValue::String("doc".into()),
    )];
    store.set_element(&(u32::MAX as u64 + 1), &attrs).unwrap();

    let provider = VirtualStorageProvider::new_memory();
    let prefix = String::from("/idx64");

    rt.block_on(async {
        store.save_with(&provider, &prefix).await.unwrap();
        let result = RoaringAttributeStore::<u32>::load_with(&provider, &prefix).await;
        assert!(result.is_err(), "expected error loading u64 file as u32");
    });
}

/// Verify that a `DocumentProvider<DefaultProvider, RoaringAttributeStore>` with a built
/// index round-trips correctly: labels assigned during insertion survive save+load.
#[test]
fn test_document_provider_round_trip() {
    let save_path = "/doc_index";
    let file_path = "/sift/siftsmall_learn_256pts.fbin";

    // --- Load training vectors ---
    let train_data = {
        let storage = VirtualStorageProvider::new_overlay(test_data_root());
        let mut reader = storage.open_reader(file_path).unwrap();
        diskann_utils::io::read_bin::<f32>(&mut reader).unwrap()
    };

    let pq_bytes = 8;
    let pq_table = diskann_async::train_pq(
        train_data.as_view(),
        pq_bytes,
        &mut create_rnd_from_seed_in_tests(0xe3c52ef001bc7ade),
        2,
    )
    .unwrap();

    let (build_config, parameters) = {
        let config = ConfigBuilder::new_with(
            32,
            MaxDegree::default_slack(),
            20,
            Metric::L2.into(),
            |_| {},
        )
        .build()
        .unwrap();
        let params = DefaultProviderParameters {
            max_points: train_data.nrows(),
            frozen_points: ONE,
            metric: Metric::L2,
            dim: train_data.ncols(),
            prefetch_lookahead: None,
            prefetch_cache_line_level: None,
            max_degree: config.max_degree_u32().get(),
        };
        (config, params)
    };

    let fp_precursor =
        CreateFullPrecision::new(parameters.dim, parameters.prefetch_cache_line_level);
    let inner_provider =
        DefaultProvider::new_empty(parameters, fp_precursor, pq_table, TableBasedDeletes).unwrap();

    // --- Wrap inner provider in DocumentProvider and build the index ---
    let doc_provider = DocumentProvider::new(inner_provider, RoaringAttributeStore::<u32>::new());
    type InnerProvider = inmem::FullPrecisionProvider<
        f32,
        diskann_providers::model::graph::provider::async_::FastMemoryQuantVectorProviderAsync,
        diskann_providers::model::graph::provider::async_::TableDeleteProviderAsync,
    >;
    type TestDocProvider = DocumentProvider<InnerProvider, RoaringAttributeStore<u32>>;
    let index = DiskANNIndex::<TestDocProvider>::new_with_current_thread_runtime(
        build_config.clone(),
        doc_provider,
    );

    let storage = VirtualStorageProvider::new_memory();
    let ctx = DefaultContext;

    // Insert each vector with a synthetic label cycling through 5 categories.
    for (i, v) in train_data.row_iter().enumerate() {
        let label = format!("category_{}", i % 5);
        let attrs = vec![Attribute::from_value(
            "category",
            AttributeValue::String(label),
        )];
        let doc = Document::new(v, &attrs);
        index
            .insert(
                DocumentInsertStrategy::new(FullPrecision),
                &ctx,
                &(i as u32),
                &doc,
            )
            .unwrap();
    }

    // --- Save ---
    // DiskANNIndex<DocumentProvider<...>> has no dedicated SaveWith impl, so we reach
    // into data_provider and call save_with directly. The start_id required by
    // DefaultProvider::save_with is read from the inner provider.
    let save_metadata = AsyncIndexMetadata::new(save_path.to_string());
    let storage_ref = &storage;
    let metadata_ref = &save_metadata;
    index
        .run(|inner| {
            let inner = Arc::clone(inner);
            async move {
                let start_ids = inner
                    .data_provider
                    .inner_provider()
                    .starting_points()
                    .unwrap();
                let start_id = *start_ids
                    .first()
                    .expect("index must have a start point after build");
                inner
                    .data_provider
                    .save_with(storage_ref, &(start_id, metadata_ref.clone()))
                    .await
            }
        })
        .unwrap();

    // --- Load DiskANNIndex<TestDocProvider> via LoadWith<(&str, IndexConfiguration)> ---
    // DiskANNIndex<DP>: LoadWith<(&str, IndexConfiguration)> for any
    // DP: DataProvider<InternalId=u32> + LoadWith<AsyncQuantLoadContext>,
    // and TestDocProvider satisfies both.
    let load_config = IndexConfiguration::new(
        Metric::L2,
        train_data.ncols(),
        train_data.nrows(),
        ONE,
        1,
        build_config,
    );

    let loaded: DiskANNIndex<TestDocProvider> =
        DiskANNIndex::load_with_current_thread_runtime(&storage, &(save_path, load_config))
            .unwrap();

    // --- Assert inner DefaultProvider loaded correctly ---
    let inner = &loaded.inner.data_provider;

    // Graph must have at least one start point after rebuild.
    let start_pts = inner.inner_provider().starting_points().unwrap();
    assert!(
        !start_pts.is_empty(),
        "loaded index should have start points"
    );

    // Every inserted external ID must map to a valid internal ID.
    let ctx = DefaultContext;
    for i in 0..train_data.nrows() as u32 {
        assert!(
            inner.to_internal_id(&ctx, &i).is_ok(),
            "external id {i} missing after round trip"
        );
    }

    // --- Assert labels survived the round trip ---
    let attr_store = inner.attribute_store();
    for i in 0..train_data.nrows() {
        assert!(
            attr_store.id_exists(&(i as u32)).unwrap(),
            "node {i} should have labels after round trip",
        );
    }
}

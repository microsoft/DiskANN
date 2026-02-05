/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{iter, sync::Arc};

use diskann_vector::distance::Metric;

use crate::graph::{
    self, AdjacencyList, DiskANNIndex,
    test::provider::{self as test_provider},
};

fn inplace_delete_setup() -> Arc<DiskANNIndex<test_provider::Provider>> {
    let provider_config = test_provider::Config::new(
        Metric::L2,
        10,
        test_provider::StartPoint::new(0, vec![0.0, 0.0]),
    )
    .unwrap();
    let provider = test_provider::Provider::new_from(
        provider_config,
        iter::once((0, AdjacencyList::new())),
        iter::empty(),
    )
    .unwrap();

    let index_config = graph::config::Builder::new(
        10,
        graph::config::MaxDegree::default_slack(),
        15,
        Metric::L2.into(),
    )
    .build()
    .unwrap();

    Arc::new(DiskANNIndex::new(index_config, provider, None))
}

/// Test that `inplace_delete()` succeeds on a simple index. The test provider will refuse to
/// translate internal/external IDs of deleted vectors once the `DataProvider::delete` call
/// returns. Inplace delete should still be able to complete successfully.
#[tokio::test(flavor = "current_thread")]
async fn basic_single() {
    let index = inplace_delete_setup();

    let ctx = test_provider::Context::default();
    let strat = test_provider::Strategy::new();

    for i in 1..6 {
        index
            .insert(strat, &ctx, &i, &[i as f32, i as f32])
            .await
            .unwrap();
    }

    index
        .inplace_delete(strat, &ctx, &3, 3, graph::InplaceDeleteMethod::OneHop)
        .await
        .unwrap();
}

/// Test that `multi_inplace_delete()` succeeds on a simple index. The test provider will refuse to
/// translate internal/external IDs of deleted vectors once the `DataProvider::delete` call
/// returns. Inplace delete should still be able to complete successfully. As the single and multi
/// in place delete logic have slightly different code paths for ID tranlsations, we have tests
/// for both.
#[tokio::test(flavor = "current_thread")]
async fn basic_multi() {
    let index = inplace_delete_setup();

    let ctx = test_provider::Context::default();
    let strat = test_provider::Strategy::new();

    for i in 1..6 {
        index
            .insert(strat, &ctx, &i, &[i as f32, i as f32])
            .await
            .unwrap();
    }

    index
        .multi_inplace_delete(
            strat,
            &ctx,
            Arc::new([3, 4]),
            3,
            graph::InplaceDeleteMethod::OneHop,
        )
        .await
        .unwrap();
}

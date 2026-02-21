/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::time::Duration;

use criterion::Criterion;
use diskann::provider::DefaultContext;
use diskann_providers::{
    index::diskann_async,
    model::graph::{
        provider::async_::{
            common::{FullPrecision, NoDeletes},
            inmem::DefaultProviderParameters,
        },
        traits::AdHoc,
    },
    storage::FileStorageProvider,
    utils::{VectorDataIterator, create_thread_pool_for_bench, load_bin},
};
use diskann_providers::storage::StorageReadProvider;
use diskann_vector::distance::Metric;
use tokio::runtime::Runtime;

pub fn benchmark_diskann_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("diskann-async-insert");
    group
        .measurement_time(Duration::from_secs(3))
        .sample_size(10);
    let rt = Runtime::new().unwrap();
    group.bench_function("DiskANN insert", |f| {
        f.iter(|| {
            rt.block_on(async {
                test_sift_256_vectors_with_quant_vectors().await;
            });
        });
    });
}

async fn test_sift_256_vectors_with_quant_vectors() {
    use diskann::graph::config;

    let l = 10;
    let target_degree = 32;
    let file_path = "test_data/sift/siftsmall_learn_256pts.fbin";

    let storage_provider = FileStorageProvider;
    let dataset_iterator = VectorDataIterator::<FileStorageProvider, AdHoc<f32>>::new(
        get_test_file_path(file_path).as_str(),
        Option::None,
        &storage_provider,
    )
    .unwrap();

    let train_data =
        load_bin::<f32>(&mut storage_provider.open_reader(get_test_file_path(file_path).as_str()).unwrap(), 0).unwrap();

    let pool = create_thread_pool_for_bench();
    let pq_chunk_table = diskann_async::train_pq(
        train_data.as_view(),
        32,
        &mut diskann_providers::utils::create_rnd_in_tests(),
        &pool,
    )
    .unwrap();

    let conf = config::Builder::new(
        target_degree,
        config::MaxDegree::default_slack(),
        l,
        (Metric::L2).into(),
    )
    .build()
    .unwrap();

    let provider_params = DefaultProviderParameters::simple(
        train_data.nrows(),
        train_data.ncols(),
        Metric::L2,
        conf.max_degree_u32().get(),
    );

    let index =
        diskann_async::new_quant_index(conf, provider_params, pq_chunk_table, NoDeletes).unwrap();

    for (pos, (vector, _associated_data)) in dataset_iterator.enumerate() {
        index
            .insert(FullPrecision, &DefaultContext, &(pos as u32), &vector)
            .await
            .unwrap();
    }
}

pub fn get_test_file_path(relative_path: &str) -> String {
    format!("{}/{}", env!("CARGO_MANIFEST_DIR"), relative_path)
}

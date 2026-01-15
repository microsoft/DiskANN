/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNResult;
use diskann_providers::storage::StorageReadProvider;
use diskann_providers::{
    model::{
        graph::traits::GraphDataType, GeneratePivotArguments, MAX_PQ_TRAINING_SET_SIZE,
        NUM_KMEANS_REPS_PQ, NUM_PQ_CENTROIDS,
    },
    storage::{
        get_disk_index_compressed_pq_file, get_disk_index_pq_pivot_file, FileStorageProvider,
        PQStorage,
    },
    utils::{load_metadata_from_file, Timer},
};
use diskann_vector::distance::Metric;
use tracing::info;

pub struct BuildPQParameters<'a> {
    pub metric: Metric,
    pub data_path: &'a str,
    pub index_path_prefix: &'a str,
    pub num_threads: usize,
    pub p_val: f64,
    pub pq_bytes: f64,
}

pub fn build_pq<Data: GraphDataType>(
    storage_provider: &impl StorageReadProvider,
    parameters: BuildPQParameters,
) -> ANNResult<()> {
    let num_pq_chunks = parameters.pq_bytes as usize;

    let data_path = parameters.data_path;
    let disk_pq_pivot_path = get_disk_index_pq_pivot_file(parameters.index_path_prefix);
    let disk_pq_compressed_data_path =
        get_disk_index_compressed_pq_file(parameters.index_path_prefix);

    let mut pq_storage = PQStorage::new(
        &disk_pq_pivot_path,
        &disk_pq_compressed_data_path,
        Some(data_path),
    );

    let metadata = load_metadata_from_file(storage_provider, parameters.data_path)?;
    info!(
        "Compressing dim-{} data into {} chunks(bytes) for PQ",
        metadata.ndims, num_pq_chunks
    );

    let p_val = MAX_PQ_TRAINING_SET_SIZE / (metadata.npoints as f64);

    let timer = Timer::new();
    let storage_provider = FileStorageProvider;
    let random_provider = diskann_providers::utils::create_rnd_provider_from_seed(42);

    let (mut train_data_vector, num_train, train_dim) = pq_storage
        .get_random_train_data_slice::<Data::VectorDataType, _>(
            p_val,
            &storage_provider,
            &mut random_provider.create_rnd(),
        )?;

    diskann_providers::model::pq::generate_pq_pivots(
        GeneratePivotArguments::new(
            num_train,
            train_dim,
            NUM_PQ_CENTROIDS,
            num_pq_chunks,
            NUM_KMEANS_REPS_PQ,
            false,
        )?,
        &mut train_data_vector,
        &pq_storage,
        &storage_provider,
        random_provider,
        parameters.num_threads,
    )?;

    diskann_providers::model::pq::generate_pq_data_from_pivots::<f32, _, _>(
        NUM_PQ_CENTROIDS,
        num_pq_chunks,
        &mut pq_storage,
        &storage_provider,
        false,
        0,
        parameters.num_threads,
    )?;

    info!(
         "PQ build completed in {:.3} seconds, {:.3}B cycles, {:.3}% CPU time, peak memory {:.3} GBs for {} chunks, using {} threads",
         timer.elapsed_seconds(),
         timer.elapsed_gcycles(),
         timer.get_average_cpu_time_in_percents(),
         timer.get_peak_memory_usage(),
         num_pq_chunks,
         parameters.num_threads
     );

    Ok(())
}

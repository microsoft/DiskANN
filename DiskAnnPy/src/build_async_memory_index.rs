/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fs,
    num::NonZeroUsize,
    path::Path,
    sync::{Arc, Mutex},
};

use diskann::{
    graph::config,
    provider::DefaultContext,
    utils::{vecid_from_usize, IntoUsize, VectorIdBoxSlice, VectorRepr, ONE},
    ANNError, ANNResult,
};
use diskann_disk::utils::instrumentation::PerfLogger;
use diskann_providers::storage::{StorageReadProvider, StorageWriteProvider};
use diskann_providers::{
    index::diskann_async::{self, train_pq, MemoryIndex, PQMemoryIndex},
    model::{
        graph::{
            provider::async_::{
                common::{FullPrecision, Hybrid, NoDeletes, TableBasedDeletes},
                inmem::{DefaultProviderParameters, SetStartPoints},
                TableDeleteProviderAsync,
            },
            traits::AdHoc,
        },
        IndexConfiguration, MAX_PQ_TRAINING_SET_SIZE, NUM_PQ_CENTROIDS,
    },
    storage::{self, FileStorageProvider, PQStorage, SaveWith},
    utils::{BridgeErr, PQPathNames, Timer, VectorDataIterator},
};
use diskann_utils::{io::Metadata, views::MatrixView};
use diskann_vector::distance::Metric;
use pyo3::prelude::*;
use rand::SeedableRng;
use tokio::task::JoinSet;

use crate::{
    async_memory_index::AsyncIndexType,
    utils::{
        common_error, get_graph_num_frozen_points, init_runtime, ANNErrorPy, DataType, MetricPy,
    },
};

// Use the type aliases in the enum definition
#[allow(dead_code)]
enum AsyncMemoryIndexEnum {
    F32(PQMemoryIndex<f32>),
    U8(PQMemoryIndex<u8>),
    Int8(PQMemoryIndex<i8>),
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn build_memory_index(
    data_type: DataType,
    metric: MetricPy,
    data_path: String,
    index_path: String,
    r: u32,
    l: u32,
    alpha: f32,
    num_start_pts: usize,
    num_threads: u32,
    build_pq_bytes: usize,
    graph_slack_factor: f32,
    max_fp_vecs_per_prune: usize,
    backedge_ratio: f32,
    num_tasks: usize,
    pq_seed: u64,
    insert_minibatch_size: usize,
) -> PyResult<()> {
    let storage_provider = FileStorageProvider;
    let result = match data_type {
        DataType::Float => build_async_in_memory_index::<f32, FileStorageProvider>(
            metric.into(),
            data_path,
            index_path,
            r,
            l,
            alpha,
            num_start_pts,
            num_threads,
            build_pq_bytes,
            graph_slack_factor,
            max_fp_vecs_per_prune,
            backedge_ratio,
            &storage_provider,
            num_tasks,
            pq_seed,
            insert_minibatch_size,
        )
        .map(AsyncMemoryIndexEnum::F32),
        DataType::Uint8 => build_async_in_memory_index::<u8, FileStorageProvider>(
            metric.into(),
            data_path,
            index_path,
            r,
            l,
            alpha,
            num_start_pts,
            num_threads,
            build_pq_bytes,
            graph_slack_factor,
            max_fp_vecs_per_prune,
            backedge_ratio,
            &storage_provider,
            num_tasks,
            pq_seed,
            insert_minibatch_size,
        )
        .map(AsyncMemoryIndexEnum::U8),
        DataType::Int8 => build_async_in_memory_index::<i8, FileStorageProvider>(
            metric.into(),
            data_path,
            index_path,
            r,
            l,
            alpha,
            num_start_pts,
            num_threads,
            build_pq_bytes,
            graph_slack_factor,
            max_fp_vecs_per_prune,
            backedge_ratio,
            &storage_provider,
            num_tasks,
            pq_seed,
            insert_minibatch_size,
        )
        .map(AsyncMemoryIndexEnum::Int8),
    };

    match result {
        Ok(_) => {
            println!("Index build completed successfully");
            Ok(())
        }
        Err(err) => {
            eprintln!("Error: {:?}", err);
            Err(PyErr::from(ANNErrorPy::new(err))) //Err(ANNErrorPy::new(err).into())
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn load_index<T>(
    metric: MetricPy,
    index_path: String,
    r: u32,
    l: u32,
    num_threads: u32,
    build_pq_bytes: usize,
    graph_slack_factor: f32,
) -> ANNResult<AsyncIndexType<T>>
where
    T: VectorRepr,
{
    let storage_provider: FileStorageProvider = FileStorageProvider;
    let num_frozen_pts = get_graph_num_frozen_points(&index_path)?;

    let metric: Metric = metric.into();

    let config = config::Builder::new(
        r.into_usize(),
        config::MaxDegree::slack(graph_slack_factor),
        l.into_usize(),
        metric.into(),
    )
    .build()?;

    let metadata =
        Metadata::read(&mut storage_provider.open_reader(&format!("{}.data", &index_path))?)?;
    let index_config = IndexConfiguration::new(
        metric,
        metadata.ndims(),
        metadata.npoints() - num_frozen_pts.get(),
        num_frozen_pts,
        num_threads.into_usize(),
        config,
    )
    .with_pseudo_rng()
    .with_prefetch_lookahead(NonZeroUsize::new(8));

    let runtime = init_runtime(num_threads as usize)
        .map_err(|_| common_error("Failed to initialize tokio runtime."))?;
    if build_pq_bytes > 0 {
        let index_pq = runtime.block_on(storage::load_pq_index_with_deletes::<T, _>(
            &storage_provider,
            &index_path,
            index_config.clone(),
        ))?;
        Ok(AsyncIndexType::PQIndex(Arc::new(index_pq)))
    } else {
        let index_nopq = runtime.block_on(storage::load_index_with_deletes::<T, _>(
            &storage_provider,
            &index_path,
            index_config.clone(),
        ))?;
        Ok(AsyncIndexType::NoPQIndex(Arc::new(index_nopq)))
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn build_empty_index<T>(
    metric: MetricPy,
    r: u32,
    l: u32,
    alpha: f32,
    graph_slack_factor: f32,
    max_points: usize,
    dim: usize,
) -> ANNResult<MemoryIndex<T, TableDeleteProviderAsync>>
where
    T: VectorRepr,
{
    let metric: diskann_vector::distance::Metric = metric.into();

    let config = config::Builder::new_with(
        r.into_usize(),
        config::MaxDegree::slack(graph_slack_factor),
        l.into_usize(),
        metric.into(),
        |builder| {
            builder.alpha(alpha).max_minibatch_par(8);
        },
    )
    .build()?;

    let provider_params = DefaultProviderParameters {
        max_points,
        frozen_points: ONE,
        metric,
        dim,
        max_degree: config.max_degree().get() as u32,
        prefetch_lookahead: Some(8),
        prefetch_cache_line_level: None,
    };

    diskann_async::new_index(config, provider_params, TableBasedDeletes)
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn build_index<T>(
    metric: MetricPy,
    data_path: String,
    index_path: String,
    r: u32,
    l: u32,
    alpha: f32,
    num_start_pts: usize,
    num_threads: u32,
    build_pq_bytes: usize,
    graph_slack_factor: f32,
    max_fp_vecs_per_prune: usize,
    backedge_ratio: f32,
    num_tasks: usize,
    pq_seed: u64,
    insert_minibatch_size: usize,
) -> ANNResult<PQMemoryIndex<T>>
where
    T: VectorRepr,
{
    let storage_provider = FileStorageProvider;
    if Path::new(&data_path).exists() {
        let _ = storage_provider.delete(&data_path);

        let compressed_file_path = format!("{}_build_pq_1_compressed.bin", data_path);
        if Path::new(&compressed_file_path).exists() {
            storage_provider
                .delete(&compressed_file_path)
                .expect("Failed to delete the compressed file");
        }

        let new_data_path = format!("{}.data", &data_path);
        if Path::new(&new_data_path).exists() {
            fs::rename(&new_data_path, &data_path).expect("Failed to rename the file");
            println!("Renamed {} to", new_data_path);
        } else {
            panic!("The file {} does not exist", new_data_path);
        }
    } else {
        panic!("The file {} does not exist", data_path);
    }
    let result = build_async_in_memory_index::<T, FileStorageProvider>(
        metric.into(),
        data_path,
        index_path,
        r,
        l,
        alpha,
        num_start_pts,
        num_threads,
        build_pq_bytes,
        graph_slack_factor,
        max_fp_vecs_per_prune,
        backedge_ratio,
        &storage_provider,
        num_tasks,
        pq_seed,
        insert_minibatch_size,
    );

    match result {
        Ok(index_enum) => {
            println!("Index build completed successfully");
            Ok(index_enum)
        }
        Err(err) => {
            eprintln!("Error: {:?}", err);
            Err(err)
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_async_in_memory_index<T, StorageType>(
    metric: Metric,
    data_path: String,
    index_path: String,
    r: u32,
    l: u32,
    alpha: f32,
    num_start_pts: usize,
    num_threads: u32,
    build_pq_bytes: usize,
    graph_slack_factor: f32,
    max_fp_vecs_per_prune: usize,
    backedge_ratio: f32,
    storage_provider: &StorageType,
    num_tasks: usize,
    pq_seed: u64,
    insert_minibatch_size: usize,
) -> ANNResult<PQMemoryIndex<T>>
where
    T: VectorRepr,
    StorageType: StorageReadProvider + StorageWriteProvider + 'static,
    <StorageType as StorageReadProvider>::Reader: Send,
{
    assert!(insert_minibatch_size > 0);
    let mut logger = PerfLogger::new("build_async_memory_index".to_string(), true);

    let metadata = Metadata::read(&mut storage_provider.open_reader(&data_path)?)?;
    let (max_points, dim) = metadata.into_dims();

    let runtime = init_runtime(num_threads.into_usize())
        .map_err(|_| common_error("Failed to initialize tokio runtime."))?;

    let index = runtime.block_on(async move {
        let pq_enabled = build_pq_bytes > 0;
        let index = {
            // only save pq pivot data
            let pq_path = PQPathNames::new(&index_path);
            let pq_storage = PQStorage::new(&pq_path.pivots, "", Some(&data_path));
            let mut rng = rand::rngs::StdRng::seed_from_u64(pq_seed);

            let p_val = MAX_PQ_TRAINING_SET_SIZE / (max_points as f64);
            let (mut train_data, num_train, dim_train) = pq_storage
                .get_random_train_data_slice::<T, StorageType>(p_val, storage_provider, &mut rng)?;

            let (pq_bytes, num_train) = if pq_enabled {
                (build_pq_bytes, num_train)
            } else {
                // Minimal training data.
                train_data.truncate(dim_train);
                (1, 1)
            };
            let pq_chunk_table = train_pq(
                MatrixView::try_from(&train_data, num_train, dim).bridge_err()?,
                pq_bytes,
                &mut rng,
                num_threads.into_usize(),
            )?;

            if pq_enabled {
                // write pivot data to disk
                let _ = pq_storage.write_pivot_data(
                    pq_chunk_table.get_pq_table(),
                    pq_chunk_table.get_centroids(),
                    pq_chunk_table.get_chunk_offsets(),
                    NUM_PQ_CENTROIDS,
                    dim,
                    storage_provider,
                );
                tracing::info!("Write PQ pivots complete.");
            }

            let num_start_pts = NonZeroUsize::new(num_start_pts).ok_or_else(|| {
                ANNError::log_index_config_error(
                    "num_start_points".to_string(),
                    "num_start_points must be atleast 1".to_string(),
                )
            })?;

            let config = config::Builder::new_with(
                r.into_usize(),
                config::MaxDegree::slack(graph_slack_factor),
                l.into_usize(),
                metric.into(),
                |builder| {
                    builder
                        .alpha(alpha)
                        .max_minibatch_par(8)
                        .backedge_ratio(backedge_ratio);
                },
            )
            .build()?;

            let provider_params = DefaultProviderParameters {
                max_points,
                frozen_points: num_start_pts,
                metric,
                dim,
                max_degree: config.max_degree().get() as u32,
                prefetch_lookahead: Some(8),
                prefetch_cache_line_level: None,
            };

            diskann_async::new_quant_index(config, provider_params, pq_chunk_table, NoDeletes)?
        };

        // Allow the iterator to be shared
        let dataset_iter = Arc::new(Mutex::new(
            VectorDataIterator::<StorageType, AdHoc<T>>::new(
                &data_path,
                Option::None,
                storage_provider,
            )?
            .enumerate(),
        ));

        {
            let temp_iter = VectorDataIterator::<StorageType, AdHoc<T>>::new(
                &data_path,
                None,
                storage_provider,
            )?;
            let start_points = temp_iter
                .take(index.provider().num_start_points())
                .map(|(v, _)| v.to_vec())
                .collect::<Vec<Vec<_>>>();
            index
                .provider()
                .set_start_points(start_points.iter().map(|x| x.as_slice()))?;
        }

        let max_fp_vecs_per_prune = if max_fp_vecs_per_prune > 0 {
            Some(max_fp_vecs_per_prune)
        } else {
            None
        };
        let hybrid = Hybrid::new(max_fp_vecs_per_prune);

        let timer = Timer::new();

        // This variable is used when insert_mini_batch_size > 1.
        // when insert_mini_batch_size is large enough to be comparable or higher than graph degree, we risk creating
        // disconnected components in the graph if we start with a batch size that is too large.
        // So we start with a batch siz eof 1 and double the batch size until we reach insert_mini_batch_size.
        let mut current_batch_size = 1;

        // Spawn the requested number of tasks.
        let mut tasks = JoinSet::new();
        for _ in 0..num_tasks {
            let index_clone = index.clone();
            let dataset_iter_clone = dataset_iter.clone();
            tasks.spawn(async move {
                let ctx = &DefaultContext;
                loop {
                    if insert_minibatch_size == 1 {
                        let result = {
                            let mut guard = dataset_iter_clone
                                .lock()
                                .map_err(|_| common_error("Poisoned mutex during construction"))?;
                            guard.next()
                        };

                        match result {
                            Some((i, (vector, _))) => {
                                let i: u32 = vecid_from_usize(i)?;
                                if pq_enabled {
                                    index_clone.insert(hybrid, ctx, &i, &vector).await?;
                                } else {
                                    index_clone.insert(FullPrecision, ctx, &i, &vector).await?;
                                }
                            }
                            None => break,
                        }
                    } else {
                        let mut vector_id_pairs =
                            Vec::<VectorIdBoxSlice<u32, T>>::with_capacity(current_batch_size);
                        {
                            let mut guard = dataset_iter_clone
                                .lock()
                                .map_err(|_| common_error("Poinsoned mutex during construction"))?;
                            for _ in 0..current_batch_size {
                                match guard.next() {
                                    Some((i, (vector, _))) => {
                                        vector_id_pairs.push(VectorIdBoxSlice::new(
                                            vecid_from_usize(i)?,
                                            vector,
                                        ));
                                    }
                                    None => break,
                                }
                            }
                        }

                        if !vector_id_pairs.is_empty() {
                            if pq_enabled {
                                index_clone
                                    .multi_insert(hybrid, ctx, vector_id_pairs.into())
                                    .await?;
                            } else {
                                index_clone
                                    .multi_insert(FullPrecision, ctx, vector_id_pairs.into())
                                    .await?;
                            }
                        } else {
                            break;
                        }
                        // Double the batch size each loop until we reach insert_mini_batch_size
                        if current_batch_size < insert_minibatch_size {
                            current_batch_size =
                                std::cmp::min(2 * current_batch_size, insert_minibatch_size);
                        }
                    }
                }
                ANNResult::Ok(())
            });
        }

        println!("Batch insert completed");

        while let Some(res) = tasks.join_next().await {
            res.map_err(|_| common_error("A spawned task failed"))??;
        }

        // index.build_from_iter(Box::new(dataset_iter)).await?;
        let diff = timer.elapsed();

        tracing::info!(
            "Number of points reachable in the graph: {}",
            index
                .count_reachable_nodes(
                    &index.provider().starting_points()?,
                    &mut index.provider().neighbors()
                )
                .await?
        );

        // #[cfg(test)]
        // tracing::info!(
        //     "Number of get vector calls per insert: {}",
        //     index.vector_provider.num_get_calls.get() as f32 / max_points as f32
        // );
        // #[cfg(test)]
        // tracing::info!(
        //     "Number of get quantized vector calls per insert: {}",
        //     index.quant_vec_provider.num_get_calls.get() as f32 / max_points as f32
        // );

        tracing::info!("Indexing time: {}", diff.as_secs_f64());

        index
            .save_with(
                storage_provider,
                &storage::AsyncIndexMetadata::new(index_path),
            )
            .await?;

        logger.log_checkpoint("async_index_created");
        ANNResult::Ok(index)
    })?;
    Ok(index)
}

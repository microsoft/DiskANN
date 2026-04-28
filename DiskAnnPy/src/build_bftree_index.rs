/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    num::NonZeroUsize,
    path::Path,
    sync::{Arc, Mutex},
};

use diskann::{
    error::IntoANNResult,
    graph::{config, DiskANNIndex},
    provider::DefaultContext,
    utils::{vecid_from_usize, IntoUsize, VectorIdBoxSlice, VectorRepr},
    ANNError, ANNResult,
};
use diskann_disk::utils::instrumentation::PerfLogger;
use diskann_providers::storage::{StorageReadProvider, StorageWriteProvider};
use diskann_providers::{
    index::diskann_async::train_pq,
    model::{
        graph::{
            provider::async_::{
                bf_tree::{
                    AsVectorDtype, BfTreePaths, BfTreeProvider, BfTreeProviderParameters, Config,
                    GraphParams,
                },
                common::{FullPrecision, Hybrid, NoStore, TableBasedDeletes},
            },
            traits::AdHoc,
        },
        MAX_PQ_TRAINING_SET_SIZE,
    },
    utils::{BridgeErr, Timer, VectorDataIterator},
};
use diskann_utils::{
    io::Metadata,
    views::{Matrix, MatrixView},
};
use diskann_vector::distance::Metric;
use rand::SeedableRng;
use tokio::task::JoinSet;

use crate::bftree_index::{BfTreeIndex, FullPrecisionBfTreeIndex, PQBfTreeIndex};

use crate::utils::{common_error, init_runtime};

pub const MIN_BUFFER_SIZE: usize = 8192; // 8KB

pub const MEMORY_SLACK: usize = 2;

// Use the type aliases in the enum definition
#[allow(dead_code)]
enum BfTreeIndexEnum {
    F32(BfTreeIndex<f32>),
    U8(BfTreeIndex<u8>),
    Int8(BfTreeIndex<i8>),
}

/// Configure vector provider page and record sizes based on vector size
fn configure_vector_provider(
    size: usize,
    path: &str,
    buffer_size: usize,
    on_disk: bool,
) -> ANNResult<Config> {
    let mut config = Config::new(Path::new(path), buffer_size);

    // leaf_page_size in config should be multiple of 64
    // cb_min_record_size cannot be larger than cb_max_record_size
    let max_record_size = (MEMORY_SLACK * size).max(1952);
    let leaf_page_size = (2 * max_record_size + 128).next_power_of_two();

    if max_record_size > 16384 {
        return Err(ANNError::log_index_config_error(
            "max_record_size".to_string(),
            format!(
                "max_record_size must be <= 16384 bytes, got {}",
                max_record_size
            ),
        ));
    }

    if leaf_page_size > 32768 {
        return Err(ANNError::log_index_config_error(
            "leaf_page_size".to_string(),
            format!(
                "leaf_page_size must be <= 32768 bytes, got {}",
                leaf_page_size
            ),
        ));
    }

    config.leaf_page_size(leaf_page_size);
    config.cb_max_record_size(max_record_size);

    // these additional bf-tree settings were suggested by Yi Shan to get the best index build performance
    config.read_promotion_rate(100).read_record_cache(false);

    if on_disk {
        config.storage_backend(::bf_tree::StorageBackend::Std);
    }

    Ok(config)
}

/// Helper function to create BfTree provider configurations
#[allow(clippy::too_many_arguments)]
fn create_bftree_provider_configs<T>(
    bf_tree_data_path: &str,
    bf_tree_graph_path: &str,
    max_points: usize,
    num_start_pts: usize,
    dim: usize,
    max_degree: usize,
    on_disk: bool,
) -> ANNResult<(Config, Config)>
where
    T: VectorRepr,
{
    let min_circular_buffer_size_in_bytes = 8192; // 8KB

    let neighbor_provider_buffer_size_in_bytes = (MEMORY_SLACK * // slack factor to account for fragmentation and other overheads
        (4 * // u32 id size
        (max_points + num_start_pts + 1) * // max points + starting points + 1 (for u32 key)
        (max_degree + 1)) // max degree + 1 (to store the length)
        .next_power_of_two())
    .max(min_circular_buffer_size_in_bytes);

    let vector_provider_buffer_size_in_bytes = (MEMORY_SLACK * // slack factor to account for fragmentation and other overheads
        ((max_points + num_start_pts) * // max points + starting points
        (dim * std::mem::size_of::<T>()) + // vector size
        4 * max_points) // u32 key size
        .next_power_of_two())
    .max(min_circular_buffer_size_in_bytes);

    let size = std::mem::size_of::<T>() * dim + 4;
    let vector_provider_config = configure_vector_provider(
        size,
        bf_tree_data_path,
        vector_provider_buffer_size_in_bytes,
        on_disk,
    )?;

    let size = 4 * max_degree + 4; // u32 id size * max degree + u32 length size
    let neighbor_list_provider_config = configure_vector_provider(
        size,
        bf_tree_graph_path,
        neighbor_provider_buffer_size_in_bytes,
        on_disk,
    )?;

    Ok((vector_provider_config, neighbor_list_provider_config))
}

#[allow(clippy::too_many_arguments)]
pub fn build_empty_bftree_index<T>(
    metric: Metric,
    r: u32,
    l: u32,
    alpha: f32,
    graph_slack_factor: f32,
    max_points: usize,
    dim: usize,
    num_start_pts: usize,
    backedge_ratio: f32,
) -> ANNResult<FullPrecisionBfTreeIndex<T>>
where
    T: VectorRepr + AsVectorDtype,
{
    let num_start_points = NonZeroUsize::new(num_start_pts).ok_or_else(|| {
        ANNError::log_index_config_error(
            "num_start_points".to_string(),
            "num_start_points must be at least 1".to_string(),
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

    // bf-tree path is memory only
    let bf_tree_data_path = format!(":memory:{}", BfTreePaths::vectors_bftree("empty").display());
    let bf_tree_graph_path = format!(
        ":memory:{}",
        BfTreePaths::neighbors_bftree("empty").display()
    );

    let (vector_provider_config, neighbor_list_provider_config) =
        create_bftree_provider_configs::<T>(
            &bf_tree_data_path,
            &bf_tree_graph_path,
            max_points,
            num_start_pts,
            dim,
            config.max_degree().get(),
            false,
        )?;

    let provider_params = BfTreeProviderParameters {
        max_points,
        num_start_points,
        dim,
        metric,
        max_fp_vecs_per_fill: None,
        max_degree: config.max_degree().get() as u32,
        vector_provider_config,
        quant_vector_provider_config: Config::default(),
        neighbor_list_provider_config,
        graph_params: Some(GraphParams {
            l_build: l.into_usize(),
            alpha,
            backedge_ratio,
            vector_dtype: T::DATA_TYPE,
        }),
    };

    // Initialize start points with zero vectors
    let start_points = Matrix::new(T::default(), num_start_pts, dim);

    let data_provider = BfTreeProvider::new(
        provider_params,
        start_points.as_view(),
        NoStore,
        TableBasedDeletes,
    )?;
    let index = Arc::new(DiskANNIndex::new(config, data_provider, None));

    Ok(FullPrecisionBfTreeIndex(index))
}

// build an index and return it
#[allow(clippy::too_many_arguments)]
pub fn build_bftree_index_inner<T, StorageType>(
    metric: Metric,
    data_path: String,
    r: u32,
    l: u32,
    alpha: f32,
    num_start_pts: usize,
    num_threads: u32,
    graph_slack_factor: f32,
    backedge_ratio: f32,
    storage_provider: &StorageType,
    num_tasks: usize,
    insert_minibatch_size: usize,
    on_disk_prefix: Option<String>,
) -> ANNResult<FullPrecisionBfTreeIndex<T>>
where
    T: VectorRepr + AsVectorDtype,
    StorageType: StorageReadProvider + StorageWriteProvider + 'static,
    <StorageType as StorageReadProvider>::Reader: Send,
{
    assert!(insert_minibatch_size > 0);
    let mut logger = PerfLogger::new("build_bftree_index".to_string(), true);

    let metadata = Metadata::read(&mut storage_provider.open_reader(&data_path)?)?;
    let (max_points, dim) = metadata.into_dims();

    let runtime = init_runtime(num_threads.into_usize())
        .map_err(|_| common_error("Failed to initialize tokio runtime."))?;

    let index = runtime.block_on(async move {
        let index = {
            let num_start_points = NonZeroUsize::new(num_start_pts).ok_or_else(|| {
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

            // bf-tree path: use on-disk prefix if provided, otherwise memory only
            let on_disk = on_disk_prefix.is_some();
            let (bf_tree_data_path, bf_tree_graph_path) = if let Some(ref prefix) = on_disk_prefix {
                (
                    BfTreePaths::vectors_bftree(prefix)
                        .to_string_lossy()
                        .to_string(),
                    BfTreePaths::neighbors_bftree(prefix)
                        .to_string_lossy()
                        .to_string(),
                )
            } else {
                (
                    format!(
                        ":memory:{}",
                        BfTreePaths::vectors_bftree(&data_path).display()
                    ),
                    format!(
                        ":memory:{}",
                        BfTreePaths::neighbors_bftree(&data_path).display()
                    ),
                )
            };

            let (vector_provider_config, neighbor_list_provider_config) =
                create_bftree_provider_configs::<T>(
                    &bf_tree_data_path,
                    &bf_tree_graph_path,
                    max_points,
                    num_start_pts,
                    dim,
                    config.max_degree().get(),
                    on_disk,
                )?;

            let provider_params = BfTreeProviderParameters {
                max_points,
                num_start_points,
                dim,
                metric,
                max_fp_vecs_per_fill: None,
                max_degree: config.max_degree().get() as u32,
                vector_provider_config,
                quant_vector_provider_config: Config::default(),
                neighbor_list_provider_config,
                graph_params: Some(GraphParams {
                    l_build: l.into_usize(),
                    alpha,
                    backedge_ratio,
                    vector_dtype: T::DATA_TYPE,
                }),
            };

            // Read the first num_start_pts vectors from dataset to use as start points
            let mut start_points = Matrix::new(T::default(), num_start_pts, dim);
            let temp_iter = VectorDataIterator::<StorageType, AdHoc<T>>::new(
                &data_path,
                None,
                storage_provider,
            )?;
            std::iter::zip(start_points.row_iter_mut(), temp_iter).for_each(|(dst, (src, _))| {
                dst.copy_from_slice(&src);
            });

            let data_provider = BfTreeProvider::new(
                provider_params,
                start_points.as_view(),
                NoStore,
                TableBasedDeletes,
            )?;
            Arc::new(DiskANNIndex::new(config, data_provider, None))
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

        let timer = Timer::new();

        // This variable is used when insert_mini_batch_size > 1.
        // when insert_mini_batch_size is large enough to be comparable or higher than graph degree, we risk creating
        // disconnected components in the graph if we start with a batch size that is too large.
        // So we start with a batch size of 1 and double the batch size until we reach insert_mini_batch_size.
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
                                index_clone.insert(FullPrecision, ctx, &i, &vector).await?;
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
                            index_clone
                                .multi_insert(FullPrecision, ctx, vector_id_pairs.into())
                                .await?;
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

        while let Some(res) = tasks.join_next().await {
            res.map_err(|_| common_error("A spawned task failed"))??;
        }

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

        tracing::info!("Indexing time: {}", diff.as_secs_f64());

        logger.log_checkpoint("async_index_created");
        ANNResult::Ok(index)
    })?;
    Ok(FullPrecisionBfTreeIndex(index))
}

// build a BfTree index with Product Quantization support
// Note: Returns the QuantIndex type from bf_tree module which uses BfTree's QuantVectorProvider
#[allow(clippy::too_many_arguments)]
pub fn build_bftree_pq_index<T, StorageType>(
    metric: Metric,
    data_path: String,
    r: u32,
    l: u32,
    alpha: f32,
    num_start_pts: usize,
    num_threads: u32,
    build_pq_bytes: usize,
    graph_slack_factor: f32,
    max_fp_vecs_per_fill: Option<usize>,
    backedge_ratio: f32,
    storage_provider: &StorageType,
    num_tasks: usize,
    pq_seed: u64,
    insert_minibatch_size: usize,
    on_disk_prefix: Option<String>,
) -> ANNResult<PQBfTreeIndex<T>>
where
    T: VectorRepr + AsVectorDtype,
    StorageType: StorageReadProvider + StorageWriteProvider + 'static,
    <StorageType as StorageReadProvider>::Reader: Send,
{
    assert!(insert_minibatch_size > 0);
    assert!(
        build_pq_bytes > 0,
        "build_pq_bytes must be > 0 for PQ index"
    );

    let mut logger = PerfLogger::new("build_bftree_pq_index".to_string(), true);

    let metadata = Metadata::read(&mut storage_provider.open_reader(&data_path)?)?;
    let (max_points, dim) = metadata.into_dims();

    let runtime = init_runtime(num_threads.into_usize())
        .map_err(|_| common_error("Failed to initialize tokio runtime."))?;

    let index = runtime.block_on(async move {
        // Train PQ
        let mut rng = rand::rngs::StdRng::seed_from_u64(pq_seed);

        let p_val = MAX_PQ_TRAINING_SET_SIZE / (max_points as f64);

        // Load training data for PQ - collect a subset of vectors for training
        let dataset_iter =
            VectorDataIterator::<StorageType, AdHoc<T>>::new(&data_path, None, storage_provider)?;

        let train_size =
            ((max_points as f64 * p_val).min(MAX_PQ_TRAINING_SET_SIZE) as usize).max(1);
        let mut train_data = Vec::with_capacity(train_size * dim);

        for (vector, _) in dataset_iter.take(train_size) {
            // Convert vector to f32 for PQ training
            let vector_f32 = T::as_f32(&vector).into_ann_result()?;
            train_data.extend_from_slice(&vector_f32);
        }

        let num_train = train_data.len() / dim;

        let pq_chunk_table = train_pq(
            MatrixView::try_from(&train_data, num_train, dim).bridge_err()?,
            build_pq_bytes,
            &mut rng,
            num_threads.into_usize(),
        )?;

        // Create index with PQ
        let num_start_points = NonZeroUsize::new(num_start_pts).ok_or_else(|| {
            ANNError::log_index_config_error(
                "num_start_points".to_string(),
                "num_start_points must be at least 1".to_string(),
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

        // bf-tree paths: use on-disk prefix if provided, otherwise memory only
        let on_disk = on_disk_prefix.is_some();
        let (bf_tree_data_path, bf_tree_graph_path, bf_tree_quant_path) =
            if let Some(ref prefix) = on_disk_prefix {
                (
                    BfTreePaths::vectors_bftree(prefix)
                        .to_string_lossy()
                        .to_string(),
                    BfTreePaths::neighbors_bftree(prefix)
                        .to_string_lossy()
                        .to_string(),
                    BfTreePaths::quant_bftree(prefix)
                        .to_string_lossy()
                        .to_string(),
                )
            } else {
                (
                    format!(
                        ":memory:{}",
                        BfTreePaths::vectors_bftree(&data_path).display()
                    ),
                    format!(
                        ":memory:{}",
                        BfTreePaths::neighbors_bftree(&data_path).display()
                    ),
                    format!(
                        ":memory:{}",
                        BfTreePaths::quant_bftree(&data_path).display()
                    ),
                )
            };

        let (vector_provider_config, neighbor_list_provider_config) =
            create_bftree_provider_configs::<T>(
                &bf_tree_data_path,
                &bf_tree_graph_path,
                max_points,
                num_start_pts,
                dim,
                config.max_degree().get(),
                on_disk,
            )?;

        // Create quant vector provider config
        let quant_buffer_size = ((MEMORY_SLACK * (4 + build_pq_bytes)) * max_points)
            .next_power_of_two()
            .max(MIN_BUFFER_SIZE);
        let size = 4 + build_pq_bytes; // u32 id size + pq bytes
        let quant_vector_provider_config =
            configure_vector_provider(size, &bf_tree_quant_path, quant_buffer_size, on_disk)?;

        let provider_params = BfTreeProviderParameters {
            max_points,
            num_start_points,
            dim,
            metric,
            max_fp_vecs_per_fill,
            max_degree: config.max_degree().get() as u32,
            vector_provider_config,
            quant_vector_provider_config,
            neighbor_list_provider_config,
            graph_params: Some(GraphParams {
                l_build: l.into_usize(),
                alpha,
                backedge_ratio,
                vector_dtype: T::DATA_TYPE,
            }),
        };

        // Read the first num_start_pts vectors from dataset to use as start points
        let mut start_points = Matrix::new(T::default(), num_start_pts, dim);
        let temp_iter =
            VectorDataIterator::<StorageType, AdHoc<T>>::new(&data_path, None, storage_provider)?;
        std::iter::zip(start_points.row_iter_mut(), temp_iter).for_each(|(dst, (src, _))| {
            dst.copy_from_slice(&src);
        });

        let data_provider = BfTreeProvider::new(
            provider_params,
            start_points.as_view(),
            pq_chunk_table,
            TableBasedDeletes,
        )?;
        let index = Arc::new(DiskANNIndex::new(config, data_provider, None));

        // Initialize dataset iterator
        let dataset_iter = Arc::new(Mutex::new(
            VectorDataIterator::<StorageType, AdHoc<T>>::new(
                &data_path,
                Option::None,
                storage_provider,
            )?
            .enumerate(),
        ));

        let hybrid = Hybrid::new(max_fp_vecs_per_fill);
        let timer = Timer::new();
        let mut current_batch_size = 1;

        // Insert vectors with PQ
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
                                index_clone.insert(hybrid, ctx, &i, &vector).await?;
                            }
                            None => break,
                        }
                    } else {
                        let mut vector_id_pairs =
                            Vec::<VectorIdBoxSlice<u32, T>>::with_capacity(current_batch_size);
                        {
                            let mut guard = dataset_iter_clone
                                .lock()
                                .map_err(|_| common_error("Poisoned mutex during construction"))?;
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
                            index_clone
                                .multi_insert(hybrid, ctx, vector_id_pairs.into())
                                .await?;
                        } else {
                            break;
                        }

                        if current_batch_size < insert_minibatch_size {
                            current_batch_size =
                                std::cmp::min(2 * current_batch_size, insert_minibatch_size);
                        }
                    }
                }
                ANNResult::Ok(())
            });
        }

        while let Some(res) = tasks.join_next().await {
            res.map_err(|_| common_error("A spawned task failed"))??;
        }

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

        tracing::info!("Indexing time: {}", diff.as_secs_f64());

        logger.log_checkpoint("bftree_pq_index_created");
        ANNResult::Ok(index)
    })?;

    Ok(PQBfTreeIndex(index))
}

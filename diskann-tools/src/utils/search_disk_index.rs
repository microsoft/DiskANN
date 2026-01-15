/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{collections::HashSet, sync::atomic::AtomicBool, time::Instant};

use diskann::utils::IntoUsize;
use diskann_disk::{
    data_model::CachingStrategy,
    search::provider::{
        disk_provider::DiskIndexSearcher, disk_vertex_provider_factory::DiskVertexProviderFactory,
    },
    storage::disk_index_reader::DiskIndexReader,
    utils::{
        aligned_file_reader::traits::AlignedReaderFactory, instrumentation::PerfLogger, statistics,
        QueryStatistics,
    },
};
use diskann_providers::storage::{StorageReadProvider, StorageWriteProvider};
use diskann_providers::{
    model::graph::traits::GraphDataType,
    storage::{get_compressed_pq_file, get_pq_pivot_file},
    utils::{create_thread_pool, load_aligned_bin, save_bin_u32, ParallelIteratorInPool},
};
use diskann_vector::distance::Metric;
use opentelemetry::global::BoxedSpan;
#[cfg(feature = "perf_test")]
use opentelemetry::{
    trace::{Span, Tracer},
    KeyValue,
};
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use tracing::{error, info};

use crate::utils::{search_index_utils, CMDResult, CMDToolError, KRecallAtN};

pub struct SearchDiskIndexParameters<'a> {
    pub metric: Metric,
    pub index_path_prefix: &'a str,
    pub result_output_prefix: &'a str,
    pub query_file: &'a str,
    pub truthset_file: &'a str,
    pub vector_filters_file: Option<&'a str>,
    pub num_threads: usize,
    pub recall_at: u32,
    pub beam_width: u32,
    pub search_io_limit: u32,
    pub l_vec: &'a [u32],
    pub fail_if_recall_below: f32,
    pub num_nodes_to_cache: usize,
    pub is_flat_search: bool,
}

pub fn search_disk_index<Data, StorageType, ReaderFactory>(
    storage_provider: &StorageType,
    parameters: SearchDiskIndexParameters,
    aligned_reader_factory: ReaderFactory,
) -> CMDResult<i32>
where
    Data: GraphDataType<VectorIdType = u32>,
    StorageType: StorageReadProvider + StorageWriteProvider,
    ReaderFactory: AlignedReaderFactory,
{
    let mut logger = PerfLogger::new("search_disk_index".to_string(), true);

    info!(
        "Search parameters: #threads: {}, recall_at {}, search_list_size: {:?}, search_io_limit: {}, fail_if_recall_below: {}, beam_width: {}",
        parameters.num_threads, parameters.recall_at, parameters.l_vec, parameters.search_io_limit, parameters.fail_if_recall_below,parameters.beam_width
    );

    // Load the query file
    let (query, query_num, _, query_aligned_dim) =
        load_aligned_bin::<Data::VectorDataType>(storage_provider, parameters.query_file)?;
    // Load the vector filters
    let vector_filters = match parameters.vector_filters_file {
        Some(vector_filters_file) => {
            search_index_utils::load_vector_filters(storage_provider, vector_filters_file)?
        }
        None => vec![HashSet::<u32>::new(); query_num],
    };

    assert_eq!(
        vector_filters.len(),
        query_num,
        "Mismatch in query and vector filter sizes"
    );

    let mut gt_dim: usize = 0;
    let mut gt_ids: Option<Vec<u32>> = None;

    let mut gt_ids_variable_length: Option<Vec<Vec<u32>>> = None;
    let mut gt_dists: Option<Vec<f32>> = None;

    // Check for ground truth
    let mut calc_recall_flag = false;
    if !parameters.truthset_file.is_empty() && storage_provider.exists(parameters.truthset_file) {
        if parameters.vector_filters_file.is_none() {
            let ret =
                search_index_utils::load_truthset(storage_provider, parameters.truthset_file)?;
            gt_ids = Some(ret.index_nodes);
            gt_dists = ret.distances;
            let gt_num = ret.index_num_points;
            gt_dim = ret.index_dimension;

            if gt_num != query_num {
                error!("Error. Mismatch in number of queries and ground truth data");
            }
        } else {
            let range_truthset = search_index_utils::load_range_truthset(
                storage_provider,
                parameters.truthset_file,
            )?;
            gt_ids_variable_length = Some(range_truthset.index_nodes);
            let gt_num = range_truthset.index_num_points;

            if gt_num != query_num {
                error!("Error. Mismatch in number of queries and ground truth data");
            }
        }

        calc_recall_flag = true;
    } else {
        error!(
            "Truthset file {} not found. Not computing recall",
            parameters.truthset_file
        );
    }

    let index_reader = DiskIndexReader::<<Data as GraphDataType>::VectorDataType>::new(
        get_pq_pivot_file(parameters.index_path_prefix),
        get_compressed_pq_file(parameters.index_path_prefix),
        storage_provider,
    )?;

    let caching_strategy = if parameters.num_nodes_to_cache > 0 {
        CachingStrategy::StaticCacheWithBfsNodes(parameters.num_nodes_to_cache)
    } else {
        CachingStrategy::None
    };
    // Create the vertex provider factory
    let vertex_provider_factory =
        DiskVertexProviderFactory::new(aligned_reader_factory, caching_strategy)?;

    let searcher = DiskIndexSearcher::<Data, DiskVertexProviderFactory<Data, ReaderFactory>>::new(
        parameters.num_threads.into_usize(),
        parameters.search_io_limit.into_usize(),
        &index_reader,
        vertex_provider_factory,
        parameters.metric,
        None,
    )?;

    logger.log_checkpoint("index_loaded");

    let recall_string = format!("Recall@{}", parameters.recall_at);
    if calc_recall_flag {
        println!(
            "{:<6}{:<12}{:<15}{:<20}{:<20}{:<12}{:<16}{:<10}{:<20}{:<12}{:<12}{:<14}{:<16}",
            "L",
            "Beamwidth",
            "QPS",
            "Mean Latency (us)",
            "99.9 Latency (us)",
            "Mean IOs",
            "Mean IO (us)",
            "CPU (us)",
            "PQ Preprocess (us)",
            "Mean Comps",
            "Mean Hops",
            "Cache Hit %",
            recall_string
        );
    } else {
        println!(
            "{:<6}{:<12}{:<15}{:<20}{:<20}{:<12}{:<16}{:<10}{:<20}{:<12}{:<12}{:<14}",
            "L",
            "Beamwidth",
            "QPS",
            "Mean Latency (us)",
            "99.9 Latency (us)",
            "Mean IOs",
            "Mean IO (us)",
            "CPU (us)",
            "PQ Preprocess (us)",
            "Mean Comparisons",
            "Mean hops",
            "Cache Hit %",
        );
    }
    println!("{:=<178}", "");

    let mut query_result_ids: Vec<Vec<u32>> = vec![vec![]; parameters.l_vec.len()];
    let mut query_result_dists: Vec<Vec<f32>> = vec![vec![]; parameters.l_vec.len()];
    let mut cmp_stats: Vec<u32> = vec![0; query_num];
    let has_any_search_failed = AtomicBool::new(false);

    let mut best_recall = 0.0;

    let pool = create_thread_pool(parameters.num_threads)?;

    for (test_id, &l) in parameters.l_vec.iter().enumerate() {
        if l < parameters.recall_at {
            println!(
                "Ignoring search with L: {} since it's smaller than K: {}",
                l, parameters.recall_at
            );
            continue;
        }

        query_result_ids[test_id].resize(parameters.recall_at as usize * query_num, 0);
        query_result_dists[test_id].resize(parameters.recall_at as usize * query_num, 0.0);

        // Assuming `QueryStats` is a struct that you have defined elsewhere
        let mut statistics: Vec<QueryStatistics> = vec![QueryStatistics::default(); query_num];
        let mut result_counts: Vec<u32> = vec![0; query_num];

        let zipped = cmp_stats
            .par_iter_mut()
            .zip(query.par_chunks(query_aligned_dim))
            .zip(vector_filters.par_iter())
            .zip(query_result_ids[test_id].par_chunks_mut(parameters.recall_at as usize))
            .zip(query_result_dists[test_id].par_chunks_mut(parameters.recall_at as usize))
            .zip(statistics.par_iter_mut())
            .zip(result_counts.par_iter_mut());

        let mut _span: BoxedSpan;
        #[cfg(feature = "perf_test")]
        {
            let tracer = opentelemetry::global::tracer("");

            // Start a span for the search iteration.
            _span = tracer.start(format!("search-with-L={}-bw={}", l, parameters.beam_width));
        }

        let test_start = Instant::now();
        zipped.for_each_in_pool(
            &pool,
            |(
                (((((_cmp, query), vector_filter), query_result_id), query_result_dist), stats),
                result_count,
            )| {
                let vector_filter_function: Box<dyn Fn(&u32) -> bool + Send + Sync> =
                    if parameters.vector_filters_file.is_none() {
                        Box::new(|_: &u32| true)
                    } else {
                        Box::new(move |vector_id: &u32| vector_filter.contains(vector_id))
                    };

                let result = searcher.search(
                    query,
                    parameters.recall_at,
                    l,
                    Some(parameters.beam_width as usize),
                    Some(vector_filter_function),
                    parameters.is_flat_search,
                );

                match result {
                    Ok(search_result) => {
                        *result_count = search_result.stats.result_count;
                        *stats = search_result.stats.query_statistics;
                        search_result
                            .results
                            .iter()
                            .take(parameters.recall_at as usize)
                            .enumerate()
                            .for_each(|(i, item)| {
                                query_result_id[i] = item.vertex_id;
                                query_result_dist[i] = item.distance;
                            });
                    }
                    Err(e) => {
                        error!("Error during search: {}", e);
                        has_any_search_failed.store(true, std::sync::atomic::Ordering::Release);
                    }
                }
            },
        );

        let diff = test_start.elapsed();
        let qps = query_num as f32 / diff.as_secs_f32();

        let mean_latency =
            statistics::get_mean_stats(&statistics, |stats| stats.total_execution_time_us as f64);

        let latency_999 = statistics::get_percentile_stats(&statistics, 0.999, |stats| {
            stats.total_execution_time_us
        });

        let mean_ios = statistics::get_mean_stats(&statistics, |stats| stats.total_io_operations);
        let mean_io_time = statistics::get_mean_stats(&statistics, |stats| stats.io_time_us as f64);
        let mean_cpus = statistics::get_mean_stats(&statistics, |stats| stats.cpu_time_us as f64);
        let mean_pq_preprocess_time = statistics::get_mean_stats(&statistics, |stats| {
            stats.query_pq_preprocess_time_us as f64
        });
        let mean_comps =
            statistics::get_mean_stats(&statistics, |stats| stats.total_comparisons as f64);
        let mean_hops = statistics::get_mean_stats(&statistics, |stats| stats.search_hops as f64);
        let total_ios = statistics::get_sum_stats(&statistics, |stats| stats.total_io_operations);
        let total_vertices_loaded =
            statistics::get_sum_stats(&statistics, |stats| stats.total_vertices_loaded);
        let cache_hit_percentage = if total_vertices_loaded > 0.0 {
            100.0 * (1.0 - (total_ios / total_vertices_loaded))
        } else {
            100.0
        };

        let mut recall = 0.0;
        if calc_recall_flag {
            recall = if let Some(gt_ids_variable_length) = &gt_ids_variable_length {
                let our_results_variable_length = query_result_ids[test_id]
                    .chunks_exact(parameters.recall_at as usize)
                    .enumerate()
                    .map(|(i, chunk)| chunk[..result_counts[i] as usize].to_vec())
                    .collect::<Vec<_>>();
                search_index_utils::calculate_filtered_search_recall(
                    query_num,
                    None,
                    gt_ids_variable_length,
                    &our_results_variable_length,
                    parameters.recall_at,
                )? as f32
            } else {
                search_index_utils::calculate_recall(
                    query_num,
                    gt_ids.as_ref().ok_or_else(|| CMDToolError {
                        details: "GroundTruth IDs not initialized".to_string(),
                    })?,
                    gt_dists.as_ref(),
                    gt_dim,
                    &query_result_ids[test_id],
                    parameters.recall_at,
                    KRecallAtN::new(parameters.recall_at, parameters.recall_at)?,
                )? as f32
            };

            best_recall = f32::from(std::cmp::max(
                OrderedFloat::<f32>(best_recall),
                OrderedFloat::<f32>(recall),
            ));
        }

        if calc_recall_flag {
            println!(
                "{:<6}{:<12.2}{:<15.2}{:<20.2}{:<20.2}{:<12.2}{:<16.2}{:<10.2}{:<20.2}{:<12.2}{:<12.2}{:<14.2}{:<16.2}",
                l,
                parameters.beam_width,
                qps,
                mean_latency,
                latency_999,
                mean_ios,
                mean_io_time,
                mean_cpus,
                mean_pq_preprocess_time,
                mean_comps,
                mean_hops,
                cache_hit_percentage,
                recall,
            );
        } else {
            println!(
                "{:<6}{:<12.2}{:<15.2}{:<20.2}{:<20.2}{:<12.2}{:<16.2}{:<10.2}{:<20.2}{:<12.2}{:<12.2}{:<14.2}",
                l,
                parameters.beam_width,
                qps,
                mean_latency,
                latency_999,
                mean_ios,
                mean_io_time,
                mean_cpus,
                mean_pq_preprocess_time,
                mean_comps,
                mean_hops,
                cache_hit_percentage,
            );
        }

        #[cfg(feature = "perf_test")]
        {
            let latency_95 = statistics::get_percentile_stats(&statistics, 0.95, |stats| {
                stats.total_execution_time_us
            });

            _span.set_attribute(KeyValue::new("qps", qps as f64));
            _span.set_attribute(KeyValue::new("mean_latency", mean_latency));
            _span.set_attribute(KeyValue::new("latency_999", latency_999 as f64));
            _span.set_attribute(KeyValue::new("latency_95", latency_95 as f64));
            _span.set_attribute(KeyValue::new("mean_cpus", mean_cpus));
            _span.set_attribute(KeyValue::new("mean_io_time", mean_io_time));
            _span.set_attribute(KeyValue::new("mean_ios", mean_ios as f64));
            _span.set_attribute(KeyValue::new("mean_comps", mean_comps));
            _span.set_attribute(KeyValue::new("mean_hops", mean_hops));
            _span.set_attribute(KeyValue::new("recall", recall as f64));
            _span.end();
        }
    }

    logger.log_checkpoint("search_completed");

    info!("Done searching. Now saving results");
    for (test_id, l_value) in parameters.l_vec.iter().enumerate() {
        if *l_value < parameters.recall_at {
            println!(
                "Ignoring all search with L: {} since it's smaller than K: {}",
                l_value, parameters.recall_at
            );
        }

        let cur_result_path = format!(
            "{}_{}_idx_uint32.bin",
            parameters.result_output_prefix, l_value
        );
        save_bin_u32(
            &mut storage_provider.create_for_write(&cur_result_path)?,
            query_result_ids[test_id].as_slice(),
            query_num,
            parameters.recall_at as usize,
            0,
        )?;
    }

    if has_any_search_failed.load(std::sync::atomic::Ordering::Acquire) {
        // Exit with error. The above stats might still be useful to the user if only a few searched failed, so allowed printing them.
        return Err(CMDToolError {
            details: "At least one search failed with error. See log for details. Exiting."
                .to_string(),
        });
    }

    if best_recall >= parameters.fail_if_recall_below {
        Ok(0)
    } else {
        println!(
            "Search failed. Best recall {} is below the threshold {}",
            best_recall, parameters.fail_if_recall_below
        );
        Ok(-1)
    }
}

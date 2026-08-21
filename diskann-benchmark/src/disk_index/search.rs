/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use rayon::prelude::*;
use std::{
    collections::HashSet, fmt, hint::black_box, mem::size_of, sync::atomic::AtomicBool,
    time::Instant,
};

use opentelemetry::{global, trace::Span, trace::Tracer};
use opentelemetry_sdk::trace::SdkTracerProvider;

use diskann::utils::VectorRepr;
use diskann_benchmark_runner::{files::InputFile, utils::MicroSeconds};
use diskann_disk::{
    data_model::{AdHoc, CachingStrategy},
    search::{
        provider::{
            disk_provider::DiskIndexSearcher,
            disk_vertex_provider_factory::DiskVertexProviderFactory,
        },
        search_mode::SearchMode,
    },
    storage::disk_index_reader::DiskIndexReader,
    utils::{instrumentation::PerfLogger, statistics, QueryStatistics},
};
use diskann_providers::storage::StorageReadProvider;
use diskann_providers::{
    storage::{
        get_compressed_pq_file, get_disk_index_file, get_pq_pivot_file, FileStorageProvider,
    },
    utils::{create_thread_pool, ParallelIteratorInPool},
};
use diskann_tools::utils::{search_index_utils, KRecallAtN};
use diskann_utils::views::Matrix;
use serde::{Deserialize, Serialize};

use crate::{
    disk_index::json_spancollector::JsonSpanCollector,
    inputs::disk::{DiskIndexLoad, DiskSearchApi, DiskSearchPhase},
    utils::{datafiles, SimilarityMeasure},
};

#[derive(Serialize, Deserialize, Debug)]
pub(super) struct DiskSearchStats {
    #[serde(default)]
    pub(super) search_api: DiskSearchApi,
    pub(super) num_threads: usize,
    pub(super) beam_width: usize,
    pub(super) recall_at: u32,
    #[serde(default)]
    pub(super) return_list_size: Option<u32>,
    pub(crate) is_flat_search: bool,
    pub(crate) distance: SimilarityMeasure,
    pub(crate) uses_vector_filters: bool,
    pub(super) num_nodes_to_cache: Option<usize>,
    pub(super) search_results_per_l: Vec<DiskSearchResult>,
    span_metrics: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug)]
pub(super) struct DiskSearchResult {
    pub(super) search_l: u32,
    pub(super) qps: f32,
    pub(super) mean_latency: f64,
    pub(super) p95_latency: MicroSeconds,
    pub(super) p999_latency: MicroSeconds,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) mean_public_api_call_latency_us: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) p95_public_api_call_latency_us: Option<MicroSeconds>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) p999_public_api_call_latency_us: Option<MicroSeconds>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) mean_returned_vector_payload_bytes: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) max_returned_vector_payload_bytes: Option<u64>,
    pub(super) mean_ios: f64,
    pub(super) mean_io_time: f64,
    pub(super) mean_cpu_time: f64,
    pub(super) mean_pq_preprocess_time: f64,
    pub(super) mean_comparisons: f64,
    pub(super) mean_hops: f64,
    pub(super) cache_hit_percentage: f64,
    pub(super) recall: f32,
}

#[derive(Debug, PartialEq)]
struct PublicApiMetrics {
    mean_call_latency_us: f64,
    p95_call_latency_us: MicroSeconds,
    p999_call_latency_us: MicroSeconds,
    mean_payload_bytes: f64,
    max_payload_bytes: u64,
}

fn percentile_from_sorted(values: &[u64], numerator: usize, denominator: usize) -> u64 {
    let index = values
        .len()
        .saturating_mul(numerator)
        .checked_div(denominator)
        .unwrap_or(0)
        .min(values.len() - 1);
    values[index]
}

fn aggregate_public_api_metrics(
    mut call_latencies_us: Vec<u64>,
    payload_bytes: &[u64],
) -> anyhow::Result<PublicApiMetrics> {
    anyhow::ensure!(
        !call_latencies_us.is_empty(),
        "cannot aggregate zero queries"
    );
    anyhow::ensure!(
        call_latencies_us.len() == payload_bytes.len(),
        "latency and payload sample counts differ"
    );

    let mean_call_latency_us = call_latencies_us
        .iter()
        .map(|&value| value as f64)
        .sum::<f64>()
        / call_latencies_us.len() as f64;
    call_latencies_us.sort_unstable();

    Ok(PublicApiMetrics {
        mean_call_latency_us,
        p95_call_latency_us: MicroSeconds::new(percentile_from_sorted(&call_latencies_us, 95, 100)),
        p999_call_latency_us: MicroSeconds::new(percentile_from_sorted(
            &call_latencies_us,
            999,
            1000,
        )),
        mean_payload_bytes: payload_bytes.iter().map(|&value| value as f64).sum::<f64>()
            / payload_bytes.len() as f64,
        max_payload_bytes: payload_bytes.iter().copied().max().unwrap_or(0),
    })
}

impl DiskSearchResult {
    #[allow(clippy::too_many_arguments)]
    fn new(
        statistics: &[QueryStatistics],
        result_ids: &[u32],
        result_counts: &[u32],
        public_api_metrics: Option<PublicApiMetrics>,
        search_l: u32,
        total_time_as_secs: f32,
        num_queries: usize,
        result_dim: u32,
        gt_context: &GroundTruthContext,
    ) -> anyhow::Result<DiskSearchResult> {
        let total_ios = statistics::get_sum_stats(statistics, |stats| stats.total_io_operations);
        let total_vertices_loaded =
            statistics::get_sum_stats(statistics, |stats| stats.total_vertices_loaded);
        let cache_hit_percentage = if total_vertices_loaded > 0.0 {
            100.0 * (1.0 - (total_ios / total_vertices_loaded))
        } else {
            100.0
        };

        let recall = if let Some(var_gt) = &gt_context.gt_ids_variable_length {
            let ours: Vec<Vec<u32>> = result_ids
                .chunks_exact(result_dim as usize)
                .enumerate()
                .map(|(qi, chunk)| {
                    let written = result_counts[qi] as usize;
                    chunk[..written.min(gt_context.recall_at as usize)].to_vec()
                })
                .collect();
            let filtered_recall = search_index_utils::calculate_filtered_search_recall(
                num_queries,
                None,
                var_gt,
                &ours,
                gt_context.recall_at,
            )?;
            filtered_recall as f32
        } else {
            let gt = gt_context
                .gt_ids
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("GT IDs missing"))?;
            let recall_value = search_index_utils::calculate_recall(
                num_queries,
                gt,
                gt_context.gt_dists.as_ref(),
                gt_context.gt_dim,
                result_ids,
                result_dim,
                KRecallAtN::new(gt_context.recall_at, gt_context.recall_at)?,
            )?;
            recall_value as f32
        };

        Ok(DiskSearchResult {
            search_l,
            qps: if total_time_as_secs > 0.0 {
                num_queries as f32 / total_time_as_secs
            } else {
                0.0
            },
            mean_latency: statistics::get_mean_stats(statistics, |s| {
                s.total_execution_time_us as f64
            }),
            p95_latency: MicroSeconds::new(
                statistics::get_percentile_stats(statistics, 0.95, |s| s.total_execution_time_us)
                    as u64,
            ),
            p999_latency: MicroSeconds::new(statistics::get_percentile_stats(
                statistics,
                0.999,
                |s| s.total_execution_time_us,
            ) as u64),
            mean_public_api_call_latency_us: public_api_metrics
                .as_ref()
                .map(|metrics| metrics.mean_call_latency_us),
            p95_public_api_call_latency_us: public_api_metrics
                .as_ref()
                .map(|metrics| metrics.p95_call_latency_us),
            p999_public_api_call_latency_us: public_api_metrics
                .as_ref()
                .map(|metrics| metrics.p999_call_latency_us),
            mean_returned_vector_payload_bytes: public_api_metrics
                .as_ref()
                .map(|metrics| metrics.mean_payload_bytes),
            max_returned_vector_payload_bytes: public_api_metrics
                .map(|metrics| metrics.max_payload_bytes),
            mean_ios: statistics::get_mean_stats(statistics, |s| s.total_io_operations),
            mean_io_time: statistics::get_mean_stats(statistics, |s| s.io_time_us as f64),
            mean_cpu_time: statistics::get_mean_stats(statistics, |stats| stats.cpu_time_us as f64),
            mean_pq_preprocess_time: statistics::get_mean_stats(statistics, |stats| {
                stats.query_pq_preprocess_time_us as f64
            }),
            mean_comparisons: statistics::get_mean_stats(statistics, |stats| {
                stats.total_comparisons as f64
            }),
            mean_hops: statistics::get_mean_stats(statistics, |s| s.search_hops as f64),
            cache_hit_percentage,
            recall,
        })
    }
}

pub(super) fn search_disk_index<T, StorageType>(
    index_load: &DiskIndexLoad,
    search_params: &DiskSearchPhase,
    storage_provider: &StorageType,
) -> anyhow::Result<DiskSearchStats>
where
    T: VectorRepr,
    StorageType: StorageReadProvider,
{
    let previous_tracer_provider = global::tracer_provider();
    let span_collector = {
        let collector = JsonSpanCollector::new();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(collector.clone())
            .build();
        global::set_tracer_provider(provider.clone());
        Some((collector, provider))
    };

    // Use PerfLogger for consistent checkpoint logging
    let mut logger = PerfLogger::new("search_disk_index", true);

    // Load the query file
    let queries: Matrix<T> = datafiles::load_dataset(datafiles::BinFile(&search_params.queries))?;
    let num_queries = queries.nrows();

    // Load the vector filters
    let vector_filters = match &search_params.vector_filters_file {
        Some(vector_filters_file) => {
            let vector_filters_file = vector_filters_file.to_string_lossy().to_string();
            search_index_utils::load_vector_filters(storage_provider, &vector_filters_file)?
        }
        None => vec![HashSet::<u32>::new(); num_queries],
    };

    if vector_filters.len() != num_queries {
        anyhow::bail!("Mismatch in query and vector filter sizes");
    }

    // Prepare ground truth context
    let gt_context = prepare_ground_truth_context(
        search_params.vector_filters_file.is_some(),
        &search_params.groundtruth,
        search_params.recall_at,
        storage_provider,
    )?;
    let return_list_size = search_params.return_list_size();

    // Setup disk index components
    let pivot_path = get_pq_pivot_file(&index_load.load_path);
    let pq_data_path = get_compressed_pq_file(&index_load.load_path);
    let disk_index_path = get_disk_index_file(&index_load.load_path);

    let index_reader = DiskIndexReader::new(pivot_path, pq_data_path, &FileStorageProvider)?;

    let caching_strategy = if let Some(num_nodes) = search_params.num_nodes_to_cache {
        CachingStrategy::StaticCacheWithBfsNodes(num_nodes)
    } else {
        CachingStrategy::None
    };

    let vertex_provider_factory =
        DiskVertexProviderFactory::from_disk_index_path(disk_index_path, caching_strategy)?;

    let searcher = &DiskIndexSearcher::<AdHoc<T>, _>::new(
        search_params.num_threads,
        if let Some(lim) = search_params.search_io_limit {
            lim
        } else {
            usize::MAX
        },
        &index_reader,
        vertex_provider_factory,
        search_params.distance.into(),
        None,
    )?;

    logger.log_checkpoint("index_loaded");

    let pool = create_thread_pool(search_params.num_threads)?;
    let mut search_results_per_l = Vec::with_capacity(search_params.search_list.len());
    let has_any_search_failed = AtomicBool::new(false);

    // Execute search iterations
    for &l in search_params.search_list.iter() {
        let mut statistics_vec: Vec<QueryStatistics> =
            vec![QueryStatistics::default(); num_queries];
        let mut result_counts: Vec<u32> = vec![0; num_queries];
        let mut result_ids: Vec<u32> = vec![0; (return_list_size as usize) * num_queries];
        let mut result_dists: Vec<f32> = vec![0.0; (return_list_size as usize) * num_queries];
        let start = Instant::now();

        let mut l_span = {
            let tracer = global::tracer("");
            let span_name = format!("search-with-L={}-bw={}", l, search_params.beam_width);
            tracer.start(span_name)
        };

        let public_api_samples = match (search_params.collect_api_metrics, search_params.search_api)
        {
            (false, DiskSearchApi::Legacy) => {
                // Keep the pre-existing hot loop unchanged for old benchmark JSON.
                let zipped = queries
                    .par_row_iter()
                    .zip(vector_filters.par_iter())
                    .zip(result_ids.par_chunks_mut(return_list_size as usize))
                    .zip(result_dists.par_chunks_mut(return_list_size as usize))
                    .zip(statistics_vec.par_iter_mut())
                    .zip(result_counts.par_iter_mut());

                zipped.for_each_in_pool(
                    pool.as_ref(),
                    |(((((q, vf), id_chunk), dist_chunk), stats), rc)| {
                        // Construct the SearchMode from the JSON-driven
                        // `adaptive_l` is now encapsulated in `DiskSearchMode`, so the
                        // benchmark only supplies the per-query filter and post-processor.
                        let has_filter = search_params.vector_filters_file.is_some();
                        let mode: SearchMode<'_> = search_params.search_mode.search_mode(
                            has_filter,
                            vf,
                            search_params.post_processor.as_ref(),
                        );

                        match searcher.search(
                            q,
                            return_list_size,
                            l,
                            Some(search_params.beam_width),
                            mode,
                        ) {
                            Ok(search_result) => {
                                *stats = search_result.stats.query_statistics;
                                let base_count = (search_result.stats.result_count as usize)
                                    .min(return_list_size as usize)
                                    .min(search_result.results.len());

                                *rc = base_count as u32;
                                id_chunk.fill(0);
                                dist_chunk.fill(0.0);

                                for (i, result_item) in
                                    search_result.results.iter().take(base_count).enumerate()
                                {
                                    id_chunk[i] = result_item.vertex_id;
                                    dist_chunk[i] = result_item.distance;
                                }
                            }
                            Err(e) => {
                                eprintln!("Search failed for query: {:?}", e);
                                *rc = 0;
                                id_chunk.fill(0);
                                dist_chunk.fill(0.0);
                                has_any_search_failed
                                    .store(true, std::sync::atomic::Ordering::Release);
                            }
                        }
                    },
                );
                None
            }
            (false, DiskSearchApi::IndexedVectors) => {
                let zipped = queries
                    .par_row_iter()
                    .zip(vector_filters.par_iter())
                    .zip(result_ids.par_chunks_mut(return_list_size as usize))
                    .zip(result_dists.par_chunks_mut(return_list_size as usize))
                    .zip(statistics_vec.par_iter_mut())
                    .zip(result_counts.par_iter_mut());

                zipped.for_each_in_pool(
                    pool.as_ref(),
                    |(((((q, vf), id_chunk), dist_chunk), stats), rc)| {
                        let has_filter = search_params.vector_filters_file.is_some();
                        let mode: SearchMode<'_> = search_params.search_mode.search_mode(
                            has_filter,
                            vf,
                            search_params.post_processor.as_ref(),
                        );

                        match searcher.search_with_indexed_vectors(
                            q,
                            return_list_size,
                            l,
                            Some(search_params.beam_width),
                            mode,
                        ) {
                            Ok(search_result) => {
                                *stats = search_result.stats.query_statistics;
                                let base_count = (search_result.stats.result_count as usize)
                                    .min(return_list_size as usize)
                                    .min(search_result.results.len());

                                *rc = base_count as u32;
                                id_chunk.fill(0);
                                dist_chunk.fill(0.0);

                                for (i, result_item) in
                                    search_result.results.iter().take(base_count).enumerate()
                                {
                                    id_chunk[i] = result_item.vertex_id;
                                    dist_chunk[i] = result_item.distance;
                                }
                                for result_item in &search_result.results {
                                    black_box(result_item.indexed_vector.as_ref());
                                }
                            }
                            Err(e) => {
                                eprintln!("Search failed for query: {:?}", e);
                                *rc = 0;
                                id_chunk.fill(0);
                                dist_chunk.fill(0.0);
                                has_any_search_failed
                                    .store(true, std::sync::atomic::Ordering::Release);
                            }
                        }
                    },
                );
                None
            }
            (true, DiskSearchApi::Legacy) => {
                let mut public_api_call_latencies_us = vec![0u64; num_queries];
                let returned_vector_payload_bytes = vec![0u64; num_queries];
                let zipped = queries
                    .par_row_iter()
                    .zip(vector_filters.par_iter())
                    .zip(result_ids.par_chunks_mut(return_list_size as usize))
                    .zip(result_dists.par_chunks_mut(return_list_size as usize))
                    .zip(statistics_vec.par_iter_mut())
                    .zip(result_counts.par_iter_mut())
                    .zip(public_api_call_latencies_us.par_iter_mut());

                zipped.for_each_in_pool(
                    pool.as_ref(),
                    |((((((q, vf), id_chunk), dist_chunk), stats), rc), call_latency_us)| {
                        let has_filter = search_params.vector_filters_file.is_some();
                        let mode: SearchMode<'_> = search_params.search_mode.search_mode(
                            has_filter,
                            vf,
                            search_params.post_processor.as_ref(),
                        );

                        let api_start = Instant::now();
                        let search_result = searcher.search(
                            q,
                            return_list_size,
                            l,
                            Some(search_params.beam_width),
                            mode,
                        );
                        let api_elapsed = api_start.elapsed();
                        *call_latency_us = api_elapsed.as_micros().min(u64::MAX as u128) as u64;

                        match search_result {
                            Ok(search_result) => {
                                *stats = search_result.stats.query_statistics;
                                let base_count = (search_result.stats.result_count as usize)
                                    .min(return_list_size as usize)
                                    .min(search_result.results.len());

                                *rc = base_count as u32;
                                id_chunk.fill(0);
                                dist_chunk.fill(0.0);

                                for (i, result_item) in
                                    search_result.results.iter().take(base_count).enumerate()
                                {
                                    id_chunk[i] = result_item.vertex_id;
                                    dist_chunk[i] = result_item.distance;
                                }
                            }
                            Err(e) => {
                                eprintln!("Search failed for query: {:?}", e);
                                *rc = 0;
                                id_chunk.fill(0);
                                dist_chunk.fill(0.0);
                                has_any_search_failed
                                    .store(true, std::sync::atomic::Ordering::Release);
                            }
                        }
                    },
                );
                Some((public_api_call_latencies_us, returned_vector_payload_bytes))
            }
            (true, DiskSearchApi::IndexedVectors) => {
                let mut public_api_call_latencies_us = vec![0u64; num_queries];
                let mut returned_vector_payload_bytes = vec![0u64; num_queries];
                let zipped = queries
                    .par_row_iter()
                    .zip(vector_filters.par_iter())
                    .zip(result_ids.par_chunks_mut(return_list_size as usize))
                    .zip(result_dists.par_chunks_mut(return_list_size as usize))
                    .zip(statistics_vec.par_iter_mut())
                    .zip(result_counts.par_iter_mut())
                    .zip(public_api_call_latencies_us.par_iter_mut())
                    .zip(returned_vector_payload_bytes.par_iter_mut());

                zipped.for_each_in_pool(
                    pool.as_ref(),
                    |(
                        ((((((q, vf), id_chunk), dist_chunk), stats), rc), call_latency_us),
                        payload_bytes,
                    )| {
                        let has_filter = search_params.vector_filters_file.is_some();
                        let mode: SearchMode<'_> = search_params.search_mode.search_mode(
                            has_filter,
                            vf,
                            search_params.post_processor.as_ref(),
                        );

                        let api_start = Instant::now();
                        let search_result = searcher.search_with_indexed_vectors(
                            q,
                            return_list_size,
                            l,
                            Some(search_params.beam_width),
                            mode,
                        );
                        let api_elapsed = api_start.elapsed();
                        *call_latency_us = api_elapsed.as_micros().min(u64::MAX as u128) as u64;

                        match search_result {
                            Ok(search_result) => {
                                *stats = search_result.stats.query_statistics;
                                let base_count = (search_result.stats.result_count as usize)
                                    .min(return_list_size as usize)
                                    .min(search_result.results.len());

                                *rc = base_count as u32;
                                id_chunk.fill(0);
                                dist_chunk.fill(0.0);

                                for (i, result_item) in
                                    search_result.results.iter().take(base_count).enumerate()
                                {
                                    id_chunk[i] = result_item.vertex_id;
                                    dist_chunk[i] = result_item.distance;
                                }
                                *payload_bytes = search_result
                                    .results
                                    .iter()
                                    .map(|result_item| {
                                        black_box(result_item.indexed_vector.as_ref());
                                        (result_item.indexed_vector.len() as u64)
                                            * (size_of::<T>() as u64)
                                    })
                                    .sum();
                                // The owned indexed vectors are dropped with `search_result` before
                                // this per-query closure returns; none are cloned or serialized.
                            }
                            Err(e) => {
                                eprintln!("Search failed for query: {:?}", e);
                                *rc = 0;
                                id_chunk.fill(0);
                                dist_chunk.fill(0.0);
                                has_any_search_failed
                                    .store(true, std::sync::atomic::Ordering::Release);
                            }
                        }
                    },
                );
                Some((public_api_call_latencies_us, returned_vector_payload_bytes))
            }
        };
        let total_time = start.elapsed();

        if has_any_search_failed.load(std::sync::atomic::Ordering::Acquire) {
            anyhow::bail!("One or more searches failed. See logs for details.");
        }

        let public_api_metrics = public_api_samples
            .map(|(call_latencies_us, payload_bytes)| {
                aggregate_public_api_metrics(call_latencies_us, &payload_bytes)
            })
            .transpose()?;
        let search_result = DiskSearchResult::new(
            &statistics_vec,
            &result_ids,
            &result_counts,
            public_api_metrics,
            l,
            total_time.as_secs_f32(),
            num_queries,
            return_list_size,
            &gt_context,
        )?;

        l_span.end();
        search_results_per_l.push(search_result);
    }

    // Log search completed checkpoint
    logger.log_checkpoint("search_completed");

    // Get span data
    let span_metrics = if let Some((collector, provider)) = span_collector {
        provider.shutdown()?;
        collector.to_hierarchical_json()
    } else {
        serde_json::json!({ "span_data": [] })
    };

    global::set_tracer_provider(previous_tracer_provider);

    Ok(DiskSearchStats {
        search_api: search_params.search_api,
        num_threads: search_params.num_threads,
        beam_width: search_params.beam_width,
        recall_at: search_params.recall_at,
        return_list_size: Some(return_list_size),
        is_flat_search: search_params.search_mode.is_flat_search,
        distance: search_params.distance,
        uses_vector_filters: search_params.vector_filters_file.is_some(),
        num_nodes_to_cache: search_params.num_nodes_to_cache,
        search_results_per_l,
        span_metrics,
    })
}

// Simplified internal structures to reduce parameter count
pub(super) struct GroundTruthContext {
    gt_ids: Option<Vec<u32>>,
    gt_ids_variable_length: Option<Vec<Vec<u32>>>,
    gt_dists: Option<Vec<f32>>,
    gt_dim: usize,
    recall_at: u32,
}

fn prepare_ground_truth_context(
    has_vector_filters: bool,
    groundtruth: &InputFile,
    recall_at: u32,
    storage: &impl StorageReadProvider,
) -> anyhow::Result<GroundTruthContext> {
    let path = groundtruth.to_string_lossy().into_owned();

    if has_vector_filters {
        let ts = search_index_utils::load_range_truthset(storage, &path)?;
        Ok(GroundTruthContext {
            gt_ids: None,
            gt_ids_variable_length: Some(ts.index_nodes),
            gt_dists: None,
            gt_dim: 0,
            recall_at,
        })
    } else {
        let ts = search_index_utils::load_truthset(storage, &path)?;
        Ok(GroundTruthContext {
            gt_ids: Some(ts.index_nodes),
            gt_ids_variable_length: None,
            gt_dists: ts.distances,
            gt_dim: ts.index_dimension,
            recall_at,
        })
    }
}

impl fmt::Display for DiskSearchStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fmt_us = |v: f64| -> String { format!("{:.1}us", v) };
        let fmt_pct = |v: f64| -> String { format!("{:.1}%", v) };

        let show_api_metrics = self.search_results_per_l.iter().all(|result| {
            result.mean_public_api_call_latency_us.is_some()
                && result.p95_public_api_call_latency_us.is_some()
                && result.p999_public_api_call_latency_us.is_some()
                && result.mean_returned_vector_payload_bytes.is_some()
                && result.max_returned_vector_payload_bytes.is_some()
        });
        let mut cols = vec![
            ("L", 2),
            ("KNN", 3),
            ("QPS", 8),
            ("Internal Mean", 13),
            ("Internal P95", 13),
            ("Internal P999", 13),
        ];
        if show_api_metrics {
            cols.extend([
                ("API Call Mean", 13),
                ("API Call P95", 13),
                ("API Call P999", 13),
                ("Vector B/q Mean", 15),
                ("Vector B/q Max", 14),
            ]);
        }
        cols.extend([
            ("IOs", 6),
            ("IO (us)", 10),
            ("CPU (us)", 10),
            ("PQ Preprocess (us)", 20),
            ("Mean Comps", 11),
            ("Mean Hops", 10),
            ("Cache Hit %", 12),
            ("Recall", 7),
        ]);

        // Build header with exact widths
        let mut header = String::new();
        for (i, (name, w)) in cols.iter().enumerate() {
            if i > 0 {
                header.push(' ');
            }
            header.push_str(&format!("{:>width$}", *name, width = *w));
        }
        let rule = "=".repeat(header.len());

        // Summary
        writeln!(f, "Search Stats")?;
        writeln!(f, "Search API,       : {}", self.search_api)?;
        writeln!(f, "Threads,          : {}", self.num_threads)?;
        writeln!(f, "Beam width,       : {}", self.beam_width)?;
        writeln!(f, "Recall at,        : {}", self.recall_at)?;
        writeln!(
            f,
            "Return K,         : {}",
            self.return_list_size.unwrap_or(self.recall_at)
        )?;
        writeln!(f, "Flat search,      : {}", self.is_flat_search)?;
        writeln!(f, "Distance,         : {}", self.distance)?;
        writeln!(f, "Vector filters,   : {}", self.uses_vector_filters)?;
        writeln!(
            f,
            "Nodes to cache,   : {}",
            self.num_nodes_to_cache
                .map(|n| n.to_string())
                .unwrap_or_else(|| "None".to_string())
        )?;

        // Table
        writeln!(f, "{rule}")?;
        writeln!(f, "{header}")?;
        writeln!(f, "{rule}")?;

        for r in &self.search_results_per_l {
            // Prepare values as strings with numeric formatting
            let mut vals = vec![
                format!("{}", r.search_l),
                format!("{}", self.recall_at),
                format!("{:.1}", r.qps),
                fmt_us(r.mean_latency),
                format!("{}", r.p95_latency),
                format!("{}", r.p999_latency),
            ];
            if show_api_metrics {
                vals.extend([
                    fmt_us(r.mean_public_api_call_latency_us.unwrap()),
                    format!("{}", r.p95_public_api_call_latency_us.unwrap()),
                    format!("{}", r.p999_public_api_call_latency_us.unwrap()),
                    format!("{:.1}", r.mean_returned_vector_payload_bytes.unwrap()),
                    format!("{}", r.max_returned_vector_payload_bytes.unwrap()),
                ]);
            }
            vals.extend([
                format!("{:.1}", r.mean_ios),
                fmt_us(r.mean_io_time),
                fmt_us(r.mean_cpu_time),
                fmt_us(r.mean_pq_preprocess_time),
                format!("{:.1}", r.mean_comparisons),
                format!("{:.1}", r.mean_hops),
                fmt_pct(r.cache_hit_percentage),
                format!("{:.3}", r.recall),
            ]);

            // Right align each value to the column width, one space between columns
            let mut line = String::new();
            for ((_, w), v) in cols.iter().zip(vals.iter()) {
                if !line.is_empty() {
                    line.push(' ');
                }
                line.push_str(&format!("{:>width$}", v, width = *w));
            }
            writeln!(f, "{line}")?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn output_without_api_metrics() -> serde_json::Value {
        json!({
            "num_threads": 1,
            "beam_width": 4,
            "recall_at": 10,
            "is_flat_search": false,
            "distance": "squared_l2",
            "uses_vector_filters": false,
            "num_nodes_to_cache": null,
            "search_results_per_l": [{
                "search_l": 10,
                "qps": 100.0,
                "mean_latency": 10.0,
                "p95_latency": 12,
                "p999_latency": 14,
                "mean_ios": 1.0,
                "mean_io_time": 2.0,
                "mean_cpu_time": 3.0,
                "mean_pq_preprocess_time": 4.0,
                "mean_comparisons": 5.0,
                "mean_hops": 6.0,
                "cache_hit_percentage": 90.0,
                "recall": 0.9
            }],
            "span_metrics": {"span_data": []}
        })
    }

    #[test]
    fn aggregates_public_api_call_latency_and_payload_metrics() {
        let metrics =
            aggregate_public_api_metrics(vec![10, 20, 30, 40, 50], &[128, 256, 384, 512, 640])
                .unwrap();
        assert_eq!(
            metrics,
            PublicApiMetrics {
                mean_call_latency_us: 30.0,
                p95_call_latency_us: MicroSeconds::new(50),
                p999_call_latency_us: MicroSeconds::new(50),
                mean_payload_bytes: 384.0,
                max_payload_bytes: 640,
            }
        );
    }

    #[test]
    fn percentiles_keep_existing_rank_semantics() {
        let metrics = aggregate_public_api_metrics(
            (0..100).rev().collect(),
            &(0..100).map(|_| 0).collect::<Vec<_>>(),
        )
        .unwrap();
        assert_eq!(metrics.p95_call_latency_us, MicroSeconds::new(95));
        assert_eq!(metrics.p999_call_latency_us, MicroSeconds::new(99));
    }

    #[test]
    fn legacy_payload_aggregation_is_zero() {
        let metrics = aggregate_public_api_metrics(vec![4, 8], &[0, 0]).unwrap();
        assert_eq!(metrics.mean_payload_bytes, 0.0);
        assert_eq!(metrics.max_payload_bytes, 0);
    }

    #[test]
    fn metric_aggregation_validates_samples() {
        assert!(aggregate_public_api_metrics(vec![], &[]).is_err());
        assert!(aggregate_public_api_metrics(vec![1], &[]).is_err());
    }

    #[test]
    fn old_output_schema_deserializes_with_legacy_and_no_api_metrics() {
        let stats: DiskSearchStats = serde_json::from_value(output_without_api_metrics()).unwrap();
        assert_eq!(stats.search_api, DiskSearchApi::Legacy);
        let result = &stats.search_results_per_l[0];
        assert_eq!(result.mean_public_api_call_latency_us, None);
        assert_eq!(result.p95_public_api_call_latency_us, None);
        assert_eq!(result.p999_public_api_call_latency_us, None);
        assert_eq!(result.mean_returned_vector_payload_bytes, None);
        assert_eq!(result.max_returned_vector_payload_bytes, None);

        let serialized = serde_json::to_value(stats).unwrap();
        assert!(serialized["search_results_per_l"][0]
            .get("mean_public_api_call_latency_us")
            .is_none());
    }

    #[test]
    fn collected_api_metrics_deserialize_as_some() {
        let mut output = output_without_api_metrics();
        output["search_api"] = json!("indexed-vectors");
        let result = &mut output["search_results_per_l"][0];
        result["mean_public_api_call_latency_us"] = json!(11.0);
        result["p95_public_api_call_latency_us"] = json!(12);
        result["p999_public_api_call_latency_us"] = json!(13);
        result["mean_returned_vector_payload_bytes"] = json!(512.0);
        result["max_returned_vector_payload_bytes"] = json!(640);

        let stats: DiskSearchStats = serde_json::from_value(output).unwrap();
        assert_eq!(stats.search_api, DiskSearchApi::IndexedVectors);
        let result = &stats.search_results_per_l[0];
        assert_eq!(result.mean_public_api_call_latency_us, Some(11.0));
        assert_eq!(
            result.p95_public_api_call_latency_us,
            Some(MicroSeconds::new(12))
        );
        assert_eq!(result.mean_returned_vector_payload_bytes, Some(512.0));
        assert_eq!(result.max_returned_vector_payload_bytes, Some(640));
    }
}

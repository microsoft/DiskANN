/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use rayon::prelude::*;
use std::{collections::HashSet, fmt, sync::atomic::AtomicBool, sync::Arc, time::Instant};

use opentelemetry::{global, trace::Span, trace::Tracer};
use opentelemetry_sdk::trace::SdkTracerProvider;

use diskann::utils::VectorRepr;
use diskann_benchmark_runner::{files::InputFile, utils::MicroSeconds};
use diskann_disk::{
    data_model::CachingStrategy,
    search::provider::{
        disk_provider::DiskIndexSearcher, disk_vertex_provider_factory::DiskVertexProviderFactory,
    },
    search::traits::VertexProviderFactory,
    storage::disk_index_reader::DiskIndexReader,
    utils::{instrumentation::PerfLogger, statistics, AlignedFileReaderFactory, QueryStatistics},
};
#[cfg(target_os = "linux")]
use diskann_disk::search::pipelined::{PipelinedSearcher, PipelinedReaderConfig};
use diskann_providers::storage::StorageReadProvider;
use diskann_providers::{
    storage::{
        get_compressed_pq_file, get_disk_index_file, get_pq_pivot_file, FileStorageProvider,
    },
    utils::{create_thread_pool, ParallelIteratorInPool},
};
use diskann_tools::utils::{search_index_utils, KRecallAtN};
use diskann_utils::views::Matrix;
use serde::Serialize;

use crate::{
    backend::disk_index::{graph_data_type::GraphData, json_spancollector::JsonSpanCollector},
    inputs::disk::{DiskIndexLoad, DiskSearchPhase, SearchMode},
    utils::{datafiles, SimilarityMeasure},
};

#[derive(Serialize, Debug)]
pub(super) struct DiskSearchStats {
    pub(super) num_threads: usize,
    pub(super) beam_width: usize,
    pub(super) recall_at: u32,
    pub(crate) is_flat_search: bool,
    pub(crate) distance: SimilarityMeasure,
    pub(crate) uses_vector_filters: bool,
    pub(super) search_mode: String,
    pub(super) num_nodes_to_cache: Option<usize>,
    pub(super) search_results_per_l: Vec<DiskSearchResult>,
    span_metrics: serde_json::Value,
}

#[derive(Serialize, Debug)]
pub(super) struct DiskSearchResult {
    pub(super) search_l: u32,
    pub(super) qps: f32,
    pub(super) mean_latency: f64,
    pub(super) p95_latency: MicroSeconds,
    pub(super) p999_latency: MicroSeconds,
    pub(super) mean_ios: f64,
    pub(super) mean_io_time: f64,
    pub(super) mean_cpu_time: f64,
    pub(super) mean_pq_preprocess_time: f64,
    pub(super) mean_comparisons: f64,
    pub(super) mean_hops: f64,
    pub(super) cache_hit_percentage: f64,
    pub(super) recall: f32,
}

impl DiskSearchResult {
    pub(super) fn new(
        statistics: &[QueryStatistics],
        result_ids: &[u32],
        result_counts: &[u32],
        search_l: u32,
        total_time_as_secs: f32,
        num_queries: usize,
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
                .chunks_exact(gt_context.recall_at as usize)
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
                gt_context.recall_at,
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

    // Setup disk index components
    let pivot_path = get_pq_pivot_file(&index_load.load_path);
    let pq_data_path = get_compressed_pq_file(&index_load.load_path);
    let disk_index_path = get_disk_index_file(&index_load.load_path);

    let index_reader = DiskIndexReader::<T>::new(pivot_path, pq_data_path, &FileStorageProvider)?;

    let caching_strategy = if let Some(num_nodes) = search_params.num_nodes_to_cache {
        CachingStrategy::StaticCacheWithBfsNodes(num_nodes)
    } else {
        CachingStrategy::None
    };

    let reader_factory = AlignedFileReaderFactory::new(disk_index_path.clone());
    let vertex_provider_factory = DiskVertexProviderFactory::new(reader_factory, caching_strategy)?;

    let pool = create_thread_pool(search_params.num_threads)?;
    let mut search_results_per_l = Vec::with_capacity(search_params.search_list.len());
    let has_any_search_failed = AtomicBool::new(false);

    match &search_params.search_mode {
        SearchMode::BeamSearch => {
            let searcher = &DiskIndexSearcher::<GraphData<T>, _>::new(
                search_params.num_threads,
                search_params.search_io_limit.unwrap_or(usize::MAX),
                &index_reader,
                vertex_provider_factory,
                search_params.distance.into(),
                None,
            )?;

            logger.log_checkpoint("index_loaded");

            for &l in search_params.search_list.iter() {
                let mut statistics_vec: Vec<QueryStatistics> =
                    vec![QueryStatistics::default(); num_queries];
                let mut result_counts: Vec<u32> = vec![0; num_queries];
                let mut result_ids: Vec<u32> =
                    vec![0; (search_params.recall_at as usize) * num_queries];
                let mut result_dists: Vec<f32> =
                    vec![0.0; (search_params.recall_at as usize) * num_queries];

                let start = Instant::now();

                let mut l_span = {
                    let tracer = global::tracer("");
                    let span_name =
                        format!("search-with-L={}-bw={}", l, search_params.beam_width);
                    tracer.start(span_name)
                };

                let zipped = queries
                    .par_row_iter()
                    .zip(vector_filters.par_iter())
                    .zip(result_ids.par_chunks_mut(search_params.recall_at as usize))
                    .zip(result_dists.par_chunks_mut(search_params.recall_at as usize))
                    .zip(statistics_vec.par_iter_mut())
                    .zip(result_counts.par_iter_mut());

                zipped.for_each_in_pool(
                    &pool,
                    |(((((q, vf), id_chunk), dist_chunk), stats), rc)| {
                        let vector_filter = if search_params.vector_filters_file.is_none() {
                            None
                        } else {
                            Some(Box::new(move |vid: &u32| vf.contains(vid))
                                as Box<dyn Fn(&u32) -> bool + Send + Sync>)
                        };

                        match searcher.search(
                            q,
                            search_params.recall_at,
                            l,
                            Some(search_params.beam_width),
                            vector_filter,
                            search_params.is_flat_search,
                        ) {
                            Ok(search_result) => {
                                *stats = search_result.stats.query_statistics;
                                *rc = search_result.results.len() as u32;
                                let actual_results = search_result
                                    .results
                                    .len()
                                    .min(search_params.recall_at as usize);
                                for (i, result_item) in search_result
                                    .results
                                    .iter()
                                    .take(actual_results)
                                    .enumerate()
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
                let total_time = start.elapsed();

                if has_any_search_failed.load(std::sync::atomic::Ordering::Acquire) {
                    anyhow::bail!("One or more searches failed. See logs for details.");
                }

                let search_result = DiskSearchResult::new(
                    &statistics_vec,
                    &result_ids,
                    &result_counts,
                    l,
                    total_time.as_secs_f32(),
                    num_queries,
                    &gt_context,
                )?;

                l_span.end();
                search_results_per_l.push(search_result);
            }
        }
        // PipeANN pipelined search — for read-only search on completed (static) indices only.
        // Searcher is created once; internal ObjectPool handles per-thread scratch allocation.
        // Build's internal search always uses BeamSearch above.
        SearchMode::PipeSearch {
            initial_beam_width,
            relaxed_monotonicity_l,
            sqpoll_idle_ms,
            iopoll,
        } => {
            #[cfg(target_os = "linux")]
            {
                let graph_header = vertex_provider_factory.get_header()?;
                let pq_data = index_reader.get_pq_data();
                let metric = search_params.distance.into();
                let search_io_limit = search_params.search_io_limit.unwrap_or(usize::MAX);
                let initial_beam_width = *initial_beam_width;
                let relaxed_monotonicity_l = *relaxed_monotonicity_l;

                let reader_config = PipelinedReaderConfig {
                    sqpoll_idle_ms: *sqpoll_idle_ms,
                    iopoll: *iopoll,
                };

                // Create searcher once — pool handles per-thread scratch allocation
                let pipe_searcher = Arc::new(PipelinedSearcher::<GraphData<T>>::new(
                    graph_header.clone(),
                    pq_data.clone(),
                    metric,
                    search_io_limit,
                    initial_beam_width,
                    relaxed_monotonicity_l,
                    disk_index_path.clone(),
                    reader_config,
                )?);

                logger.log_checkpoint("index_loaded");

                for &l in search_params.search_list.iter() {
                    let mut statistics_vec: Vec<QueryStatistics> =
                        vec![QueryStatistics::default(); num_queries];
                    let mut result_counts: Vec<u32> = vec![0; num_queries];
                    let mut result_ids: Vec<u32> =
                        vec![0; (search_params.recall_at as usize) * num_queries];
                    let mut result_dists: Vec<f32> =
                        vec![0.0; (search_params.recall_at as usize) * num_queries];

                    let start = Instant::now();

                    let mut l_span = {
                        let tracer = global::tracer("");
                        let span_name =
                            format!("pipesearch-with-L={}-bw={}", l, search_params.beam_width);
                        tracer.start(span_name)
                    };

                    let pipe_searcher = pipe_searcher.clone(); // Arc clone for this L iteration

                    let zipped = queries
                        .par_row_iter()
                        .zip(result_ids.par_chunks_mut(search_params.recall_at as usize))
                        .zip(result_dists.par_chunks_mut(search_params.recall_at as usize))
                        .zip(statistics_vec.par_iter_mut())
                        .zip(result_counts.par_iter_mut());

                    zipped.for_each_in_pool(
                        &pool,
                        |((((q, id_chunk), dist_chunk), stats), rc)| {
                            match pipe_searcher.search(
                                q,
                                search_params.recall_at,
                                l,
                                search_params.beam_width,
                            ) {
                                Ok(search_result) => {
                                    *stats = search_result.stats.query_statistics;
                                    *rc = search_result.results.len() as u32;
                                    let actual_results = search_result
                                        .results
                                        .len()
                                        .min(search_params.recall_at as usize);
                                    for (i, result_item) in search_result
                                        .results
                                        .iter()
                                        .take(actual_results)
                                        .enumerate()
                                    {
                                        id_chunk[i] = result_item.vertex_id;
                                        dist_chunk[i] = result_item.distance;
                                    }
                                }
                                Err(e) => {
                                    eprintln!("PipeSearch failed for query: {:?}", e);
                                    *rc = 0;
                                    id_chunk.fill(0);
                                    dist_chunk.fill(0.0);
                                    has_any_search_failed
                                        .store(true, std::sync::atomic::Ordering::Release);
                                }
                            }
                        },
                    );
                    let total_time = start.elapsed();

                    if has_any_search_failed.load(std::sync::atomic::Ordering::Acquire) {
                        anyhow::bail!("One or more searches failed. See logs for details.");
                    }

                    let search_result = DiskSearchResult::new(
                        &statistics_vec,
                        &result_ids,
                        &result_counts,
                        l,
                        total_time.as_secs_f32(),
                        num_queries,
                        &gt_context,
                    )?;

                    l_span.end();
                    search_results_per_l.push(search_result);
                }
            }
            #[cfg(not(target_os = "linux"))]
            {
                let _ = (initial_beam_width, relaxed_monotonicity_l, sqpoll_idle_ms, iopoll);
                anyhow::bail!("PipeSearch is only supported on Linux");
            }
        }
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
        num_threads: search_params.num_threads,
        beam_width: search_params.beam_width,
        recall_at: search_params.recall_at,
        is_flat_search: search_params.is_flat_search,
        distance: search_params.distance,
        uses_vector_filters: search_params.vector_filters_file.is_some(),
        search_mode: format!("{:?}", search_params.search_mode),
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

        let cols: [(&str, usize); 14] = [
            ("L", 2),
            ("KNN", 3),
            ("QPS", 8),
            ("Mean Latency", 13),
            ("95% Latency", 13),
            ("99.9 Latency", 13),
            ("IOs", 6),
            ("IO (us)", 10),
            ("CPU (us)", 10),
            ("PQ Preprocess (us)", 20),
            ("Mean Comps", 11),
            ("Mean Hops", 10),
            ("Cache Hit %", 12),
            ("Recall", 7),
        ];

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
        writeln!(f, "Threads,          : {}", self.num_threads)?;
        writeln!(f, "Beam width,       : {}", self.beam_width)?;
        writeln!(f, "Recall at,        : {}", self.recall_at)?;
        writeln!(f, "Flat search,      : {}", self.is_flat_search)?;
        writeln!(f, "Distance,         : {}", self.distance)?;
        writeln!(f, "Vector filters,   : {}", self.uses_vector_filters)?;
        writeln!(f, "Search mode,      : {}", self.search_mode)?;
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
            let vals: [String; 14] = [
                format!("{}", r.search_l),
                format!("{}", self.recall_at),
                format!("{:.1}", r.qps),
                fmt_us(r.mean_latency),
                format!("{}", r.p95_latency),
                format!("{}", r.p999_latency),
                format!("{:.1}", r.mean_ios),
                fmt_us(r.mean_io_time),
                fmt_us(r.mean_cpu_time),
                fmt_us(r.mean_pq_preprocess_time),
                format!("{:.1}", r.mean_comparisons),
                format!("{:.1}", r.mean_hops),
                fmt_pct(r.cache_hit_percentage),
                format!("{:.3}", r.recall),
            ];

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

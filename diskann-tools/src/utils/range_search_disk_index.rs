/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::{path::Path, time::Instant};

use diskann_providers::{
    index::DiskIndexSearcher,
    model::{
        aligned_file_reader::AlignedFileReaderFactory,
        graph::{
            graph_data_model::CachingStrategy, provider::disk::DiskVertexProviderFactory,
            traits::GraphDataType,
        },
        statistics, QueryStatistics,
    },
    storage::{get_disk_index_file, DiskIndexReader, FileStorageProvider},
    utils::{create_thread_pool, load_aligned_bin, ParallelIteratorInPool},
};
use diskann::ANNResult;
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use diskann_vector::distance::Metric;

use crate::utils::search_index_utils;

#[allow(clippy::too_many_arguments, clippy::unwrap_used)]
pub fn range_search_disk_index<Data>(
    metric: Metric,
    index_path_prefix: &str,
    query_file: &str,
    truthset_file: &str,
    num_threads: usize,
    range_threshold: f32,
    beam_width: u32,
    search_io_limit: u32,
    l_vec: &[u32],
    num_nodes_to_cache: usize,
) -> ANNResult<(Vec<Vec<Vec<f32>>>, f32)>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    println!(
        "Search parameters: #threads: {}, range_threshold {}, search_list_size: {:?}, search_io_limit: {}, beam_width: {}",
        num_threads, range_threshold, l_vec, search_io_limit,beam_width
    );

    let storage_provider = FileStorageProvider;

    // Load the query file
    let (query, query_num, _, query_aligned_dim) =
        load_aligned_bin::<Data::VectorDataType>(&storage_provider, query_file)?;
    let mut gt_ids: Option<Vec<Vec<u32>>> = None;

    // Check for ground truth
    let mut calc_recall_flag = false;
    if !truthset_file.is_empty() && Path::new(truthset_file).exists() {
        let ret = search_index_utils::load_range_truthset(&storage_provider, truthset_file)?;
        gt_ids = Some(ret.index_nodes);
        let gt_num = ret.index_num_points;

        if gt_num != query_num {
            println!("Error. Mismatch in number of queries and ground truth data");
        }

        calc_recall_flag = true;
    } else {
        println!(
            "Truthset file {} not found. Not computing recall",
            truthset_file
        );
    }

    let index_reader = DiskIndexReader::<<Data as GraphDataType>::VectorDataType>::new(
        format!("{}_pq_pivots.bin", index_path_prefix),
        format!("{}_pq_compressed.bin", index_path_prefix),
        &storage_provider,
    )?;

    // Create the vertex provider factor
    let caching_strategy = if num_nodes_to_cache > 0 {
        CachingStrategy::StaticCacheWithBfsNodes(num_nodes_to_cache)
    } else {
        CachingStrategy::None
    };

    let aligned_file_reader_factory =
        AlignedFileReaderFactory::new(get_disk_index_file(index_path_prefix));
    // Create the vertex provider factory
    let vertex_provider_factory =
        DiskVertexProviderFactory::new(aligned_file_reader_factory, caching_strategy)?;

    let searcher =
        DiskIndexSearcher::<Data, DiskVertexProviderFactory<Data, AlignedFileReaderFactory>>::new(
            num_threads,
            search_io_limit,
            beam_width,
            &index_reader,
            vertex_provider_factory,
            metric,
        )?;

    let range_threshold_string = format!("Recall for Range Threshold={}", range_threshold);
    if calc_recall_flag {
        println!(
            "{:<6}{:<12}{:<16}{:<20}{:<20}{:<16}{:<20}{:<16}{:<16}",
            "L",
            "Beamwidth",
            "QPS",
            "Mean Latency (us)",
            "99.9 Latency (us)",
            "Mean IOs",
            "Mean IO (us)",
            "CPU (us)",
            range_threshold_string
        );
    } else {
        println!(
            "{:<6}{:<12}{:<16}{:<20}{:<20}{:<16}{:<20}{:<16}",
            "L",
            "Beamwidth",
            "QPS",
            "Mean Latency (us)",
            "99.9 Latency (us)",
            "Mean IOs",
            "Mean IO (us)",
            "CPU (us)"
        );
    }
    println!("{:=<140}", "");

    let mut query_result_ids: Vec<Vec<Vec<u32>>> = vec![vec![vec![]; query_num]; l_vec.len()];
    let mut query_result_dists: Vec<Vec<Vec<f32>>> = vec![vec![vec![]; query_num]; l_vec.len()];
    let mut res_counts: Vec<u32> = vec![0; query_num];

    let mut best_recall = 0.0;
    let max_search_list_size = index_reader.get_num_points() as u32;

    let pool = create_thread_pool(num_threads)?;

    for (test_id, &l) in l_vec.iter().enumerate() {
        // Assuming `QueryStats` is a struct that you have defined elsewhere
        let mut statistics: Vec<QueryStatistics> = vec![QueryStatistics::default(); query_num];

        let zipped = res_counts
            .par_iter_mut()
            .zip(query.par_chunks(query_aligned_dim))
            .zip(query_result_ids[test_id].par_iter_mut())
            .zip(query_result_dists[test_id].par_iter_mut())
            .zip(statistics.par_iter_mut());

        let test_start = Instant::now();
        zipped.for_each_in_pool(
            &pool,
            |((((res_count, query), query_result_id), query_result_dist), stats)| {
                let mut associated_data = vec![];

                *res_count = searcher
                    .range_search(
                        query,
                        range_threshold,
                        l,
                        max_search_list_size,
                        beam_width,
                        query_result_id,
                        query_result_dist,
                        stats,
                        &mut associated_data,
                    )
                    .unwrap();

                associated_data.resize(*res_count as usize, Data::AssociatedDataType::default());
                query_result_dist.resize(*res_count as usize, 0.0);
                query_result_id.resize(*res_count as usize, 0);
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

        let mut recall = 0.0;
        if calc_recall_flag {
            recall = search_index_utils::calculate_range_search_recall(
                query_num as u32,
                gt_ids.as_ref().unwrap(),
                &query_result_ids[test_id],
            )? as f32;

            best_recall = f32::from(std::cmp::max(
                OrderedFloat::<f32>(best_recall),
                OrderedFloat::<f32>(recall),
            ));
        }

        if calc_recall_flag {
            println!(
                "{:<6}{:<12.2}{:<16.2}{:<20.2}{:<20.2}{:<16.2}{:<20.2}{:<16.2}{:<16.2}",
                l,
                beam_width,
                qps,
                mean_latency,
                latency_999,
                mean_ios,
                mean_io_time,
                mean_cpus,
                recall
            );
        } else {
            println!(
                "{:<6}{:<12.2}{:<20.2}{:<20.2}{:<16.2}{:<16.2}{:<20.2}{:<16.2}",
                l, beam_width, qps, mean_latency, latency_999, mean_ios, mean_io_time, mean_cpus
            );
        }
    }

    Ok((query_result_dists, best_recall))
}

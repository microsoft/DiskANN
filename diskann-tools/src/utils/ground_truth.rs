/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use bit_set::BitSet;
use diskann_label_filter::{eval_query_expr, read_and_parse_queries, read_baselabels};

use std::{io::Write, mem::size_of, str::FromStr};

use bytemuck::cast_slice;
use diskann::{
    neighbor::{Neighbor, NeighborPriorityQueue},
    utils::VectorRepr,
};
use diskann_providers::storage::{StorageReadProvider, StorageWriteProvider};
use diskann_providers::{
    common::AlignedBoxWithSlice,
    model::graph::traits::GraphDataType,
    utils::{
        create_thread_pool, file_util, write_metadata, ParallelIteratorInPool, VectorDataIterator,
    },
};
use diskann_utils::views::Matrix;
use diskann_vector::{distance::Metric, DistanceFunction};
use itertools::Itertools;
use rayon::prelude::*;

use crate::utils::{search_index_utils, CMDResult, CMDToolError};

pub fn read_labels_and_compute_bitmap(
    base_label_filename: &str,
    query_label_filename: &str,
) -> CMDResult<Vec<BitSet>> {
    // Read base labels
    let base_labels = read_baselabels(base_label_filename)?;

    // Parse queries and evaluate against labels
    let parsed_queries = read_and_parse_queries(query_label_filename)?;

    // using the global threadpool is fine here
    #[allow(clippy::disallowed_methods)]
    let query_bitmaps: Vec<BitSet> = parsed_queries
        .par_iter()
        .map(|(_query_id, query_expr)| {
            let mut bitmap = BitSet::new();
            for base_label in base_labels.iter() {
                if eval_query_expr(query_expr, &base_label.label) {
                    bitmap.insert(base_label.doc_id);
                }
            }
            bitmap
        })
        .collect();

    Ok(query_bitmaps)
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::panic)]
/// Computes the true nearest neighbors for a set of queries and writes them to a file.
///
/// # Arguments
///
/// * `distance_function` - e.g. L2
/// * `base_file` - The file containing the base vectors.
/// * `query_file` - The file containing the query vectors.
/// * `ground_truth_file` - The file to write the ground truth results to.
/// * `recall_at` - The number of neighbors to compute for each query.
/// * `insert_file` - Optional file containing more dataset vectors. This may be useful if you are testing recall for an index that has points dynamically inserted into it.
/// * `skip_base` - Optional number of base points to skip. This is useful if you want to compute the ground truth for a set where the first skip_base points are deleted from the index.
pub fn compute_ground_truth_from_datafiles<
    Data: GraphDataType,
    StorageProvider: StorageReadProvider + StorageWriteProvider,
>(
    storage_provider: &StorageProvider,
    distance_function: Metric,
    base_file: &str,
    query_file: &str,
    ground_truth_file: &str,
    vector_filters_file: Option<&str>,
    recall_at: u32,
    insert_file: Option<&str>,
    skip_base: Option<usize>,
    associated_data_file: Option<String>,
    base_file_labels: Option<&str>,
    query_file_labels: Option<&str>,
) -> CMDResult<()> {
    let dataset_iterator = VectorDataIterator::<StorageProvider, Data>::new(
        base_file,
        associated_data_file.clone(),
        storage_provider,
    )?;

    // both base_file_labels and query_file_labels are provided or both are not provided
    if !((base_file_labels.is_some() && query_file_labels.is_some())
        || (base_file_labels.is_none() && query_file_labels.is_none()))
    {
        return Err(CMDToolError {
            details: "Both base_file_labels and query_file_labels must be provided or both must be not provided.".to_string(),
        });
    }

    if base_file_labels.is_some() && vector_filters_file.is_some() {
        return Err(CMDToolError {
            details: "Both base_file_labels and vector_filters_file cannot be provided."
                .to_string(),
        });
    }

    let insert_iterator = match insert_file {
        Some(insert_file) => {
            let i = VectorDataIterator::<StorageProvider, Data>::new(
                insert_file,
                Option::None,
                storage_provider,
            )?;
            Some(i)
        }
        None => None,
    };

    // Load the query file
    let (raw_query_data, query_num, query_dim) = file_util::load_bin::<
        Data::VectorDataType,
        StorageProvider,
    >(storage_provider, query_file, 0)?;

    let mut query_bitmaps: Option<Vec<BitSet>> = None;
    if let (Some(base_file_labels), Some(query_file_labels)) = (base_file_labels, query_file_labels)
    {
        query_bitmaps = Some(read_labels_and_compute_bitmap(
            base_file_labels,
            query_file_labels,
        )?);
    }

    let queries: Vec<_> = raw_query_data.chunks(query_dim).collect();

    // Load the vector filters
    let vector_filters = match vector_filters_file {
        Some(vector_filters_file) => {
            let filters =
                search_index_utils::load_vector_filters(storage_provider, vector_filters_file)?;

            assert_eq!(
                filters.len(),
                queries.len(),
                "Mismatch in query and vector filter sizes"
            );

            Some(filters)
        }
        None => None,
    };

    let has_vector_filters = vector_filters.is_some();
    let has_query_bitmaps = query_bitmaps.is_some();

    if has_vector_filters {
        // copy vector_filters to query_bitmaps one item at a time
        if let Some(filters) = vector_filters {
            let mut bitmaps = vec![BitSet::new(); queries.len()];
            for (idx_query, filter) in filters.iter().enumerate() {
                for item in filter.iter() {
                    if let Ok(idx) = (*item).try_into() {
                        bitmaps[idx_query].insert(idx);
                    }
                }
            }
            query_bitmaps = Some(bitmaps)
        }
    }

    let query_aligned_dim = query_dim.next_multiple_of(8);
    let ground_truth_result = compute_ground_truth_from_data::<
        Data,
        StorageProvider,
        VectorDataIterator<StorageProvider, Data>,
    >(
        distance_function,
        dataset_iterator,
        queries,
        query_aligned_dim,
        recall_at,
        insert_iterator,
        skip_base,
        query_bitmaps,
    );
    assert!(
        &ground_truth_result.is_ok(),
        "Ground-truth computation failed"
    );
    let (ground_truth, id_to_associated_data) = ground_truth_result?;

    assert_ne!(ground_truth.len(), 0, "No ground-truth results computed");

    if has_vector_filters || has_query_bitmaps {
        let ground_truth_collection = ground_truth
            .into_iter()
            .map(|npq| npq.into_iter().collect())
            .collect();
        write_range_search_ground_truth(
            storage_provider,
            ground_truth_file,
            query_num,
            ground_truth_collection,
        )
    } else {
        // Write results and return
        let id_to_associated_data = associated_data_file.map(|_| id_to_associated_data);
        write_ground_truth::<Data>(
            storage_provider,
            ground_truth_file,
            query_num,
            recall_at as usize,
            ground_truth,
            id_to_associated_data,
        )
    }
}

#[derive(Debug, Clone)]
pub enum MultivecAggregationMethod {
    AveragePairwise,
    MinPairwise,
    AvgofMins,
}

#[derive(Debug)]
pub enum ParseAggrError {
    InvalidFormat(String),
}

impl std::fmt::Display for ParseAggrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFormat(str) => write!(f, "Invalid format for Aggregation Method: {}", str),
        }
    }
}

impl std::error::Error for ParseAggrError {}

impl FromStr for MultivecAggregationMethod {
    type Err = ParseAggrError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "average_pairwise" => Ok(MultivecAggregationMethod::AveragePairwise),
            "min_pairwise" => Ok(MultivecAggregationMethod::MinPairwise),
            "avg_of_mins" => Ok(MultivecAggregationMethod::AvgofMins),
            _ => Err(ParseAggrError::InvalidFormat(String::from(s))),
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::panic)]
/// Computes the true nearest neighbors for a set of queries and writes them to a file.
///
/// # Arguments
///
/// * `distance_function` - e.g. L2
/// * `aggregation_method` - e.g. Average or Min
/// * `base_file` - The file containing the base vectors.
/// * `query_file` - The file containing the query vectors.
/// * `ground_truth_file` - The file to write the ground truth results to.
/// * `recall_at` - The number of neighbors to compute for each query.
/// * `base_file_labels` - Optional labels file for the base vectors to filter which base vectors to consider per query.
/// * `query_file_labels` - Optional labels file for the query vectors to filter which base vectors to consider per query.
pub fn compute_multivec_ground_truth_from_datafiles<
    Data: GraphDataType,
    StorageProvider: StorageReadProvider + StorageWriteProvider,
>(
    storage_provider: &StorageProvider,
    distance_function: Metric,
    aggregation_method: MultivecAggregationMethod,
    base_file: &str,
    query_file: &str,
    ground_truth_file: &str,
    recall_at: u32,
    base_file_labels: Option<&str>,
    query_file_labels: Option<&str>,
) -> CMDResult<()> {
    let (base_vectors, _, _, _) = file_util::load_multivec_bin::<
        Data::VectorDataType,
        StorageProvider,
    >(storage_provider, base_file)?;

    let (query_vectors, query_num, query_dim, _) = file_util::load_multivec_bin::<
        Data::VectorDataType,
        StorageProvider,
    >(storage_provider, query_file)?;

    // both base_file_labels and query_file_labels are provided or both are not provided
    if !((base_file_labels.is_some() && query_file_labels.is_some())
        || (base_file_labels.is_none() && query_file_labels.is_none()))
    {
        return Err(CMDToolError {
            details: "Both base_file_labels and query_file_labels must be provided or both must be not provided.".to_string(),
        });
    }

    let mut query_bitmaps: Option<Vec<BitSet>> = None;
    if let (Some(base_file_labels), Some(query_file_labels)) = (base_file_labels, query_file_labels)
    {
        query_bitmaps = Some(read_labels_and_compute_bitmap(
            base_file_labels,
            query_file_labels,
        )?);
    }

    let has_query_bitmaps = query_bitmaps.is_some();

    let ground_truth =
        compute_multivec_ground_truth_from_data::<Data::VectorDataType, StorageProvider>(
            distance_function,
            aggregation_method,
            base_vectors,
            query_vectors,
            query_dim,
            recall_at,
            query_bitmaps,
        )?;

    if has_query_bitmaps {
        let ground_truth_collection = ground_truth
            .into_iter()
            .map(|npq| npq.into_iter().collect())
            .collect();
        write_range_search_ground_truth(
            storage_provider,
            ground_truth_file,
            query_num,
            ground_truth_collection,
        )
    } else {
        // Write results and return
        write_ground_truth::<Data>(
            storage_provider,
            ground_truth_file,
            query_num,
            recall_at as usize,
            ground_truth,
            Option::None,
        )
    }
}

pub fn compute_range_search_ground_truth_from_datafiles<
    Data: GraphDataType,
    StorageProvider: StorageReadProvider + StorageWriteProvider,
>(
    storage_provider: &StorageProvider,
    distance_function: Metric,
    base_file: &str,
    query_file: &str,
    ground_truth_file: &str,
    range_threshold: f32,
    tags_file: &str,
) -> CMDResult<()> {
    if !tags_file.is_empty() {
        // We have not implemented tags yet so let the user know!
        return Err(CMDToolError {
            details: "Tag files are not implemented for the ground_truth computation yet."
                .to_string(),
        });
    }

    let dataset_iterator = VectorDataIterator::<StorageProvider, Data>::new(
        base_file,
        Option::None,
        storage_provider,
    )?;

    // Load the query file
    let (raw_query_data, query_num, query_dim) = file_util::load_bin::<
        Data::VectorDataType,
        StorageProvider,
    >(storage_provider, query_file, 0)?;
    let queries: Vec<_> = raw_query_data.chunks(query_dim).collect();

    let query_aligned_dim = query_dim.next_multiple_of(8);
    let ground_truth_result = compute_range_search_ground_truth_from_data::<
        Data,
        StorageProvider,
        VectorDataIterator<StorageProvider, Data>,
    >(
        distance_function,
        dataset_iterator,
        queries,
        query_aligned_dim,
        range_threshold,
    );
    assert!(
        &ground_truth_result.is_ok(),
        "Ground-truth computation failed"
    );
    let ground_truth = ground_truth_result?;

    assert_ne!(ground_truth.len(), 0, "No ground-truth results computed");

    // Write results
    let _res = write_range_search_ground_truth(
        storage_provider,
        ground_truth_file,
        query_num,
        ground_truth,
    );

    Ok(())
}

fn write_range_search_ground_truth<StorageProvider: StorageReadProvider + StorageWriteProvider>(
    storage_provider: &StorageProvider,
    ground_truth_file: &str,
    number_of_queries: usize,
    ground_truth: Vec<Vec<Neighbor<u32>>>,
) -> CMDResult<()> {
    let mut file = storage_provider.create_for_write(ground_truth_file)?;

    let queue_sizes: Vec<u32> = ground_truth
        .iter()
        .map(|queue| queue.len() as u32)
        .collect();
    let total_number_of_neighbors: usize = queue_sizes.iter().sum::<u32>() as usize;

    // Metadata
    write_metadata(&mut file, number_of_queries, total_number_of_neighbors)?;

    // Write queue sizes array.
    let mut queue_sizes_buffer = vec![0; queue_sizes.len() * size_of::<u32>()];
    queue_sizes_buffer.clone_from_slice(cast_slice::<u32, u8>(&queue_sizes));
    file.write_all(&queue_sizes_buffer)?;

    let mut neighbor_ids: Vec<u32> = Vec::with_capacity(total_number_of_neighbors);

    // Write the neighbor IDs array.
    for query_neighbors in ground_truth {
        for neighbor in query_neighbors.iter() {
            neighbor_ids.push(neighbor.id);
        }
    }

    // Write neighbor IDs
    let mut id_buffer = vec![0; total_number_of_neighbors * size_of::<u32>()];
    id_buffer.clone_from_slice(cast_slice::<u32, u8>(&neighbor_ids));
    file.write_all(&id_buffer)?;

    // Make sure everything is written to disk
    file.flush()?;

    Ok(())
}

/// Writes out a ground truth file.  ground_truth is a vector of NeighborPriorityQueue objects
/// where the order of queue objects corresponds to the order of queries used to compute this
/// ground truth.
fn write_ground_truth<Data: GraphDataType>(
    storage_provider: &impl StorageWriteProvider,
    ground_truth_file: &str,
    number_of_queries: usize,
    number_of_neighbors: usize,
    ground_truth: Vec<NeighborPriorityQueue<u32>>,
    id_to_associated_data: Option<Vec<Data::AssociatedDataType>>,
) -> CMDResult<()> {
    let mut file = storage_provider.create_for_write(ground_truth_file)?;

    write_metadata(&mut file, number_of_queries, number_of_neighbors)?;

    let mut gt_ids: Vec<u32> = Vec::with_capacity(number_of_neighbors * number_of_queries);
    let mut gt_distances: Vec<f32> = Vec::with_capacity(number_of_neighbors * number_of_queries);

    // In the file, we write the neighbor IDs array first, then write the distances array.
    for mut query_neighbors in ground_truth {
        while query_neighbors.has_notvisited_node() {
            let closest_node = query_neighbors.closest_notvisited();

            gt_ids.push(closest_node.id);
            gt_distances.push(closest_node.distance);
        }
    }

    // Write neighbor IDs or Associated Data
    if let Some(id_to_associated_data) = id_to_associated_data {
        let mut associated_data_buffer = Vec::<u8>::new();
        for id in gt_ids {
            let associated_data = id_to_associated_data[id as usize];
            let serialized_associated_data =
                bincode::serialize(&associated_data).map_err(|e| CMDToolError {
                    details: format!("Failed to serialize associated data: {}", e),
                })?;
            associated_data_buffer.extend_from_slice(serialized_associated_data.as_slice());
        }
        file.write_all(&associated_data_buffer)?;
    } else {
        let mut id_buffer = vec![0; number_of_queries * number_of_neighbors * size_of::<u32>()];
        id_buffer.clone_from_slice(cast_slice::<u32, u8>(&gt_ids));
        file.write_all(&id_buffer)?;
    }

    // Write neighbor distances
    let mut distance_buffer = vec![0; number_of_queries * number_of_neighbors * size_of::<f32>()];
    distance_buffer.clone_from_slice(cast_slice::<f32, u8>(&gt_distances));
    file.write_all(&distance_buffer)?;

    // Make sure everything is written to disk
    file.flush()?;

    Ok(())
}

type Npq = Vec<NeighborPriorityQueue<u32>>;
/// Computes the true nearest neighbors for a set of queries and dataset iterators
///
/// # Arguments
///
/// * `distance_function` - e.g. L2
/// * `dataset_iter` - The iterator over the dataset vectors, associated data, and
/// * `queries` - A vector of query vectors
/// * `query_aligned_dimmensions` - The number of dimensions to align the query vectors to for optimized distance comparison.
/// * `recall_at` - The number of neighbors to compute for each query.
/// * `insert_iterator` - Optional iterator containing more dataset vectors. This may be useful if you are testing recall for an index that has points dynamically inserted into it.
/// * `skip_base` - Optional number of base points to skip. This is useful if you want to compute the ground truth for a set where the first skip_base points are deleted from the index.
#[allow(clippy::too_many_arguments)]
pub fn compute_ground_truth_from_data<Data, VectorReader, VectorIteratorType>(
    distance_function: Metric,
    dataset_iter: VectorDataIterator<VectorReader, Data>,
    queries: Vec<&[Data::VectorDataType]>,
    query_aligned_dimmensions: usize,
    recall_at: u32,
    insert_iter: Option<VectorDataIterator<VectorReader, Data>>,
    skip_base: Option<usize>,
    query_bitmaps: Option<Vec<BitSet>>,
) -> CMDResult<(Npq, Vec<Data::AssociatedDataType>)>
where
    Data: GraphDataType,
    VectorReader: StorageReadProvider,
{
    let query_num = queries.len();

    let mut aligned_queries = Vec::with_capacity(query_num);
    let mut neighbor_queues: Vec<NeighborPriorityQueue<u32>> = Vec::with_capacity(query_num);
    for query in queries {
        let mut aligned_query = AlignedBoxWithSlice::new(query_aligned_dimmensions, 32)?;
        aligned_query[..query.len()].copy_from_slice(query);
        aligned_queries.push(aligned_query);
        neighbor_queues.push(NeighborPriorityQueue::new(recall_at as usize));
    }
    let mut queries_and_neighbor_queue: Vec<_> = aligned_queries
        .iter()
        .zip(neighbor_queues.iter_mut())
        .collect();

    let distance_comparer =
        Data::VectorDataType::distance(distance_function, Some(query_aligned_dimmensions));

    let batch_size = 10_000;
    let mut aligned_data_batch = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        aligned_data_batch.push(AlignedBoxWithSlice::new(query_aligned_dimmensions, 32)?);
    }

    let pool = create_thread_pool(0)?;

    let mut num_base_points: usize = 0;
    let mut id_to_associated_data = Vec::<Data::AssociatedDataType>::new();
    let skip_base = skip_base.unwrap_or(0);
    // Loop over all the raw data
    for chunk in dataset_iter
        .skip(skip_base)
        .enumerate()
        .chunks(batch_size)
        .into_iter()
    {
        let mut points = 0;
        for (idx, (data_vector, associated_data)) in chunk {
            aligned_data_batch[idx % batch_size][..data_vector.len()].copy_from_slice(&data_vector);
            id_to_associated_data.push(associated_data);
            points += 1;
        }

        if points == 0 {
            continue;
        }

        // For each node in the raw data, calculate the distance to each query vector and store it in the priority queue for that query.  This will find the closest N neighbors for each query.
        queries_and_neighbor_queue
            .par_iter_mut()
            .enumerate()
            .for_each_in_pool(
                &pool,
                |(idx_query, (aligned_query, ref mut neighbor_queue))| {
                    for (idx_in_batch, aligned_data) in
                        aligned_data_batch[..points].iter().enumerate()
                    {
                        let idx = (num_base_points + idx_in_batch) as u32;

                        let allowed_by_bitmap = if let Some(ref bitmaps) = query_bitmaps {
                            if let Ok(idx_usize) = idx.try_into() {
                                bitmaps[idx_query].contains(idx_usize)
                            } else {
                                false
                            }
                        } else {
                            true
                        };

                        if allowed_by_bitmap {
                            let distance = distance_comparer
                                .evaluate_similarity(&**aligned_data, aligned_query);
                            neighbor_queue.insert(Neighbor { id: idx, distance });
                        }
                    }
                },
            );

        num_base_points += points;
    }

    let mut aligned_data = AlignedBoxWithSlice::new(query_aligned_dimmensions, 32)?;

    if let Some(insert_iter) = insert_iter {
        for (insert_idx, (data_vector, _associated_data)) in insert_iter.enumerate() {
            aligned_data[..data_vector.len()].copy_from_slice(&data_vector);
            // For each node in the raw data, calculate the distance to each query vector and store it in the priority queue for that query.  This will find the closest N neighbors for each query.
            for (idx_query, (aligned_query, ref mut neighbor_queue)) in
                queries_and_neighbor_queue.iter_mut().enumerate()
            {
                let idx = (num_base_points + insert_idx) as u32;

                let allowed_by_bitmap = if let Some(ref bitmaps) = query_bitmaps {
                    if let Ok(idx_usize) = idx.try_into() {
                        bitmaps[idx_query].contains(idx_usize)
                    } else {
                        false
                    }
                } else {
                    true
                };

                if allowed_by_bitmap {
                    let distance =
                        distance_comparer.evaluate_similarity(&*aligned_data, aligned_query);
                    neighbor_queue.insert(Neighbor { id: idx, distance })
                }
            }
        }
    }

    Ok((neighbor_queues, id_to_associated_data))
}

#[allow(clippy::too_many_arguments)]
pub fn compute_multivec_ground_truth_from_data<T, VectorReader>(
    distance_function: Metric,
    aggregation_method: MultivecAggregationMethod,
    base_vectors: Vec<Matrix<T>>,
    queries: Vec<Matrix<T>>,
    query_dim: usize,
    recall_at: u32,
    query_bitmaps: Option<Vec<BitSet>>,
) -> CMDResult<Vec<NeighborPriorityQueue<u32>>>
where
    T: VectorRepr,
    VectorReader: StorageReadProvider,
{
    let query_num = queries.len();

    let mut neighbor_queues: Vec<NeighborPriorityQueue<u32>> = Vec::with_capacity(query_num);
    //
    for _ in 0..query_num {
        neighbor_queues.push(NeighborPriorityQueue::new(recall_at as usize));
    }
    let mut query_multivecs_and_neighbor_queue: Vec<_> =
        queries.iter().zip(neighbor_queues.iter_mut()).collect();

    let distance_comparer = T::distance(distance_function, Some(query_dim));

    let pool = create_thread_pool(0)?;

    // for each query multivec, compute chamfer distance in parallel

    query_multivecs_and_neighbor_queue
        .par_iter_mut()
        .enumerate()
        .for_each_in_pool(&pool, |(query_idx, (query_multivec, neighbor_queue))| {
            for (idx_base, base_multivec) in base_vectors.iter().enumerate() {
                // check if calculation is allowed by bitmap if present
                let allowed_by_bitmap = if let Some(ref bitmaps) = query_bitmaps {
                    bitmaps[query_idx].contains(idx_base)
                } else {
                    true
                };

                if allowed_by_bitmap {
                    // compute distance between query_multivec and base_multivec
                    let distance = match aggregation_method {
                        MultivecAggregationMethod::AveragePairwise => {
                            let mut total_distance = 0.0;
                            for query_vec in query_multivec.row_iter() {
                                for base_vec in base_multivec.row_iter() {
                                    let dist =
                                        distance_comparer.evaluate_similarity(query_vec, base_vec);
                                    total_distance += dist;
                                }
                            }
                            total_distance / (query_multivec.nrows() * base_multivec.nrows()) as f32
                        }
                        MultivecAggregationMethod::MinPairwise => {
                            let mut min_distance = f32::MAX;
                            for query_vec in query_multivec.row_iter() {
                                for base_vec in base_multivec.row_iter() {
                                    let dist =
                                        distance_comparer.evaluate_similarity(query_vec, base_vec);
                                    min_distance = min_distance.min(dist);
                                }
                            }
                            min_distance
                        }
                        MultivecAggregationMethod::AvgofMins => {
                            let mut distance = 0_f32;
                            for query_vec in query_multivec.row_iter() {
                                let mut local_min = f32::MAX;
                                for base_vec in base_multivec.row_iter() {
                                    let dist =
                                        distance_comparer.evaluate_similarity(query_vec, base_vec);
                                    local_min = local_min.min(dist);
                                }
                                distance += local_min;
                            }
                            distance / query_multivec.nrows() as f32
                        }
                    };
                    // insert into neighbor queue
                    let idx = idx_base as u32;
                    neighbor_queue.insert(Neighbor { id: idx, distance });
                }
            }
        });

    Ok(neighbor_queues)
}

pub fn compute_range_search_ground_truth_from_data<Data, VectorReader, VectorIteratorType>(
    distance_function: Metric,
    dataset_iter: VectorDataIterator<VectorReader, Data>,
    queries: Vec<&[Data::VectorDataType]>,
    query_aligned_dimmensions: usize,
    range_threshold: f32,
) -> CMDResult<Vec<Vec<Neighbor<u32>>>>
where
    Data: GraphDataType,
    VectorReader: StorageReadProvider,
{
    let query_num = queries.len();
    let mut neighbor_queues: Vec<Vec<Neighbor<u32>>> = Vec::with_capacity(query_num);
    for _ in 0..query_num {
        neighbor_queues.push(Vec::new());
    }

    let mut queries_and_neighbor_queue: Vec<_> =
        queries.iter().zip(neighbor_queues.iter_mut()).collect();

    let distance_comparer =
        Data::VectorDataType::distance(distance_function, Some(query_aligned_dimmensions));

    let mut aligned_data = AlignedBoxWithSlice::new(query_aligned_dimmensions, 32)?;
    let mut aligned_query = AlignedBoxWithSlice::new(query_aligned_dimmensions, 32)?;

    for (idx, (data_vector, _associated_data)) in dataset_iter.enumerate() {
        aligned_data[..data_vector.len()].copy_from_slice(&data_vector);
        for (query, ref mut neighbor_queue) in queries_and_neighbor_queue.iter_mut() {
            aligned_query[..query.len()].copy_from_slice(query);
            let distance = distance_comparer.evaluate_similarity(&*aligned_data, &aligned_query);
            if distance <= range_threshold {
                neighbor_queue.push(Neighbor {
                    id: idx as u32,
                    distance,
                })
            }
        }
    }

    Ok(neighbor_queues)
}

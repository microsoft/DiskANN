/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::utils::compute_bitmap::compute_query_bitmaps;
use bit_set::BitSet;
use diskann_label_filter::{read_and_parse_queries, read_baselabels};

use std::{io::Write, mem::size_of, str::FromStr};

use bytemuck::cast_slice;
use diskann::{
    neighbor::{Neighbor, NeighborPriorityQueue},
    utils::VectorRepr,
};
use diskann_disk::data_model::GraphDataType;
use diskann_providers::storage::{StorageReadProvider, StorageWriteProvider};
use diskann_providers::utils::{
    create_thread_pool, file_util, ParallelIteratorInPool, VectorDataIterator,
};
use diskann_utils::{
    io::{read_bin, Metadata},
    views::Matrix,
};
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

    // Read and parse queries
    let parsed_queries = read_and_parse_queries(query_label_filename)?;

    // Compute the query bitmaps
    let query_bitmaps = compute_query_bitmaps(base_labels, parsed_queries);

    match query_bitmaps {
        Ok(bitmaps) => Ok(bitmaps),
        Err(e) => Err(CMDToolError {
            details: format!("Error computing query bitmaps: {}", e),
        }),
    }
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
    let dataset_iterator = VectorDataIterator::<
        StorageProvider,
        Data::VectorDataType,
        Data::AssociatedDataType,
    >::new(base_file, associated_data_file.clone(), storage_provider)?;

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
            let i = VectorDataIterator::<
                StorageProvider,
                Data::VectorDataType,
                Data::AssociatedDataType,
            >::new(insert_file, Option::None, storage_provider)?;
            Some(i)
        }
        None => None,
    };

    // Load the query file
    let query_data =
        read_bin::<Data::VectorDataType>(&mut storage_provider.open_reader(query_file)?)?;
    let query_num = query_data.nrows();

    let mut query_bitmaps: Option<Vec<BitSet>> = None;
    if let (Some(base_file_labels), Some(query_file_labels)) = (base_file_labels, query_file_labels)
    {
        query_bitmaps = Some(read_labels_and_compute_bitmap(
            base_file_labels,
            query_file_labels,
        )?);
    }

    // Load the vector filters
    let vector_filters = match vector_filters_file {
        Some(vector_filters_file) => {
            let filters =
                search_index_utils::load_vector_filters(storage_provider, vector_filters_file)?;

            assert_eq!(
                filters.len(),
                query_num,
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
            let mut bitmaps = vec![BitSet::new(); query_num];
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

    let ground_truth_result = compute_ground_truth_from_data::<Data, StorageProvider>(
        distance_function,
        dataset_iterator,
        &query_data,
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

    let ground_truth = compute_multivec_ground_truth_from_data::<Data::VectorDataType>(
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
    Metadata::new(number_of_queries, total_number_of_neighbors)?.write(&mut file)?;

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

    Metadata::new(number_of_queries, number_of_neighbors)?.write(&mut file)?;

    let mut gt_ids: Vec<u32> = Vec::with_capacity(number_of_neighbors * number_of_queries);
    let mut gt_distances: Vec<f32> = Vec::with_capacity(number_of_neighbors * number_of_queries);

    // In the file, we write the neighbor IDs array first, then write the distances array.
    for mut query_neighbors in ground_truth {
        while let Some(closest_node) = query_neighbors.closest_notvisited() {
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
/// * `dataset_iter` - The iterator over the dataset vectors and associated data.
/// * `queries` - Query vectors as a row-major `Matrix` of shape `num_queries × query_dim`.
///   `query_dim` is inferred from `queries.ncols()`.
/// * `recall_at` - The number of neighbors to compute for each query.
/// * `insert_iter` - Optional iterator containing more dataset vectors. This may be useful if you are testing recall for an index that has points dynamically inserted into it.
/// * `skip_base` - Optional number of base points to skip. This is useful if you want to compute the ground truth for a set where the first skip_base points are deleted from the index.
/// * `query_bitmaps` - Optional per-query bitmaps restricting which base point ids contribute to that query's neighbors.
#[allow(clippy::too_many_arguments)]
pub fn compute_ground_truth_from_data<Data, VectorReader>(
    distance_function: Metric,
    dataset_iter: VectorDataIterator<VectorReader, Data::VectorDataType, Data::AssociatedDataType>,
    queries: &Matrix<Data::VectorDataType>,
    recall_at: u32,
    insert_iter: Option<
        VectorDataIterator<VectorReader, Data::VectorDataType, Data::AssociatedDataType>,
    >,
    skip_base: Option<usize>,
    query_bitmaps: Option<Vec<BitSet>>,
) -> CMDResult<(Npq, Vec<Data::AssociatedDataType>)>
where
    Data: GraphDataType,
    VectorReader: StorageReadProvider,
{
    let query_num = queries.nrows();
    let query_dim = queries.ncols();

    let mut neighbor_queues: Vec<NeighborPriorityQueue<u32>> = (0..query_num)
        .map(|_| NeighborPriorityQueue::new(recall_at as usize))
        .collect();
    let mut queries_and_neighbor_queue: Vec<_> =
        queries.row_iter().zip(neighbor_queues.iter_mut()).collect();

    let distance_comparer = Data::VectorDataType::distance(distance_function, Some(query_dim));

    let batch_size = 10_000;
    let mut data_batch: Vec<Box<[Data::VectorDataType]>> = Vec::with_capacity(batch_size);

    let pool = create_thread_pool(0)?;

    let mut num_base_points: usize = 0;
    let mut id_to_associated_data = Vec::<Data::AssociatedDataType>::new();
    let skip_base = skip_base.unwrap_or(0);
    // Loop over all the raw data
    for chunk in dataset_iter.skip(skip_base).chunks(batch_size).into_iter() {
        data_batch.clear();
        for (data_vector, associated_data) in chunk {
            data_batch.push(data_vector);
            id_to_associated_data.push(associated_data);
        }
        let points = data_batch.len();

        if points == 0 {
            continue;
        }

        // For each node in the raw data, calculate the distance to each query vector and store it in the priority queue for that query.  This will find the closest N neighbors for each query.
        queries_and_neighbor_queue
            .par_iter_mut()
            .enumerate()
            .for_each_in_pool(
                pool.as_ref(),
                |(idx_query, (query, ref mut neighbor_queue))| {
                    for (idx_in_batch, data) in data_batch.iter().enumerate() {
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
                            let distance = distance_comparer.evaluate_similarity(data, query);
                            neighbor_queue.insert(Neighbor { id: idx, distance });
                        }
                    }
                },
            );

        num_base_points += points;
    }

    if let Some(insert_iter) = insert_iter {
        for (insert_idx, (data_vector, _associated_data)) in insert_iter.enumerate() {
            // For each node in the raw data, calculate the distance to each query vector and store it in the priority queue for that query.  This will find the closest N neighbors for each query.
            for (idx_query, (query, ref mut neighbor_queue)) in
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
                    let distance = distance_comparer.evaluate_similarity(&data_vector, query);
                    neighbor_queue.insert(Neighbor { id: idx, distance })
                }
            }
        }
    }

    Ok((neighbor_queues, id_to_associated_data))
}

#[allow(clippy::too_many_arguments)]
pub fn compute_multivec_ground_truth_from_data<T>(
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
        .for_each_in_pool(
            pool.as_ref(),
            |(query_idx, (query_multivec, neighbor_queue))| {
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
                                        let dist = distance_comparer
                                            .evaluate_similarity(query_vec, base_vec);
                                        total_distance += dist;
                                    }
                                }
                                total_distance
                                    / (query_multivec.nrows() * base_multivec.nrows()) as f32
                            }
                            MultivecAggregationMethod::MinPairwise => {
                                let mut min_distance = f32::MAX;
                                for query_vec in query_multivec.row_iter() {
                                    for base_vec in base_multivec.row_iter() {
                                        let dist = distance_comparer
                                            .evaluate_similarity(query_vec, base_vec);
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
                                        let dist = distance_comparer
                                            .evaluate_similarity(query_vec, base_vec);
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
            },
        );

    Ok(neighbor_queues)
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskann_disk::data_model::AdHoc;
    use diskann_providers::storage::VirtualStorageProvider;
    use std::io::Read;

    type GraphDataF32 = AdHoc<f32>;

    /// Write a `.bin` vector file: 8-byte header (npts, dim) then row-major f32 data.
    fn write_vectors(
        provider: &impl StorageWriteProvider,
        path: &str,
        dim: usize,
        rows: &[Vec<f32>],
    ) {
        let mut w = provider.create_for_write(path).unwrap();
        Metadata::new(rows.len(), dim)
            .unwrap()
            .write(&mut w)
            .unwrap();
        for row in rows {
            w.write_all(cast_slice::<f32, u8>(row)).unwrap();
        }
        w.flush().unwrap();
    }

    /// Write a range-truthset (vector filters) file: header (npts, total_ids),
    /// then `npts` i32 counts, then concatenated u32 ids.
    fn write_filters(provider: &impl StorageWriteProvider, path: &str, rows: &[Vec<u32>]) {
        let total: usize = rows.iter().map(|r| r.len()).sum();
        let mut w = provider.create_for_write(path).unwrap();
        Metadata::new(rows.len(), total)
            .unwrap()
            .write(&mut w)
            .unwrap();
        for row in rows {
            w.write_all(&(row.len() as i32).to_le_bytes()).unwrap();
        }
        for row in rows {
            w.write_all(cast_slice::<u32, u8>(row)).unwrap();
        }
        w.flush().unwrap();
    }

    /// Read back a standard ground-truth file: returns (npts, dim, ids, distances).
    fn read_ground_truth(
        provider: &impl StorageReadProvider,
        path: &str,
    ) -> (usize, usize, Vec<u32>, Vec<f32>) {
        let mut f = provider.open_reader(path).unwrap();
        let (npts, dim) = Metadata::read(&mut f).unwrap().into_dims();
        let mut id_bytes = vec![0u8; npts * dim * size_of::<u32>()];
        f.read_exact(&mut id_bytes).unwrap();
        let mut dist_bytes = vec![0u8; npts * dim * size_of::<f32>()];
        f.read_exact(&mut dist_bytes).unwrap();
        (
            npts,
            dim,
            cast_slice::<u8, u32>(&id_bytes).to_vec(),
            cast_slice::<u8, f32>(&dist_bytes).to_vec(),
        )
    }

    #[test]
    fn test_compute_ground_truth_basic() {
        let provider = VirtualStorageProvider::new_memory();
        // 4 base points on a line; single query at origin.
        write_vectors(
            &provider,
            "/base.bin",
            2,
            &[
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![2.0, 0.0],
                vec![10.0, 0.0],
            ],
        );
        write_vectors(&provider, "/query.bin", 2, &[vec![0.0, 0.0]]);

        compute_ground_truth_from_datafiles::<GraphDataF32, _>(
            &provider,
            Metric::L2,
            "/base.bin",
            "/query.bin",
            "/gt.bin",
            None,
            2,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let (npts, dim, ids, dists) = read_ground_truth(&provider, "/gt.bin");
        assert_eq!(npts, 1);
        assert_eq!(dim, 2);
        // Closest two points to origin are ids 0 and 1.
        assert_eq!(ids, vec![0, 1]);
        assert_eq!(dists[0], 0.0);
        assert_eq!(dists[1], 1.0);
    }

    #[test]
    fn test_compute_ground_truth_with_skip_base() {
        let provider = VirtualStorageProvider::new_memory();
        write_vectors(
            &provider,
            "/base.bin",
            2,
            &[vec![0.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]],
        );
        write_vectors(&provider, "/query.bin", 2, &[vec![0.0, 0.0]]);

        // Skip the first base point; nearest remaining is id 1.
        compute_ground_truth_from_datafiles::<GraphDataF32, _>(
            &provider,
            Metric::L2,
            "/base.bin",
            "/query.bin",
            "/gt.bin",
            None,
            1,
            None,
            Some(1),
            None,
            None,
            None,
        )
        .unwrap();

        let (_, _, ids, _) = read_ground_truth(&provider, "/gt.bin");
        // After skipping the first base point, remaining points are re-indexed
        // from 0, so the nearest (original [1,0]) is reported as id 0.
        assert_eq!(ids, vec![0]);
    }

    #[test]
    fn test_compute_ground_truth_with_insert_file() {
        let provider = VirtualStorageProvider::new_memory();
        write_vectors(&provider, "/base.bin", 2, &[vec![5.0, 0.0]]);
        // The inserted vector is the true nearest neighbor; it gets id 1.
        write_vectors(&provider, "/insert.bin", 2, &[vec![0.0, 0.0]]);
        write_vectors(&provider, "/query.bin", 2, &[vec![0.0, 0.0]]);

        compute_ground_truth_from_datafiles::<GraphDataF32, _>(
            &provider,
            Metric::L2,
            "/base.bin",
            "/query.bin",
            "/gt.bin",
            None,
            1,
            Some("/insert.bin"),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let (_, _, ids, dists) = read_ground_truth(&provider, "/gt.bin");
        assert_eq!(ids, vec![1]);
        assert_eq!(dists[0], 0.0);
    }

    #[test]
    fn test_compute_ground_truth_with_vector_filters() {
        let provider = VirtualStorageProvider::new_memory();
        write_vectors(
            &provider,
            "/base.bin",
            2,
            &[
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![2.0, 0.0],
                vec![3.0, 0.0],
            ],
        );
        write_vectors(&provider, "/query.bin", 2, &[vec![0.0, 0.0]]);
        // Restrict the single query to base ids {2, 3}.
        write_filters(&provider, "/filters.bin", &[vec![2, 3]]);

        compute_ground_truth_from_datafiles::<GraphDataF32, _>(
            &provider,
            Metric::L2,
            "/base.bin",
            "/query.bin",
            "/gt.bin",
            Some("/filters.bin"),
            2,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Filtered output uses the range-search format: header then per-query
        // counts then ids. Nearest allowed point to origin is id 2.
        let mut f = provider.open_reader("/gt.bin").unwrap();
        let (num_queries, _) = Metadata::read(&mut f).unwrap().into_dims();
        assert_eq!(num_queries, 1);
        let mut count_bytes = vec![0u8; num_queries * size_of::<u32>()];
        f.read_exact(&mut count_bytes).unwrap();
        let counts = cast_slice::<u8, u32>(&count_bytes);
        assert_eq!(counts, &[2]);
        let mut id_bytes = vec![0u8; 2 * size_of::<u32>()];
        f.read_exact(&mut id_bytes).unwrap();
        let ids = cast_slice::<u8, u32>(&id_bytes);
        assert_eq!(ids[0], 2);
    }

    #[test]
    fn test_error_only_base_labels_provided() {
        let provider = VirtualStorageProvider::new_memory();
        write_vectors(&provider, "/base.bin", 2, &[vec![0.0, 0.0]]);
        write_vectors(&provider, "/query.bin", 2, &[vec![0.0, 0.0]]);

        let err = compute_ground_truth_from_datafiles::<GraphDataF32, _>(
            &provider,
            Metric::L2,
            "/base.bin",
            "/query.bin",
            "/gt.bin",
            None,
            1,
            None,
            None,
            None,
            Some("/base_labels.txt"),
            None,
        )
        .unwrap_err();
        assert!(err
            .details
            .contains("must be provided or both must be not provided"));
    }

    #[test]
    fn test_error_base_labels_and_vector_filters() {
        let provider = VirtualStorageProvider::new_memory();
        write_vectors(&provider, "/base.bin", 2, &[vec![0.0, 0.0]]);
        write_vectors(&provider, "/query.bin", 2, &[vec![0.0, 0.0]]);

        let err = compute_ground_truth_from_datafiles::<GraphDataF32, _>(
            &provider,
            Metric::L2,
            "/base.bin",
            "/query.bin",
            "/gt.bin",
            Some("/filters.bin"),
            1,
            None,
            None,
            None,
            Some("/base_labels.txt"),
            Some("/query_labels.txt"),
        )
        .unwrap_err();
        assert!(err
            .details
            .contains("base_file_labels and vector_filters_file cannot be provided"));
    }

    /// Write a multivec `.bin` file: header (num_points, dim, total_results),
    /// then `num_points` u32 per-point vector counts, then concatenated
    /// row-major f32 data (each point is `count * dim` values).
    fn write_multivec(
        provider: &impl StorageWriteProvider,
        path: &str,
        dim: usize,
        points: &[Vec<Vec<f32>>],
    ) {
        let total: usize = points.iter().map(|p| p.len()).sum();
        let mut w = provider.create_for_write(path).unwrap();
        w.write_all(&(points.len() as u32).to_le_bytes()).unwrap();
        w.write_all(&(dim as u32).to_le_bytes()).unwrap();
        w.write_all(&(total as u32).to_le_bytes()).unwrap();
        for p in points {
            w.write_all(&(p.len() as u32).to_le_bytes()).unwrap();
        }
        for p in points {
            for row in p {
                w.write_all(cast_slice::<f32, u8>(row)).unwrap();
            }
        }
        w.flush().unwrap();
    }

    #[test]
    fn test_compute_multivec_ground_truth_basic() {
        let provider = VirtualStorageProvider::new_memory();
        // Three base "points", each a single vector on a line.
        write_multivec(
            &provider,
            "/mbase.bin",
            2,
            &[
                vec![vec![0.0, 0.0]],
                vec![vec![1.0, 0.0]],
                vec![vec![5.0, 0.0]],
            ],
        );
        // One query point with a single vector at the origin.
        write_multivec(&provider, "/mquery.bin", 2, &[vec![vec![0.0, 0.0]]]);

        compute_multivec_ground_truth_from_datafiles::<GraphDataF32, _>(
            &provider,
            Metric::L2,
            MultivecAggregationMethod::MinPairwise,
            "/mbase.bin",
            "/mquery.bin",
            "/mgt.bin",
            2,
            None,
            None,
        )
        .unwrap();

        let (npts, dim, ids, _) = read_ground_truth(&provider, "/mgt.bin");
        assert_eq!(npts, 1);
        assert_eq!(dim, 2);
        // Closest two base points to origin are ids 0 and 1.
        assert_eq!(ids, vec![0, 1]);
    }

    #[test]
    fn test_compute_multivec_aggregation_methods() {
        for method in [
            MultivecAggregationMethod::AveragePairwise,
            MultivecAggregationMethod::MinPairwise,
            MultivecAggregationMethod::AvgofMins,
        ] {
            let provider = VirtualStorageProvider::new_memory();
            // Base points are multi-vector sets.
            write_multivec(
                &provider,
                "/mbase.bin",
                2,
                &[
                    vec![vec![0.0, 0.0], vec![0.5, 0.0]],
                    vec![vec![9.0, 0.0], vec![10.0, 0.0]],
                ],
            );
            write_multivec(&provider, "/mquery.bin", 2, &[vec![vec![0.0, 0.0]]]);

            compute_multivec_ground_truth_from_datafiles::<GraphDataF32, _>(
                &provider,
                Metric::L2,
                method.clone(),
                "/mbase.bin",
                "/mquery.bin",
                "/mgt.bin",
                1,
                None,
                None,
            )
            .unwrap();

            let (_, _, ids, _) = read_ground_truth(&provider, "/mgt.bin");
            // The nearest base point to the origin is always id 0.
            assert_eq!(ids, vec![0], "method {:?} picked wrong neighbor", method);
        }
    }

    #[test]
    fn test_compute_multivec_error_only_one_label_file() {
        let provider = VirtualStorageProvider::new_memory();
        write_multivec(&provider, "/mbase.bin", 2, &[vec![vec![0.0, 0.0]]]);
        write_multivec(&provider, "/mquery.bin", 2, &[vec![vec![0.0, 0.0]]]);

        let err = compute_multivec_ground_truth_from_datafiles::<GraphDataF32, _>(
            &provider,
            Metric::L2,
            MultivecAggregationMethod::MinPairwise,
            "/mbase.bin",
            "/mquery.bin",
            "/mgt.bin",
            1,
            Some("/base_labels.txt"),
            None,
        )
        .unwrap_err();
        assert!(err
            .details
            .contains("must be provided or both must be not provided"));
    }

    #[test]
    fn test_aggregation_method_from_str() {
        assert!(matches!(
            "average_pairwise".parse::<MultivecAggregationMethod>(),
            Ok(MultivecAggregationMethod::AveragePairwise)
        ));
        assert!(matches!(
            "MIN_PAIRWISE".parse::<MultivecAggregationMethod>(),
            Ok(MultivecAggregationMethod::MinPairwise)
        ));
        assert!(matches!(
            "avg_of_mins".parse::<MultivecAggregationMethod>(),
            Ok(MultivecAggregationMethod::AvgofMins)
        ));

        let err = "nope".parse::<MultivecAggregationMethod>().unwrap_err();
        assert_eq!(
            format!("{}", err),
            "Invalid format for Aggregation Method: nope"
        );
    }

    #[test]
    fn test_compute_multivec_ground_truth_with_labels() {
        let provider = VirtualStorageProvider::new_memory();
        // 3 base points, 2 query points (single vector each).
        write_multivec(
            &provider,
            "/mbase.bin",
            2,
            &[
                vec![vec![0.0, 0.0]],
                vec![vec![1.0, 0.0]],
                vec![vec![2.0, 0.0]],
            ],
        );
        write_multivec(
            &provider,
            "/mquery.bin",
            2,
            &[vec![vec![0.0, 0.0]], vec![vec![2.0, 0.0]]],
        );

        // Base/query label files live on the real filesystem (read via std::fs).
        let dir = tempfile::TempDir::new().unwrap();
        let base_labels = dir.path().join("base.jsonl");
        let query_labels = dir.path().join("query.jsonl");
        {
            let mut f = std::fs::File::create(&base_labels).unwrap();
            writeln!(f, r#"{{"doc_id": 0, "g": "a"}}"#).unwrap();
            writeln!(f, r#"{{"doc_id": 1, "g": "b"}}"#).unwrap();
            writeln!(f, r#"{{"doc_id": 2, "g": "a"}}"#).unwrap();
        }
        {
            let mut f = std::fs::File::create(&query_labels).unwrap();
            writeln!(f, r#"{{"query_id": 0, "filter": {{"g": {{"$eq": "a"}}}}}}"#).unwrap();
            writeln!(f, r#"{{"query_id": 1, "filter": {{"g": {{"$eq": "a"}}}}}}"#).unwrap();
        }

        compute_multivec_ground_truth_from_datafiles::<GraphDataF32, _>(
            &provider,
            Metric::L2,
            MultivecAggregationMethod::MinPairwise,
            "/mbase.bin",
            "/mquery.bin",
            "/mgt_range.bin",
            2,
            Some(base_labels.to_str().unwrap()),
            Some(query_labels.to_str().unwrap()),
        )
        .unwrap();

        // Range-search GT format: header (num_queries, total_neighbors),
        // then num_queries u32 queue sizes, then total_neighbors u32 ids.
        let mut f = provider.open_reader("/mgt_range.bin").unwrap();
        let (num_queries, total_neighbors) = Metadata::read(&mut f).unwrap().into_dims();
        assert_eq!(num_queries, 2);

        let mut size_bytes = vec![0u8; num_queries * size_of::<u32>()];
        f.read_exact(&mut size_bytes).unwrap();
        let sizes = cast_slice::<u8, u32>(&size_bytes).to_vec();
        assert_eq!(sizes.iter().sum::<u32>() as usize, total_neighbors);

        let mut id_bytes = vec![0u8; total_neighbors * size_of::<u32>()];
        f.read_exact(&mut id_bytes).unwrap();
        let ids = cast_slice::<u8, u32>(&id_bytes).to_vec();
        // Only base docs with g=="a" (ids 0 and 2) may appear in the results.
        assert!(ids.iter().all(|&id| id == 0 || id == 2), "ids: {:?}", ids);
    }
}

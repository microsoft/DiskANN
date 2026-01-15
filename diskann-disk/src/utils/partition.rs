/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use diskann::{error::IntoANNResult, utils::VectorRepr, ANNError, ANNResult};
use diskann_providers::storage::{StorageReadProvider, StorageWriteProvider};
use diskann_providers::{
    forward_threadpool,
    utils::{
        compute_closest_centers, gen_random_slice, k_meanspp_selecting_pivots, run_lloyds,
        AsThreadPool, RayonThreadPool, READ_WRITE_BLOCK_SIZE,
    },
};
use rand::Rng;
use tracing::info;

use crate::{
    disk_index_build_parameter::BYTES_IN_GB,
    storage::{CachedReader, CachedWriter, DiskIndexWriter},
};

/// Block size for reading/processing large files and matrices in blocks
const BLOCK_SIZE_LARGE_FILE: u32 = 10_000;

#[allow(clippy::too_many_arguments)]
pub fn partition_with_ram_budget<T, StorageProvider, Pool, F>(
    dataset_file: &str,
    dim: usize,
    sampling_rate: f64,
    ram_budget_in_bytes: f64,
    k_base: usize,
    merged_index_prefix: &str,
    storage_provider: &StorageProvider,
    rng: &mut impl Rng,
    pool: Pool,
    ram_estimator: F,
) -> ANNResult<usize>
where
    T: VectorRepr,
    StorageProvider: StorageReadProvider + StorageWriteProvider,
    Pool: AsThreadPool,
    F: Fn(u64, u64) -> f64,
{
    forward_threadpool!(pool = pool);
    // Find partition size and get pivot data
    let (num_parts, pivot_data, train_dim) = find_partition_size::<T, StorageProvider, F>(
        dataset_file,
        sampling_rate,
        ram_budget_in_bytes,
        k_base,
        storage_provider,
        rng,
        pool,
        &ram_estimator,
    )?;

    info!("Saving shard data into clusters, with only ids");

    shard_data_into_clusters_only_ids::<T, StorageProvider>(
        dataset_file,
        &pivot_data,
        num_parts,
        dim,
        train_dim,
        k_base,
        merged_index_prefix,
        storage_provider,
        pool,
    )?;

    Ok(num_parts)
}

#[allow(clippy::too_many_arguments)]
fn find_partition_size<T, StorageProvider, F>(
    dataset_file: &str,
    sampling_rate: f64,
    ram_budget_in_bytes: f64,
    k_base: usize,
    storage_provider: &StorageProvider,
    rng: &mut impl Rng,
    pool: &RayonThreadPool,
    ram_estimator: &F,
) -> ANNResult<(usize, Vec<f32>, usize)>
where
    T: VectorRepr,
    StorageProvider: StorageReadProvider + StorageWriteProvider,
    F: Fn(u64, u64) -> f64,
{
    const MAX_K_MEANS_REPS: usize = 10;

    let (train_data_float, num_train, train_dim) =
        gen_random_slice::<T, StorageProvider>(dataset_file, sampling_rate, storage_provider, rng)?;
    info!("Loaded {} points for train, dim: {}", num_train, train_dim);

    let (test_data_float, num_test, test_dim) =
        gen_random_slice::<T, StorageProvider>(dataset_file, sampling_rate, storage_provider, rng)?;
    info!("Loaded {} points for test, dim: {}", num_test, test_dim);

    // Calculate total points accounting for sampling rate
    let total_points = (num_train as f64 / sampling_rate) as u64;
    // Get initial partition count estimate
    let initial_num_parts = estimate_initial_partition_count::<F>(
        total_points,
        train_dim as u64,
        k_base,
        ram_budget_in_bytes,
        ram_estimator,
    );

    let mut num_parts = initial_num_parts;
    let mut fit_in_ram = false;
    let mut pivot_data = Vec::new();
    // Iteratively find the right number of parts, kmeans_partitioning on training data
    while !fit_in_ram {
        fit_in_ram = true;

        let mut max_ram_usage_in_bytes = 0.0;

        pivot_data = vec![0.0; num_parts * train_dim];

        // Process Global k-means for kmeans_partitioning Step
        info!("Processing global k-means (kmeans_partitioning Step)");
        k_meanspp_selecting_pivots(
            &train_data_float,
            num_train,
            train_dim,
            &mut pivot_data,
            num_parts,
            rng,
            &mut (false),
            pool,
        )?;

        run_lloyds(
            &train_data_float,
            num_train,
            train_dim,
            &mut pivot_data,
            num_parts,
            MAX_K_MEANS_REPS,
            &mut (false),
            pool,
        )?;

        // now pivots are ready. need to stream base points and assign them to closest clusters.

        let mut cluster_sizes = Vec::new();
        estimate_cluster_sizes(
            &test_data_float,
            num_test,
            &pivot_data,
            num_parts,
            test_dim,
            k_base,
            &mut cluster_sizes,
            pool,
        )?;

        let mut partition_stats = Vec::with_capacity(num_parts);
        for p in &cluster_sizes {
            // to account for the fact that p is the size of the shard over the testing sample.
            let p = (*p as f64 / sampling_rate) as u64;
            let cur_shard_ram_estimate_in_bytes = ram_estimator(p, train_dim as u64);
            partition_stats.push((p, cur_shard_ram_estimate_in_bytes));

            if cur_shard_ram_estimate_in_bytes > max_ram_usage_in_bytes {
                max_ram_usage_in_bytes = cur_shard_ram_estimate_in_bytes;
            }
        }

        info!(
            "Partition RAM estimates (GB): {}",
            partition_stats
                .iter()
                .map(|(size, ram)| format!("#{}: {:.2}", size, ram / BYTES_IN_GB))
                .collect::<Vec<_>>()
                .join(", ")
        );

        info!(
            "With {} parts, max estimated RAM usage: {:.2} GB, budget given is {:.2} GB",
            num_parts,
            max_ram_usage_in_bytes / BYTES_IN_GB,
            ram_budget_in_bytes / BYTES_IN_GB
        );
        if max_ram_usage_in_bytes > ram_budget_in_bytes {
            fit_in_ram = false;
            num_parts += 2;
        } else {
            info!(
                "Found optimal partition count: [parts={}, initial={}, max_ram={:.2}GB, budget={:.2}GB]",
                num_parts,
                initial_num_parts,
                max_ram_usage_in_bytes / BYTES_IN_GB,
                ram_budget_in_bytes / BYTES_IN_GB
            );
        }
    }

    Ok((num_parts, pivot_data, train_dim))
}

/// Initial estimation of partition count based on dataset characteristics and RAM budget
fn estimate_initial_partition_count<F>(
    total_points: u64,
    dimension: u64,
    k_base: usize,
    ram_budget_in_bytes: f64,
    ram_estimator: &F,
) -> usize
where
    F: Fn(u64, u64) -> f64,
{
    // Calculate total RAM needed without partitioning
    let total_ram_estimate = ram_estimator(total_points * k_base as u64, dimension);

    let mut partition_count = (total_ram_estimate / ram_budget_in_bytes).ceil() as usize;

    // Ensure minimum of 3 partitions and odd number for balanced splitting
    partition_count = std::cmp::max(3, partition_count);
    if partition_count.is_multiple_of(2) {
        partition_count += 1;
    }

    info!(
        "Estimated initial partition count: {} (total points: {}, dimension: {}, k_base: {}, total_ram_estimate: {:.2} GB, ram_budget: {:.2} GB)",
        partition_count,
        total_points,
        dimension,
        k_base,
        total_ram_estimate / BYTES_IN_GB,
        ram_budget_in_bytes / BYTES_IN_GB
    );

    partition_count
}

#[allow(clippy::too_many_arguments)]
fn shard_data_into_clusters_only_ids<T, StorageProvider>(
    dataset_file: &str,
    pivot_data: &[f32],
    num_parts: usize,
    dim: usize,
    full_dim: usize,
    k_base: usize,
    merged_index_prefix: &str,
    storage_provider: &StorageProvider,
    pool: &RayonThreadPool,
) -> ANNResult<()>
where
    T: VectorRepr,
    StorageProvider: StorageReadProvider + StorageWriteProvider,
{
    let mut dataset_reader = CachedReader::<StorageProvider>::new(
        dataset_file,
        READ_WRITE_BLOCK_SIZE,
        storage_provider,
    )?;
    let num_points = dataset_reader.read_u32()?;
    let base_dim = dataset_reader.read_u32()?;
    if base_dim != dim as u32 {
        return Err(ANNError::log_index_error(
            "dimensions dont match for train set and base set",
        ));
    }

    let mut shard_counts = vec![0; num_parts];
    let shard_idmaps_names = (0..num_parts)
        .map(|shard| {
            DiskIndexWriter::get_merged_index_subshard_id_map_file(merged_index_prefix, shard)
        })
        .collect::<Vec<String>>();

    // 8KB cache for small ID map files - matches default BufWriter size
    const WRITE_ID_CACHE_SIZE: u64 = 8 * 1024;
    let mut shard_idmap_cached_writers = Vec::new();
    for name in &shard_idmaps_names {
        let writer = storage_provider.create_for_write(name)?;
        let cached_writer =
            CachedWriter::<StorageProvider>::new(name, WRITE_ID_CACHE_SIZE, writer)?;
        shard_idmap_cached_writers.push(cached_writer);
    }

    let dummy_size: u32 = 0;
    let const_one: u32 = 1;
    for writer in shard_idmap_cached_writers.iter_mut() {
        writer.write(&dummy_size.to_le_bytes())?;
        writer.write(&const_one.to_le_bytes())?;
    }

    let block_size = if num_points <= BLOCK_SIZE_LARGE_FILE {
        num_points
    } else {
        BLOCK_SIZE_LARGE_FILE
    };

    let num_blocks = num_points.div_ceil(block_size);

    let mut block_closest_centers = vec![0u32; block_size as usize * k_base];
    let mut block_data_t: Vec<u8> = vec![0; block_size as usize * dim * std::mem::size_of::<T>()];
    let mut block_data_float: Vec<f32> = vec![0.0; full_dim * block_size as usize];

    for block in 0..num_blocks {
        let start_id = (block * block_size) as usize;
        let end_id = std::cmp::min((block + 1) * block_size, num_points) as usize;
        let cur_blk_size = end_id - start_id;

        dataset_reader.read(&mut block_data_t[..cur_blk_size * dim * std::mem::size_of::<T>()])?;

        // convert data from type T to f32
        let cur_vector_t: &[T] =
            bytemuck::cast_slice(&block_data_t[..cur_blk_size * dim * std::mem::size_of::<T>()]);

        for (v, dst) in cur_vector_t
            .chunks_exact(dim)
            .zip(block_data_float.chunks_exact_mut(full_dim))
        {
            T::as_f32_into(v, dst).into_ann_result()?;
        }

        compute_closest_centers(
            &block_data_float[..full_dim * cur_blk_size],
            cur_blk_size,
            full_dim,
            pivot_data,
            num_parts,
            k_base,
            &mut block_closest_centers,
            None,
            None,
            pool,
        )?;

        for p in 0..cur_blk_size {
            for p1 in 0..k_base {
                let shard_id = block_closest_centers[p * k_base + p1] as usize;
                let original_point_map_id = (start_id + p) as u32;
                shard_idmap_cached_writers[shard_id].write(&original_point_map_id.to_le_bytes())?;
                shard_counts[shard_id] += 1;
            }
        }
    }

    let mut total_count = 0;

    for i in 0..num_parts {
        let cur_shard_count = shard_counts[i] as u32;
        info!(" shard_{} with npts : {} ", i, cur_shard_count);
        total_count += cur_shard_count;
        shard_idmap_cached_writers[i].reset()?;
        shard_idmap_cached_writers[i].write(&cur_shard_count.to_le_bytes())?;
        shard_idmap_cached_writers[i].flush()?;
    }

    info!(
        "Partitioned {} with replication factor {} to get {} points across {} shards",
        num_points, k_base, total_count, num_parts
    );

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn estimate_cluster_sizes(
    data_float: &[f32],
    num_pts: usize,
    pivot_data: &[f32],
    num_centers: usize,
    dim: usize,
    k_base: usize,
    cluster_sizes: &mut Vec<u32>,
    pool: &RayonThreadPool,
) -> ANNResult<()> {
    cluster_sizes.clear();
    let mut shard_counts = vec![0; num_centers];

    let block_size = if num_pts <= BLOCK_SIZE_LARGE_FILE as usize {
        num_pts
    } else {
        BLOCK_SIZE_LARGE_FILE as usize
    };

    let mut block_closest_centers = vec![0; block_size * k_base];

    let num_blocks = num_pts.div_ceil(block_size);

    for block in 0..num_blocks {
        let start_id = block * block_size;
        let end_id = std::cmp::min((block + 1) * block_size, num_pts);
        let cur_blk_size = end_id - start_id;

        let block_data_float = &data_float[start_id * dim..(start_id + cur_blk_size) * dim];

        compute_closest_centers(
            block_data_float,
            cur_blk_size,
            dim,
            pivot_data,
            num_centers,
            k_base,
            &mut block_closest_centers,
            None,
            None,
            pool,
        )?;

        for p in 0..cur_blk_size {
            for p1 in 0..k_base {
                let shard_id = block_closest_centers[p * k_base + p1] as usize;
                shard_counts[shard_id] += 1;
            }
        }
    }

    (0..num_centers).for_each(|i| {
        let cur_shard_count = shard_counts[i] as u32;
        cluster_sizes.push(cur_shard_count);
    });
    info!("Estimated cluster sizes: {:?}", cluster_sizes);
    Ok(())
}

#[cfg(test)]
mod partition_test {
    use std::io::Read;

    use diskann_providers::storage::VirtualStorageProvider;
    use diskann_providers::utils::create_thread_pool_for_test;
    use diskann_utils::test_data_root;
    use vfs::{MemoryFS, OverlayFS, PhysicalFS};

    use super::*;

    #[test]
    fn test_estimate_cluster_sizes() {
        let data_float = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let num_pts = 3;
        let pivot_data = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let num_centers = 3;
        let dim = 2;
        let k_base = 2;
        let mut cluster_sizes = vec![];
        let pool = create_thread_pool_for_test();

        estimate_cluster_sizes(
            &data_float,
            num_pts,
            pivot_data,
            num_centers,
            dim,
            k_base,
            &mut cluster_sizes,
            &pool,
        )
        .unwrap();

        assert_eq!(cluster_sizes.len(), num_centers);
        assert_eq!(cluster_sizes, &[2, 3, 1]);
    }

    #[test]
    fn test_shard_data_into_clusters_only_ids() {
        // create a temporary file for the dataset
        let dataset_path = "/dataset_file";
        // write some dummy data to the dataset file
        let mut data_float = Vec::new();
        let num_points: u32 = 100;
        let dim: usize = 10;

        let base_filesystem = PhysicalFS::new(test_data_root());
        let memory_filesystem = MemoryFS::new();
        let vfs = OverlayFS::new(&[memory_filesystem.into(), base_filesystem.into()]);
        let storage_provider = VirtualStorageProvider::new(vfs);
        {
            let writer = storage_provider.create_for_write(dataset_path).unwrap();
            let mut dataset_writer = CachedWriter::<VirtualStorageProvider<MemoryFS>>::new(
                dataset_path,
                READ_WRITE_BLOCK_SIZE,
                writer,
            )
            .unwrap();
            dataset_writer.write(&num_points.to_le_bytes()).unwrap();
            dataset_writer.write(&dim.to_le_bytes()).unwrap();
            for i in 0..num_points {
                for j in 0..dim {
                    let val = (i * dim as u32 + j as u32) as f32;
                    data_float.push(val);
                    dataset_writer.write(&val.to_le_bytes()).unwrap();
                }
            }
        }

        // create some dummy pivot data
        let k_base: usize = 2;
        let num_parts = 3;

        // generate pivot data
        let pivot_data: [f32; 30] = [
            820.0, 821.0, 822.0, 823.0, 824.0, 825.0, 826.0, 827.0, 828.0, 829.0, 155.0, 156.0,
            157.0, 158.0, 159.0, 160.0, 161.0, 162.0, 163.0, 164.0, 480.0, 481.0, 482.0, 483.0,
            484.0, 485.0, 486.0, 487.0, 488.0, 489.0,
        ];

        // create a temporary prefix for the merged index prefix
        let merged_index_prefix = "/merged_index";
        let pool = create_thread_pool_for_test();
        // call the function being tested
        shard_data_into_clusters_only_ids::<f32, VirtualStorageProvider<OverlayFS>>(
            dataset_path,
            &pivot_data,
            num_parts,
            dim,
            dim,
            k_base,
            merged_index_prefix,
            &storage_provider,
            &pool,
        )
        .unwrap();

        // check that the output is as expected
        let expected_prefix = "/partition/id_maps/merged_index_expected";
        for shard in 0..num_parts {
            let path1 =
                DiskIndexWriter::get_merged_index_subshard_id_map_file(merged_index_prefix, shard);
            let path2 =
                DiskIndexWriter::get_merged_index_subshard_id_map_file(expected_prefix, shard);
            let file1 =
                load_file_to_vec::<VirtualStorageProvider<OverlayFS>>(&path1, &storage_provider);
            let file2 =
                load_file_to_vec::<VirtualStorageProvider<OverlayFS>>(&path2, &storage_provider);

            assert_eq!(file1.len(), file2.len());
            assert_eq!(file1[..], file2[..]);

            // clean up the temporary files and directory
            storage_provider.delete(&path1).unwrap();
        }

        storage_provider.delete(dataset_path).unwrap();
    }

    fn load_file_to_vec<StorageProvider>(
        file_path: &str,
        storage_provider: &StorageProvider,
    ) -> Vec<u8>
    where
        StorageProvider: StorageReadProvider,
    {
        let mut file = storage_provider.open_reader(file_path).unwrap();
        let mut buffer = vec![];
        file.read_to_end(&mut buffer).unwrap();
        buffer
    }

    #[test]
    fn test_partition_with_ram_budget() -> ANNResult<()> {
        let base_filesystem = PhysicalFS::new(test_data_root());
        let memory_filesystem = MemoryFS::new();
        let vfs = OverlayFS::new(&[memory_filesystem.into(), base_filesystem.into()]);
        let storage_provider = VirtualStorageProvider::new(vfs);
        let dataset_file = "/sift/siftsmall_learn.bin";
        let mut file = storage_provider.open_reader(dataset_file).unwrap();
        let mut data = vec![];
        file.read_to_end(&mut data).unwrap();

        let sampling_rate = 1.0;
        let ram_budget_in_bytes = 15_000_000.0;
        let max_degree = 64;
        let k_base = 2;
        let merged_index_prefix = "/test_merged_index_prefix";
        let pool = create_thread_pool_for_test();

        let num_parts = partition_with_ram_budget::<f32, _, _, _>(
            dataset_file,
            128, //sift is 128 dimensions
            sampling_rate,
            ram_budget_in_bytes,
            k_base,
            merged_index_prefix,
            &storage_provider,
            &mut diskann_providers::utils::create_rnd_in_tests(),
            &pool,
            |num_points, dim| {
                // Simple RAM estimation for test - capture datasize and graph_degree from context
                use diskann_providers::model::GRAPH_SLACK_FACTOR;

                let datasize = std::mem::size_of::<f32>() as u64;
                let graph_degree = max_degree as u64;
                let dataset_size = (num_points * dim.next_multiple_of(8u64) * datasize) as f64;
                let graph_size = (num_points * graph_degree * 4) as f64 * GRAPH_SLACK_FACTOR;
                1.1 * (dataset_size + graph_size)
            },
        )?;

        assert!(num_parts >= 3);

        for i in 0..num_parts {
            let idmap_filename =
                DiskIndexWriter::get_merged_index_subshard_id_map_file(merged_index_prefix, i);
            storage_provider.delete(&idmap_filename)?;
        }

        Ok(())
    }
}

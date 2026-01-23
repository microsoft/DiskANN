/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{utils::VectorRepr, ANNError};
use diskann_providers::storage::StorageReadProvider;
use diskann_providers::{model::graph::traits::GraphDataType, utils::file_util::load_bin};
use rand::Rng;

use crate::utils::{CMDResult, CMDToolError};

fn squared_distance<Data: GraphDataType>(
    v1: &[Data::VectorDataType],
    v2: &[Data::VectorDataType],
) -> CMDResult<f32> {
    let v1 = &*<Data::VectorDataType>::as_f32(v1)
        .map_err(|x| CMDToolError::from(Into::<ANNError>::into(x)))?;
    let v2 = &*<Data::VectorDataType>::as_f32(v2)
        .map_err(|x| CMDToolError::from(Into::<ANNError>::into(x)))?;
    Ok(v1
        .iter()
        .zip(v2)
        .map(|(a, b)| {
            let diff = *a - *b;
            diff * diff
        })
        .sum())
}

fn average_squared_distance<Data: GraphDataType>(
    query: &[Data::VectorDataType],
    base: &[Vec<Data::VectorDataType>],
    num_random_samples: usize,
) -> CMDResult<f32> {
    let mut rng = rand::rng();
    let n = base.len();
    let mut sum_dist = 0.0;
    for _ in 0..num_random_samples {
        let r = rng.random_range(0..n);
        sum_dist += squared_distance::<Data>(query, &base[r])?;
    }
    Ok(sum_dist / num_random_samples as f32)
}

pub fn compute_relative_contrast<Data: GraphDataType, StorageProvider: StorageReadProvider>(
    storage_provider: &StorageProvider,
    base_file: &str,
    query_file: &str,
    gt_file: &str,
    recall_at: usize,
    num_random_samples: usize,
) -> CMDResult<f32> {
    // Load base, query, and ground truth data
    let (base_flat, nb, dim) = load_bin::<Data::VectorDataType, _>(storage_provider, base_file, 0)?;
    let (query_flat, nq, _) = load_bin::<Data::VectorDataType, _>(storage_provider, query_file, 0)?;
    let (gt_flat, _, ngt) = load_bin::<u32, _>(storage_provider, gt_file, 0)?;

    tracing::info!(
        "Loaded base: {} points, query: {} points, dimension: {}, ground truth neighbors: {}",
        nb,
        nq,
        dim,
        ngt
    );

    // Reshape flat vectors into 2D vectors
    let base: Vec<Vec<Data::VectorDataType>> = base_flat.chunks(dim).map(|x| x.to_vec()).collect();
    let query: Vec<Vec<Data::VectorDataType>> =
        query_flat.chunks(dim).map(|x| x.to_vec()).collect();
    let gt: Vec<Vec<u32>> = gt_flat.chunks(ngt).map(|x| x.to_vec()).collect();

    let mut mean_rc = 0.0;

    for (i, q) in query.iter().enumerate() {
        // Compute numerator: average squared distance to random samples
        let numerator = average_squared_distance::<Data>(q, &base, num_random_samples)?;

        // Compute denominator: average squared distance to ground truth neighbors
        let mut denominator = 0.0;
        for &idx in gt[i].iter().take(recall_at) {
            denominator += squared_distance::<Data>(q, &base[idx as usize])?;
        }
        denominator /= recall_at as f32;

        // Compute relative contrast for this query
        let rc = numerator / denominator;
        mean_rc += rc / nq as f32;
    }

    if (1.5..2.0).contains(&mean_rc) {
        tracing::info!(
            "Mean relative contrast = {}. The dataset is suitable for ANN.",
            mean_rc
        );
    } else {
        tracing::info!(
            "Mean relative contrast = {}. The dataset is not suitable for ANN.",
            mean_rc
        );
    }
    Ok(mean_rc)
}

#[cfg(test)]
mod relative_contrast_tests {
    use diskann_providers::storage::{StorageWriteProvider, VirtualStorageProvider};
    use diskann_providers::utils::write_metadata;
    use diskann_vector::distance::Metric;
    use half::f16;
    use rand::Rng;
    use vfs::MemoryFS;

    use super::*;
    use crate::utils::{ground_truth::compute_ground_truth_from_datafiles, GraphDataHalfVector};

    /// Test for compute_relative_contrast function with random data
    /// Generate 1000 random vectors and 10 queries, both random samples/recall_at = 5
    /// Expectation: relative contrast < 1.2
    #[test]
    fn test_compute_relative_contrast_with_random_data() {
        let filesystem = MemoryFS::new();
        let storage_provider = VirtualStorageProvider::new(filesystem);

        // Generate 1000 random vectors of fp16 data type with 384 dimensions
        let num_vectors = 1000;
        let dim = 384;
        let mut rng = rand::rng();
        let base: Vec<f16> = (0..num_vectors * dim)
            .map(|_| f16::from_f32(rng.random_range(0.0..1.0)))
            .collect();

        // Generate 10 query vectors of fp16 data type with 384 dimensions
        let num_queries = 10;
        let query: Vec<f16> = (0..num_queries * dim)
            .map(|_| f16::from_f32(rng.random_range(0.0..1.0)))
            .collect();

        // Write base vectors to base.bin
        let base_file_path = "/base.bin";
        {
            let mut base_writer = storage_provider.create_for_write(base_file_path).unwrap();
            write_metadata(&mut base_writer, num_vectors, dim).unwrap();
            for value in &base {
                base_writer.write_all(&value.to_le_bytes()).unwrap();
            }
        }

        // Write query vectors to query.bin
        let query_file_path = "/query.bin";
        {
            let mut query_writer = storage_provider.create_for_write(query_file_path).unwrap();
            write_metadata(&mut query_writer, num_queries, dim).unwrap();
            for value in &query {
                query_writer.write_all(&value.to_le_bytes()).unwrap();
            }
        }

        // Generate ground truth file using compute_ground_truth_from_datafiles
        let gt_file_path = "/ground_truth.bin";
        let recall_at = 5;
        compute_ground_truth_from_datafiles::<GraphDataHalfVector, _>(
            &storage_provider,
            Metric::L2,
            base_file_path,
            query_file_path,
            gt_file_path,
            None,
            recall_at as u32,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Run compute_relative_contrast with the generated files
        let num_random_samples = 5;
        let mean_rc = compute_relative_contrast::<GraphDataHalfVector, _>(
            &storage_provider,
            base_file_path,
            query_file_path,
            gt_file_path,
            recall_at,
            num_random_samples,
        )
        .unwrap();
        println!("Mean relative contrast: {}", mean_rc);

        // expect the relative contrast to be close to 1.0
        assert!(
            mean_rc > 1.0 && mean_rc < 1.2,
            "Mean relative contrast is out of range: {}",
            mean_rc
        );
    }

    /// Test for compute_relative_contrast function with siftsmall data
    /// 256 vectors and 10 queries, both random samples/recall_at = 3
    /// Expectation: relative contrast > 1.5
    #[test]
    fn test_compute_relative_contrast_with_sift_files() {
        let storage_provider =
            VirtualStorageProvider::new_overlay(diskann_utils::test_data_root().join("sift"));
        let base_file_path = "/siftsmall_learn_256pts.fbin";

        assert!(
            storage_provider.exists(base_file_path),
            "Base file does not exist"
        );

        let num_queries = 10;
        let dim = 128;
        let mut rng = rand::rng();
        let query: Vec<f16> = (0..num_queries * dim)
            .map(|_| f16::from_f32(rng.random_range(0.0..1.0)))
            .collect();

        let query_file_path = "/query.bin";

        {
            let mut query_writer = storage_provider
                .create_for_write(query_file_path)
                .expect("Failed to create query file in memory");
            write_metadata(&mut query_writer, num_queries, dim).expect("Failed to write metadata");
            for value in &query {
                query_writer
                    .write_all(&value.to_le_bytes())
                    .expect("Failed to write query vector");
            }
        }

        // Generate ground truth file using compute_ground_truth_from_datafiles
        let gt_file_path = "/ground_truth.bin";
        let recall_at = 3;
        compute_ground_truth_from_datafiles::<GraphDataHalfVector, _>(
            &storage_provider,
            Metric::L2,
            base_file_path,
            query_file_path,
            gt_file_path,
            None,
            recall_at as u32,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Run compute_relative_contrast with the generated files
        let num_random_samples = 3;
        let mean_rc = compute_relative_contrast::<GraphDataHalfVector, _>(
            &storage_provider,
            base_file_path,
            query_file_path,
            gt_file_path,
            recall_at,
            num_random_samples,
        )
        .unwrap();
        println!("Mean relative contrast: {}", mean_rc);

        storage_provider
            .delete(query_file_path)
            .expect("Failed to delete query file in disk");
        storage_provider
            .delete(gt_file_path)
            .expect("Failed to delete ground truth file in disk");

        // expect the relative contrast to be greater than 1.5
        assert!(
            mean_rc > 1.5 && mean_rc < 2.0,
            "Mean relative contrast is out of range: {}",
            mean_rc
        );
    }
}

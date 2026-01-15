/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! Aligned allocator

use std::{cmp::Ordering, collections::BinaryHeap};

use diskann::{ANNError, ANNResult};
use diskann_linalg::{self, Transpose};
use num_traits::FromPrimitive;
use rand::Rng;
use rand_distr::StandardNormal;
use rayon::prelude::*;
use tracing::info;

use super::{AsThreadPool, ParallelIteratorInPool, RayonThreadPool};
use crate::forward_threadpool;

// This is the chunk size applied when computing the closest centers in a block.
// The chunk size is the number of points to process in a single iteration to reduce memory usage of
// distance_matrix.
// 1200 is a number we tested to be optimal for the number of points in a chunk that
// * Large enough to take advantage of BLAS operations
// * Small enough to avoid hefty memory allocations
// the experiment performance of pq construction:
// | Chunk Size   |1087932vector384dim  |8717820vector384dim  |
// |--------------|---------------------|---------------------|
// | 1            | 169.082s/3.181GB    | 202.175s/2.892GB    |
// | 2            | 156.726s/1.704GB    | 189.860s/1.444GB    |
// | 8            | 151.853s/0.996GB    | 185.035s/0.838GB    |
// | 16           | 145.725s/0.995GB    | 185.756s/0.831GB    |
// | 32           | 122.644s/0.996GB    | 141.831s/0.841GB    |
// | 64           | 83.927s/0.994GB     | 97.761s/0.840GB     |
// | 128          | 64.404s/0.994GB     | 79s/0.841GB         |
// | 256          | 59.662s/0.995GB     | 73s/0.841GB         |
// | 512          | 58.331s/0.996GB     | 70.552s/0.819GB     |
// we are currently using the chunk size of 256 (about 1200 (256000 train data / 256))
// test results are collected from i9-10900X 3.7GHz 10 cores 20 threads 32GB RAM
// key parameters -M 1000 -R 59 -L 64 -T 8 -B 0.195 --dist_fn CosineNormalized
const POINTS_PER_CHUNK: usize = 1200;

struct PivotContainer {
    piv_id: usize,
    piv_dist: f32,
}

/// The PartialOrd trait is for types that can be partially ordered, i.e., where some pairs of values are incomparable (like with floating-point numbers when one of them is NaN).
/// So the correct way to implement PartialOrd for a type that has Ord is to use self.cmp(other) directly.
impl PartialOrd for PivotContainer {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// The Ord trait is for types that have a total order, where every pair of values is comparable.
impl Ord for PivotContainer {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Treat NaN as less than all other values.
        // piv_dist should never be NaN.
        other
            .piv_dist
            .partial_cmp(&self.piv_dist)
            .unwrap_or(Ordering::Less)
    }
}

impl PartialEq for PivotContainer {
    fn eq(&self, other: &Self) -> bool {
        self.piv_dist == other.piv_dist
    }
}

impl Eq for PivotContainer {}

/// Compute L2-squared norms of data stored in row-major num_points * dim,
/// need to be pre-allocated
pub fn compute_vecs_l2sq<Pool: AsThreadPool>(
    vecs_l2sq: &mut [f32],
    data: &[f32],
    num_points: usize,
    dim: usize,
    pool: Pool,
) -> ANNResult<()> {
    if data.len() != num_points * dim {
        return Err(ANNError::log_pq_error(format_args!(
            "data.len() {} should be num_points {} * dim {}",
            data.len(),
            num_points,
            dim
        )));
    }

    if dim < 5 {
        for (i, vec_l2sq) in vecs_l2sq.iter_mut().enumerate() {
            *vec_l2sq = compute_vec_l2sq(data, i, dim);
        }
    } else {
        forward_threadpool!(pool = pool);
        vecs_l2sq
            .par_iter_mut()
            .enumerate()
            .for_each_in_pool(pool, |(i, vec_l2sq)| {
                *vec_l2sq = compute_vec_l2sq(data, i, dim);
            });
    }

    Ok(())
}

/// The implementation of computing L2-squared norm of a vector
pub fn compute_vec_l2sq(data: &[f32], index: usize, dim: usize) -> f32 {
    let start = index * dim;
    let slice = unsafe { std::slice::from_raw_parts(data.as_ptr().add(start), dim) };
    let mut sum_squared = 0.0;
    for &value in slice {
        sum_squared += value * value;
    }

    sum_squared
}

/// Calculate k closest centers to data of num_points * dim (row-major)
/// Centers is num_centers * dim (row-major)
/// data_l2sq has pre-computed squared norms of data
/// centers_l2sq has pre-computed squared norms of centers
/// Pre-allocated center_index will contain id of nearest center
/// Pre-allocated dist_matrix should be num_points * num_centers and contain squared distances
/// Default value of k is 1
/// Ideally used only by compute_closest_centers
#[allow(clippy::too_many_arguments)]
pub fn compute_closest_centers_in_block(
    data: &[f32],
    num_points: usize,
    dim: usize,
    centers: &[f32],
    num_centers: usize,
    docs_l2sq: &[f32],
    centers_l2sq: &[f32],
    center_index: &mut [u32],
    dist_matrix: &mut [f32],
    k: usize,
    pool: &RayonThreadPool,
) -> ANNResult<()> {
    if k > num_centers {
        return Err(ANNError::log_index_error(format_args!(
            "ERROR: k ({}) > num_centers({})",
            k, num_centers
        )));
    }

    let ones_a: Vec<f32> = vec![1.0; num_centers];
    let ones_b: Vec<f32> = vec![1.0; num_points];

    diskann_linalg::sgemm(
        Transpose::None,
        Transpose::Ordinary,
        num_points,
        num_centers,
        1,
        1.0,
        docs_l2sq,
        &ones_a,
        None, // Initialize the destination matrix
        dist_matrix,
    );

    diskann_linalg::sgemm(
        Transpose::None,
        Transpose::Ordinary,
        num_points,
        num_centers,
        1,
        1.0,
        &ones_b,
        centers_l2sq,
        Some(1.0), // Add to the destination matrix
        dist_matrix,
    );

    diskann_linalg::sgemm(
        Transpose::None,
        Transpose::Ordinary,
        num_points,
        num_centers,
        dim,
        -2.0,
        data,
        centers,
        Some(1.0), // Add to the destination matrix.
        dist_matrix,
    );

    if k == 1 {
        center_index
            .par_iter_mut()
            .enumerate()
            .for_each_in_pool(pool, |(i, center_idx)| {
                let mut min = f32::MAX;
                let current = &dist_matrix[i * num_centers..(i + 1) * num_centers];
                let mut min_idx = 0;
                for (j, &distance) in current.iter().enumerate() {
                    if distance < min {
                        min = distance;
                        min_idx = j;
                    }
                }
                *center_idx = min_idx as u32;
            });
    } else {
        center_index
            .par_chunks_mut(k)
            .enumerate()
            .for_each_in_pool(pool, |(i, center_chunk)| {
                let current = &dist_matrix[i * num_centers..(i + 1) * num_centers];
                let mut top_k_queue = BinaryHeap::new();
                for (j, &distance) in current.iter().enumerate() {
                    let this_piv = PivotContainer {
                        piv_id: j,
                        piv_dist: distance,
                    };
                    top_k_queue.push(this_piv);
                }
                for center_idx in center_chunk.iter_mut() {
                    if let Some(this_piv) = top_k_queue.pop() {
                        *center_idx = this_piv.piv_id as u32;
                    } else {
                        break;
                    }
                }
            });
    }

    Ok(())
}

/// Given data in num_points * new_dim row major
/// Pivots stored in full_pivot_data as num_centers * new_dim row major
/// Calculate the k closest pivot for each point and store it in vector
/// closest_centers_ivf (row major, num_points*k) (which needs to be allocated
/// outside) Additionally, if inverted index is not null (and pre-allocated),
/// it will return inverted index for each center, assuming each of the inverted
/// indices is an empty vector. Additionally, if pts_norms_squared is not null,
/// then it will assume that point norms are pre-computed and use those values
#[allow(clippy::too_many_arguments)]
pub fn compute_closest_centers<Pool: AsThreadPool>(
    data: &[f32],
    num_points: usize,
    dim: usize,
    pivot_data: &[f32],
    num_centers: usize,
    k: usize,
    closest_centers_ivf: &mut [u32],
    mut inverted_index: Option<&mut Vec<Vec<usize>>>,
    pts_norms_squared: Option<&[f32]>,
    pool: Pool,
) -> ANNResult<()> {
    if k > num_centers {
        return Err(ANNError::log_index_error(format_args!(
            "ERROR: k ({}) > num_centers({})",
            k, num_centers
        )));
    }

    forward_threadpool!(pool = pool);

    let pts_norms_squared = if let Some(pts_norms) = pts_norms_squared {
        pts_norms.to_vec()
    } else {
        let mut norms_squared = vec![0.0; num_points];
        compute_vecs_l2sq(&mut norms_squared, data, num_points, dim, pool)?;
        norms_squared
    };

    let mut pivs_norms_squared = vec![0.0; num_centers];
    compute_vecs_l2sq(&mut pivs_norms_squared, pivot_data, num_centers, dim, pool)?;

    let mut distance_matrix = vec![0.0; POINTS_PER_CHUNK * num_centers];
    let mut closest_center_indices = vec![0; POINTS_PER_CHUNK * k];
    let pts_norms_squared_chunks = pts_norms_squared.chunks(POINTS_PER_CHUNK);

    for (chunk_index, (data_chunk, pts_norms_squared_chunk)) in data
        .chunks(dim * POINTS_PER_CHUNK)
        .zip(pts_norms_squared_chunks)
        .enumerate()
    {
        // actual chunk size maybe less than the pt_num_per_chunk for the last chunk
        let chunk_size = data_chunk.len() / dim;

        // Potentially shrink scratch data structures.
        let this_distance_matrix = &mut distance_matrix[..num_centers * chunk_size];
        let this_closest_center_indices = &mut closest_center_indices[..k * chunk_size];

        compute_closest_centers_in_block(
            data_chunk,
            chunk_size,
            dim,
            pivot_data,
            num_centers,
            pts_norms_squared_chunk,
            &pivs_norms_squared,
            this_closest_center_indices,
            this_distance_matrix,
            k,
            pool,
        )?;

        let point_start_index = chunk_index * POINTS_PER_CHUNK;

        for point_index in point_start_index..point_start_index + chunk_size {
            for l in 0..k {
                let center_chunk_index = (point_index - point_start_index) * k + l;
                let ivf_index = point_index * k + l;

                let this_center_index = closest_center_indices[center_chunk_index];
                closest_centers_ivf[ivf_index] = this_center_index;

                if let Some(inverted_index) = &mut inverted_index {
                    inverted_index[this_center_index as usize].push(point_index);
                }
            }
        }
    }
    Ok(())
}

/// If to_subtract is true, will subtract nearest center from each row.
/// Else will add.
/// Output will be in data_load itself.
/// Nearest centers need to be provided in closest_centers.
#[allow(clippy::too_many_arguments)]
pub fn process_residuals(
    data_load: &mut [f32],
    num_points: usize,
    dim: usize,
    cur_pivot_data: &[f32],
    num_centers: usize,
    closest_centers: &[u32],
    to_subtract: bool,
    pool: &RayonThreadPool,
) {
    info!(
        "Processing residuals of {} points in {} dimensions using {} centers",
        num_points, dim, num_centers
    );

    data_load
        .par_chunks_mut(dim)
        .enumerate()
        .for_each_in_pool(pool, |(n_iter, chunk)| {
            let cur_pivot_index = closest_centers[n_iter] as usize * dim;
            for d_iter in 0..dim {
                if to_subtract {
                    chunk[d_iter] -= cur_pivot_data[cur_pivot_index + d_iter];
                } else {
                    chunk[d_iter] += cur_pivot_data[cur_pivot_index + d_iter];
                }
            }
        })
}

/// Generate num vectors with given dimension and norm
pub fn generate_vectors_with_norm<VectorDataType>(
    num: usize,
    dimension: usize,
    norm: f32,
    rng: &mut impl Rng,
) -> ANNResult<Vec<Vec<VectorDataType>>>
where
    VectorDataType: FromPrimitive,
{
    let mut result = Vec::<Vec<VectorDataType>>::with_capacity(num);

    for _ in 0..num {
        let vector = generate_vector_with_norm::<VectorDataType>(dimension, norm, rng)?;
        result.push(vector);
    }

    Ok(result)
}

/// Generate a vector with the given norm and dimension using the provided random number generator and normal distribution
fn generate_vector_with_norm<VectorDataType>(
    dim: usize,
    norm: f32,
    rng: &mut impl Rng,
) -> ANNResult<Vec<VectorDataType>>
where
    VectorDataType: FromPrimitive,
{
    let f32_vector: Vec<f32> = (0..dim).map(|_| rng.sample(StandardNormal)).collect();
    let normalization_factor = norm / f32_vector.iter().map(|x| x * x).sum::<f32>().sqrt();

    let mut result_vector = Vec::<VectorDataType>::with_capacity(dim);
    for x in f32_vector {
        let normalized_x = x * normalization_factor;
        match VectorDataType::from_f32(normalized_x) {
            Some(x) => result_vector.push(x),
            None => {
                // For unsigned types, we need to ensure that the value is non-negative
                match VectorDataType::from_f32(normalized_x.abs()) {
                    Some(x) => result_vector.push(x),
                    None => {
                        return Err(ANNError::log_index_error(format_args!(
                            "Failed to convert f32 to {}. Please choose a norm that can fit in {}",
                            std::any::type_name::<VectorDataType>(),
                            std::any::type_name::<VectorDataType>()
                        )));
                    }
                }
            }
        }
    }

    Ok(result_vector)
}

/// Converts usize to u64
#[inline]
pub fn convert_usize_to_u64(value: usize) -> u64 {
    value as u64
}

#[cfg(test)]
mod math_util_test {
    use approx::assert_abs_diff_eq;
    use diskann_vector::Half;

    use super::*;
    use crate::utils::create_thread_pool_for_test;

    #[test]
    fn partial_ord_test() {
        let pviot1 = PivotContainer {
            piv_id: 2,
            piv_dist: f32::NAN,
        };
        let pivot2 = PivotContainer {
            piv_id: 1,
            piv_dist: 1.0,
        };

        assert_eq!(pviot1.partial_cmp(&pivot2), Some(Ordering::Less));
    }

    #[test]
    fn ord_test() {
        let pviot1 = PivotContainer {
            piv_id: 1,
            piv_dist: f32::NAN,
        };
        let pivot2 = PivotContainer {
            piv_id: 2,
            piv_dist: 1.0,
        };

        assert_eq!(pviot1.cmp(&pivot2), Ordering::Less);
    }

    #[test]
    fn compute_vecs_l2sq_small_dim_test() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let num_points = 2;
        let dim = 3;
        let mut vecs_l2sq = vec![0.0; num_points];
        let pool = create_thread_pool_for_test();

        compute_vecs_l2sq(&mut vecs_l2sq, &data, num_points, dim, &pool).unwrap();

        let expected = [14.0, 77.0];

        assert_eq!(vecs_l2sq.len(), num_points);
        assert_abs_diff_eq!(vecs_l2sq[0], expected[0], epsilon = 1e-6);
        assert_abs_diff_eq!(vecs_l2sq[1], expected[1], epsilon = 1e-6);
    }

    #[test]
    fn compute_vecs_l2sq_large_dim_test() {
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let num_points = 2;
        let dim = 8;
        let mut vecs_l2sq = vec![0.0; num_points];
        let pool = create_thread_pool_for_test();
        compute_vecs_l2sq(&mut vecs_l2sq, &data, num_points, dim, &pool).unwrap();

        let expected = [204.0, 1292.0];

        assert_eq!(vecs_l2sq.len(), num_points);
        assert_abs_diff_eq!(vecs_l2sq[0], expected[0], epsilon = 1e-6);
        assert_abs_diff_eq!(vecs_l2sq[1], expected[1], epsilon = 1e-6);
    }

    #[test]
    fn compute_closest_centers_in_block_test() {
        let num_points = 10;
        let dim = 5;
        let num_centers = 3;
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0,
            45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
        ];
        let centers = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 21.0, 22.0, 23.0, 24.0, 25.0, 31.0, 32.0, 33.0, 34.0, 35.0,
        ];
        let mut docs_l2sq = vec![0.0; num_points];
        let pool = create_thread_pool_for_test();
        compute_vecs_l2sq(&mut docs_l2sq, &data, num_points, dim, &pool).unwrap();
        let mut centers_l2sq = vec![0.0; num_centers];
        compute_vecs_l2sq(&mut centers_l2sq, &centers, num_centers, dim, &pool).unwrap();
        let mut center_index = vec![0; num_points];
        let mut dist_matrix = vec![0.0; num_points * num_centers];
        let k = 1;

        compute_closest_centers_in_block(
            &data,
            num_points,
            dim,
            &centers,
            num_centers,
            &docs_l2sq,
            &centers_l2sq,
            &mut center_index,
            &mut dist_matrix,
            k,
            &pool,
        )
        .unwrap();

        assert_eq!(center_index.len(), num_points);
        let expected_center_index = vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 2];
        assert_abs_diff_eq!(*center_index, expected_center_index);

        assert_eq!(dist_matrix.len(), num_points * num_centers);
        let expected_dist_matrix = vec![
            0.0, 2000.0, 4500.0, 125.0, 1125.0, 3125.0, 500.0, 500.0, 2000.0, 1125.0, 125.0,
            1125.0, 2000.0, 0.0, 500.0, 3125.0, 125.0, 125.0, 4500.0, 500.0, 0.0, 6125.0, 1125.0,
            125.0, 8000.0, 2000.0, 500.0, 10125.0, 3125.0, 1125.0,
        ];
        assert_abs_diff_eq!(*dist_matrix, expected_dist_matrix, epsilon = 1e-2);
    }

    #[test]
    fn compute_closest_centers_in_block_test_k_equals_two() {
        let num_points = 2;
        let dim = 5;
        let num_centers = 4;
        let data = vec![41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0];
        let centers = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 21.0, 22.0, 23.0, 24.0, 25.0, 31.0, 32.0, 33.0, 34.0, 35.0,
            46.0, 47.0, 48.0, 49.0, 50.0,
        ];
        let mut docs_l2sq = vec![0.0; num_points];
        let pool = create_thread_pool_for_test();
        compute_vecs_l2sq(&mut docs_l2sq, &data, num_points, dim, &pool).unwrap();
        let mut centers_l2sq = vec![0.0; num_centers];
        compute_vecs_l2sq(&mut centers_l2sq, &centers, num_centers, dim, &pool).unwrap();
        let k = 2;
        let mut center_index = vec![0; num_points * k];
        let mut dist_matrix = vec![0.0; num_points * num_centers];

        compute_closest_centers_in_block(
            &data,
            num_points,
            dim,
            &centers,
            num_centers,
            &docs_l2sq,
            &centers_l2sq,
            &mut center_index,
            &mut dist_matrix,
            k,
            &pool,
        )
        .unwrap();

        assert_eq!(center_index.len(), num_points * k);
        let expected_center_index = vec![3, 2, 3, 2];
        assert_abs_diff_eq!(*center_index, expected_center_index);

        assert_eq!(dist_matrix.len(), num_points * num_centers);
        // obviously, the order of distance [8000.0, 2000.0, 500.0, 125.0], is #3, #2, #1, #0
        // so the top 2 closest centers for the first point are #3, #2
        // obviously, the order of distance [10125.0, 3125.0, 1125.0, 0.0], is #3, #2, #1, #0
        // so the top 2 closest centers for the second point are #3, #2
        let expected_dist_matrix = vec![8000.0, 2000.0, 500.0, 125.0, 10125.0, 3125.0, 1125.0, 0.0];
        assert_abs_diff_eq!(*dist_matrix, expected_dist_matrix, epsilon = 1e-2);
    }

    #[test]
    fn test_compute_closest_centers() {
        let num_points = 4;
        let dim = 3;
        let num_centers = 2;
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let pivot_data = vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0];
        let k = 1;

        let mut closest_centers_ivf = vec![0u32; num_points * k];
        let mut inverted_index: Vec<Vec<usize>> = vec![vec![], vec![]];
        let pool = create_thread_pool_for_test();
        compute_closest_centers(
            &data,
            num_points,
            dim,
            &pivot_data,
            num_centers,
            k,
            &mut closest_centers_ivf,
            Some(&mut inverted_index),
            None,
            &pool,
        )
        .unwrap();

        assert_eq!(closest_centers_ivf, vec![0, 0, 1, 1]);

        for vec in inverted_index.iter_mut() {
            vec.sort_unstable();
        }
        assert_eq!(inverted_index, vec![vec![0, 1], vec![2, 3]]);
    }

    #[test]
    fn process_residuals_test() {
        let mut data_load = vec![1.0, 2.0, 3.0, 4.0];
        let num_points = 2;
        let dim = 2;
        let cur_pivot_data = vec![0.5, 1.5, 2.5, 3.5];
        let num_centers = 2;
        let closest_centers = vec![0, 1];
        let to_subtract = true;
        let pool = create_thread_pool_for_test();

        process_residuals(
            &mut data_load,
            num_points,
            dim,
            &cur_pivot_data,
            num_centers,
            &closest_centers,
            to_subtract,
            &pool,
        );

        assert_eq!(data_load, vec![0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_generate_vectors_with_norm_u8() {
        let num = 1;
        let dimension = 2;
        let norm = 5.0;

        let result = generate_vectors_with_norm::<u8>(
            num,
            dimension,
            norm,
            &mut crate::utils::create_rnd_in_tests(),
        )
        .unwrap();

        assert_eq!(result.len(), num);
        for vec in result.iter() {
            assert_eq!(vec.len(), dimension);

            let actual_norm_sq = vec
                .iter()
                .fold(0.0, |acc, &x| acc + (x as f32) * (x as f32));
            let actual_norm = actual_norm_sq.sqrt();

            assert_abs_diff_eq!(actual_norm, norm, epsilon = 1.0); // There will be a significant loss of precision when converting f32 to u8
        }
    }

    #[test]
    fn test_generate_vectors_with_norm_f32() {
        let num = 3;
        let dimension = 384;
        let norm = 10000.0;

        let result = generate_vectors_with_norm::<f32>(
            num,
            dimension,
            norm,
            &mut crate::utils::create_rnd_in_tests(),
        )
        .unwrap();

        assert_eq!(result.len(), num);
        for vec in result.iter() {
            assert_eq!(vec.len(), dimension);

            let actual_norm = vec.iter().fold(0.0, |acc, &x| acc + x * x).sqrt();

            assert_abs_diff_eq!(actual_norm, norm, epsilon = 0.01);
        }
    }

    #[test]
    fn test_generate_vectors_with_norm_f16() {
        let num = 1;
        let dimension = 128;
        let norm = 500.0;

        let result = generate_vectors_with_norm::<Half>(
            num,
            dimension,
            norm,
            &mut crate::utils::create_rnd_in_tests(),
        )
        .unwrap();

        assert_eq!(result.len(), num);
        for vec in result.iter() {
            assert_eq!(vec.len(), dimension);

            let actual_norm = vec
                .iter()
                .fold(0.0, |acc, &x| acc + x.to_f32() * x.to_f32())
                .sqrt();

            assert_abs_diff_eq!(actual_norm, norm, epsilon = 0.1);
        }
    }

    #[test]
    fn test_generate_vectors_with_norm_i8() {
        let num = 1;
        let dimension = 8;
        let norm = 127.0;

        let result = generate_vectors_with_norm::<i8>(
            num,
            dimension,
            norm,
            &mut crate::utils::create_rnd_in_tests(),
        )
        .unwrap();

        assert_eq!(result.len(), num);
        for vec in result.iter() {
            assert_eq!(vec.len(), dimension);

            let actual_norm_sq = vec
                .iter()
                .fold(0.0, |acc, &x| acc + (x as f32) * (x as f32));
            let actual_norm = actual_norm_sq.sqrt();

            assert_abs_diff_eq!(actual_norm, norm, epsilon = 8.0); // There will be a significant loss of precision when converting f32 to i8
        }
    }

    #[test]
    fn test_convert_usize_to_u64() {
        assert_eq!(convert_usize_to_u64(0), 0);
        assert_eq!(convert_usize_to_u64(1), 1);
        assert_eq!(convert_usize_to_u64(2), 2);
    }

    #[test]
    fn test_convert_usize_max_to_u64() {
        convert_usize_to_u64(usize::MAX);
    }
}

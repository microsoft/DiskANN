/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! K-means clustering and related utilities

use std::{
    cmp::{min, Ordering},
    collections::BinaryHeap,
};

use diskann::{ANNError, ANNResult};
use diskann_linalg::{self, Transpose};
use diskann_vector::{PureDistanceFunction, distance::SquaredL2};
use hashbrown::HashSet;
use rand::{
    Rng,
    distr::{StandardUniform, Uniform},
    prelude::Distribution,
};
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

/// Run Lloyds one iteration
/// Given data in row-major num_points * dim, and centers in row-major
/// num_centers * dim and squared lengths of ata points, output the closest
/// center to each data point, update centers, and also return inverted index.
/// If closest_centers == NULL, will allocate memory and return.
/// Similarly, if closest_docs == NULL, will allocate memory and return.
#[allow(clippy::too_many_arguments)]
fn lloyds_iter(
    data: &[f32],
    num_points: usize,
    dim: usize,
    centers: &mut [f32],
    num_centers: usize,
    docs_l2sq: &[f32],
    closest_docs: &mut Vec<Vec<usize>>,
    closest_center: &mut [u32],
    pool: &RayonThreadPool,
) -> ANNResult<f32> {
    let compute_residual = true;

    closest_docs.iter_mut().for_each(|doc| doc.clear());

    compute_closest_centers(
        data,
        num_points,
        dim,
        centers,
        num_centers,
        1,
        closest_center,
        Some(closest_docs),
        Some(docs_l2sq),
        pool,
    )?;

    centers.fill(0.0);

    centers
        .par_chunks_mut(dim)
        .enumerate()
        .for_each_in_pool(pool, |(c, center)| {
            let mut cluster_sum = vec![0.0; dim];
            for &doc_index in &closest_docs[c] {
                let current = &data[doc_index * dim..(doc_index + 1) * dim];
                for (j, current_val) in current.iter().enumerate() {
                    cluster_sum[j] += *current_val as f64;
                }
            }
            if !closest_docs[c].is_empty() {
                for (i, sum_val) in cluster_sum.iter().enumerate() {
                    center[i] = (*sum_val / closest_docs[c].len() as f64) as f32;
                }
            }
        });

    let mut residual = 0.0;
    if compute_residual {
        let buf_pad: usize = 32;
        let chunk_size: usize = 2 * 8192;
        let nchunks = usize::div_ceil(num_points, chunk_size);

        let mut residuals: Vec<f32> = vec![0.0; nchunks * buf_pad];

        residuals
            .par_iter_mut()
            .enumerate()
            .for_each_in_pool(pool, |(chunk, res)| {
                for d in (chunk * chunk_size)..min(num_points, (chunk + 1) * chunk_size) {
                    let cc = closest_center[d] as usize;
                    let dist: f32 = SquaredL2::evaluate(
                        &data[d * dim..(d + 1) * dim],
                        &centers[cc * dim..(cc + 1) * dim],
                    );
                    *res += dist;
                }
            });

        for chunk in 0..nchunks {
            residual += residuals[chunk * buf_pad];
        }
    }

    Ok(residual)
}

/// Run Lloyds until max_reps or stopping criterion
/// If you pass NULL for closest_docs and closest_center, it will NOT return
/// the results, else it will assume appropriate allocation as `closest_docs =
/// new vec<usize> [num_centers]`, and `closest_center = new size_t[num_points]`
/// Final centers are output in centers as row-major num_centers * dim.
#[allow(clippy::too_many_arguments)]
pub fn run_lloyds<Pool: AsThreadPool>(
    data: &[f32],
    num_points: usize,
    dim: usize,
    centers: &mut [f32],
    num_centers: usize,
    max_reps: usize,
    cancellation_token: &mut bool,
    pool: Pool,
) -> ANNResult<(Vec<Vec<usize>>, Vec<u32>, f32)> {
    let mut residual = f32::MAX;

    let mut closest_docs = vec![Vec::new(); num_centers];
    let mut closest_center = vec![0; num_points];

    let mut docs_l2sq = vec![0.0; num_points];

    forward_threadpool!(pool = pool);
    compute_vecs_l2sq(&mut docs_l2sq, data, num_points, dim, pool)?;

    let mut old_residual;

    for i in 0..max_reps {
        if *cancellation_token {
            return Err(ANNError::log_pq_error(
                "Error: Cancellation requested by caller.",
            ));
        }

        old_residual = residual;

        residual = lloyds_iter(
            data,
            num_points,
            dim,
            centers,
            num_centers,
            &docs_l2sq,
            &mut closest_docs,
            &mut closest_center,
            pool,
        )?;

        if (i != 0 && (old_residual - residual) / residual < 0.00001) || (residual < f32::EPSILON) {
            break;
        }
    }

    Ok((closest_docs, closest_center, residual))
}

/// Select random num_centers points as pivots
/// `pivot_data` must be initialized with [f32; num_centers * dim]
/// Returns an error if num_points < num_centers
#[cfg(test)]
fn select_random_pivots(
    data: &[f32],
    num_points: usize,
    dim: usize,
    pivot_data: &mut [f32],
    num_centers: usize,
    rng: &mut impl Rng,
) -> ANNResult<()> {
    if num_points < num_centers {
        return Err(ANNError::log_kmeans_error(format!(
            "Number of points {} is less than number of centers {}",
            num_points, num_centers
        )));
    }
    if pivot_data.len() != num_centers * dim {
        return Err(ANNError::log_kmeans_error(format!(
            "Pivot data buffer should be of size num_centers * dim = {} * {} = {}",
            num_centers,
            dim,
            num_centers * dim
        )));
    }

    let mut picked = HashSet::new();
    let distribution = Uniform::try_from(0..num_points).unwrap();

    for _ in 0..num_centers {
        let mut pivot = distribution.sample(rng);
        while picked.contains(&pivot) {
            pivot = distribution.sample(rng);
        }
        picked.insert(pivot);
        let data_offset = pivot * dim;
        let pivot_offset = (picked.len() - 1) * dim;
        pivot_data[pivot_offset..pivot_offset + dim]
            .copy_from_slice(&data[data_offset..data_offset + dim]);
    }
    Ok(())
}

/// Select pivots using k-means++ algorithm.
/// Points that are farther away from the already chosen centroids
/// have a higher probability of being selected as the next centroid.
/// The k-means++ algorithm helps avoid poor initial centroid
/// placement that can result in suboptimal clustering.
///
/// `data` is the input data in row-major format with num_points * dim elements.
/// `num_points` is the number of data points.
/// `dim` is the dimension of the data points.
/// `pivot_data` buffer allocated to [f32; num_centers * dim]
/// `num_centers` is the number of pivots to select.
/// `rng` is the random number generator.
///
///
/// Returns an error if num_points > 8388608 (2^23).
/// If there are are fewer than num_center distinct points, pick all unique points as pivots,
/// and sample data randomly for the remaining pivots.
#[allow(clippy::too_many_arguments)]
pub fn k_meanspp_selecting_pivots<Pool: AsThreadPool>(
    data: &[f32],
    num_points: usize,
    dim: usize,
    pivot_data: &mut [f32],
    num_centers: usize,
    rng: &mut impl Rng,
    cancellation_token: &mut bool,
    pool: Pool,
) -> ANNResult<()> {
    if num_points > (1 << 23) {
        return Err(ANNError::log_kmeans_error(format!(
            "Number of points {} is greater than 8388608, and k-means++ can not process this.
            Try selecting_random_pivots instead.",
            num_points
        )));
    }
    if pivot_data.len() != num_centers * dim {
        return Err(ANNError::log_kmeans_error(format!(
            "Pivot data buffer should be of size num_centers * dim = {} * {} = {}",
            num_centers,
            dim,
            num_centers * dim
        )));
    }

    if *cancellation_token {
        return Err(ANNError::log_pq_error(
            "Error: Cancellation requested by caller.",
        ));
    }

    // 'picked' contains the distinct node ids that have been selected as pivot.
    let mut picked = HashSet::with_capacity(num_centers);

    let real_distribution = StandardUniform;
    let int_distribution = Uniform::new(0, num_points)
        .map_err(|_| ANNError::log_kmeans_error("cannot cluster an empty dataset".into()))?;

    // Randomly select a node as the first pivot.
    let init_id = int_distribution.sample(rng);
    picked.insert(init_id);

    // Copy the data of the first pivot to the result pivot_data.
    let init_data_offset = init_id * dim;
    pivot_data[0..dim].copy_from_slice(&data[init_data_offset..init_data_offset + dim]);

    let mut dist = vec![0.0; num_points];

    forward_threadpool!(pool = pool);
    // Calculate the distance between each node and the first pivot and store the result in dist.
    dist.par_iter_mut()
        .enumerate()
        .for_each_in_pool(pool, |(i, dist_i)| {
            *dist_i = SquaredL2::evaluate(
                &data[i * dim..(i + 1) * dim],
                &data[init_id * dim..(init_id + 1) * dim],
            );
        });

    // Loop starts from 1 since we already picked the first pivot.
    // In which loop iteration, we pick one pivot or return with an error
    // At the end of the loop we should have num_centers pivots.
    for _ in 1..num_centers {
        if *cancellation_token {
            return Err(ANNError::log_pq_error(
                "Error: Cancellation requested by caller.",
            ));
        }

        // Calculate the sum of distances of all the nodes to their nearest selected pivot.
        let sum: f64 = dist
            .iter()
            .map(|&x| { if x == f32::INFINITY { f32::MAX } else { x } } as f64)
            .sum();

        // All unique points are picked as pivots.
        // Exit this k-means++ sampling loop and sample data randomly for the remaining pivots.
        if sum == 0.0 {
            break;
        }

        // Pick a dart_val in [0..sum] range using uniform distribution.
        let sample: f64 = real_distribution.sample(rng);
        let dart_val: f64 = sample * sum;

        // In the below for loop, we will select the next pivot based on where dart_val falls in prefix_sum.
        let mut prefix_sum: f64 = 0.0;
        let mut picked_pivot_id = num_points; // An invalid number we will set later.
        for (i, pivot_dist) in dist.iter().enumerate() {
            // Select i as pivot when dart_val is in [prefix_sum, prefix_sum + min_distance(i, chosen_pivots)).
            // If pivot_dist is >0 but so small that prefix_sum + pivot_dist == prefix_sum, we should still pick it.
            if dart_val >= prefix_sum
                && (dart_val < prefix_sum + *pivot_dist as f64
                    || (dart_val <= prefix_sum && *pivot_dist != 0.0f32))
            {
                if picked.contains(&i) {
                    return Err(ANNError::log_kmeans_error(
                        "A pivot was sampled again, the condition on dart_val range should not have happened".to_string(),
                    ));
                }
                picked.insert(i);
                picked_pivot_id = i;
                break;
            }
            // Increment the prefix_sum
            prefix_sum += *pivot_dist as f64;
        }
        if prefix_sum > sum {
            return Err(ANNError::log_kmeans_error(
                "Prefix sum should not be greater than sum.
            If the for loop above ran to conclusion without break,
            prefix_sum shoule be equal to sum"
                    .to_string(),
            ));
        }
        // We should have picked a pivot in this loop.
        // If not, there is a corner condition we might have missed and we should fix this function.
        if picked_pivot_id == num_points {
            return Err(ANNError::log_kmeans_error(
                "Did not pick a pivot in this loop".to_string(),
            ));
        }

        // Copy the data of the selected pivot to the result pivot_data.
        let pivot_offset = (picked.len() - 1) * dim;
        let data_offset = picked_pivot_id * dim;
        pivot_data[pivot_offset..pivot_offset + dim]
            .copy_from_slice(&data[data_offset..data_offset + dim]);

        // Now, update the distance between each node and the selected pivots and store the result in dist. This needs to be done since we just selected a new pivot.
        dist.par_iter_mut()
            .enumerate()
            .for_each_in_pool(pool, |(i, dist_i)| {
                *dist_i = (*dist_i).min(SquaredL2::evaluate(
                    &data[i * dim..(i + 1) * dim],
                    &data[picked_pivot_id * dim..(picked_pivot_id + 1) * dim],
                ));
            });
    }

    // If we have fewer than num_center distinct points, pick the remaining pivots randomly.
    let mut num_picked = picked.len();
    while num_picked < num_centers {
        let random_id = int_distribution.sample(rng);
        num_picked += 1;
        let pivot_offset = (num_picked - 1) * dim;
        let data_offset = random_id * dim;
        pivot_data[pivot_offset..pivot_offset + dim]
            .copy_from_slice(&data[data_offset..data_offset + dim]);
    }

    Ok(())
}

/// k-means algorithm interface
#[allow(clippy::too_many_arguments)]
pub fn k_means_clustering<Pool: AsThreadPool>(
    data: &[f32],
    num_points: usize,
    dim: usize,
    centers: &mut [f32],
    num_centers: usize,
    max_reps: usize,
    rng: &mut impl Rng,
    cancellation_token: &mut bool,
    pool: Pool,
) -> ANNResult<(Vec<Vec<usize>>, Vec<u32>, f32)> {
    forward_threadpool!(pool = pool);

    k_meanspp_selecting_pivots(
        data,
        num_points,
        dim,
        centers,
        num_centers,
        rng,
        cancellation_token,
        pool,
    )?;
    let (closest_docs, closest_center, residual) = run_lloyds(
        data,
        num_points,
        dim,
        centers,
        num_centers,
        max_reps,
        cancellation_token,
        pool,
    )?;

    Ok((closest_docs, closest_center, residual))
}

#[cfg(test)]
mod kmeans_test {
    use crate::storage::{StorageReadProvider, VirtualStorageProvider};
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use diskann::ANNErrorKind;
    use diskann_utils::test_data_root;
    use rstest::rstest;

    use super::*;
    use crate::utils::create_thread_pool_for_test;

    #[test]
    fn lloyds_iter_test() {
        let dim = 2;
        let num_points = 10;
        let num_centers = 3;

        let data: Vec<f32> = (1..=num_points * dim).map(|x| x as f32).collect();
        let mut centers = [1.0, 2.0, 7.0, 8.0, 19.0, 20.0];

        let mut closest_docs: Vec<Vec<usize>> = vec![vec![]; num_centers];
        let mut closest_center: Vec<u32> = vec![0; num_points];
        let docs_l2sq: Vec<f32> = data
            .chunks(dim)
            .map(|chunk| chunk.iter().map(|val| val.powi(2)).sum())
            .collect();

        let pool = create_thread_pool_for_test();
        let residual = lloyds_iter(
            &data,
            num_points,
            dim,
            &mut centers,
            num_centers,
            &docs_l2sq,
            &mut closest_docs,
            &mut closest_center,
            &pool,
        )
        .unwrap();

        let expected_centers: [f32; 6] = [2.0, 3.0, 9.0, 10.0, 17.0, 18.0];
        let expected_closest_docs: Vec<Vec<usize>> =
            vec![vec![0, 1], vec![2, 3, 4, 5, 6], vec![7, 8, 9]];
        let expected_closest_center: [u32; 10] = [0, 0, 1, 1, 1, 1, 1, 2, 2, 2];
        let expected_residual: f32 = 100.0;

        // sort data for assert
        centers.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for inner_vec in &mut closest_docs {
            inner_vec.sort();
        }
        closest_center.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_eq!(centers, expected_centers);
        assert_eq!(closest_docs, expected_closest_docs);
        assert_eq!(closest_center, expected_closest_center);
        assert_relative_eq!(residual, expected_residual, epsilon = 1.0e-6_f32);
    }

    #[test]
    fn run_lloyds_test() {
        let dim = 2;
        let num_points = 10;
        let num_centers = 3;
        let max_reps = 5;

        let data: Vec<f32> = (1..=num_points * dim).map(|x| x as f32).collect();
        let mut centers = [1.0, 2.0, 7.0, 8.0, 19.0, 20.0];
        let pool = create_thread_pool_for_test();

        let (mut closest_docs, mut closest_center, residual) = run_lloyds(
            &data,
            num_points,
            dim,
            &mut centers,
            num_centers,
            max_reps,
            &mut (false),
            &pool,
        )
        .unwrap();

        let expected_centers: [f32; 6] = [3.0, 4.0, 10.0, 11.0, 17.0, 18.0];
        let expected_closest_docs: Vec<Vec<usize>> =
            vec![vec![0, 1, 2], vec![3, 4, 5, 6], vec![7, 8, 9]];
        let expected_closest_center: [u32; 10] = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2];
        let expected_residual: f32 = 72.0;

        // sort data for assert
        centers.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for inner_vec in &mut closest_docs {
            inner_vec.sort();
        }
        closest_center.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_eq!(centers, expected_centers);
        assert_eq!(closest_docs, expected_closest_docs);
        assert_eq!(closest_center, expected_closest_center);
        assert_relative_eq!(residual, expected_residual, epsilon = 1.0e-6_f32);
    }

    #[test]
    fn run_lloyds_return_err_when_canceled() {
        let dim = 2;
        let num_points = 10;
        let num_centers = 3;
        let max_reps = 5;
        let cancellation_token = &mut true; // Cancellation requested

        let data: Vec<f32> = (1..=num_points * dim).map(|x| x as f32).collect();
        let mut centers = [1.0, 2.0, 7.0, 8.0, 19.0, 20.0];
        let pool = create_thread_pool_for_test();

        let err = run_lloyds(
            &data,
            num_points,
            dim,
            &mut centers,
            num_centers,
            max_reps,
            cancellation_token,
            &pool,
        )
        .unwrap_err();

        assert_eq!(err.kind(), ANNErrorKind::PQError);
        assert!(
            err.to_string()
                .contains("Error: Cancellation requested by caller.")
        );
    }

    #[test]
    fn selecting_random_pivots_test() {
        let dim = 2;
        let num_points = 10;
        let num_centers = 3;

        // Generate some random data points
        let mut rng = crate::utils::create_rnd_in_tests();
        let data: Vec<f32> = (0..num_points * dim).map(|_| rng.random()).collect();

        let mut pivot_data = vec![0.0; num_centers * dim];
        select_random_pivots(
            &data,
            num_points,
            dim,
            &mut pivot_data,
            num_centers,
            &mut crate::utils::create_rnd_in_tests(),
        )
        .unwrap();

        // Verify that each pivot point corresponds to a point in the data
        for i in 0..num_centers {
            let pivot_offset = i * dim;
            let pivot = &pivot_data[pivot_offset..(pivot_offset + dim)];

            // Make sure the pivot is found in the data
            let mut found = false;
            for j in 0..num_points {
                let data_offset = j * dim;
                let point = &data[data_offset..(data_offset + dim)];

                if pivot == point {
                    found = true;
                    break;
                }
            }
            assert!(found, "Pivot not found in data");
        }
    }

    #[test]
    fn selecting_random_pivots_return_err_when_too_less_input_points() {
        let dim = 2;
        let num_points = 10;
        let num_centers = 11; // More centers than points
        let use_correct_buffer_size = true;

        let expected_error_message = "Number of points 10 is less than number of centers 11";

        selecting_random_pivots_test_error_internal(
            dim,
            num_points,
            num_centers,
            use_correct_buffer_size,
            expected_error_message,
        );
    }

    #[test]
    fn selecting_random_pivots_return_err_when_mismatched_buffer_size() {
        let dim = 2;
        let num_points = 10;
        let num_centers = 3;
        let use_correct_buffer_size = false; // Buffer size is 1 less than required

        let expected_error_message =
            "Pivot data buffer should be of size num_centers * dim = 3 * 2 = 6";

        selecting_random_pivots_test_error_internal(
            dim,
            num_points,
            num_centers,
            use_correct_buffer_size,
            expected_error_message,
        );
    }

    fn selecting_random_pivots_test_error_internal(
        dim: usize,
        num_points: usize,
        num_centers: usize,
        use_correct_buffer_size: bool,
        expected_error_message: &str,
    ) {
        // Generate some random data points
        let mut rng = crate::utils::create_rnd_in_tests();
        let data: Vec<f32> = (0..num_points * dim).map(|_| rng.random()).collect();

        let pivot_data_size = if use_correct_buffer_size {
            num_centers * dim
        } else {
            num_centers * dim - 1
        };
        let mut pivot_data = vec![0.0; pivot_data_size];
        let err = select_random_pivots(
            &data,
            num_points,
            dim,
            &mut pivot_data,
            num_centers,
            &mut crate::utils::create_rnd_in_tests(),
        )
        .unwrap_err();

        assert_eq!(err.kind(), ANNErrorKind::KMeansError);
        assert!(err.to_string().contains(expected_error_message));
    }

    #[rstest]
    #[case(2, 10, 3)]
    #[case(2, 10, 10)]
    fn k_meanspp_selects_pivots_in_dataset(
        #[case] dim: usize,
        #[case] num_points: usize,
        #[case] num_centers: usize,
    ) {
        // Generate some random data points
        let mut rng = crate::utils::create_rnd_in_tests();
        let data: Vec<f32> = (0..num_points * dim).map(|_| rng.random()).collect();

        let mut pivot_data = vec![0.0; num_centers * dim];
        let pool = create_thread_pool_for_test();
        k_meanspp_selecting_pivots(
            &data,
            num_points,
            dim,
            &mut pivot_data,
            num_centers,
            &mut crate::utils::create_rnd_in_tests(),
            &mut (false),
            &pool,
        )
        .unwrap();

        // Verify that each pivot point corresponds to a point in the data
        for i in 0..num_centers {
            let pivot_offset = i * dim;
            let pivot = &pivot_data[pivot_offset..pivot_offset + dim];

            // Make sure the pivot is found in the data
            let mut found = false;
            for j in 0..num_points {
                let data_offset = j * dim;
                let point = &data[data_offset..data_offset + dim];

                if pivot == point {
                    found = true;
                    break;
                }
            }
            assert!(found, "Pivot not found in data");
        }
    }

    #[test]
    fn k_meanspp_selecting_pivots_should_not_hang() {
        let test_data_path: &str = "/kmeans_test_data_file.fbin";
        let dim = 1;
        let num_points = 256;
        let num_centers = 75; // Number of unique points in this dataset
        let mut data: Vec<f32> = Vec::with_capacity(256);

        let storage_provider = VirtualStorageProvider::new_overlay(test_data_root());
        let mut reader =
            std::io::BufReader::new(storage_provider.open_reader(test_data_path).unwrap());
        for _ in 0..256 {
            let float =
                byteorder::ReadBytesExt::read_f32::<byteorder::LittleEndian>(&mut reader).unwrap();
            data.push(float);
        }
        let pool = create_thread_pool_for_test();

        // Should work with num_centers=75
        let mut pivot_data = vec![0.0; num_centers * dim];
        k_meanspp_selecting_pivots(
            &data,
            num_points,
            dim,
            &mut pivot_data,
            num_centers,
            &mut crate::utils::create_rnd_in_tests(),
            &mut (false),
            &pool,
        )
        .unwrap();

        // Should work with num_centers=75+1
        pivot_data = vec![0.0; (num_centers + 1) * dim];
        k_meanspp_selecting_pivots(
            &data,
            num_points,
            dim,
            &mut pivot_data,
            num_centers + 1,
            &mut crate::utils::create_rnd_in_tests(),
            &mut (false),
            &pool,
        )
        .unwrap();

        // Should work with num_centers=256
        pivot_data = vec![0.0; num_points * dim];
        k_meanspp_selecting_pivots(
            &data,
            num_points,
            dim,
            &mut pivot_data,
            num_points,
            &mut crate::utils::create_rnd_in_tests(),
            &mut (false),
            &pool,
        )
        .unwrap();
    }

    #[rstest]
    #[case(3, 10, 5, 5)]
    #[case(3, 15, 5, 7)]
    #[case(4, 6, 1, 1)]
    #[case(5, 6, 1, 3)]
    #[case(5, 3, 3, 6)]
    #[case(5, 3, 2, 6)]
    fn k_meanspp_all_unique_data_should_be_sampled_for_pivots(
        #[case] dim: usize,
        #[case] num_points: usize,
        #[case] num_unique_points: usize,
        #[case] num_centers: usize,
    ) {
        let unique_data: Vec<f32> = (1..=num_unique_points * dim).map(|x| x as f32).collect();
        let mut data: Vec<f32> = vec![0.0; num_points * dim];

        // Sample each unique point at least once
        for i in 0..num_unique_points {
            let unique_data_offset = i * dim;
            let data_offset = i * dim;
            data[data_offset..data_offset + dim]
                .copy_from_slice(&unique_data[unique_data_offset..unique_data_offset + dim]);
        }

        // Fill the rest with copies of randomly sampled unique points
        let mut rng = crate::utils::create_rnd_provider_from_seed_in_tests(42).create_rnd();
        for i in num_unique_points..num_points {
            let random_index = rng.random_range(0..num_unique_points);
            let data_offset = i * dim;
            let unique_data_offset = random_index * dim;
            data[data_offset..data_offset + dim]
                .copy_from_slice(&unique_data[unique_data_offset..unique_data_offset + dim]);
        }

        let mut pivot_data: Vec<f32> = vec![0.0; num_centers * dim];
        let pool = create_thread_pool_for_test();

        k_meanspp_selecting_pivots(
            &data,
            num_points,
            dim,
            &mut pivot_data,
            num_centers,
            &mut crate::utils::create_rnd_in_tests(),
            &mut (false),
            &pool,
        )
        .unwrap();

        // Verify that each unique point has been chosen as a pivot
        for i in 0..num_unique_points {
            let unique_data_offset = i * dim;
            let mut found = false;
            for j in 0..num_centers {
                let pivot_offset = j * dim;
                let pivot = &pivot_data[pivot_offset..pivot_offset + dim];
                if pivot == &unique_data[unique_data_offset..unique_data_offset + dim] {
                    found = true;
                    break;
                }
            }
            assert!(found, "Unique point not found in pivots");
        }
    }

    #[test]
    fn k_meanspp_selecting_pivots_return_err_when_too_many_input_points() {
        let dim = 2;
        let num_points = (1 << 23) + 1; // More than 8388608 (1<<23) points
        let num_centers = 3;
        let use_correct_buffer_size = true;
        let cancellation_token = &mut false;

        let expected_error_type = ANNErrorKind::KMeansError;
        let expected_error_message =
            "Number of points 8388609 is greater than 8388608, and k-means++ can not process this.";

        k_meanspp_selecting_pivots_test_error_internal(
            dim,
            num_points,
            num_centers,
            use_correct_buffer_size,
            cancellation_token,
            expected_error_type,
            expected_error_message,
        );
    }

    #[test]
    fn k_meanspp_selecting_pivots_return_err_when_mismatched_buffer_size() {
        let dim = 2;
        let num_points = 10;
        let num_centers = 3;
        let use_correct_buffer_size = false; // Buffer size is 1 less than required
        let cancellation_token = &mut false;

        let expected_error_type = ANNErrorKind::KMeansError;
        let expected_error_message =
            "Pivot data buffer should be of size num_centers * dim = 3 * 2 = 6";

        k_meanspp_selecting_pivots_test_error_internal(
            dim,
            num_points,
            num_centers,
            use_correct_buffer_size,
            cancellation_token,
            expected_error_type,
            expected_error_message,
        );
    }

    #[test]
    fn k_meanspp_selecting_pivots_return_err_when_canceled() {
        let dim = 2;
        let num_points = 10;
        let num_centers = 3;
        let use_correct_buffer_size = true;
        let cancellation_token = &mut true;

        let expected_error_type = ANNErrorKind::PQError;
        let expected_error_message = "Error: Cancellation requested by caller.";

        k_meanspp_selecting_pivots_test_error_internal(
            dim,
            num_points,
            num_centers,
            use_correct_buffer_size,
            cancellation_token,
            expected_error_type,
            expected_error_message,
        );
    }

    fn k_meanspp_selecting_pivots_test_error_internal(
        dim: usize,
        num_points: usize,
        num_centers: usize,
        use_correct_buffer_size: bool,
        cancellation_token: &mut bool,
        expected_error_type: ANNErrorKind,
        expected_error_message: &str,
    ) {
        // Generate some random data points
        let mut rng = crate::utils::create_rnd_in_tests();
        let data: Vec<f32> = (0..num_points * dim).map(|_| rng.random()).collect();

        let pivot_data_size = if use_correct_buffer_size {
            num_centers * dim
        } else {
            1
        };
        let mut pivot_data = vec![0.0; pivot_data_size];
        let pool = create_thread_pool_for_test();
        let err = k_meanspp_selecting_pivots(
            &data,
            num_points,
            dim,
            &mut pivot_data,
            num_centers,
            &mut crate::utils::create_rnd_in_tests(),
            cancellation_token,
            &pool,
        )
        .unwrap_err();

        assert_eq!(err.kind(), expected_error_type);
        assert!(err.to_string().contains(expected_error_message));
    }

    use proptest::{prelude::*, test_runner::Config};

    proptest! {
        #![proptest_config(Config {
                cases: 10,
                ..Default::default()
            })]
        #[test]
        // Property test to verify the kmeans pivot selection.
        fn k_meansspp_selection_should_work_for_pq_dim_1(data: [f32; 5]) {
            let pq_dim = 1;
            let num_points = 5;
            let num_centers = 5;
            let mut pivot_data = vec![0.0; num_centers * pq_dim];
            let pool = create_thread_pool_for_test();
            k_meanspp_selecting_pivots(&data, num_points, pq_dim, &mut pivot_data,  num_centers, &mut crate::utils::create_rnd_in_tests(), &mut (false),&pool).unwrap();
        }
    }
    proptest! {
        #![proptest_config(Config {
                cases: 10,
                ..Default::default()
            })]
        fn k_meansspp_selection_should_work_for_pq_dim_10(data: [f32; 50]) {
            let pq_dim = 10;
            let num_points = 5;
            let num_centers = 5;
            let mut pivot_data = vec![0.0; num_centers * pq_dim];
            let pool = create_thread_pool_for_test();
            k_meanspp_selecting_pivots(&data, num_points, pq_dim, &mut pivot_data, num_centers, &mut crate::utils::create_rnd_in_tests(), &mut (false),&pool).unwrap();
        }
    }
    proptest! {
        #![proptest_config(Config {
                cases: 10,
                ..Default::default()
            })]
        fn k_meansspp_selection_should_work_centers_more_than_points(data: [f32; 15]) {
            let pq_dim = 5;
            let num_points = 3;
            let num_centers = 5;
            let mut pivot_data = vec![0.0; num_centers * pq_dim];
            let pool = create_thread_pool_for_test();
            k_meanspp_selecting_pivots(&data, num_points, pq_dim, &mut pivot_data, num_centers, &mut crate::utils::create_rnd_in_tests(), &mut (false),&pool).unwrap();
        }
    }

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
}

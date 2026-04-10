/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! Mathematical utilities for processing residuals and generating random vectors

use diskann::{ANNError, ANNResult};
use num_traits::FromPrimitive;
use rand::Rng;
use rand_distr::StandardNormal;
use rayon::prelude::*;
use tracing::info;

use super::{ParallelIteratorInPool, RayonThreadPool};

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

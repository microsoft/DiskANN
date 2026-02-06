/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations)]
use std::{
    io::{Seek, SeekFrom, Write},
    mem::size_of,
    sync::atomic::AtomicBool,
    vec,
};

use crate::storage::{StorageReadProvider, StorageWriteProvider};
use diskann::{
    ANNError, ANNResult,
    error::IntoANNResult,
    utils::{VectorRepr, read_exact_into},
};
use diskann_linalg::{self, Transpose};
use diskann_quantization::{
    CompressInto,
    product::{BasicTableView, TransposedTable, train::TrainQuantizer},
};
use diskann_utils::views::{MatrixView, MutMatrixView};
use rand::{Rng, distr::Distribution};
use rayon::prelude::*;
use tracing::info;

use crate::{
    forward_threadpool,
    model::GeneratePivotArguments,
    storage::PQStorage,
    utils::{
        AsThreadPool, BridgeErr, ParallelIteratorInPool, RandomProvider, Timer,
        create_rnd_provider_from_seed, k_means_clustering, read_metadata, run_lloyds,
    },
};

/// Max size of PQ training set
pub const MAX_PQ_TRAINING_SET_SIZE: f64 = 50_000f64;

/// Number of PQ centroids for each chunk
/// Caution: The PQ logic is designed and hardcoded to work with 256 centroids, be cafeul when changing this value.
pub const NUM_PQ_CENTROIDS: usize = 256;

/// number of k-means repetitions to run for PQ
pub const NUM_KMEANS_REPS_PQ: usize = 12;

/// Maximum number of iterations of the product quantization algorithm when calculating the optimum product
/// quantization
const MAX_OPQ_ITERATIONS: usize = 20;

impl<R> diskann_quantization::random::RngBuilder<usize> for RandomProvider<R>
where
    R: Rng,
{
    type Rng = R;
    fn build_rng(&self, chunk_index: usize) -> Self::Rng {
        self.create_rnd_from_seed(chunk_index as u64)
    }
}

/// given training data in train_data of dimensions num_train * dim, generate
/// PQ pivots using k-means algorithm to partition the co-ordinates into
/// num_pq_chunks (if it divides dimension, else rounded) chunks, and runs
/// k-means in each chunk to compute the PQ pivots and stores in bin format in
/// file pq_pivots_path as a s num_centers*dim floating point binary file
/// PQ pivot table layout: {pivot offsets data: METADATA_SIZE}{pivot vector:[dim; num_centroid]}{centroid vector:[dim; 1]}{chunk offsets:[chunk_num+1; 1]}
pub fn generate_pq_pivots<Storage, Random, Pool>(
    parameters: GeneratePivotArguments,
    train_data: &mut [f32],
    pq_storage: &PQStorage,
    storage_provider: &Storage,
    random_provider: RandomProvider<Random>,
    pool: Pool,
) -> ANNResult<()>
where
    Storage: StorageWriteProvider + StorageReadProvider,
    Random: Rng,
    Pool: AsThreadPool,
{
    if pq_storage.pivot_data_exist(storage_provider) {
        let (file_num_centers, file_dim) =
            pq_storage.read_existing_pivot_metadata(storage_provider)?;
        if file_dim == parameters.dim() && file_num_centers == parameters.num_centers() {
            // PQ pivot file exists. Not generating again.
            return Ok(());
        }
    }

    let mut centroid: Vec<f32> = vec![0.0; parameters.dim()];
    if parameters.translate_to_center() {
        move_train_data_by_centroid(
            train_data,
            parameters.num_train(),
            parameters.dim(),
            &mut centroid,
        );
    }

    let mut chunk_offsets: Vec<usize> = vec![0; parameters.num_pq_chunks() + 1];
    calculate_chunk_offsets(
        parameters.dim(),
        parameters.num_pq_chunks(),
        &mut chunk_offsets,
    );

    forward_threadpool!(pool = pool);
    let trainer = diskann_quantization::product::train::LightPQTrainingParameters::new(
        parameters.num_centers(),
        parameters.max_k_means_reps(),
    );

    let full_pivot_data = pool.install(|| -> Result<Vec<f32>, ANNError> {
        let result = trainer
            .train(
                MatrixView::try_from(train_data, parameters.num_train(), parameters.dim())
                    .bridge_err()?,
                diskann_quantization::views::ChunkOffsetsView::new(chunk_offsets.as_slice())
                    .bridge_err()?,
                diskann_quantization::Parallelism::Rayon,
                &random_provider,
                &diskann_quantization::cancel::DontCancel,
            )
            .map_err(|err| ANNError::log_pq_error(diskann_quantization::error::format(&err)))?
            .flatten();
        Ok(result)
    })?;

    pq_storage.write_pivot_data(
        &full_pivot_data,
        &centroid,
        &chunk_offsets,
        parameters.num_centers(),
        parameters.dim(),
        storage_provider,
    )?;

    Ok(())
}

/// Given `train_data_slice` of dimensions num_train * dim,
/// partition the dimensions in `num_pq_chunks` chunks and generate PQ pivots
/// using k-means++ init and k-means algorithm.
/// This API doesn't involve reading/writing to disk and is used for in-memory.
///
/// If make_zero_mean is true, calculate the `centroid` of all the points and
/// subtract the centroid from each point before running k-means.
/// `centroid` is stored in the `centroid` vector which must be of size `dim`.
///
/// The `offsets` vector is used to store the start and end offsets of each chunk.
/// The size of the `offsets` vector must be `num_pq_chunks + 1`.
///
/// Result is stored in the `full_pivot_data`, which must be of size `num_centers * dim`.
#[allow(clippy::too_many_arguments)]
pub fn generate_pq_pivots_from_membuf<T: Copy + Into<f32>, Pool: AsThreadPool>(
    parameters: &GeneratePivotArguments,
    train_data_slice: &[T],
    centroid: &mut [f32],
    offsets: &mut [usize],
    full_pivot_data: &mut [f32],
    rng: &mut (impl Rng + ?Sized),
    cancellation_token: &mut bool,
    pool: Pool,
) -> ANNResult<()> {
    if full_pivot_data.len() != parameters.num_centers() * parameters.dim() {
        return Err(ANNError::log_pq_error(
            "Error: full_pivot_data size is not num_centers * dim.",
        ));
    }

    if centroid.len() != parameters.dim() {
        return Err(ANNError::log_pq_error(
            "Error: centroid size is not equal to dim.",
        ));
    }

    if offsets.len() != parameters.num_pq_chunks() + 1 {
        return Err(ANNError::log_pq_error(
            "Error: invalid offsets buffer input size.",
        ));
    }

    if *cancellation_token {
        return Err(ANNError::log_pq_error(
            "Error: Cancellation requested by caller.",
        ));
    }

    // Convert train_data to f32
    let mut train_data = train_data_slice
        .iter()
        .map(|x| (*x).into())
        .collect::<Vec<f32>>();

    // Calculate the centroid if needed and move the train_data to the centroid
    if parameters.translate_to_center() {
        move_train_data_by_centroid(
            &mut train_data,
            parameters.num_train(),
            parameters.dim(),
            centroid,
        );
    } else {
        for val in centroid.iter_mut() {
            *val = 0.0;
        }
    }

    // Calculate the chunk offsets
    calculate_chunk_offsets(parameters.dim(), parameters.num_pq_chunks(), offsets);

    forward_threadpool!(pool = pool);
    let trainer = diskann_quantization::product::train::LightPQTrainingParameters::new(
        parameters.num_centers(),
        parameters.max_k_means_reps(),
    );

    let rng_builder = create_rnd_provider_from_seed(rand::distr::StandardUniform {}.sample(rng));
    let trained = pool.install(|| -> Result<Vec<f32>, ANNError> {
        // SAFETY: The pointer for `cancellation_token` is valid for this local lifetime,
        // and we do not otherwise access `cancellation_token`.
        //
        // The particular semantics of `cancellation_token` make it useless for cancellation
        // in a pure Rust implementation, but the other side of the reference is held by
        // C++, which has no qualms about multiple mutators.
        //
        // Since this occupies only a single byte - the fallout of mixing non-atomic (C++)
        // and atomic (Rust) accesses should be minimal on x86 and Arm.
        //
        // If this type were more than 1 byte, however, we would have concerns about the
        // non-atomic side technically being allowed to tear writes.
        let atomic_bool: &AtomicBool = unsafe { AtomicBool::from_ptr(cancellation_token) };
        let cancelation = diskann_quantization::cancel::AtomicCancelation::new(atomic_bool);

        let result = trainer
            .train(
                MatrixView::try_from(
                    train_data.as_slice(),
                    parameters.num_train(),
                    parameters.dim(),
                )
                .bridge_err()?,
                diskann_quantization::views::ChunkOffsetsView::new(offsets).bridge_err()?,
                diskann_quantization::Parallelism::Rayon,
                &rng_builder,
                &cancelation,
            )
            .map_err(|err| ANNError::log_pq_error(diskann_quantization::error::format(&err)))?
            .flatten();
        Ok(result)
    })?;

    full_pivot_data.copy_from_slice(&trained);
    Ok(())
}

/// Copy centroids from this chunk to the full table.
///
/// Function extracted from existing functions so marked as inline.
#[inline]
fn copy_chunk_centroids_to_full_table(
    parameters: &GeneratePivotArguments,
    chunk_offsets: &[usize],
    full_pivot_data: &mut [f32],
    chunk_index: usize,
    chunk_size: &usize,
    cur_pivot_data: &[f32],
) {
    for center_index in 0..parameters.num_centers() {
        let current_chunk_offset = chunk_offsets[chunk_index];
        let next_chunk_offset = chunk_offsets[chunk_index + 1];
        full_pivot_data[center_index * parameters.dim() + current_chunk_offset
            ..center_index * parameters.dim() + next_chunk_offset]
            .copy_from_slice(
                &cur_pivot_data[center_index * chunk_size..(center_index + 1) * chunk_size],
            );
    }
}

/// Given `train_data`  of dimensions num_train * dim, generate
/// PQ pivots using k-means algorithm to partition the co-ordinates into
/// num_pq_chunks (if it divides dimension, else rounded) chunks, and runs
/// k-means in each chunk to compute the PQ pivots and stores in bin format in
/// file pq_pivots_path as a s num_centers*dim floating point binary file
/// PQ pivot table layout:
/// {pivot offsets data: METADATA_SIZE}
/// {pivot vector:[dim; num_centroid]}
/// {centroid vector:[dim; 1]}
/// {chunk offsets:[chunk_num+1; 1]}
#[allow(dead_code)] // keeping this for now since we may not want to delete this feature
fn generate_optimized_pq_pivots<Storage, Pool>(
    parameters: GeneratePivotArguments,
    train_data: &mut [f32],
    pq_storage: &PQStorage,
    storage_provider: &Storage,
    rng: &mut impl Rng,
    pool: Pool,
) -> ANNResult<()>
where
    Storage: StorageWriteProvider + StorageReadProvider,
    Pool: AsThreadPool,
{
    if pq_storage.pivot_data_exist(storage_provider) {
        let (file_num_centers, file_dim) =
            pq_storage.read_existing_pivot_metadata(storage_provider)?;
        if file_dim == parameters.dim() && file_num_centers == parameters.num_centers() {
            // PQ pivot file exists. Not generating again.
            return Ok(());
        }
    }

    let mut centroid: Vec<f32> = vec![0.0; parameters.dim()];
    if parameters.translate_to_center() {
        move_train_data_by_centroid(
            train_data,
            parameters.num_train(),
            parameters.dim(),
            &mut centroid,
        );
    }

    let mut chunk_offsets: Vec<usize> = vec![0; parameters.num_pq_chunks() + 1];
    calculate_chunk_offsets(
        parameters.dim(),
        parameters.num_pq_chunks(),
        &mut chunk_offsets,
    );

    // Create the initial rotation matrix as an identity matrix of size dim x dim.
    let mut rotation_matrix: Vec<f32> = vec![0.0; parameters.dim() * parameters.dim()];
    for index in 0..parameters.dim() {
        // Put 1.0 along the diagonal of the matrix
        rotation_matrix[index + (index * parameters.dim())] = 1.0;
    }

    // If we use Vector::with_capacity then the vector length is not set, so we must create the vectors
    // with zeros to make sure the vector elements are writable using indexing.
    let mut full_pivot_data: Vec<f32> = vec![0.0; parameters.num_centers() * parameters.dim()];
    let mut rotated_training_data: Vec<f32> = vec![0.0; parameters.num_train() * parameters.dim()];
    let mut quantized_data_results: Vec<f32> = vec![0.0; parameters.num_train() * parameters.dim()];
    let mut correlation_matrix: Vec<f32> = vec![0.0; parameters.dim() * parameters.dim()];

    // LAPACKE sgesdd variables
    let mut u_matrix: Vec<f32> = vec![0.0; parameters.dim() * parameters.dim()];
    let mut vt_matrix: Vec<f32> = vec![0.0; parameters.dim() * parameters.dim()];
    let mut singular_values: Vec<f32> = vec![0.0; parameters.dim()];

    forward_threadpool!(pool = pool);
    for iteration_number in 0..MAX_OPQ_ITERATIONS {
        // Transform the training data by the rotation matrix.
        diskann_linalg::sgemm(
            Transpose::None,            // Do not transpose matrix 'a'
            Transpose::None,            // Do not transpose matrix 'b'
            parameters.num_train(),     // m (number of rows in matrices 'a' and 'c')
            parameters.dim(),           // n (number of columns in matrices 'b' and 'c')
            parameters.dim(), // k (number of columns in matrix 'a', number of rows in matrix 'b')
            1.0,              // alpha (scaling factor for the product of matrices 'a' and 'b')
            train_data,       // matrix 'a'
            &rotation_matrix, // matrix 'b'
            None,             // beta (scaling factor for matrix 'c')
            &mut rotated_training_data, // matrix 'c' (result matrix)
        );

        // Quantize in the rotated space.
        opq_quantize_all_chunks(
            &parameters,
            &chunk_offsets,
            &mut full_pivot_data,
            &rotated_training_data,
            &mut quantized_data_results,
            iteration_number,
            rng,
            pool,
        )?;

        // compute the correlation matrix between the original data and the
        // quantized data to compute the new rotation matrix.
        diskann_linalg::sgemm(
            Transpose::Ordinary,     // Transpose matrix 'a' (flip rows and columns)
            Transpose::None,         // Do not transpose matrix 'b'
            parameters.dim(), // m (number of rows in matrices 'a' (after transpose) and 'ic')
            parameters.dim(), // n (number of columns in matrices 'b' and 'c')
            parameters.num_train(), // k (number of columns in matrix 'a' (after transpose), number of rows in matrix 'b')
            1.0,                    // scaling factor for the product of matrices 'a' and 'b'
            train_data,             // matrix 'a' (before transpose)
            &quantized_data_results, // matrix 'b'
            None,                   // beta (scaling factor for matrix 'c')
            &mut correlation_matrix, // matrix 'c' (result matrix)
        );

        let result = diskann_linalg::svd_into(
            parameters.dim(),        // Number of rows in `a`
            parameters.dim(),        // Number of columns in `a`.
            &mut correlation_matrix, // Matrix `a`
            &mut singular_values,    // The singular values of `a` in desceding order.
            &mut u_matrix,           // Matrix `u` (row major)
            &mut vt_matrix,          // matrix v` (column major)
        );

        result.map_err(|err| {
            ANNError::log_opq_error(format!(
                "SVD failed on iteration {} with error: {}",
                iteration_number, err
            ))
        })?;

        // Compute the new rotation matrix from the singular vectors as R^T = U * V^T
        diskann_linalg::sgemm(
            Transpose::None,      // Matrix 'a' is not transposed
            Transpose::None,      // Matrix 'b' is not transposed
            parameters.dim(),     // m (number of rows in matrices 'a' and 'c')
            parameters.dim(),     // n (number of columns in matrices 'b' and 'c')
            parameters.dim(), // k (number of columns in matrix 'a', number of rows in matrix 'b')
            1.0,              // alpha (scaling factor for the product of matrices 'a' and 'b')
            &u_matrix,        // matrix 'a'
            &vt_matrix,       // matrix 'b'
            None,             // beta (scaling factor for matrix 'c')
            &mut rotation_matrix, // matrix 'c' (result matrix)
        );
    }

    // Write the pivot data
    pq_storage.write_pivot_data(
        &full_pivot_data,
        &centroid,
        &chunk_offsets,
        parameters.num_centers(),
        parameters.dim(),
        storage_provider,
    )?;

    // Write out the rotation matrix
    pq_storage.write_rotation_matrix_data(&rotation_matrix, parameters.dim(), storage_provider)?;

    Ok(())
}

/// Determines quantization for each chunk.  Quantization is the process of mapping a set of values
/// to a smaller set of known values.
///
/// Function marked as inline because it was refactored out of larger function.  This function was
/// made for code clarity reasons and is unlikely to be reused.
#[inline]
#[allow(clippy::too_many_arguments)]
fn opq_quantize_all_chunks<Pool: AsThreadPool>(
    parameters: &GeneratePivotArguments,
    chunk_offsets: &[usize],
    full_pivot_data: &mut [f32],
    rotated_training_data: &[f32],
    quantized_data_results: &mut [f32],
    rotation_iteration_number: usize,
    rng: &mut impl Rng,
    pool: Pool,
) -> ANNResult<()> {
    forward_threadpool!(pool = pool);

    for chunk_index in 0..parameters.num_pq_chunks() {
        let chunk_end_offset = chunk_offsets[chunk_index + 1];
        let chunk_start_offset = chunk_offsets[chunk_index];
        let chunk_size = chunk_end_offset - chunk_start_offset;

        if chunk_size == 0 {
            continue;
        }

        let current_chunk_train_data = get_chunk_from_training_data(
            rotated_training_data,
            parameters.num_train(),
            parameters.dim(),
            chunk_size,
            chunk_start_offset,
        );

        // Run kmeans to get the centroids/pivots of this chunk.
        let mut cur_pivot_data: Vec<f32> = vec![0.0; parameters.num_centers() * chunk_size];

        // On first rotation, do k-means and lloyds.  On second iteration, just lloyds.
        let closest_center = if rotation_iteration_number == 0 {
            // run k_means and lloyds
            let (_closest_docs, closest_center, _residual) = k_means_clustering(
                &current_chunk_train_data,
                parameters.num_train(),
                chunk_size,
                &mut cur_pivot_data,
                parameters.num_centers(),
                parameters.max_k_means_reps(),
                rng,
                &mut (false),
                pool,
            )?;

            closest_center
        } else {
            // map the full pivot data to cur_pivot_data.  cur_pivot_data contains a list of the current chunk
            // per center so each center only has one chunk.
            for current_center in 0..parameters.num_centers() {
                let current_center_index = current_center * parameters.dim();
                let full_pivot_slice = &full_pivot_data[current_center_index + chunk_start_offset
                    ..current_center_index + chunk_end_offset];

                let current_center_start = current_center * chunk_size;
                // Find the next start.
                let current_center_end = current_center_start + chunk_size;

                // copy chunk from full_pivot_slice to cur_pivot data
                cur_pivot_data[current_center_start..current_center_end]
                    .copy_from_slice(full_pivot_slice);
            }

            let (_closest_docs, closest_center, _residual) = run_lloyds(
                &current_chunk_train_data,
                parameters.num_train(),
                chunk_size,
                &mut cur_pivot_data,
                parameters.num_centers(),
                parameters.max_k_means_reps(),
                &mut (false),
                pool,
            )?;

            closest_center
        };

        // Copy centroids from this chunk table to full table
        copy_chunk_centroids_to_full_table(
            parameters,
            chunk_offsets,
            full_pivot_data,
            chunk_index,
            &chunk_size,
            &cur_pivot_data,
        );

        // copy cur_pivot_data to the rotated train data
        for index in 0..parameters.num_train() {
            let source_slice = &cur_pivot_data[closest_center[index] as usize * chunk_size
                ..(closest_center[index] as usize + 1) * chunk_size];

            quantized_data_results[(index * parameters.dim()) + chunk_start_offset
                ..(index * parameters.dim()) + chunk_end_offset]
                .copy_from_slice(source_slice);
        }
    }
    Ok(())
}

/// Gets all instances of a chunk from the training data for all records in the training data.  Each vector in the
/// training dataset is divided into chunks and the PQ algorithm handles each vector chunk individually.  This method
/// gets the same chunk from each vector in the training data and creates a new vector out of all of them.
///
/// # Example
/// See tests for examples
#[inline]
pub fn get_chunk_from_training_data(
    train_data: &[f32],
    num_train: usize,
    raw_vector_dim: usize,
    chunk_size: usize,
    chunk_offset: usize,
) -> Vec<f32> {
    let mut result: Vec<f32> = vec![0.0; num_train * chunk_size];

    result
        // group empty result data into chunks of chunk_size
        .chunks_mut(chunk_size)
        .enumerate()
        // for each chunk, copy the chunk from the training data into the result vector
        .for_each(|(chunk_number, result_chunk)| {
            let train_data_start = chunk_number * raw_vector_dim + chunk_offset;
            let train_data_end = train_data_start + chunk_size;
            result_chunk.copy_from_slice(&train_data[train_data_start..train_data_end]);
        });
    result
}

/// Calculates the centroid if needed and moves the train_data to to the centroid
/// # Arguments
/// * `train_data` Dataset
/// * `num_points` Number of points in the dataset
/// * `dimensions` Number of dimensions in the dataset
/// * `translate_to_center` True to move the training data to the calculated centroid.  False to leave the training
///   data in its original location
/// * `centroid` An output vector of centroids, where the size is equal to the number of dimensions.
///
/// # Panics
/// Panics under the following condition:
///
/// * `train_data.len() != num_points * dimensions`.
/// * `centroid.len() != dimensions`.
#[inline]
pub fn move_train_data_by_centroid(
    train_data: &mut [f32],
    num_points: usize,
    dimensions: usize,
    centroid: &mut [f32],
) {
    assert_eq!(train_data.len(), num_points * dimensions);
    assert_eq!(centroid.len(), dimensions);

    // Calculate centroid and center of the training data
    // If we use L2 distance, there is an option to
    // translate all vectors to make them centered and
    // then compute PQ. This needs to be set to false
    // when using PQ for MIPS as such translations dont
    // preserve inner products.
    centroid.fill(0.0);
    for row in train_data.chunks_exact_mut(dimensions) {
        for (c, r) in std::iter::zip(centroid.iter_mut(), row.iter()) {
            *c += *r;
        }
    }
    centroid.iter_mut().for_each(|c| *c /= num_points as f32);

    // Remove the mean from each chunk in the input train data.
    for row in train_data.chunks_exact_mut(dimensions) {
        for (r, c) in std::iter::zip(row.iter_mut(), centroid.iter()) {
            *r -= *c;
        }
    }
}

/// Calculate the number of chunks for the product quantization algorithm.  Returns a vector of offsets where
/// each offset corresponds to a chunk based on the index of the chunk in the vector.
///
/// # Arguments
/// * `dimensions` Number of dimensions of the input data
/// * `num_pq_chunks` - Number of chunks that will be used in the PQ calculation.  Each vector will be split into these
///   number of chunks and each chunk will be compressed down to one byte.
/// * `offsets` - An output vector of offsets, where the size is equal to the number of pq chunks + 1.
#[inline]
pub fn calculate_chunk_offsets(dimensions: usize, num_pq_chunks: usize, offsets: &mut [usize]) {
    // Calculate each chunk's offset
    // If we have 8 dimension and 3 chunks then offsets would be [0,3,6,8]
    let mut chunk_offset: usize = 0;
    offsets[0] = chunk_offset;
    for chunk_index in 0..num_pq_chunks {
        chunk_offset += dimensions / num_pq_chunks;
        if chunk_index < (dimensions % num_pq_chunks) {
            chunk_offset += 1;
        }
        offsets[chunk_index + 1] = chunk_offset;
    }
}

pub fn calculate_chunk_offsets_auto(dimensions: usize, num_pq_chunks: usize) -> Vec<usize> {
    let mut offsets = vec![0; num_pq_chunks + 1];
    calculate_chunk_offsets(dimensions, num_pq_chunks, offsets.as_mut_slice());
    offsets
}

/// Add the row `y` to every row in `x`.
///
/// # Panics
///
/// Panics if `y.len() != x.ncols()`.
pub fn accum_row_inplace<T>(mut x: MutMatrixView<T>, y: &[T])
where
    T: Copy + std::ops::AddAssign,
{
    assert_eq!(x.ncols(), y.len());
    x.row_iter_mut().for_each(|row| {
        std::iter::zip(row.iter_mut(), y.iter()).for_each(|(a, b)| {
            *a += *b;
        });
    });
}

/// streams the base file (data_file), and computes the closest centers in each
/// chunk to generate the compressed data_file and stores it in
/// pq_compressed_vectors_path.
/// If the numbber of centers is < 256, it stores as byte vector, else as
/// 4-byte vector in binary format.
/// Compressed PQ table layout: {num_points: usize}{num_chunks: usize}{compressed pq table: [num_points; num_chunks]}
/// It will start from the start_vector_id and compress the data_file in chunks.
/// It validates the existing compressed data_file is consistent with the start_vector_id.
#[allow(clippy::too_many_arguments)]
pub fn generate_pq_data_from_pivots<T, Storage, Pool>(
    num_centers: usize,
    num_pq_chunks: usize,
    pq_storage: &mut PQStorage,
    storage_provider: &Storage,
    use_opq: bool,
    offset: usize,
    pool: Pool,
) -> ANNResult<()>
where
    T: Copy + VectorRepr,
    Storage: StorageWriteProvider + StorageReadProvider,
    Pool: AsThreadPool,
{
    let timer = Timer::new();

    info!("Generating PQ data starting from offset {}", offset);

    let uncompressed_data_reader =
        &mut storage_provider.open_reader(pq_storage.get_data_path()?)?;

    let mut compressed_data_writer = if offset > 0 {
        storage_provider.open_writer(pq_storage.get_compressed_data_path())?
    } else {
        storage_provider.create_for_write(pq_storage.get_compressed_data_path())?
    };

    let metadata = read_metadata(uncompressed_data_reader)?;
    let (num_points, dim) = (metadata.npoints, metadata.ndims);

    let mut full_pivot_data: Vec<f32>;
    let centroid: Vec<f32>;
    let chunk_offsets: Vec<usize>;
    let opq_rotation_matrix: Vec<f32>;
    let full_dim: usize;

    if !pq_storage.pivot_data_exist(storage_provider) {
        return Err(ANNError::log_pq_error(
            "ERROR: PQ k-means pivot file not found.",
        ));
    } else {
        (_, full_dim) = pq_storage.read_existing_pivot_metadata(storage_provider)?;
        (
            full_pivot_data,
            centroid,
            chunk_offsets,
            opq_rotation_matrix,
        ) = pq_storage.load_existing_pivot_data(
            &num_pq_chunks,
            &num_centers,
            &full_dim,
            storage_provider,
            use_opq,
        )?;
    }

    // Instead of subtracting the center from each data set component, we instead
    // add it to each center.
    //
    // This centering is only done if OPQ is not being used.
    let mut full_pivot_data_mat =
        MutMatrixView::try_from(full_pivot_data.as_mut_slice(), num_centers, full_dim)
            .bridge_err()?;
    if !use_opq {
        accum_row_inplace(full_pivot_data_mat.as_mut_view(), centroid.as_slice());
    }

    pq_storage.write_compressed_pivot_metadata::<Storage>(
        num_points,
        num_pq_chunks,
        &mut compressed_data_writer,
    )?;

    const CHUNKING_BLOCK_SIZE: usize = 10_000;

    let block_size = if num_points <= CHUNKING_BLOCK_SIZE {
        num_points
    } else {
        CHUNKING_BLOCK_SIZE
    };

    let num_points_to_compress = num_points - offset;
    let num_blocks = (num_points_to_compress / block_size)
        + !num_points_to_compress.is_multiple_of(block_size) as usize;

    uncompressed_data_reader.seek(SeekFrom::Start(
        (size_of::<i32>() * 2 + offset * dim * size_of::<T>()) as u64,
    ))?;

    // The compression table.
    let table = TransposedTable::from_parts(
        full_pivot_data_mat.as_view(),
        diskann_quantization::views::ChunkOffsetsView::new(&chunk_offsets)
            .bridge_err()?
            .to_owned(),
    )
    .map_err(|err| ANNError::log_pq_error(diskann_quantization::error::format(&err)))?;

    let mut buffer = vec![0.0; full_dim * block_size];

    forward_threadpool!(pool = pool);
    for block_index in 0..num_blocks {
        let start_index: usize = offset + block_index * block_size;
        let end_index: usize = std::cmp::min(start_index + block_size, num_points);
        let cur_block_size: usize = end_index - start_index;

        let mut block_compressed_base: Vec<u8> = vec![0; cur_block_size * num_pq_chunks];

        let block_data: Vec<T> = read_exact_into(uncompressed_data_reader, cur_block_size * dim)?;

        for (dst, src) in buffer
            .chunks_exact_mut(full_dim)
            .zip(block_data.chunks_exact(dim))
        {
            T::as_f32_into(src, dst).into_ann_result()?;
        }

        let block_data = &buffer[..cur_block_size * full_dim];

        if !use_opq {
            // We need some batch size of data to pass to `compress`. There is a balance
            // to achieve here. It must be:
            //
            // 1. Small enough to allow for parallelism across threads/tasks.
            // 2. Large enough to take advantage of cache locality in `compress`.
            //
            // A value of 128 is a somewhat arbitrary compromise, meaning each task will
            // process `BATCH_SIZE` many dataset vectors at a time.
            const BATCH_SIZE: usize = 128;

            // Wrap the data in `MatrixViews` so we do not need to manually construct view
            // in the compression loop.
            let mut compressed_block =
                MutMatrixView::try_from(&mut block_compressed_base, cur_block_size, num_pq_chunks)
                    .bridge_err()?;
            let base_block =
                MatrixView::try_from(block_data, cur_block_size, full_dim).bridge_err()?;

            base_block
                .par_window_iter(BATCH_SIZE)
                .zip_eq(compressed_block.par_window_iter_mut(BATCH_SIZE))
                .try_for_each_in_pool(pool, |(src, dst)| {
                    table.compress_into(src, dst).map_err(|err| {
                        ANNError::log_pq_error(diskann_quantization::error::format(&err))
                    })
                })?;
        } else {
            // Otherwise, we need to convert for F32 and apply the transformation.
            // NOTE: Don't remove the center in thie case because that step should
            // not be applied to OPQ.

            let mut adjusted_block_data_output: Vec<f32> = vec![0.0; cur_block_size * full_dim];

            // Rotate the current block with the trained rotation matrix before continuing with PQ
            diskann_linalg::sgemm(
                Transpose::None,                 // Do not transpose matrix 'a'
                Transpose::None,                 // Do not transpose matrix 'b'
                cur_block_size,                  // m (number of rows in matrices 'a' and 'c')
                full_dim,                        // n (number of columns in matrices 'b' and 'c')
                full_dim, // k (number of columns in matrix 'a', number of rows in matrix 'b')
                1.0,      // alpha (scaling factor for the product of matrices 'a' and 'b')
                block_data, // matrix 'a'
                &opq_rotation_matrix, // matrix 'b'
                None,     // beta (scaling factor for matrix 'c')
                &mut adjusted_block_data_output, // matrix 'c' (result matrix)
            );

            block_compressed_base
                .par_chunks_mut(num_pq_chunks)
                .zip(adjusted_block_data_output.par_chunks(full_dim))
                .try_for_each_in_pool(pool, |(dst, src)| {
                    table.compress_into(src, dst).map_err(|err| {
                        ANNError::log_pq_error(diskann_quantization::error::format(&err))
                    })
                })?;
        }

        let offset = start_index * num_pq_chunks + std::mem::size_of::<i32>() * 2;
        compressed_data_writer.seek(SeekFrom::Start(offset as u64))?;
        compressed_data_writer.write_all(&block_compressed_base)?;
    }

    info!(
        "PQ data generation took {} seconds",
        timer.elapsed().as_secs_f64()
    );

    Ok(())
}

/// Compute the PQ codes for a single vector argument.
///
/// Given training data in train_data of dimensions `dim` and
/// PQ pivots computed earlier, partition the co-ordinates into
/// `num_pq_chunks`, and find the closest pivots for each point in each chunk.
/// This API doesn't involve reading/writing to disk and is used for in-memory.
///
/// If `centroid` is `Some(_)` subtract the centroid from each point before finding the
/// closest pivots.
///
/// Output `pq_out` which must be pre-allocated and will be used to determine
/// `num_pq_chunks`.
///
/// # Arguments
/// * `vector_data` - A single vector to be encoded.
/// * `pivot_data` - A logical 2-dimensional array containing the PQ pivots in row-major
///   order.
/// * `num_pivots` - The size of the first dimension of the `pivot_data` matrix.
/// * `centroid` - An optional centroid to use for zero centering `vector_data`.
///
///   If `Some(_)`, then `vector_data` will be transformed by subtracting each component by
///   its corresponding entry in `centroid`.
///
///   If `None`, then no centering will take place.
/// * `offsets` - A prefix-sum style encoding of the start and stop positions of each
///   chunk in `pivot_data`.
/// * `pq_out` - Output buffer for the PQ codes.
///
/// # Returns
/// An `ANNResult<()>` indicating success or failure.
pub fn generate_pq_data_from_pivots_from_membuf<T: Copy + Into<f32>>(
    vector_data: &[T],
    pivot_data: &[f32],
    num_pivots: usize,
    centroid: Option<&[f32]>,
    offsets: &[usize],
    pq_out: &mut [u8],
) -> ANNResult<()> {
    // Number of dimensions in the vector to encode.
    let dim = vector_data.len();

    // Create a `BasicTableView` of the pivots.
    //
    // This does not allocate memory, but does validate the following invariants:
    // * `pivot_data.len() == num_pivots * dim`.
    // * `offsets` begins at zero, ends at `dim`, and is monotonic.
    let table = BasicTableView::new(
        MatrixView::try_from(pivot_data, num_pivots, dim).bridge_err()?,
        diskann_quantization::views::ChunkOffsetsView::new(offsets).bridge_err()?,
    )
    .map_err(|err| ANNError::log_pq_error(diskann_quantization::error::format(&err)))?;

    let mut data = vector_data
        .iter()
        .map(|x| (*x).into())
        .collect::<Vec<f32>>();

    // Validate centroid dimensionality is correct (if provided).
    // Furthermore, if the centroid is provided, use it to adjust our local copy of the
    // data.
    centroid.map_or(Ok(()), |centroid_unwrapped| -> ANNResult<()> {
        if centroid_unwrapped.len() != vector_data.len() {
            return Err(ANNError::log_pq_error(
                "Error: centroids vector size does not match dimension!",
            ));
        }
        for (dim_index, item) in data.iter_mut().enumerate() {
            *item -= centroid_unwrapped[dim_index];
        }
        Ok(())
    })?;

    table
        .compress_into(data.as_slice(), pq_out)
        .map_err(|err| ANNError::log_pq_error(diskann_quantization::error::format(&err)))
}

/// Legacy compatibility function for providing an batch data generation.
///
/// Compute the PQ codes for a single vector argument.
///
/// Given training data in train_data of dimensions `dim` and
/// PQ pivots computed earlier, partition the co-ordinates into
/// `num_pq_chunks`, and find the closest pivots for each point in each chunk.
/// This API doesn't involve reading/writing to disk and is used for in-memory.
pub fn generate_pq_data_from_pivots_from_membuf_batch<
    T: Copy + Sync + Into<f32>,
    Pool: AsThreadPool,
>(
    parameters: &GeneratePivotArguments,
    vector_data: &[T],
    pivot_data: &[f32],
    centroid: &[f32],
    offsets: &[usize],
    pq_out: &mut [u8],
    pool: Pool,
) -> ANNResult<()> {
    // Perform minimal error checking at this level, mainly on the sizes of `vector_data`
    // and `pq_out`.
    //
    // More dimentionality checking is deferred to the inner function.
    let num_train = parameters.num_train();
    let num_pq_chunks = parameters.num_pq_chunks();
    let dim = parameters.dim();

    if vector_data.len() != num_train * dim {
        return Err(ANNError::log_pq_error(
            "Error: Vector data length has the incorrect size!",
        ));
    }
    if pq_out.len() != num_train * num_pq_chunks {
        return Err(ANNError::log_pq_error(
            "Error: Invalid PQ buffer input size.",
        ));
    }
    let translate_to_center = parameters.translate_to_center();
    let centroid_option: Option<&[f32]> = translate_to_center.then_some(centroid);

    forward_threadpool!(pool = pool);

    pq_out
        .par_chunks_mut(num_pq_chunks)
        .zip(vector_data.par_chunks(dim))
        .try_for_each_in_pool(pool, |(pq_slice, vector_slice)| {
            generate_pq_data_from_pivots_from_membuf(
                vector_slice,
                pivot_data,
                parameters.num_centers(),
                centroid_option,
                offsets,
                pq_slice,
            )
        })
}

#[cfg(test)]
mod pq_test {
    use std::{f32, io::Write};

    use crate::storage::VirtualStorageProvider;
    use approx::assert_relative_eq;
    use diskann::utils::IntoUsize;
    use diskann_utils::test_data_root;
    use rand_distr::{Distribution, Uniform};
    use rstest::rstest;
    use vfs::{MemoryFS, OverlayFS};

    use super::*;
    use crate::{
        model::{
            FixedChunkPQTable,
            pq::{METADATA_SIZE, convert_types, debug},
        },
        utils::{ParallelIteratorInPool, create_thread_pool_for_test, load_bin},
    };

    #[test]
    fn test_move_train_data_by_centroid() {
        let dim = 20;
        let num_data = 200;
        let val: f32 = 1.0;

        let mut data = vec![val; dim * num_data];
        let mut centroid = vec![0.0; dim];
        move_train_data_by_centroid(&mut data, num_data, dim, &mut centroid);
        assert!(centroid.iter().all(|&i| i == val));
        assert!(data.iter().all(|&i| i == 0.0));
    }

    #[test]
    fn generate_pq_pivots_test() {
        let storage_provider = VirtualStorageProvider::new(MemoryFS::default());
        type ReaderType = <VirtualStorageProvider<MemoryFS> as StorageReadProvider>::Reader;

        let pivot_file_name = "/generate_pq_pivots_test3.bin";
        let compressed_file_name = "/compressed2.bin";
        let pq_training_file_name = "/file_not_used.bin";
        let pq_storage: PQStorage = PQStorage::new(
            pivot_file_name,
            compressed_file_name,
            Some(pq_training_file_name),
        );
        let mut train_data: Vec<f32> = vec![
            1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 2.0f32, 2.0f32, 2.0f32,
            2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.1f32, 2.1f32, 2.1f32, 2.1f32, 2.1f32, 2.1f32,
            2.1f32, 2.1f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32,
            100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32,
        ];
        let pool = create_thread_pool_for_test();
        generate_pq_pivots(
            GeneratePivotArguments::new(5, 8, 2, 2, 5, true).unwrap(),
            &mut train_data,
            &pq_storage,
            &storage_provider,
            crate::utils::create_rnd_provider_from_seed_in_tests(42),
            &pool,
        )
        .unwrap();

        let (data, nr, nc) = load_bin::<u64, ReaderType>(
            &mut storage_provider.open_reader(pivot_file_name).unwrap(),
            0,
        )
        .unwrap();
        let file_offset_data = convert_types(&data, nr * nc, |x: u64| x.into_usize());
        assert_eq!(file_offset_data[0], METADATA_SIZE);
        assert_eq!(nr, 4);
        assert_eq!(nc, 1);

        let (data, nr, nc) = load_bin::<f32, ReaderType>(
            &mut storage_provider.open_reader(pivot_file_name).unwrap(),
            file_offset_data[0],
        )
        .unwrap();

        let full_pivot_data = data.to_vec();
        assert_eq!(full_pivot_data.len(), 16);
        assert_eq!(nr, 2);
        assert_eq!(nc, 8);

        let (data, nr, nc) = load_bin::<f32, ReaderType>(
            &mut storage_provider.open_reader(pivot_file_name).unwrap(),
            file_offset_data[1],
        )
        .unwrap();
        let centroid = data.to_vec();
        assert_eq!(
            centroid[0],
            (1.0f32 + 2.0f32 + 2.1f32 + 2.2f32 + 100.0f32) / 5.0f32
        );
        assert_eq!(nr, 8);
        assert_eq!(nc, 1);

        let (data, nr, nc) = load_bin::<u32, ReaderType>(
            &mut storage_provider.open_reader(pivot_file_name).unwrap(),
            file_offset_data[2],
        )
        .unwrap();
        let chunk_offsets = convert_types(&data, nr * nc, |x: u32| x.into_usize());
        assert_eq!(chunk_offsets[0], 0);
        assert_eq!(chunk_offsets[1], 4);
        assert_eq!(chunk_offsets[2], 8);
        assert_eq!(nr, 3);
        assert_eq!(nc, 1);
    }

    #[test]
    fn generate_optimized_pq_pivots_test() {
        let storage_provider = VirtualStorageProvider::new(MemoryFS::default());
        type ReaderType = <VirtualStorageProvider<MemoryFS> as StorageReadProvider>::Reader;

        let pivot_file_name = "/generate_pq_pivots_test3.bin";
        let compressed_file_name = "/compressed2.bin";
        let pq_training_file_name = "/file_not_used.bin";
        let pq_storage: PQStorage = PQStorage::new(
            pivot_file_name,
            compressed_file_name,
            Some(pq_training_file_name),
        );
        let mut train_data: Vec<f32> = vec![
            1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 2.0f32, 2.0f32, 2.0f32,
            2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.1f32, 2.1f32, 2.1f32, 2.1f32, 2.1f32, 2.1f32,
            2.1f32, 2.1f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32,
            100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32,
        ];
        let pool = create_thread_pool_for_test();
        generate_optimized_pq_pivots(
            GeneratePivotArguments::new(5, 8, 2, 2, 5, true).unwrap(),
            &mut train_data,
            &pq_storage,
            &storage_provider,
            &mut crate::utils::create_rnd_in_tests(),
            &pool,
        )
        .unwrap();

        let (data, nr, nc) = load_bin::<u64, ReaderType>(
            &mut storage_provider.open_reader(pivot_file_name).unwrap(),
            0,
        )
        .unwrap();
        let file_offset_data = convert_types(&data, nr * nc, |x: u64| x.into_usize());
        assert_eq!(file_offset_data[0], METADATA_SIZE);
        assert_eq!(nr, 4);
        assert_eq!(nc, 1);

        let (data, nr, nc) = load_bin::<f32, ReaderType>(
            &mut storage_provider.open_reader(pivot_file_name).unwrap(),
            file_offset_data[0],
        )
        .unwrap();

        let full_pivot_data = data.to_vec();
        assert_eq!(full_pivot_data.len(), 16);
        assert_eq!(nr, 2);
        assert_eq!(nc, 8);

        let (data, nr, nc) = load_bin::<f32, ReaderType>(
            &mut storage_provider.open_reader(pivot_file_name).unwrap(),
            file_offset_data[1],
        )
        .unwrap();
        let centroid = data.to_vec();
        assert_eq!(
            centroid[0],
            (1.0f32 + 2.0f32 + 2.1f32 + 2.2f32 + 100.0f32) / 5.0f32
        );
        assert_eq!(nr, 8);
        assert_eq!(nc, 1);

        let (data, nr, nc) = load_bin::<u32, ReaderType>(
            &mut storage_provider.open_reader(pivot_file_name).unwrap(),
            file_offset_data[2],
        )
        .unwrap();
        let chunk_offsets = convert_types(&data, nr * nc, |x: u32| x.into_usize());
        assert_eq!(chunk_offsets[0], 0);
        assert_eq!(chunk_offsets[1], 4);
        assert_eq!(chunk_offsets[2], 8);
        assert_eq!(nr, 3);
        assert_eq!(nc, 1);
    }

    #[rstest]
    #[case(false, 2)]
    #[case(true, 2)]
    #[case(false, 3)]
    #[case(true, 3)]
    fn generate_pq_pivots_membuf_test(#[case] make_zero_mean: bool, #[case] num_pq_chunks: usize) {
        let num_train = 5;
        let dim = 8;
        let num_centers = 2;

        let train_data: [f32; 40] = [
            1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 2.0f32, 2.0f32, 2.0f32,
            2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.1f32, 2.1f32, 2.1f32, 2.1f32, 2.1f32, 2.1f32,
            2.1f32, 2.1f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32,
            100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32,
        ];

        let mut full_pivot_data: Vec<f32> = vec![0.0; num_centers * dim];
        let mut centroids: Vec<f32> = vec![0.0; dim];
        let mut offsets: Vec<usize> = vec![0; num_pq_chunks + 1];
        let pool = create_thread_pool_for_test();
        let result = generate_pq_pivots_from_membuf(
            &GeneratePivotArguments::new(
                num_train,
                dim,
                num_centers,
                num_pq_chunks,
                5,
                make_zero_mean,
            )
            .unwrap(),
            &train_data, // train_data
            &mut centroids,
            &mut offsets,
            &mut full_pivot_data,
            &mut crate::utils::create_rnd_in_tests(),
            &mut (false),
            &pool,
        );

        assert!(result.is_ok());
        assert_eq!(full_pivot_data.len(), 16);
    }

    #[test]
    fn read_pivot_metadata_existing_test() {
        // no real data except pivot data.
        const DATA_FILE: &str = "/test/test/fake.bin";
        const PQ_PIVOT_PATH: &str = "/test_data/sift/siftsmall_learn_pq_pivots.bin";
        const PQ_COMPRESSED_PATH: &str = "/test/test/fake.bin";

        let mut train_data = vec![0.0; 10 * 5];
        let num_train = 10;
        let dim = 128;
        let num_centers = 256;
        let num_pq_chunks = dim - 1;
        let max_k_means_reps = 10;
        let storage_provider = VirtualStorageProvider::new_overlay(test_data_root());
        let pq_storage = PQStorage::new(PQ_PIVOT_PATH, PQ_COMPRESSED_PATH, Some(DATA_FILE));
        let pool = create_thread_pool_for_test();
        let result = generate_pq_pivots(
            GeneratePivotArguments::new(
                num_train,
                dim,
                num_centers,
                num_pq_chunks,
                max_k_means_reps,
                true,
            )
            .unwrap(),
            &mut train_data,
            &pq_storage,
            &storage_provider,
            crate::utils::create_rnd_provider_from_seed_in_tests(42),
            &pool,
        );

        // still succeed without training data
        assert!(result.is_ok());
    }

    #[test]
    fn generate_pq_data_from_pivots_test() {
        let file_system = MemoryFS::new(); // Assuming you have a FileSystem struct
        let storage_provider = VirtualStorageProvider::new(file_system);
        let data_file = "/generate_pq_data_from_pivots_test_data.bin";
        //npoints=5, dim=8, 5 vectors [1.0;8] [2.0;8] [2.1;8] [2.2;8] [100.0;8]
        let mut train_data: Vec<f32> = vec![
            1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 2.0f32, 2.0f32, 2.0f32,
            2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.1f32, 2.1f32, 2.1f32, 2.1f32, 2.1f32, 2.1f32,
            2.1f32, 2.1f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32,
            100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32,
        ];
        // Casting Pod types (f32, i32) to bytes always succeeds (u8 has alignment of 1)
        let my_nums_unstructured: &[u8] = bytemuck::must_cast_slice(&train_data);
        let meta: Vec<i32> = vec![5, 8];
        let meta_unstructured: &[u8] = bytemuck::must_cast_slice(&meta);
        {
            let mut data_file_writer = storage_provider.create_for_write(data_file).unwrap();
            data_file_writer
                .write_all(meta_unstructured)
                .expect("Failed to write sample file");
            data_file_writer
                .write_all(my_nums_unstructured)
                .expect("Failed to write sample file");
        }
        let pq_pivots_path = "/generate_pq_data_from_pivots_test_pivot.bin";
        let pq_compressed_vectors_path = "/generate_pq_data_from_pivots_test.bin";
        let mut pq_storage =
            PQStorage::new(pq_pivots_path, pq_compressed_vectors_path, Some(data_file));
        let pool = create_thread_pool_for_test();
        generate_pq_pivots(
            GeneratePivotArguments::new(5, 8, 2, 2, 5, true).unwrap(),
            &mut train_data,
            &pq_storage,
            &storage_provider,
            crate::utils::create_rnd_provider_from_seed_in_tests(42),
            &pool,
        )
        .unwrap();
        generate_pq_data_from_pivots::<f32, _, _>(
            2,
            2,
            &mut pq_storage,
            &storage_provider,
            false,
            0,
            &pool,
        )
        .unwrap();
        let (data, nr, nc) = load_bin::<u8, _>(
            &mut storage_provider
                .open_reader(pq_compressed_vectors_path)
                .unwrap(),
            0,
        )
        .unwrap();
        assert_eq!(nr, 5);
        assert_eq!(nc, 2);
        assert_eq!(data[0], data[2]);
        assert_ne!(data[0], data[8]);

        storage_provider.delete(data_file).unwrap();
        storage_provider.delete(pq_pivots_path).unwrap();
        storage_provider.delete(pq_compressed_vectors_path).unwrap();
    }

    #[rstest]
    #[case(false, 2)]
    #[case(true, 2)]
    #[case(false, 3)]
    #[case(true, 3)]
    fn generate_pq_data_from_pivots_membuf_test(
        #[case] make_zero_mean: bool,
        #[case] num_pq_chunks: usize,
    ) {
        let num_train: usize = 5;
        let dim: usize = 8;
        let num_centers: usize = 2;
        let max_k_means_reps: usize = 5;

        let train_data: Vec<f32> = vec![
            1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 2.0f32, 2.0f32, 2.0f32,
            2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.1f32, 2.1f32, 2.1f32, 2.1f32, 2.1f32, 2.1f32,
            2.1f32, 2.1f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32,
            100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32,
        ];

        let mut centroids: Vec<f32> = vec![f32::MAX; dim];
        let mut offsets: Vec<usize> = vec![usize::MAX; num_pq_chunks + 1];
        let mut pivot_data: Vec<f32> = vec![f32::MAX; num_centers * dim];
        let pool = create_thread_pool_for_test();
        generate_pq_pivots_from_membuf(
            &GeneratePivotArguments::new(
                num_train,
                dim,
                num_centers,
                num_pq_chunks,
                max_k_means_reps,
                make_zero_mean,
            )
            .unwrap(),
            &train_data,
            &mut centroids,
            &mut offsets,
            &mut pivot_data,
            &mut crate::utils::create_rnd_in_tests(),
            &mut (false),
            &pool,
        )
        .unwrap();

        let mut pq: Vec<u8> = vec![0; num_pq_chunks];
        for i in 0..num_train {
            generate_pq_data_from_pivots_from_membuf(
                &train_data[dim * i..dim * (i + 1)],
                &pivot_data,
                num_centers,
                make_zero_mean.then_some(&centroids),
                &offsets,
                &mut pq,
            )
            .unwrap();
        }

        // Check if any value is equal to max
        assert!(
            !centroids.contains(&f32::MAX),
            "centroids contains max value!"
        );
        assert!(
            !offsets.contains(&usize::MAX),
            "offsets contains max value!"
        );
        assert!(
            !pivot_data.contains(&f32::MAX),
            "pivot_data contains max value!"
        );

        if !make_zero_mean {
            assert!(
                centroids.iter().all(|&x| x == 0.0),
                "centroids is not all 0"
            );
        }
    }

    #[rstest]
    #[case(true, 16)]
    #[case(true, 32)]
    #[case(true, 17)]
    #[case(true, 13)]
    fn verify_identical_results_for_membuf_api(
        #[case] make_zero_mean: bool,
        #[case] num_pq_chunks: usize,
    ) {
        // Creates a new filesystem using a read/write MemoryFS with PhysicalFS as a fall-back read-only filesystem.
        let storage_provider = VirtualStorageProvider::new_overlay(test_data_root());

        let data_file = "/test_data/sift/siftsmall_learn.bin";
        let pq_pivots_path = "/pq_pivots_validation.bin";
        let pq_compressed_vectors_path = "/pq_validation.bin";
        let mut pq_storage: PQStorage =
            PQStorage::new(pq_pivots_path, pq_compressed_vectors_path, Some(data_file));
        let pool = create_thread_pool_for_test();
        // use original function to generate pq pivots and pq data

        let (mut full_data_vector, num_train, train_dim) = pq_storage
            .get_random_train_data_slice::<f32, VirtualStorageProvider<OverlayFS>>(
                1.0,
                &storage_provider,
                &mut crate::utils::create_rnd_in_tests(),
            )
            .unwrap();

        generate_pq_pivots(
            GeneratePivotArguments::new(
                num_train,
                train_dim,
                NUM_PQ_CENTROIDS,
                num_pq_chunks,
                NUM_KMEANS_REPS_PQ,
                false,
            )
            .expect("Failed to create pivot parameters"),
            &mut full_data_vector,
            &pq_storage,
            &storage_provider,
            crate::utils::create_rnd_provider_from_seed_in_tests(42),
            &pool,
        )
        .expect("Failed to generate pivots");

        generate_pq_data_from_pivots::<f32, _, _>(
            NUM_PQ_CENTROIDS,
            num_pq_chunks,
            &mut pq_storage,
            &storage_provider,
            false,
            0,
            &pool,
        )
        .expect("Failed to generate quantized data");

        // use membuf function to generate pq

        // use pivot data generated by original function
        let (full_pivot_data, centroid, offsets, _) = pq_storage
            .load_existing_pivot_data(
                &num_pq_chunks,
                &NUM_PQ_CENTROIDS,
                &train_dim,
                &storage_provider,
                false,
            )
            .unwrap();

        let mut membuf_pq_data: Vec<u8> = vec![0; num_pq_chunks * num_train];

        // `from_membuf` switched to an implementation optimized for a single vector.
        // To accomodate comparison, we need to invoke this method for each vector in our
        // dataset.

        membuf_pq_data
            .par_chunks_mut(num_pq_chunks)
            .enumerate()
            .for_each_in_pool(&pool, |(i, membuf_slice)| {
                generate_pq_data_from_pivots_from_membuf(
                    &full_data_vector[train_dim * i..train_dim * (i + 1)],
                    &full_pivot_data,
                    NUM_PQ_CENTROIDS,
                    make_zero_mean.then_some(&centroid),
                    &offsets,
                    membuf_slice,
                )
                .unwrap();
            });

        // use pq generated by original function as the gt
        let (original_pq_data, _nr, _nc) =
            load_bin::<u8, <VirtualStorageProvider<OverlayFS> as StorageReadProvider>::Reader>(
                &mut storage_provider
                    .open_reader(pq_compressed_vectors_path)
                    .unwrap(),
                0,
            )
            .unwrap();

        let membuf_view =
            MatrixView::try_from(membuf_pq_data.as_slice(), num_train, num_pq_chunks).unwrap();

        let original_view =
            MatrixView::try_from(original_pq_data.as_slice(), num_train, num_pq_chunks).unwrap();

        // Pre-emptively construct an offset view to compare mismatched slices.
        // We want to check that the difference in the mismatched chunks is small.
        let mut offsets = vec![0; num_pq_chunks + 1];
        calculate_chunk_offsets(train_dim, num_pq_chunks, &mut offsets);
        let offset_view = diskann_quantization::views::ChunkOffsetsView::new(&offsets).unwrap();
        let full_data =
            MatrixView::try_from(full_data_vector.as_slice(), num_train, train_dim).unwrap();
        let pivot_view =
            MatrixView::try_from(full_pivot_data.as_slice(), NUM_PQ_CENTROIDS, train_dim).unwrap();

        // Due to difference in numerical rounding, the results between the two APIs can
        // vary slightly.
        //
        // In our checking routine, we do a bunch of work to determining the centers that
        // were mismatched and to compute the relative error between these two centers.
        //
        // IF the relative error between the two is small enough, then we are okay with
        // the mismatch.
        let max_relative_error = 2.05e-5;
        // The maximum number of mismatches allowed.
        let max_mismatches = 6;

        let mismatch_records = debug::compare_pq(
            full_data,
            offset_view,
            pivot_view,
            &centroid,
            membuf_view,
            original_view,
        );

        let mut max_relative_error_seen: f32 = 0.0;
        mismatch_records.iter().enumerate().for_each(|(i, r)| {
            println!("Mismatch {} of {}\n", i + 1, mismatch_records.len());
            println!("{}", r);
            let relative_error = (r.squared_l2_a - r.squared_l2_b).abs() / (r.squared_l2_b);
            println!("relative error = {relative_error}");
            max_relative_error_seen = max_relative_error_seen.max(relative_error)
        });

        assert!(
            max_relative_error_seen <= max_relative_error,
            "observed max relative error {max_relative_error_seen} exceeds the configured \
                 upper bound of {max_relative_error}"
        );
        assert!(
            mismatch_records.len() <= max_mismatches,
            "observed {} mismatches when a maximum of {} was allowed",
            mismatch_records.len(),
            max_mismatches
        );
    }

    enum RandGenStrategy {
        HundredMaxMin,
        ZeroToHundred,
        UnitSphere,
        RandDivByRand,
    }

    #[rstest]
    #[case(16, 8)]
    #[case(8, 3)]
    #[case(7, 5)]
    #[case(10, 5)]
    #[case(20, 2)]
    #[case(3, 3)]
    fn check_pq_api_for_membuf_runs_with_rand_f32_data(
        #[values(
            RandGenStrategy::HundredMaxMin,
            RandGenStrategy::ZeroToHundred,
            RandGenStrategy::UnitSphere,
            RandGenStrategy::RandDivByRand
        )]
        rand_strategy: RandGenStrategy,
        #[values(false, true)] make_zero_mean: bool,
        #[values(256)] npts: usize,
        #[case] dim: usize,
        #[case] num_pq_chunks: usize,
    ) {
        let mut rng = crate::utils::create_rnd_provider_from_seed(42).create_rnd();
        let full_data_vector: Vec<f32> = match rand_strategy {
            RandGenStrategy::HundredMaxMin => (0..npts * dim)
                .map(|_| rng.random_range(-100.0..100.0))
                .collect(),
            RandGenStrategy::ZeroToHundred => (0..npts * dim)
                .map(|_| rng.random_range(0.0..100.0))
                .collect(),
            RandGenStrategy::UnitSphere => {
                let mut data: Vec<f32> = (0..npts * dim)
                    .map(|_| rng.random_range(-100.0..100.0))
                    .collect();
                let norms: Vec<f32> = data
                    .chunks(dim)
                    .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt())
                    .collect();
                for (slice, norm) in data.chunks_mut(dim).zip(norms) {
                    for iter in slice.iter_mut() {
                        *iter /= norm;
                    }
                }
                data
            }
            RandGenStrategy::RandDivByRand => (0..npts * dim)
                .map(|_| rng.random_range(0.0..100.0) / rng.random_range(f32::EPSILON..100.0))
                .collect(),
        };

        // Generate pivot data
        let mut full_pivot_data: Vec<f32> = vec![0.0; NUM_PQ_CENTROIDS * dim];
        let mut centroids: Vec<f32> = vec![0.0; dim];
        let mut offsets: Vec<usize> = vec![0; num_pq_chunks + 1];
        let pool = create_thread_pool_for_test();
        let result = generate_pq_pivots_from_membuf(
            &GeneratePivotArguments::new(
                npts,
                dim,
                NUM_PQ_CENTROIDS,
                num_pq_chunks,
                crate::model::pq::pq_construction::NUM_KMEANS_REPS_PQ,
                make_zero_mean,
            )
            .unwrap(),
            &full_data_vector,
            &mut centroids,
            &mut offsets,
            &mut full_pivot_data,
            &mut crate::utils::create_rnd_in_tests(),
            &mut (false),
            &pool,
        );
        assert!(result.is_ok());

        let mut membuf_pq_data: Vec<u8> = vec![0; num_pq_chunks];
        for i in 0..npts {
            let result = generate_pq_data_from_pivots_from_membuf(
                &full_data_vector[(dim * i)..(dim * (i + 1))],
                &full_pivot_data,
                NUM_PQ_CENTROIDS,
                make_zero_mean.then_some(&centroids),
                &offsets,
                &mut membuf_pq_data,
            );
            assert!(result.is_ok());
        }
    }

    #[test]
    fn pq_end_to_end_validation_with_codebook_test() {
        // Creates a new filesystem using a read/write MemoryFS with PhysicalFS as a fall-back read-only filesystem.
        let storage_provider = VirtualStorageProvider::new_overlay(test_data_root());
        type ReaderType = <VirtualStorageProvider<OverlayFS> as StorageReadProvider>::Reader;

        let data_file = "/test_data/sift/siftsmall_learn.bin";
        let pq_pivots_path = "/test_data/sift/siftsmall_learn_pq_pivots.bin";
        let ground_truth_path = "/test_data/sift/siftsmall_learn_pq_compressed.bin";
        let pq_compressed_vectors_path = "/validation.bin";
        let mut pq_storage =
            PQStorage::new(pq_pivots_path, pq_compressed_vectors_path, Some(data_file));

        let pool = create_thread_pool_for_test();

        generate_pq_data_from_pivots::<f32, _, _>(
            NUM_PQ_CENTROIDS,
            1,
            &mut pq_storage,
            &storage_provider,
            false,
            0,
            &pool,
        )
        .expect("Failed to generate quantized data");

        let (data, nr, nc) = load_bin::<u8, ReaderType>(
            &mut storage_provider
                .open_reader(pq_compressed_vectors_path)
                .unwrap(),
            0,
        )
        .unwrap();
        let (gt_data, gt_nr, gt_nc) = load_bin::<u8, ReaderType>(
            &mut storage_provider.open_reader(ground_truth_path).unwrap(),
            0,
        )
        .unwrap();
        assert_eq!(nr, gt_nr);
        assert_eq!(nc, gt_nc);
        for i in 0..data.len() {
            assert_eq!(data[i], gt_data[i]);
        }
    }

    #[test]
    fn get_chunk_from_training_data_chunk0() {
        // train_data contains 2 vectors of dimension 7.  [0.0-0.6] and [1.0-1.6]
        let train_data = vec![
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        ];

        let result = get_chunk_from_training_data(
            &train_data,
            2, /* 2 vectors in dataset */
            7, /* Each vector is dimension 7 */
            3, /* Current chunk is size 3 */
            0, /* Get chunk 0 for each vector */
        );

        // 0.0, 0.1, 0.2 are the first chunk of the first vector
        // 1.0, 1.1, 1.2 are the first chunk of the second vector
        assert_eq!(result, vec!(0.0, 0.1, 0.2, 1.0, 1.1, 1.2));
    }

    #[test]
    fn get_chunk_from_training_data_chunk1() {
        // train_data contains 2 vectors of dimension 7.  [0.0-0.6] and [1.0-1.6]
        let train_data = vec![
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        ];

        let chunk_id = 1;
        let chunk_size = 3;
        let chunk_offset = chunk_size * chunk_id;

        let result = get_chunk_from_training_data(
            &train_data,
            2,            /* 2 vectors in dataset */
            7,            /* Each vector is dimension 7 */
            chunk_size,   /* Current chunk is size 3 */
            chunk_offset, /* Get chunk 1 for each vector */
        );

        // 0.0, 0.1, 0.2 are the first chunk of the first vector
        // 1.0, 1.1, 1.2 are the first chunk of the second vector
        assert_eq!(result, vec!(0.3, 0.4, 0.5, 1.3, 1.4, 1.5));
    }

    #[rstest]
    #[case(true, "l2", 16)]
    #[case(false, "l2", 16)]
    #[case(false, "inner_product", 16)]
    #[case(true, "l2", 32)]
    #[case(false, "l2", 32)]
    #[case(false, "inner_product", 32)]
    #[case(true, "l2", 31)]
    #[case(false, "l2", 31)]
    #[case(false, "inner_product", 31)]
    fn rerankingtest_with_membuf_pq_functions(
        #[case] make_zero_mean: bool,
        #[case] distance_function: String,
        #[case] num_pq_chunks: usize,
    ) {
        // Creates a new filesystem using a read/write MemoryFS with PhysicalFS as a fall-back read-only filesystem.
        let storage_provider = VirtualStorageProvider::new_overlay(test_data_root());

        let data_file = "/test_data/sift/siftsmall_learn.bin";
        let pq_pivots_path = "/pq_pivots_validation.bin";
        let pq_compressed_vectors_path = "/pq_validation.bin";
        let pq_storage: PQStorage =
            PQStorage::new(pq_pivots_path, pq_compressed_vectors_path, Some(data_file));
        let num_runs = 10;
        let num_closest_pq_vectors = 100;
        let num_closest_gt_vectors = 10;
        let p_val = 0.1;

        let (train_data_vector, train_size, train_dim) = pq_storage
            .get_random_train_data_slice::<f32, VirtualStorageProvider<OverlayFS>>(
                p_val,
                &storage_provider,
                &mut crate::utils::create_rnd_in_tests(),
            )
            .unwrap();

        // Generate pivot data
        let mut full_pivot_data: Vec<f32> = vec![0.0; NUM_PQ_CENTROIDS * train_dim];
        let mut centroid: Vec<f32> = vec![0.0; train_dim];
        let mut offsets: Vec<usize> = vec![0; num_pq_chunks + 1];
        let pivot_args = GeneratePivotArguments::new(
            train_size,
            train_dim,
            NUM_PQ_CENTROIDS,
            num_pq_chunks,
            crate::model::pq::pq_construction::NUM_KMEANS_REPS_PQ,
            make_zero_mean,
        )
        .unwrap();
        let pool = create_thread_pool_for_test();

        generate_pq_pivots_from_membuf(
            &pivot_args,
            &train_data_vector,
            &mut centroid,
            &mut offsets,
            &mut full_pivot_data,
            &mut crate::utils::create_rnd_in_tests(),
            &mut (false),
            &pool,
        )
        .unwrap();

        // use membuf function to generate pq data
        let (mut full_data_vector, train_size, train_dim) = pq_storage
            .get_random_train_data_slice::<f32, VirtualStorageProvider<OverlayFS>>(
                1.0,
                &storage_provider,
                &mut crate::utils::create_rnd_in_tests(),
            )
            .unwrap();

        // Update the pivot arguments with the full number of dataset elements.
        let pivot_args = GeneratePivotArguments::new(
            train_size,
            pivot_args.dim(),
            pivot_args.num_centers(),
            pivot_args.num_pq_chunks(),
            pivot_args.max_k_means_reps(),
            pivot_args.translate_to_center(),
        )
        .unwrap();

        let pool = create_thread_pool_for_test();
        let mut pq_data: Vec<u8> = vec![0; num_pq_chunks * train_size];
        generate_pq_data_from_pivots_from_membuf_batch(
            &pivot_args,
            &full_data_vector,
            &full_pivot_data,
            &centroid,
            &offsets,
            &mut pq_data,
            &pool,
        )
        .unwrap();

        let fixed_chunk_pq_table = FixedChunkPQTable::new(
            train_dim,
            full_pivot_data.into(),
            centroid.clone().into(),
            offsets.into(),
            None,
        )
        .unwrap();

        // Hook into here to test pairwise distances.
        let pairs = [(0, 1), (1, 0), (10, 10), (23, 42)];
        for (a, b) in pairs {
            let left = &pq_data[a * num_pq_chunks..(a + 1) * num_pq_chunks];
            let right = &pq_data[b * num_pq_chunks..(b + 1) * num_pq_chunks];

            let self_l2 = fixed_chunk_pq_table.qq_l2_distance(left, right);

            let mut inflated = fixed_chunk_pq_table.inflate_vector(left);
            fixed_chunk_pq_table.preprocess_query(&mut inflated);

            let from_inflated = fixed_chunk_pq_table.l2_distance(&inflated, right);
            assert_relative_eq!(self_l2, from_inflated, max_relative = 1e-6);
        }

        let mut rng = crate::utils::create_rnd_in_tests();
        let int_distribution = Uniform::try_from(0..train_size).unwrap();

        let mut counter_sum = vec![0; num_runs];

        // Average over num_runs runs
        for item in counter_sum.iter_mut().take(num_runs) {
            let query_index = int_distribution.sample(&mut rng);

            let mut query_vec =
                full_data_vector[train_dim * query_index..train_dim * (query_index + 1)].to_vec();
            let query = query_vec.as_mut_slice();

            if make_zero_mean {
                fixed_chunk_pq_table.preprocess_query(query);
            }

            let mut distance_map: Vec<(f32, usize)> = Vec::new();

            // Calculate the PQ distance with the PQ-compressed vectors
            for i in 0..train_size {
                if i == query_index {
                    continue;
                }
                let compressed_data = pq_data[i * num_pq_chunks..(i + 1) * num_pq_chunks].to_vec();
                match distance_function.as_str() {
                    "l2" => {
                        let distance = fixed_chunk_pq_table.l2_distance(query, &compressed_data);
                        distance_map.push((distance, i));
                    }
                    "inner_product" => {
                        let distance = fixed_chunk_pq_table.inner_product(query, &compressed_data);
                        distance_map.push((distance, i));
                    }
                    _ => panic!("Invalid distance function"),
                }
            }

            // Find the closest num_closest_pq_vectors vectors
            distance_map.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let closest_pq_vectors: Vec<usize> = distance_map
                .into_iter()
                .take(num_closest_pq_vectors)
                .map(|(_, value)| value)
                .collect();

            // Adding centroid value again because we are computing gronud truth distance with full vectors later
            if make_zero_mean {
                for i in 0..train_dim {
                    query[i] += centroid[i];
                }
            }

            // Calculate the ground truth distance with the original data vectors
            let mut gt_map: Vec<(f32, usize)> = Vec::new();
            for i in 0..train_size {
                if i == query_index {
                    continue;
                }

                let data_vector = &mut full_data_vector[i * train_dim..(i + 1) * train_dim];
                let mut distance = 0.0;
                match distance_function.as_str() {
                    "l2" => {
                        for j in 0..train_dim {
                            let diff = query[j] - data_vector[j];
                            distance += diff * diff;
                        }
                        gt_map.push((distance, i));
                    }
                    "inner_product" => {
                        for j in 0..train_dim {
                            distance += query[j] * data_vector[j];
                        }
                        gt_map.push((-distance, i));
                    }
                    _ => panic!("Invalid distance function"),
                }
            }

            // Find the closest 10 vectors
            gt_map.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let closest_gt_vectors: Vec<usize> = gt_map
                .into_iter()
                .take(num_closest_gt_vectors)
                .map(|(_, value)| value)
                .collect();

            // Find the closest 10 vectors that are also in the closest 100 PQ vectors
            let counter = closest_gt_vectors
                .iter()
                .filter(|&point| closest_pq_vectors.contains(point))
                .count();

            //counter_sum += counter;
            *item = counter;
        }

        // Calculate the average (over num_runs) recall as a percentage
        let recall_percentage: f32 = ((counter_sum.iter().sum::<usize>() as f32 / num_runs as f32)
            / num_closest_gt_vectors as f32)
            * 100.0;
        println!(
            "\n\nOriginal data dimension: {}, Number of PQ chunks: {}",
            train_dim, num_pq_chunks
        );
        println!(
            "Data file: {}, Make Zero Mean: {}, Distance function: {}, Recall: {}",
            data_file, make_zero_mean, distance_function, recall_percentage
        );
        assert!(recall_percentage > 90.0);
    }
}

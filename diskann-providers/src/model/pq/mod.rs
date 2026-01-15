/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
mod fixed_chunk_pq_table;
pub use fixed_chunk_pq_table::{
    FixedChunkPQTable, compute_pq_distance, compute_pq_distance_for_pq_coordinates,
    direct_distance_impl, pq_dist_lookup_single,
};

mod pq_construction;
pub use pq_construction::{
    MAX_PQ_TRAINING_SET_SIZE, NUM_KMEANS_REPS_PQ, NUM_PQ_CENTROIDS, accum_row_inplace,
    calculate_chunk_offsets, calculate_chunk_offsets_auto, generate_pq_data_from_pivots,
    generate_pq_data_from_pivots_from_membuf, generate_pq_data_from_pivots_from_membuf_batch,
    generate_pq_pivots, generate_pq_pivots_from_membuf, get_chunk_from_training_data,
    move_train_data_by_centroid,
};

/// all metadata of individual sub-component files is written in first 4KB for unified files
pub(crate) const METADATA_SIZE: usize = 4096;

mod pq_compressed_data;
pub use pq_compressed_data::PQCompressedData;

pub(crate) mod pq_dataset;
pub use pq_dataset::PQData;

pub mod debug;
pub mod distance;
pub mod strided;
pub mod views;

pub mod generate_pivot_arguments;
pub use generate_pivot_arguments::{GeneratePivotArguments, GeneratePivotArgumentsError};

pub mod quantizer_preprocess;
pub use quantizer_preprocess::quantizer_preprocess;

/// Convert all types within `src` using the provided closure.
pub(crate) fn convert_types<F, T, U>(src: &[T], max: usize, f: F) -> Vec<U>
where
    T: Copy,
    F: Fn(T) -> U,
{
    src.iter().copied().take(max).map(f).collect()
}

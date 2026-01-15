/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
pub mod file_util;
pub use file_util::{
    copy_aligned_data_from_file, file_exists, get_file_size, load_aligned_bin,
    load_ids_to_delete_from_file, load_metadata_from_bytes, load_metadata_from_file,
    open_file_to_write,
};

mod normalizing_util;
pub use normalizing_util::{
    normalize_data_file, normalize_data_internal, normalize_data_internal_no_cblas,
};

#[allow(clippy::module_inception)]
mod utils;
pub use utils::DatasetDto;

mod bridge_error;
pub use bridge_error::{Bridge, BridgeErr};

pub mod rayon_util;
pub use rayon_util::{
    AsThreadPool, ParallelIteratorInPool, RayonThreadPool, create_thread_pool,
    create_thread_pool_for_bench, create_thread_pool_for_test, execute_with_rayon,
};

mod timer;
pub use timer::Timer;

pub mod math_util;
pub use math_util::{
    compute_closest_centers, compute_closest_centers_in_block, compute_vec_l2sq, compute_vecs_l2sq,
    convert_usize_to_u64, generate_vectors_with_norm, process_residuals,
};

mod kmeans;
pub use kmeans::{k_means_clustering, k_meanspp_selecting_pivots, run_lloyds};

/// Read/write block size (64 MB) for cached I/O operations such as CachedReader and CachedWriter
pub const READ_WRITE_BLOCK_SIZE: u64 = 64 * 1024 * 1024;

mod generate_structured_data;
pub use generate_structured_data::{
    generate_1d_grid_adj_list, generate_1d_grid_vectors_f32, generate_1d_grid_vectors_i8,
    generate_1d_grid_vectors_u8, generate_3d_grid_vectors_f32, generate_3d_grid_vectors_i8,
    generate_3d_grid_vectors_u8, generate_4d_grid_adj_list, generate_4d_grid_vectors_f32,
    generate_4d_grid_vectors_i8, generate_4d_grid_vectors_u8, generate_circle_adj_list,
    generate_circle_vectors, generate_circle_with_various_radii_vectors,
    genererate_3d_grid_adj_list, map_ijk_to_grid, map_ijkl_to_grid,
};

pub mod random;
pub use random::{
    DEFAULT_SEED_FOR_TESTS, RandomProvider, StandardRng, create_rnd, create_rnd_from_optional_seed,
    create_rnd_from_seed, create_rnd_from_seed_in_tests, create_rnd_in_tests, create_rnd_provider,
    create_rnd_provider_from_optional_seed, create_rnd_provider_from_seed,
    create_rnd_provider_from_seed_in_tests,
};

mod vector_data_iterator;
pub use vector_data_iterator::VectorDataIterator;

pub mod generate_synthetic_labels_utils;

mod storage_utils;
pub use storage_utils::{
    Metadata, MetadataError, copy_aligned_data, load_bin, load_vector_ids, read_metadata,
    save_bin_f32, save_bin_u32, save_bin_u64, save_bytes, save_data_in_base_dimensions,
    write_metadata,
};

mod sampling;
pub use sampling::{SampleVectorReader, SamplingDensity, gen_random_slice};

mod pq_path_names;
pub use pq_path_names::PQPathNames;

mod medoid;
pub use medoid::{MAX_MEDOID_SAMPLE_SIZE, find_medoid_from_file, find_medoid_with_sampling};

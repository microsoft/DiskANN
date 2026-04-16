/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
pub mod file_util;
pub use file_util::{file_exists, load_metadata_from_file};

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
    load_vector_ids, read_bin_from, save_bytes, save_data_in_base_dimensions, write_bin_from,
};

mod sampling;
pub use sampling::{SampleVectorReader, SamplingDensity, gen_random_slice};

mod pq_path_names;
pub use pq_path_names::PQPathNames;

mod medoid;
pub use medoid::{MAX_MEDOID_SAMPLE_SIZE, find_medoid_from_file, find_medoid_with_sampling};

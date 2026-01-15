/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::{io::Read, mem::size_of, num::NonZeroUsize};

use diskann::{ANNError, ANNResult};
use diskann_providers::storage::StorageReadProvider;

pub fn get_graph_num_frozen_points(
    storage_provider: &impl StorageReadProvider,
    graph_file: &str,
) -> ANNResult<NonZeroUsize> {
    let mut file = storage_provider.open_reader(graph_file)?;
    let mut usize_buffer = [0; size_of::<usize>()];
    let mut u32_buffer = [0; size_of::<u32>()];

    file.read_exact(&mut usize_buffer)?;
    file.read_exact(&mut u32_buffer)?;
    file.read_exact(&mut u32_buffer)?;
    file.read_exact(&mut usize_buffer)?;
    let file_frozen_pts = usize::from_le_bytes(usize_buffer);

    NonZeroUsize::new(file_frozen_pts).ok_or_else(|| {
        ANNError::log_index_config_error(
            "num_frozen_pts".to_string(),
            "num_frozen_pts is zero in saved file".to_string(),
        )
    })
}

pub fn get_graph_max_observed_degree(
    storage_provider: &impl StorageReadProvider,
    graph_file: &str,
) -> ANNResult<u32> {
    let mut file = storage_provider.open_reader(graph_file)?;
    let mut usize_buffer = [0; size_of::<usize>()];
    let mut u32_buffer = [0; size_of::<u32>()];

    file.read_exact(&mut usize_buffer)?;
    file.read_exact(&mut u32_buffer)?;
    let max_observed_degree = u32::from_le_bytes(u32_buffer);

    Ok(max_observed_degree)
}

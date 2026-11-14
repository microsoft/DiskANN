/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
const DEFAULT_DATA_COMPRESSION_CHUNK_VECTOR_COUNT: usize = 25_000;

mod disk_index_build_config;
pub use disk_index_build_config::{DiskIndexBuildConfig, GraphBuildConfig};

pub mod disk_index_build_parameter;
pub use disk_index_build_parameter::{DiskIndexBuildParameters, MemoryBudget, NumPQChunks};

pub mod filter_parameter;

pub mod quantization_types;
pub use quantization_types::{QuantizationType, SphericalBits};

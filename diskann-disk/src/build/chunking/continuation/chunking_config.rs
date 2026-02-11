/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use core::fmt;

use super::continuation_tracker::{ContinuationTrackerTrait, NaiveContinuationTracker};

const PQ_DEFAULT_BATCH_SIZE: usize = 5000000;
const PQ_COMPRESSION_DEFAULT_CHUNK_SIZE: usize = 25_000;

/// Configuraton used for chunked index build.
pub struct ChunkingConfig {
    // Continuation grant provider to be used for getting continuation grants during chunk intervals.
    pub continuation_checker: Box<dyn ContinuationTrackerTrait>,

    // The size of each chunk in terms of number of vectors during pq compression.
    pub data_compression_chunk_vector_count: usize,

    // The size of each chunk in terms of number of vectors during in-memory index build.
    pub inmemory_build_chunk_vector_count: usize,
}

impl Default for ChunkingConfig {
    // Default ChunkingConfig that tries to build the entire index in one go.
    fn default() -> Self {
        ChunkingConfig {
            continuation_checker: Box::<NaiveContinuationTracker>::default(),
            data_compression_chunk_vector_count: PQ_COMPRESSION_DEFAULT_CHUNK_SIZE,
            inmemory_build_chunk_vector_count: PQ_DEFAULT_BATCH_SIZE,
        }
    }
}

impl Clone for ChunkingConfig {
    fn clone(&self) -> Self {
        ChunkingConfig {
            continuation_checker: self.continuation_checker.clone_box(),
            data_compression_chunk_vector_count: self.data_compression_chunk_vector_count,
            inmemory_build_chunk_vector_count: self.inmemory_build_chunk_vector_count,
        }
    }
}

impl fmt::Display for ChunkingConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ChunkingConfig: data_compression_chunk_vector_count: {}, inmemory_build_chunk_vector_count: {} }}",
            self.data_compression_chunk_vector_count, self.inmemory_build_chunk_vector_count
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunking_config_default() {
        let config = ChunkingConfig::default();
        assert_eq!(config.data_compression_chunk_vector_count, PQ_COMPRESSION_DEFAULT_CHUNK_SIZE);
        assert_eq!(config.inmemory_build_chunk_vector_count, PQ_DEFAULT_BATCH_SIZE);
    }

    #[test]
    fn test_chunking_config_clone() {
        let config = ChunkingConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.data_compression_chunk_vector_count, config.data_compression_chunk_vector_count);
        assert_eq!(cloned.inmemory_build_chunk_vector_count, config.inmemory_build_chunk_vector_count);
    }

    #[test]
    fn test_chunking_config_display() {
        let config = ChunkingConfig::default();
        let display_str = format!("{}", config);
        assert!(display_str.contains("ChunkingConfig"));
        assert!(display_str.contains(&PQ_COMPRESSION_DEFAULT_CHUNK_SIZE.to_string()));
        assert!(display_str.contains(&PQ_DEFAULT_BATCH_SIZE.to_string()));
    }

    #[test]
    fn test_chunking_config_custom_values() {
        let mut config = ChunkingConfig::default();
        config.data_compression_chunk_vector_count = 10000;
        config.inmemory_build_chunk_vector_count = 20000;
        
        assert_eq!(config.data_compression_chunk_vector_count, 10000);
        assert_eq!(config.inmemory_build_chunk_vector_count, 20000);
    }

    #[test]
    fn test_constants() {
        assert_eq!(PQ_DEFAULT_BATCH_SIZE, 5000000);
        assert_eq!(PQ_COMPRESSION_DEFAULT_CHUNK_SIZE, 25_000);
    }
}

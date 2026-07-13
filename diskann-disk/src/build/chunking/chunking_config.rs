/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

const PQ_COMPRESSION_DEFAULT_CHUNK_SIZE: usize = 25_000;

/// Controls the number of vectors held in memory during data compression.
#[derive(Clone, Debug)]
pub struct ChunkingConfig {
    /// The number of vectors processed per PQ compression chunk.
    pub data_compression_chunk_vector_count: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            data_compression_chunk_vector_count: PQ_COMPRESSION_DEFAULT_CHUNK_SIZE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunking_config_default() {
        let config = ChunkingConfig::default();

        assert_eq!(
            config.data_compression_chunk_vector_count,
            PQ_COMPRESSION_DEFAULT_CHUNK_SIZE
        );
    }
}

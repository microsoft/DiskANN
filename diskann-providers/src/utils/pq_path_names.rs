/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// Generate canonical path-names for saving PQ data.
#[derive(Debug, Clone)]
pub struct PQPathNames {
    pub pivots: String,
    pub compressed_data: String,
}

impl PQPathNames {
    /// Generate canonical path names from a path-prefix and number of PQ chunks.
    pub fn new(prefix: &str) -> Self {
        PQPathNames {
            pivots: format!("{}_build_pq_pivots.bin", prefix),
            compressed_data: format!("{}_build_pq_compressed.bin", prefix),
        }
    }

    pub fn for_disk_index(prefix: &str) -> Self {
        PQPathNames {
            pivots: format!("{}_pq_pivots.bin", prefix),
            compressed_data: format!("{}_pq_compressed.bin", prefix),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_path_names_new() {
        let prefix = "test_prefix";

        let pq_path_names = PQPathNames::new(prefix);

        assert_eq!(pq_path_names.pivots, "test_prefix_build_pq_pivots.bin");
        assert_eq!(
            pq_path_names.compressed_data,
            "test_prefix_build_pq_compressed.bin"
        );

        let pq_path_names = PQPathNames::for_disk_index(prefix);

        assert_eq!(pq_path_names.pivots, "test_prefix_pq_pivots.bin");
        assert_eq!(
            pq_path_names.compressed_data,
            "test_prefix_pq_compressed.bin"
        );
    }
}

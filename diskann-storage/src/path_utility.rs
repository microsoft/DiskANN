/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Canonical file-path naming conventions for DiskANN index artifacts.
//!
//! Every function in this module takes a path prefix (e.g. `"/index/my_index"`)
//! and appends the appropriate suffix for the artifact type. This ensures that
//! all components in the system agree on file naming.

/// Return the path for the in-memory graph index file.
pub fn get_mem_index_file(index_path_prefix: &str) -> String {
    format!("{index_path_prefix}_mem.index")
}

/// Return the path for the full-precision vector data file.
pub fn get_mem_index_data_file(mem_index_path: &str) -> String {
    format!("{mem_index_path}.data")
}

/// Return the path for the disk-based graph index file.
pub fn get_disk_index_file(index_path_prefix: &str) -> String {
    format!("{index_path_prefix}_disk.index")
}

/// Return the path for the PQ pivot table file.
pub fn get_pq_pivot_file(index_path_prefix: &str) -> String {
    format!("{index_path_prefix}_pq_pivots.bin")
}

/// Return the path for the PQ compressed data file.
pub fn get_compressed_pq_file(index_path_prefix: &str) -> String {
    format!("{index_path_prefix}_pq_compressed.bin")
}

/// Return the path for the disk-index PQ pivot table file.
pub fn get_disk_index_pq_pivot_file(index_path_prefix: &str) -> String {
    format!("{index_path_prefix}_disk.index_pq_pivots.bin")
}

/// Return the path for the disk-index PQ compressed data file.
pub fn get_disk_index_compressed_pq_file(index_path_prefix: &str) -> String {
    format!("{index_path_prefix}_disk.index_pq_compressed.bin")
}

/// Return the path for the label file.
pub fn get_label_file(index_path_prefix: &str) -> String {
    format!("{index_path_prefix}_labels.txt")
}

/// Return the path for the label-to-medoid mapping file.
pub fn get_label_medoids_file(index_path_prefix: &str) -> String {
    format!("{index_path_prefix}_labels_to_medoids.txt")
}

/// Return the path for the universal label file.
pub fn get_universal_label_file(index_path_prefix: &str) -> String {
    format!("{index_path_prefix}_universal_label.txt")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mem_index_file() {
        assert_eq!(get_mem_index_file("test_prefix"), "test_prefix_mem.index");
    }

    #[test]
    fn mem_index_data_file() {
        assert_eq!(get_mem_index_data_file("test_prefix"), "test_prefix.data");
    }

    #[test]
    fn disk_index_file() {
        assert_eq!(get_disk_index_file("test_prefix"), "test_prefix_disk.index");
    }

    #[test]
    fn pq_pivot_file() {
        assert_eq!(
            get_pq_pivot_file("test_prefix"),
            "test_prefix_pq_pivots.bin"
        );
    }

    #[test]
    fn compressed_pq_file() {
        assert_eq!(
            get_compressed_pq_file("test_prefix"),
            "test_prefix_pq_compressed.bin"
        );
    }

    #[test]
    fn disk_index_pq_pivot_file() {
        assert_eq!(
            get_disk_index_pq_pivot_file("test_prefix"),
            "test_prefix_disk.index_pq_pivots.bin"
        );
    }

    #[test]
    fn disk_index_compressed_pq_file() {
        assert_eq!(
            get_disk_index_compressed_pq_file("test_prefix"),
            "test_prefix_disk.index_pq_compressed.bin"
        );
    }

    #[test]
    fn label_file() {
        assert_eq!(get_label_file("test_prefix"), "test_prefix_labels.txt");
    }

    #[test]
    fn label_medoids_file() {
        assert_eq!(
            get_label_medoids_file("test_prefix"),
            "test_prefix_labels_to_medoids.txt"
        );
    }

    #[test]
    fn universal_label_file() {
        assert_eq!(
            get_universal_label_file("test_prefix"),
            "test_prefix_universal_label.txt"
        );
    }

    #[test]
    fn empty_prefix() {
        assert_eq!(get_mem_index_file(""), "_mem.index");
        assert_eq!(get_label_file(""), "_labels.txt");
    }

    #[test]
    fn prefix_with_path_separators() {
        assert_eq!(
            get_mem_index_file("/data/index/sift"),
            "/data/index/sift_mem.index"
        );
        assert_eq!(
            get_pq_pivot_file("C:\\data\\index"),
            "C:\\data\\index_pq_pivots.bin"
        );
    }
}

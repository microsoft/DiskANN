/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub fn get_mem_index_file(index_path_prefix: &str) -> String {
    index_path_prefix.to_string() + "_mem.index"
}

pub fn get_mem_index_data_file(mem_index_path: &str) -> String {
    format!("{}.data", mem_index_path)
}

pub fn get_disk_index_file(index_path_prefix: &str) -> String {
    index_path_prefix.to_string() + "_disk.index"
}

pub fn get_pq_pivot_file(index_path_prefix: &str) -> String {
    index_path_prefix.to_string() + "_pq_pivots.bin"
}

pub fn get_compressed_pq_file(index_path_prefix: &str) -> String {
    index_path_prefix.to_string() + "_pq_compressed.bin"
}

pub fn get_disk_index_pq_pivot_file(index_path_prefix: &str) -> String {
    index_path_prefix.to_string() + "_disk.index_pq_pivots.bin"
}

pub fn get_disk_index_compressed_pq_file(index_path_prefix: &str) -> String {
    index_path_prefix.to_string() + "_disk.index_pq_compressed.bin"
}

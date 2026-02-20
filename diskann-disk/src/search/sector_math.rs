/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Shared sector-layout arithmetic used by both beam search and pipelined search.

/// Compute the sector index that contains the given vertex.
///
/// The first sector (index 0) is reserved for the graph header, so data sectors
/// start at index 1.
#[inline]
pub fn node_sector_index(
    vertex_id: u32,
    num_nodes_per_sector: u64,
    num_sectors_per_node: usize,
) -> u64 {
    1 + if num_nodes_per_sector > 0 {
        vertex_id as u64 / num_nodes_per_sector
    } else {
        vertex_id as u64 * num_sectors_per_node as u64
    }
}

/// Compute the byte offset of a node within its sector.
#[inline]
pub fn node_offset_in_sector(vertex_id: u32, num_nodes_per_sector: u64, node_len: u64) -> usize {
    if num_nodes_per_sector == 0 {
        0
    } else {
        (vertex_id as u64 % num_nodes_per_sector * node_len) as usize
    }
}

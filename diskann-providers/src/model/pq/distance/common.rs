/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::utils::object_pool::{self, ObjectPool};

use crate::model::pq::fixed_chunk_pq_table::FixedChunkPQTable;

// Return the expected size of a LookupTable based on a `FixedChunkPQTable`.
pub(super) fn get_lookup_table_size(table: &FixedChunkPQTable) -> usize {
    let num_chunks = table.get_num_chunks();
    let num_centers = table.get_num_centers();
    num_chunks * num_centers
}

pub fn distance_table_pool(table: &FixedChunkPQTable) -> ObjectPool<Vec<f32>> {
    let vec_size = get_lookup_table_size(table);
    ObjectPool::new(object_pool::Undef::new(vec_size), 0, None)
}

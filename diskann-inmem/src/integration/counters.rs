/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// A snapshot of global [`Counters`](crate::counters::Counters).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CounterSnapshot {
    pub query_distance: u64,
    pub distance: u64,
    pub get_vector: u64,
    pub set_vector: u64,
    pub get_neighbors: u64,
    pub set_neighbors: u64,
    pub append_neighbors: u64,
}

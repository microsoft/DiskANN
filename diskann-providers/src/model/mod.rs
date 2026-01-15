/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub const GRAPH_SLACK_FACTOR: f64 = 1.3;

pub mod configuration;
pub use configuration::IndexConfiguration;

pub mod scratch;
pub use scratch::{
    ArcConcurrentBoxedQueue, ConcurrentQueue, FP_VECTOR_MEM_ALIGN, PQScratch, concurrent_queue,
};

pub mod pq;
pub use pq::{
    FixedChunkPQTable, GeneratePivotArguments, MAX_PQ_TRAINING_SET_SIZE, NUM_KMEANS_REPS_PQ,
    NUM_PQ_CENTROIDS, PQCompressedData, PQData, accum_row_inplace, calculate_chunk_offsets_auto,
    compute_pq_distance, compute_pq_distance_for_pq_coordinates, direct_distance_impl, distance,
    generate_pq_data_from_pivots_from_membuf, generate_pq_data_from_pivots_from_membuf_batch,
    generate_pq_pivots, generate_pq_pivots_from_membuf,
};

pub mod statistics;
pub use statistics::DegreeStats;

pub mod graph;

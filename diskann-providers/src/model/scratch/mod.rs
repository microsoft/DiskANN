/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
pub mod concurrent_queue;
pub use concurrent_queue::{ArcConcurrentBoxedQueue, ConcurrentQueue};

pub mod pq_scratch;
pub use pq_scratch::PQScratch;

pub const FP_VECTOR_MEM_ALIGN: usize = 32;

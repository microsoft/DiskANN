/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNError;
use tokio::runtime::Runtime;

pub struct VectorIdBoxSliceWrapper<T> {
    pub id: u32,
    pub value: Vec<T>,
}

pub fn init_runtime(num_threads: usize) -> std::io::Result<Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_threads)
        .build()
}

pub fn common_error(message: &str) -> ANNError {
    ANNError::log_index_error(message)
}

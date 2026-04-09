/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Product quantization types and functions used during disk-based search.

mod pq_scratch;
pub use pq_scratch::PQScratch;

pub(crate) mod pq_dataset;
pub use pq_dataset::PQData;
pub use pq_dataset::PQTable;

mod quantizer_preprocess;
pub use quantizer_preprocess::quantizer_preprocess;

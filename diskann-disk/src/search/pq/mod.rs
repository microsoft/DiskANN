/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Product quantization types and functions used during disk-based search.

mod pq_scratch;
pub(crate) use pq_scratch::{PQBatchScratch, PQQueryComputerArgs, PQQueryComputerStorage};
pub use pq_scratch::{PQQueryComputer, PQScratch};

pub(crate) use crate::storage::quant::pq::PQData;

mod quantizer_preprocess;
pub(crate) use quantizer_preprocess::prepare_query;
pub use quantizer_preprocess::quantizer_preprocess;

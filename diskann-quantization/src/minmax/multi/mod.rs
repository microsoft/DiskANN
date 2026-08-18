// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Multi-vector support for MinMax quantized vectors.

mod max_sim;
mod meta;

pub use max_sim::MinMaxKernel;
pub use meta::MinMaxMeta;

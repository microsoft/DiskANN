// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Multi-vector support for MinMax quantized vectors.

mod factory;
mod kernel;
mod max_sim;
mod meta;

pub use factory::build_minmax_max_sim;
pub use kernel::{MinMaxErase, MinMaxMaxSimKernel};
pub use max_sim::MinMaxKernel;
pub use meta::MinMaxMeta;

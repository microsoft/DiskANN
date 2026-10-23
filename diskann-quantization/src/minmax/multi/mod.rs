/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Multi-vector support for MinMax quantized vectors.

mod factory;
mod kernel;
mod max_sim;
mod meta;
mod prepared;
mod reference;

pub use factory::{MinMaxKernelFactory, build_minmax_max_sim};
pub use kernel::{MinMaxErase, MinMaxMaxSimKernel};
pub use max_sim::MinMaxKernel;
pub use meta::MinMaxMeta;

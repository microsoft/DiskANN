/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A moderately functional utility for making simple benchmarking CLI applications.

#[doc(hidden)]
extern crate self as diskann_benchmark_runner;

pub mod benchmark;
mod checker;
mod internal;
mod jobs;
mod result;

pub mod app;
pub mod files;
pub mod input;
pub mod output;
pub mod reflect;
pub mod registry;
pub mod utils;

pub use app::App;
pub use benchmark::Benchmark;
pub use checker::Checker;
pub use input::Input;
pub use output::Output;
pub use reflect::Reflect;
pub use registry::{Registry, RegistryError};
pub use result::Checkpoint;

#[cfg(any(test, feature = "test-app"))]
pub mod test;

#[cfg(any(test, feature = "ux-tools"))]
#[doc(hidden)]
pub mod ux;

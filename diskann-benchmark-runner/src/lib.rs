/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A moderately functional utility for making simple benchmarking CLI applications.

mod benchmark;
mod checker;
mod jobs;
mod result;

pub mod any;
pub mod app;
pub mod dispatcher;
pub mod files;
pub mod input;
pub mod output;
pub mod registry;
pub mod utils;

pub use any::Any;
pub use app::App;
pub use benchmark::Benchmark;
pub use checker::{CheckDeserialization, Checker};
pub use input::Input;
pub use output::Output;
pub use result::Checkpoint;

#[cfg(any(test, feature = "test-app"))]
pub mod test;

#[cfg(any(test, feature = "ux-tools"))]
#[doc(hidden)]
pub mod ux;

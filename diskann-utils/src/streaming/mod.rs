/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # Support for Streaming Operations.
//!
//! Streaming operations are defined by sequences of insertions, deletions, and replacements
//! with the goal to help study how an algorithm performs under dynamic workloads.

mod api;
pub use api::{AnyStream, Arguments, Executor, Stream};

pub mod executors;

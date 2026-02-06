/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # Support for Streaming Operations.
//!
//! Streaming operations are defined by sequences of insertions, deletions, and replacements
//! with the goal to help study how an algorithm performs under dynamic workloads.
//!
//! Unlike the components defined in [`crate::build`] and [`crate::search`], which usually define
//! a single operation to benchmark, the streaming interface is a little more free-form.
//!
//! Index algorithms should implement [`Stream`] to define how to process different operations.
//! The [`Stream`] trait is designed to be layered, with various adaptors that modify the function
//! argument types as needed.
//!
//! Execution of streaming workloads is handled by the [`Executor`] trait.
//!
//! ## Built-in Executors
//!
//! - [`executors::bigann::RunBook`] - An executor for the BigANN style runbooks.
//!
//! ## Stream Adaptors
//!
//! - [`executors::bigann::WithData`] - Adapt the raw ranges in a BigANN runbook to data points.
//!
//! ## Built-in Utilities
//!
//! - [`graph`]: Tools for working with [`diskann::graph::DiskANNIndex`].
//!   - [`graph::InplaceDelete`]: An implementation of [`crate::build::Build`] for invoking
//!     the inplace delete method. This is meant to be used in a higher-level [`Stream`] implementation.
//!   
//!   - [`graph::DropDeleted`]: A tool for cleaning up deleted neighbors after deletions.
//!     Like [`graph::InplaceDelete`], this is a building block for [`Stream`] implementations.

mod api;
pub use api::{AnyStream, Arguments, Executor, Stream};

pub mod executors;
pub mod graph;

/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # Shareable Infrastructure for Benchmarking Vector Indexing
//!
//! The purpose of this crate is to create abstractions and implementations for benchmarking
//! DiskANN vector indexing operations. We try to facilitate infrastructure that can be
//! shared across a range of [`diskann::provider::DataProvider`]s with stable APIs to enable
//!
//! * A tight benchmarking loop for developers performing performance optimization.
//! * Creating standalone binaries for CI benchmarking jobs.
//! * Shared infrastructure to facilitate developing new providers.
//!
//! # Algorithms
//!
//! - [`build`]: Tools for running parallelized index builds.
//!   - [`build::graph`]: Built-in utilities for working with [`diskann::graph::DiskANNIndex`].
//!
//! - [`search`]: Tools for running parallelized search operations.
//!   - [`search::graph`]: Built-in utilities for working with [`diskann::graph::DiskANNIndex`].
//!
//! - [`streaming`]: Tools for running streaming workloads consisting of inserts, deletes,
//!   replaces, searches, etc.
//!   - [`streaming::runbooks`]: Built-in [`streaming::Executor`]s for dynamic operations.
//!     - [`streaming::runbooks::bigann`]: BigANN style runbook support.
//!   - [`streaming::graph`]: Built-in utilities for working with [`diskann::graph::DiskANNIndex`].
//!
//! # Tools
//!
//! - [`recall`]: KNN-Recall and other accuracy measures.
//! - [`tokio`]: Quickly create new [`tokio::runtime::Runtime]`s.
//!
//! # Error Handling
//!
//! Index benchmark operations typically live high in a program's call stack and need to
//! support a wide variety of index implementations and thus error types. To that end,
//! [`anyhow::Error`] is typically used at API boundaries. While this does hide the ways
//! in which method can fail, the [`anyhow::Error`] type balances generality and fidelity.

mod internal;
pub mod utils;

// Public Utility Modules
pub mod recall;
pub mod tokio;

// Algorithms
pub mod build;
pub mod search;
pub mod streaming;
